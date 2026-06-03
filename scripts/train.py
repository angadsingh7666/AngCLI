import os
import torch
import typer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

app = typer.Typer(help="🏗️ SFT Training for Qwen3-Coder-3B (GPU-ONLY, Optimized)")

@app.command()
def train(
    model_path: str = typer.Option("./models/Qwen3-Coder-3B", help="Base model directory or HF repo ID"),
    train_data: str = typer.Option("data/train/sft_train.jsonl", help="Training JSONL path"),
    val_data: str = typer.Option("data/val/sft_val.jsonl", help="Validation JSONL path"),
    output_dir: str = typer.Option("outputs/checkpoints", help="LoRA checkpoint save directory"),
    merge_dir: str = typer.Option("outputs/final_models", help="Merged model save directory"),
    
    # --- Training Dynamics (Optimized for 3B models) ---
    epochs: int = typer.Option(2, help="Training epochs (1-2 is usually enough for high-quality data)"),
    batch_size: int = typer.Option(2, help="Per-device batch size"),
    grad_accum: int = typer.Option(8, help="Gradient accumulation steps (Effective batch = batch_size * grad_accum * GPUs)"),
    lr: float = typer.Option(1e-4, help="Learning rate (Lowered from 2e-4 to prevent catastrophic forgetting)"),
    max_seq_len: int = typer.Option(4096, help="Max sequence length"),
    warmup_ratio: float = typer.Option(0.05, help="Warmup ratio to prevent early divergence"),
    
    # --- QLoRA / Precision ---
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Enable 4-bit QLoRA"),
    lora_r: int = typer.Option(32, help="LoRA rank (Increased to 32 for better code pattern capture)"),
    lora_alpha: int = typer.Option(64, help="LoRA alpha (Best practice: 2 * lora_r)"),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout"),
    
    # --- Evaluation & Logging ---
    eval_steps: int = typer.Option(100, help="Evaluate and save every N steps (safer than 'epoch' for small datasets)"),
    logging_steps: int = typer.Option(10, help="Log metrics every N steps"),
    report_to: str = typer.Option("none", help="Logging backend: 'none', 'wandb', or 'tensorboard'"),
):
    # 🔒 STRICT GPU ENFORCEMENT
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU required. This script has NO CPU fallback.")
    
    effective_batch_size = batch_size * grad_accum
    print(f"🚀 Starting SFT training | Model: {model_path} | QLoRA: {qlora}")
    print(f"📊 Config: Epochs={epochs} | Eff. Batch={effective_batch_size} | LR={lr} | MaxLen={max_seq_len}")

    # 1️⃣ Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2️⃣ Load Model (GPU-Only, 4-bit or FP16/BF16)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=qlora,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if qlora else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map={"": 0},  # Forces GPU 0
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "sdpa", # Speed boost
    )

    if qlora:
        model = prepare_model_for_kbit_training(model)

    # 3️⃣ Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4️⃣ Load & Format Dataset
    dataset = load_dataset("json", data_files={"train": train_data, "validation": val_data})
    
    def format_to_text(examples):
        # add_generation_prompt=False is CRITICAL for SFT: it ensures the template includes 
        # the assistant's response, allowing the model to learn to generate it.
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
        return {"text": texts}
    
    dataset = dataset.map(format_to_text, batched=True, remove_columns=["messages"])
    print(f"📈 Train samples: {len(dataset['train'])} | Val samples: {len(dataset['validation'])}")

    # 5️⃣ Trainer Configuration (Upgraded for stability)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_seq_length=max_seq_len,
        packing=True, # Packs multiple short sequences to max_seq_len for efficiency
        
        # Evaluation & Checkpointing
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,       # Automatically loads the checkpoint with the lowest eval_loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Optimization
        optim="paged_adamw_8bit" if qlora else "adamw_torch_fused", # Prevents CPU RAM spikes
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        
        # Misc
        logging_steps=logging_steps,
        report_to=report_to,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataset_text_field="text",
    )

    # 6️⃣ Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )

    print("🔥 Starting training loop...")
    train_result = trainer.train()
    metrics = train_result.metrics
    print(f"✅ Training complete | Final Train Loss: {metrics.get('train_loss', 'N/A'):.4f} | Eval Loss: {metrics.get('eval_loss', 'N/A'):.4f}")

    # Save LoRA adapters
    trainer.save_model(output_dir)
    print(f"💾 LoRA adapters saved to {output_dir}")

    # 7️⃣ Merge & Export Final Model
    print("🔄 Merging LoRA weights into base model...")
    # merge_and_unload safely combines the adapter weights with the base model
    merged_model = model.merge_and_unload()
    
    # safe_serialization=True uses the modern .safetensors format (faster, more secure)
    merged_model.save_pretrained(merge_dir, safe_serialization=True)
    tokenizer.save_pretrained(merge_dir)
    
    print(f"✅ Merged model saved to {merge_dir}")
    print("🎉 Training pipeline finished. Use this path for inference:")
    print(f"   python inference.py chat --model-path {merge_dir} --prompt 'Write a Python script to...'")

if __name__ == "__main__":
    app()