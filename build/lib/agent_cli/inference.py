import torch
import gc
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class HFInferenceEngine:
    def __init__(self, model_id: str, use_4bit: bool = True):
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA GPU is REQUIRED.")

        self.device = torch.device("cuda:0")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        quant_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if use_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="sdpa" # Memory-efficient attention
        )
        self.model.eval()
        print(f"✅ Loaded {model_id} on {self.device} | 4bit={use_4bit} | dtype={dtype}")

    # 🔄 CHANGED: Now returns a Tuple of (Text, Stats Dictionary)
    def generate(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, Dict[str, int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        defaults = {
            "max_new_tokens": 2048, 
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        defaults.update(gen_kwargs)

        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **defaults)

        input_len = inputs.input_ids.shape[1]
        output_len = outputs.shape[1] - input_len
        
        # Calculate Stats
        stats = {
            "input_tokens": input_len,
            "output_tokens": output_len,
            "total_tokens": input_len + output_len
        }

        response_text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return response_text, stats