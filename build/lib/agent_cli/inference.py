import torch
import gc
import re
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ContextLimitExceeded(Exception):
    """Custom exception raised when token or VRAM limits are hit."""
    pass

class HFInferenceEngine:
    def __init__(
        self, 
        model_id: str, 
        use_4bit: bool = True, 
        max_total_tokens: int = 10000, 
        max_vram_gb: float = 12.0  # Adjust based on your GPU (e.g., 8.0 for 8GB GPU)
    ):
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
            attn_implementation="sdpa"
        )
        self.model.eval()
        
        # Limit tracking
        self.max_total_tokens = max_total_tokens
        self.max_vram_gb = max_vram_gb
        self.total_tokens_used = 0
        
        print(f"✅ Loaded {model_id} on {self.device} | 4bit={use_4bit} | dtype={dtype}")
        print(f"🛡️ Limits set: Max Tokens={max_total_tokens}, Max VRAM={max_vram_gb}GB")

    def _check_limits(self, input_len: int, estimated_output_len: int):
        """Checks if the next generation will exceed VRAM or Token limits."""
        vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
        if vram_used_gb > self.max_vram_gb:
            raise ContextLimitExceeded(f"❌ VRAM limit exceeded: {vram_used_gb:.2f} GB > {self.max_vram_gb} GB")
        
        projected_total = self.total_tokens_used + input_len + estimated_output_len
        if projected_total > self.max_total_tokens:
            raise ContextLimitExceeded(f"❌ Token limit exceeded: Projected {projected_total} > {self.max_total_tokens}")

    def generate(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, Dict[str, int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        defaults = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2
        }
        defaults.update(gen_kwargs)
        
        # 1. Check limits BEFORE generating
        self._check_limits(input_len, defaults["max_new_tokens"])

        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **defaults)

        output_len = outputs.shape[1] - input_len
        self.total_tokens_used += input_len + output_len
        
        response_text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        
        # 2. Strip <think>...</think> blocks for clean output
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        stats = {
            "input_tokens": input_len,
            "output_tokens": output_len,
            "total_tokens": self.total_tokens_used,
            "vram_gb": torch.cuda.memory_allocated() / (1024**3)
        }
        
        return response_text, stats