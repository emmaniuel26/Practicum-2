"""
Model backend for Hugging Face causal language models.

This version avoids accelerate/device_map and works on CPU fallback.
If CUDA is available, it will use it. Otherwise, it stays on CPU.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME


@dataclass
class GenerationResult:
    """
    Output from one generation run.
    """
    output_text: str
    input_tokens: int
    output_tokens: int


class HFBackend:
    """
    Simple Hugging Face backend for prompt -> generation.
    """

    def __init__(self, model_name: str = MODEL_NAME, max_new_tokens: int = 128):

        # Store model settings

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Select available hardware device
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use float16 on GPU for efficiency, float32 on CPU
        
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[INFO] Loading model '{model_name}' on device: {self.device}")

        # Load tokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure tokenizer has valid padding token
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model without device_map / accelerate dependency
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
        )

        # Move model to selected device
        
        self.model.to(self.device)

        # Set model to evaluation mode
        
        self.model.eval()

    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate model output for a single prompt.
        """
        # Tokenize input prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move tokenized inputs to active device
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Record input token count
        
        input_len = int(inputs["input_ids"].shape[1])

        # Run model inference without gradient tracking
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False, # Deterministic generation
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Full generated token sequence
        
        total_tokens = outputs[0]

        # Calculate newly generated output tokens
        
        output_len = int(total_tokens.shape[0] - input_len)

        # Decode generated text only
        
        decoded = self.tokenizer.decode(
            total_tokens[input_len:],
            skip_special_tokens=True
        )

        # Return structured generation result
        
        return GenerationResult(
            output_text=decoded.strip(),
            input_tokens=input_len,
            output_tokens=output_len,
        )
