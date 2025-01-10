import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu' and torch.backends.mps.is_available():
    device = torch.device('mps')

print(f"Using device: {device}")

# Model configuration
MODEL_ID = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"

class LocalLLM:
    def __init__(self, model_id: str = MODEL_ID):
        # Initialize config with specific settings
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_id, 
            hidden_activation="gelu_pytorch_tanh", 
            token=HUGGINGFACE_API_KEY
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=HUGGINGFACE_API_KEY,
            clean_up_tokenization_spaces=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config=self.config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            token=HUGGINGFACE_API_KEY
        )
        
        self.model.to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.3) -> str:
        """
        Generate a response using the loaded model.
        
        Args:
            prompt (str): The input prompt
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            
        Returns:
            str: The generated response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=temperature,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    top_k=50,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.2,
                )
            
            output_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            answer = output_text.split("Answer:")[-1].strip()
            answer = answer.split("Answer:")[0].strip()
            
            return answer
            
        except Exception as e:
            return f"An error occurred: {str(e)}. Please try again or rephrase your question."
