import torch
import sys

class LLMModel():
    def __init__(self, llm_model_id, 
                 llm_device):
        self.llm_model_id = llm_model_id
        self.llm_device = llm_device

    def load_model(self):
        if self.llm_device == "cuda":
            try:
                print(f"Loading quantized LLM model ({self.llm_model_id}) onto {self.llm_device}...")
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id)
                llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_id,
                    quantization_config=quantization_config,
                    device_map="auto", # Let transformers handle device mapping
                )
                llm_model.eval()
                print("Quantized LLM model loaded successfully.")
            except ImportError:
                print("Error: Loading LLM requires 'transformers' and 'bitsandbytes'. Please install them.")
                sys.exit(1)
            except Exception as e:
                print(f"Error: Failed to load LLM model: {e}")
                # Consider adding more specific error handling (e.g., for gated repo access)
                sys.exit(1)
        else:
            print("Warning: CUDA not detected or GPU not selected, LLM will run on CPU (very slow).")
            # Add CPU loading logic here if needed

        return llm_model, llm_tokenizer