from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv
dotenv.load_dotenv()

HF_TOKEN=os.getenv("HF_TOKEN")

class HfCausalModel(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path")
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=1024)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
        parser.add_argument('--quant', choices=["none", "4bit", "8bit"], default='none')
        parser.add_argument('--flash_atten_2', action='store_true', help="enable flash attention 2")
        
    def __init__(self, args):
        self.args = args
    
    def token_len(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, token=HF_TOKEN,
        trust_remote_code=True, 
        use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(self.args.model_path, device_map="auto", token=HF_TOKEN, torch_dtype=self.DTYPE.get(self.args.dtype, None), load_in_8bit = self.args.quant == "8bit", load_in_4bit = self.args.quant == "4bit", trust_remote_code=True, use_flash_attention_2 = self.args.flash_atten_2)
        self.generator = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
        self.maximun_token = self.tokenizer.model_max_length
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens, handle_long_generation="hole")
        return outputs[0]['generated_text'] # type: ignore