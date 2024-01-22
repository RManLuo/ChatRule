
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from transformers import AutoTokenizer
from .start_fastchat_api import start_fastchat_api
import dotenv
from .chatgpt import get_token_limit
import tiktoken

dotenv.load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

class LLMProxy(object):
    
    @staticmethod
    def regist_args(parser):
        parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf') # Llama-2-7b-chat-hf
        parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
        parser.add_argument("--conv_template", type=str, default="llama-2")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--disable_auto_start", action="store_true")
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        
    def __init__(self, args) -> None:
        self.args = args
        if "gpt-4" in args.model_name or "gpt-3.5" in args.model_name:
            # Load key for OpenAI API
            load_dotenv()
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.organization = os.getenv("OPENAI_ORG")
            self.maximun_token = get_token_limit(self.model_name)
        else:
            # Use local API
            if not args.disable_auto_start:
                start_fastchat_api(args.model_name, args.model_path, args.conv_template, args.host, args.port)
            openai.api_key = "EMPTY"
            openai.base_url = f"http://{args.host}:{args.port}/v1"
        self.retry = args.retry
        self.model_name = args.model_name
        
    def prepare_for_inference(self):
        client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
        )
        self.client = client
        if "gpt-4" not in self.model_name and "gpt-3.5" not in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, token=HF_TOKEN,
            trust_remote_code=True, 
            use_fast=False)
            self.maximun_token = self.tokenizer.model_max_length
    
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                num_tokens = len(encoding.encode(text))
            except KeyError:
                raise KeyError(f"Warning: model {self.model_name} not found.")
        else:
            num_tokens = len(self.tokenizer.tokenize(text))
        return num_tokens
    
    def generate_sentence(self, llm_input):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = query,
                    timeout=60,
                    temperature=0.0
                    )
                result = response.choices[0].message.content.strip() # type: ignore
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.token_len(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None