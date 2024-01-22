import time
import os
from openai import OpenAI
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
dotenv.load_dotenv()

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']

def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit

PROMPT = """{instruction}

{input}"""

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        parser.add_argument('--model_path', type=str, default='None')
           
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except KeyError:
            raise KeyError(f"Warning: model {self.model_name} not found.")
        return num_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
        )
        self.client = client
    
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        return query
    
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
