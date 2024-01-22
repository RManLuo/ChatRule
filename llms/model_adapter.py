from .base_hf_causal_model import HfCausalModel
from .conv_prompt import *

class Llama(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("llama-2")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
    def prepare_for_inference(self):
        super().prepare_for_inference()
        self.maximun_token = 4096
    
    
class Mistral(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("mistral")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
    def prepare_for_inference(self):
        super().prepare_for_inference()
        self.maximun_token = 8192
    
class Qwen(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("qwen")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
class Vicuna(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()