import random
import time
# import openai
# from dotenv import load_dotenv
import tiktoken
import os
import numpy as np
import re

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")
# os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'


def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")

def camel_to_normal(camel_string):
    # 使用正则表达式将驼峰字符串转换为正常字符串
    normal_string = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_string).lower()
    return normal_string

def clean_symbol_in_rel(rel):
    '''
    clean symbol in relation

    Args:
        rel (str): relation name
    '''
    
    rel = rel.strip("_") # Remove heading
    # Replace inv_ with inverse
    # rel = rel.replace("inv_", "inverse ")
    if "/" in rel:
        if "inverse" in rel:
            rel = rel.replace("inverse ", "")
            rel = "inverse " + fb15k_rel_map[rel]
        else:
            rel = fb15k_rel_map[rel]
    # WN-18RR
    elif "_" in rel:
        rel = rel.replace("_", " ") # Replace _ with space
    # UMLS
    elif "&" in rel:
        rel = rel.replace("&", " ") # Replace & with space
    # YAGO 
    else:
        rel = camel_to_normal(rel)
    return rel

def query(message, model="gpt-4"):
    '''
    Query ChatGPT API
    :param message:
    :return:
    '''
    # Chekc if the input is too long
    maximun_token, tokenizer = get_token_limit(model)
    input_length = len(tokenizer.encode(message))
    if input_length > maximun_token:
        print(f"Input lengt {input_length} is too long. The maximum token is {maximun_token}.\n Right tuncate the input to {maximun_token} tokens.")
        message = message[:maximun_token]
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                request_timeout=180,
            )
            result = response["choices"][0]["message"]["content"].strip()
            return result
        except Exception as e:
            print(e)
            time.sleep(60)
            continue
 
       


def check_prompt_length(prompt, list_of_paths, model):
    '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
    all_paths = "\n".join(list_of_paths)
    all_tokens = prompt + all_paths
    maximun_token = model.maximun_token
    if model.token_len(all_tokens) < maximun_token:
        return all_paths
    else:
        # Shuffle the paths
        random.shuffle(list_of_paths)
        new_list_of_paths = []
        # check the length of the prompt
        for p in list_of_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            tmp_all_tokens = prompt + tmp_all_paths
            if model.token_len(tmp_all_tokens) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)

def num_tokens_from_message(path_string, model):
    """Returns the number of tokens used by a list of messages."""
    messages = [{"role": "user", "content": path_string}]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ["gpt-3.5-turbo", 'gpt-3.5-turbo-16k']:
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    elif model == "gpt-4":
        tokens_per_message = 3
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


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
    tokenizer = tiktoken.encoding_for_model(model)
    return num_tokens_limit, tokenizer


def split_path_list(path_list, token_limit, model):
    """
    Split the path list into several lists, each list can be fed into the model.
    """
    output_list = []
    current_list = []
    current_token_count = 4

    for path in path_list:
        path += '\n'
        path_token_count = num_tokens_from_message(path, model) - 4
        if current_token_count + path_token_count > token_limit:  # If the path makes the current list exceed the token limit
            output_list.append(current_list)
            current_list = [path]  # Start a new list.
            current_token_count = path_token_count + 4
        else:  # The new path fits into the current list without exceeding the limit
            current_list.append(path)  # Just add it there.
            current_token_count += path_token_count
    # Add the last list of tokens, if it's non-empty.
    if current_list:  # The last list not exceed the limit but no more paths
        output_list.append(current_list)
    return output_list


def shuffle_split_path_list(path_content_list, prompt_len, model):
    """
    First shuffle the path_content list, then split the path list into a list of several lists
    Each list can be directly fed into the model
    """
    token_limitation = get_token_limit(model)  # Get input token limitation for current model
    token_limitation -= prompt_len + 4  # minus prompt length for path length
    all_path_content = '\n'.join(path_content_list)
    token_num_all_path = num_tokens_from_message(all_path_content, model)
    random.shuffle(path_content_list)
    if token_num_all_path > token_limitation:
        list_of_paths = split_path_list(path_content_list, token_limitation, model)
    else:
        list_of_paths = [[path + '\n' for path in path_content_list]]
    return list_of_paths


def ill_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank

def harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank

def balance_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    if pred[ent2idx[q_t]]!=0:
        pred_ranks = np.argsort(pred)[::-1]    

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t!=ent2idx[q_t]]

        filtered_ranks = []
        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth:
                filtered_ranks.append(idx)

        rank = filtered_ranks.index(ent2idx[q_t])+1
    else:
        truth = gt[(q_h, q_r)]

        filtered_pred = []

        for i in range(len(pred)):
            if i not in truth:
                filtered_pred.append(pred[i])
        n_non_zero = np.count_nonzero(filtered_pred)
        rank = n_non_zero+1
    return rank

def random_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    truth.append(ent2idx[q_t])
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            if (pred[idx] == pred[ent2idx[q_t]]) and (np.random.uniform() < 0.5):
                filtered_ranks.append(idx)
            else:
                filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank
