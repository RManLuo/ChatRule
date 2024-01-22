import argparse
import os
from data import *
from utils import *
import re
from difflib import get_close_matches


def extract_rules(content_list):
    """ Extract the rules in the content without any explanation and the leading number if it has."""
    rule_pattern = re.compile(r".*\(X,\s?Y\) <--.*")  # The rule always has (X,Y) <--
    extracted_rules = [s.strip() for s in content_list if rule_pattern.match(s)]
    number_pattern = re.compile(r"^\d+\. ")
    cleaned_rules = [number_pattern.sub('', s) for s in extracted_rules]
    return list(set(cleaned_rules)) # Remove duplicates by converting to set and back to list


def summarize_rules_prompt(relname, k):
    """
    Generate prompt for the relation in the content_list
    """

    if k != 0:
        prompt = f'\n\nPlease identify the most important {k} rules from the following rules for the rule head: "{relname}(X,Y)". '
    else:  # k ==0
        prompt = f'\n\nPlease identify as many of the most important rules for the rule head: "{relname}(X,Y)" as possible. '

    prompt += 'You can summarize the rules that have similar meanings as one rule, if you think they are important. ' \
              'Return the rules only without any explanations. '
    return prompt


def get_valid_rules(input_filepath, output_filepath, valid_response_filepath):
    with open(input_filepath, "r") as f:
        sum_rule_list = [line.strip() for line in f]
        f.close()
    valid_prompt = ("Logical rules define the relationship between two entities: X and Y.\n"
                    "Now please analyse this relation rule path step by step to check whether it is correct. \n"
                    "If the rules is correct please write (Correct) at the end of your analysis, otherwise please write (Incorrect).\n\n")

    with open(output_filepath, "w") as f1, open(valid_response_filepath, 'w') as f2:
        for sum_rule in sum_rule_list:
            message = valid_prompt + sum_rule
            response = query(message, model="gpt-4")
            print(response)
            f2.write("Input Rule: " + sum_rule + "\n")
            f2.write("GPT-4 Response: \n" + response + '\n')
            f2.write("\n=======================================\n")
            if "incorrect" not in response.lower():
                f1.write(sum_rule + '\n')


def check_sample_times(content_list):
    """
    Determine the sample time, return True if only sample once
    """
    sample_times = 0
    for line in content_list:
        match = re.search(r'Sample \d+ time:', line)
        if match:
            sample_times += 1
    return sample_times == 1


def summarize_rule(file, args):
    """
    Summarize the rules
    """
    with open(file, 'r') as f:  # Load files
        content = f.read()
        results = re.match(r"Rule_head:\s(.*)", content)
        rel_name = results.group(1)
        # rel_name = clean_symbol_in_rel(rel_name)
    content_list = content.split('\n')
    is_sample_once = check_sample_times(content_list)
    rule_list = extract_rules(content_list)  # Extract rules and remove any explanations
    if (is_sample_once or args.model == 'none') and not args.force_summarize:  # just return the whole rule_list
        return rule_list
    else:  # Do summarization and correct the spelling error
        summarize_prompt = summarize_rules_prompt(rel_name, args.k)
        summarize_prompt_len = num_tokens_from_message(summarize_prompt, args.model)
        list_of_rule_lists = shuffle_split_path_list(rule_list, summarize_prompt_len, args.model)
        response_list = []
        for rule_list in list_of_rule_lists:
            message = '\n'.join(rule_list) + summarize_prompt
            print('prompt: ', message)
            response = query(message, model=args.model)
            response_list.extend(response.split('\n'))
        response_rules = extract_rules(response_list) # Extract rules and remove any explanations from summarized response
            
        return response_rules


def clean_rules(summarized_file_path, all_rels):
    """
    Clean error rules and remove rules with error relation.
    """
    with open(summarized_file_path, 'r') as f:
        input_rules = [line.strip() for line in f]
    cleaned_rules = list()
    # Correct spelling error/grammar error for the relation in the rules and Remove rules with error relation.
    for rule in input_rules:
        if rule == "":
            continue
        try:
            # Get rule head
            match = re.search(r'([\w\s\'-\.]+)\(X,\s?Y\)', rule)
            if not match:
                continue

            head = match.group(1).strip()
            if head not in all_rels:
                best_match = get_close_matches(head, all_rels, n=1)
                if not best_match:
                    print("Cannot correctify this rule, head not in relation: ", rule)
                    continue
                head = best_match[0].strip()

            # Get rule conditions and check if they are in the relation list
            condition_string = rule.split('<--')[1].strip()
            matches = re.findall(r"([\w\s'-\.]+)\((\w+),\s*(\w+)\)", condition_string)
            last_subject = "X"
            body_list = []
            correctyfied = True if len(matches) > 0 else False
            for match in matches:
                predicate = match[0].strip()
                subject = match[1].strip()
                object = match[2].strip()
                if predicate not in all_rels:
                    best_match = get_close_matches(predicate, all_rels, n=1)
                    if not best_match:
                        correctyfied = False
                        print(f"Cannot correctify this rule, body: {predicate} not in relaiton: ", rule)
                        break
                    predicate = best_match[0].strip()
                # Make sure the rule is in the chain-like format
                if subject == last_subject:
                    body_list.append(predicate)
                    last_subject = object
                else:
                    last_subject = subject
                    if "inv_" in predicate:
                        body_list.append(predicate.replace("inv_", ""))
                    else:
                        body_list.append(f"inv_{predicate}")

            # Add corrected rule to cleaned_rules if it's valid
            if correctyfied:
                cleaned_rules.append(f"{head} <-- {', '.join(body_list)}")

        except Exception as e:
            print(f"Processing {rule} failed.\n Error: {str(e)}")
    return cleaned_rules


def write_clean_rules_to_file(cleaned_rules, output_filepath, all_rels):
    """
    Write cleaned rules to output file in simplified format.
    """
    with open(output_filepath, "w") as output_file:
        for rule in cleaned_rules:
            try:
                match = re.search(r'([\w\s\'-\.]+)\(X,\s?Y\)', rule)  # Get rule head
                if match:
                    head = match.group(1).strip()
                    if head not in all_rels:
                        raise KeyError(f"Key {head} not found in all_rels dictionary")
                else:
                    continue

                # Get rule conditions and write to file in simplified format
                condition_string = rule.split('<--')[1].strip()
                matches = re.findall(r"([\w\s'-\.]+)\(", condition_string)
                conditions = []
                for match in matches:
                    match = match.strip()
                    if match in all_rels:
                        conditions.append(match)
                    else:
                        raise KeyError(f"Key {match} not found in all_rels dictionary")

                # Write to file
                output_file.write(f"{head} <-- {', '.join(conditions)}\n")

            except KeyError as e:
                print(f"Skipping rule {rule} due to error: {e}")
                continue


def clean(args):
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Dataset(data_root=data_path, inv=True)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())
    input_folder = os.path.join(args.rule_path, args.dataset, args.p)
    output_folder = os.path.join(args.output_path, args.dataset, args.p, args.model)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt") and "query" not in filename:
            input_filepath = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            summarized_filepath = os.path.join(output_folder, f"{name}_summarized_rules.txt")
            clean_filename = name + '_cleaned_rules.txt'
            clean_filepath = os.path.join(output_folder, clean_filename)
            
            if not args.clean_only:
                # Step 1: Summarize rules from the input file
                print("Start summarize: ", filename)
                # Summarize rules
                summarized_rules = summarize_rule(input_filepath, args)
                print("write file", summarized_filepath)
                with open(summarized_filepath, "w") as f:
                    f.write('\n'.join(summarized_rules))

            # Step 2: Clean summarized rules and keep format
            print(f"Clean file {summarized_filepath} with keeping the format")
            cleaned_rules = clean_rules(summarized_filepath, all_rels)
            
            with open(clean_filepath, "w") as f:
                f.write('\n'.join(cleaned_rules))
            

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='datasets', help='data directory')
    args.add_argument("--rule_path", default="gen_rules", type=str, help="path to rule file")
    args.add_argument("--output_path", default="clean_rules", type=str, help="path to output file")
    args.add_argument('--dataset', default='family')
    args.add_argument('--model', default='none', help='model name', choices=['none', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    args.add_argument('-p', default='gpt-3.5-turbo-top-0-f-5-l-3', help='rule prefix')
    args.add_argument('-k', type=int, default=0, help='Number of summarized rules')
    args.add_argument('--clean_only', action='store_true', help='Load summarized rules then clean rules only')
    args.add_argument('--valid_clean', action='store_true', help='gpt-4 validation for rules')
    args.add_argument('--force_summarize', action='store_true', help='force summarize rules')
    args = args.parse_args()
    clean(args)
