import json
import os.path
import pickle
from data import *
import re
import torch.multiprocessing as mp
import numpy as np
from scipy import sparse
from collections import defaultdict
import argparse
from utils import *
import glob
from tqdm import tqdm

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2mrr = defaultdict(list)
head2hit_5 = defaultdict(list)
head2hit_10 = defaultdict(list)
head2hit_1 = defaultdict(list)


def sortSparseMatrix(m, r, rev=True, only_indices=False):
    """ Sort a row in matrix row and return column index
    """
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res


def remove_var(r):
    """R1(A, B), R2(B, C) --> R1, R2"""
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r


def parse_rule(r):
    """parse a rule into body and head"""
    head, body = r.split(" <-- ")
    head_list = head.split("\t")
    score = [ float(s) for s in head_list[:-1] ]
    head = head_list[-1]
    body = body.split(", ")
    return score, head, body


def load_rules(rule_path, all_rules, all_heads):
    for input_filepath in glob.glob(os.path.join(rule_path, "*.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for i_, rule in enumerate(rules):
                score, head, body = parse_rule(rule.strip('\n'))
                # Skip zero support rules
                if score[0] == 0.0:
                    continue
                if head not in all_rules:
                    all_rules[head] = []
                all_rules[head].append((head, body, score))

                if head not in all_heads:
                    all_heads.append(head)

def get_gt(dataset):
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = defaultdict(list)
    all_rdf = fact_rdf + train_rdf + valid_rdf + test_rdf
    for rdf in all_rdf:
        h, r, t = parse_rdf(rdf)
        gt[(h, r)].append(ent2idx[t])
    return gt


def kg_completion(rules, dataset, args):
    """
    Input a set of rules
    Complete Querys from test_rdf based on rules and fact_rdf
    """
    # rdf_data
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    # groud truth
    gt = get_gt(dataset)
    # relation
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    rel2idx, idx2rel = rdict.rel2idx, rdict.idx2rel
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    e_num = len(idx2ent)
    # construct relation matrix (following Neural-LP)
    r2mat = construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf + train_rdf + valid_rdf)
    # Test rdf grouped by head
    test = {}
    for rdf in test_rdf:
        query = parse_rdf(rdf)
        q_h, q_r, q_t = query
        if q_r not in test:
            test[q_r] = [query]
        else:
            test[q_r].append(query)
    
    # output_folder = os.path.join(args.output_path, args.dataset, args.p)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # output_pred_path =os.path.join(output_folder, 'output_predict.pkl')
    
    mrr, hits_1, hits_5, hits_10 = [], [], [], []

    # if args.rank_only and os.path.exists(output_pred_path):
    #     with open(output_pred_path, 'rb') as f:
    #         output_pred = pickle.load(f)
    # else:
    output_pred = {}
    
    score_name_to_id = {"support": 0, "coverage": 1, "confidence": 2, "pca_confidence": 3, "none": -1}
    score_id = score_name_to_id[args.score]
    threshold_score_id = score_name_to_id[args.threshold_score]
    for head in tqdm(test.keys()):
        if not args.rank_only:
            output_pred[head] = {}
        if head not in rules:
            continue
        _rules = rules[head]
        if not args.rank_only:
            path_count = sparse.dok_matrix((e_num, e_num))
            if score_id != -1:
                sorted_rules = sorted(_rules, key=lambda x: x[2][score_id], reverse=True)
                if args.top > 0:
                    _rules = sorted_rules[:args.top]
                if args.threshold > 0:
                    _rules = [rule for rule in sorted_rules if rule[2][threshold_score_id] > args.threshold]
            for rule in _rules:
                head, body, score = rule
                if score_id != -1:
                    score = score[score_id]
                else:
                    score = 1.0
                body_adj = sparse.eye(e_num)
                for b_rel in body:
                    body_adj = body_adj * r2mat[b_rel]

                body_adj = body_adj * score
                path_count += body_adj
            
        for q_i, query_rdf in enumerate(test[head]):
            query = parse_rdf(query_rdf)
            q_h, q_r, q_t = query
            if args.debug:
                print("{}\t{}\t{}".format(q_h, q_r, q_t))
                
            if not args.rank_only:
                pred = np.squeeze(np.array(path_count[ent2idx[q_h]].todense()))
                output_pred[head][(q_h, q_r, q_t)] = pred
            else:
                pred = output_pred[head][(q_h, q_r, q_t)]
            
            # ill-rank
            if args.rank_mode == 'ill':
                rank = ill_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            elif args.rank_mode == 'harsh':
                rank = harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            elif args.rank_mode == 'balance':
                rank = balance_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            else:
                rank = random_rank(pred, gt, ent2idx, q_h, q_t, q_r)
                    
            mrr.append(1.0 / rank)
            head2mrr[q_r].append(1.0 / rank)

            hits_1.append(1 if rank <= 1 else 0)
            hits_5.append(1 if rank <= 5 else 0)
            hits_10.append(1 if rank <= 10 else 0)
            head2hit_1[q_r].append(1 if rank <= 1 else 0)
            head2hit_5[q_r].append(1 if rank <= 5 else 0)
            head2hit_10[q_r].append(1 if rank <= 10 else 0)
            if args.debug:
                print("rank at {}: {}".format(q_i, rank))
    
    # if not args.rank_only:
    #     with open(output_pred_path, 'wb') as f:
    #         pickle.dump(output_pred, f)
        
    return mrr, hits_1, hits_5, hits_10


def load_results(head):
    input_file_name = os.path.join(args.output_path, args.dataset, args.p, 'output_predict.pkl')
    with open(input_file_name, 'rb') as f:
        pred_results_dict = pickle.load(f)
    return pred_results_dict[head]


def feq(relation, fact_rdf):
    count = 0
    for rdf in fact_rdf:
        h, r, t = parse_rdf(rdf)
        if r == relation:
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='ranked_rules')
    parser.add_argument("--dataset", default="family")
    parser.add_argument('--output_path', default='pred_results', type=str, help='path to save pred results')
    parser.add_argument("-p", default='ranked_rules/family/gpt-3.5-turbo-top-0-f-5-l-3/none/all', help="rule path")
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all",
                        help="evaluate on all or only test set")
    parser.add_argument('--cpu_num', type=int, default=mp.cpu_count() // 2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rank_mode", choices=['ill', 'harsh', 'balance'], default='harsh')
    parser.add_argument("--rank_only", action="store_true")
    parser.add_argument("--threshold_score", choices=['pca_confidence', 'confidence', 'coverage', 'support'], default='support')
    parser.add_argument("--score", choices=['pca_confidence', 'confidence', 'coverage', 'support', 'none'], default='pca_confidence')
    args = parser.parse_args()
    dataset = Dataset(data_root='datasets/{}/'.format(args.dataset), inv=True)
    all_rules = {}
    all_rule_heads = []

    print("Rule path is {}".format(args.p))
    load_rules(args.p, all_rules, all_rule_heads)

    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    test_mrr, test_hits_1, test_hits_5, test_hits_10 = kg_completion(all_rules, dataset, args)

    if args.debug:
        print_msg("distribution of test query")
        for head in all_rule_heads:
            count = feq(head, test_rdf)
            print("Head: {} Count: {}".format(head, count))

        print_msg("distribution of train query")
        for head in all_rule_heads:
            count = feq(head, fact_rdf + valid_rdf + train_rdf)
            print("Head: {} Count: {}".format(head, count))

        
        all_results = {"mrr": [], "hits_1": [], "hits_5": [],"hits_10": []}
        print_msg("Stat on head and hit@1")
        for head, hits in head2hit_1.items():
            print(head, np.mean(hits))
            all_results["hits_1"].append(np.mean(hits))

        print_msg("Stat on head and hit@5")
        for head, hits in head2hit_5.items():
            print(head, np.mean(hits))
            all_results["hits_5"].append(np.mean(hits))
        
        print_msg("Stat on head and hit@10")
        for head, hits in head2hit_10.items():
            print(head, np.mean(hits))
            all_results["hits_10"].append(np.mean(hits))

        print_msg("Stat on head and mrr")
        for head, mrr in head2mrr.items():
            print(head, np.mean(mrr))
            all_results["mrr"].append(np.mean(mrr))
    dataset_name = args.dataset + ": " + args.p
    
    result_dict = {"mrr": np.mean(test_mrr), "hits_1": np.mean(test_hits_1), "hits_5": np.mean(test_hits_5), "hits_10": np.mean(test_hits_10)}
    output_dir = args.p.replace(args.input_folder, args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "result_dict.json"), 'w') as f:
            json.dump(result_dict, f)
    # print("{}: MRR: {} Hits@1: {} Hits@5: {} Hits@10: {}".format(dataset_name, np.mean(test_mrr), np.mean(test_hits_1), np.mean(test_hits_5),np.mean(test_hits_10)))
    print("MRR	Hit@1	Hit@5	Hit@10")
    print("{}	{}	{}	{}".format(np.mean(test_mrr), np.mean(test_hits_1), np.mean(test_hits_5),np.mean(test_hits_10)))