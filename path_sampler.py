import argparse
import json
import os
from data import *
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

def sample_paths(max_path_len, anchor_num, fact_rdf, entity2desced, rdict, cores, output_path):
    print("Sampling training data...")
    print("Number of head relation:{}".format((rdict.__len__() - 1) // 2))
    # print("Maximum paths per head: {}".format(anchor_num))
    fact_dict = construct_fact_dict(fact_rdf)
    with open(os.path.join(output_path, "closed_rel_paths.jsonl"), "w") as f:
        for head in tqdm(rdict.rel2idx):
            paths = set()
            if head == "None" or "inv_" in head:
                continue
            # Sample anchor
            sampled_rdf = sample_anchor_rdf(fact_dict[head], num=anchor_num)
            with Pool(cores) as p:
                for path_seq in p.map(partial(search_closed_rel_paths, entity2desced=entity2desced, max_path_len=max_path_len), sampled_rdf):
                    paths = paths.union(set(path_seq))
            paths = list(paths)
            tqdm.write("Head relation: {}".format(head))
            tqdm.write("Number of paths: {}".format(len(paths)))
            tqdm.write("Saving paths...")
            json.dump({"head": head, "paths": paths}, f)
            f.write("\n")
            f.flush()


def main(args):
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Dataset(data_root=data_path, sparsity=args.sparsity, inv=True)
    rdict = dataset.get_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    entity2desced = construct_descendant(all_rdf)
    # Sample training data
    max_path_len = args.max_path_len
    n_anchor = args.anchor
    # Save paths
    output_path = os.path.join(args.output_path, args.dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sample_paths(max_path_len, n_anchor, all_rdf, entity2desced, rdict, args.cores, output_path)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='data directory')
    parser.add_argument('--dataset', type=str, default='family', help='dataset')
    parser.add_argument("--max_path_len", type=int, default=3, help="max sampled path length")
    parser.add_argument("--anchor", type=int, default=5, help="anchor facts for each relation")
    parser.add_argument("--output_path", type=str, default="sampled_path", help="output path")
    parser.add_argument("--sparsity", type=float, default=1, help="dataset sampling sparsity")
    parser.add_argument("--cores", type=int, default=5, help="dataset sampling sparsity")
    args = parser.parse_args()

    main(args)

