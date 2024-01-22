import copy
import random
from random import sample
import networkx as nx
from scipy import sparse

def construct_nx(idx2rel, idx2ent, ent2idx, fact_rdf):
    G = nx.Graph()
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        h_idx, t_idx = ent2idx[h], ent2idx[t]
        G.add_edge(h_idx, t_idx, relation=r)
    return G

def construct_fact_dict(fact_rdf):
    fact_dict = {}
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        if r not in fact_dict:
            fact_dict[r] = []
        fact_dict[r].append(rdf)

    return fact_dict


def construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf):
    e_num = len(idx2ent)
    r2mat = {}
    # initialize rmat
    for idx, rel in idx2rel.items():
        mat = sparse.dok_matrix((e_num, e_num))
        r2mat[rel] = mat
    # fill rmat
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        h_idx, t_idx = ent2idx[h], ent2idx[t]
        r2mat[r][h_idx, t_idx] = 1
    return r2mat

class RuleDataset(object):
    def __init__(self, r2mat, rules, e_num, idx2rel, args):
        self.e_num = e_num
        self.r2mat = r2mat
        self.rules = rules
        self.idx2rel = idx2rel
        self.len = len(self.rules)
        self.args = args

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rel = self.idx2rel[idx]
        _rules = self.rules[rel]
        path_count = sparse.dok_matrix((self.e_num, self.e_num))
        for rule in _rules:
            head, body, conf_1, conf_2 = rule

            body_adj = sparse.eye(self.e_num)
            for b_rel in body:
                body_adj = body_adj * self.r2mat[b_rel]

            body_adj = body_adj * conf_1
            path_count += body_adj

        return rel, path_count

    @staticmethod
    def collate_fn(data):
        head = [_[0] for _ in data]
        path_count = [_[1] for _ in data]
        return head, path_count

def parse_rdf(rdf):
    """
        return: head, relation, tail
    """
    return rdf
    # rdf_tail, rdf_rel, rdf_head = rdf
    # return rdf_head, rdf_rel, rdf_tail


class Dictionary(object):
    def __init__(self):
        self.rel2idx_ = {}
        self.idx2rel_ = {}
        self.idx = 0

    def add_relation(self, rel):
        if rel not in self.rel2idx_.keys():
            self.rel2idx_[rel] = self.idx
            self.idx2rel_[self.idx] = rel
            self.idx += 1

    @property
    def rel2idx(self):
        return self.rel2idx_

    @property
    def idx2rel(self):
        return self.idx2rel_

    def __len__(self):
        return len(self.idx2rel_)


def load_entities(path):
    idx2ent, ent2idx = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            e = line.strip()
            ent2idx[e] = idx
            idx2ent[idx] = e
    return idx2ent, ent2idx


class Dataset(object):
    def __init__(self, data_root, sparsity=1, inv=False):
        # Construct entity_list
        entity_path = data_root + 'entities.txt'
        self.idx2ent_, self.ent2idx_ = load_entities(entity_path)
        # Construct rdict which contains relation2idx & idx2relation2
        relation_path = data_root + 'relations.txt'
        self.rdict = Dictionary()
        self.load_relation_dict(relation_path)
        # head relation
        self.head_rdict = Dictionary()
        self.head_rdict = copy.deepcopy(self.rdict)
        # load (h, r, t) tuples
        fact_path = data_root + 'facts.txt'
        train_path = data_root + 'train.txt'
        valid_path = data_root + 'valid.txt'
        test_path = data_root + 'test.txt'
        if inv:
            fact_path += '.inv'
        self.rdf_data_ = self.load_data_(fact_path, train_path, valid_path, test_path, sparsity)
        self.fact_rdf_, self.train_rdf_, self.valid_rdf_, self.test_rdf_ = self.rdf_data_
        # inverse
        if inv:
            # add inverse relation to rdict
            rel_list = list(self.rdict.rel2idx_.keys())
            for rel in rel_list:
                inv_rel = "inv_" + rel
                self.rdict.add_relation(inv_rel)
                self.head_rdict.add_relation(inv_rel)
                # add None
        self.head_rdict.add_relation("None")

    def load_rdfs(self, path):
        rdf_list = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tuples = line.strip().split('\t')
                rdf_list.append(tuples)
        return rdf_list

    def load_data_(self, fact_path, train_path, valid_path, test_path, sparsity):
        fact = self.load_rdfs(fact_path)
        fact = sample(fact, int(len(fact) * sparsity))
        train = self.load_rdfs(train_path)
        valid = self.load_rdfs(valid_path)
        test = self.load_rdfs(test_path)
        return fact, train, valid, test

    def load_relation_dict(self, relation_path):
        """
        Read relation.txt to relation dictionary
        """
        with open(relation_path, encoding='utf-8') as f:
            rel_list = f.readlines()
            for r in rel_list:
                relation = r.strip()
                self.rdict.add_relation(relation)
                # self.head_dict.add_relation(relation)

    def get_relation_dict(self):
        return self.rdict

    def get_head_relation_dict(self):
        return self.head_rdict

    @property
    def idx2ent(self):
        return self.idx2ent_

    @property
    def ent2idx(self):
        return self.ent2idx_

    @property
    def fact_rdf(self):
        return self.fact_rdf_

    @property
    def train_rdf(self):
        return self.train_rdf_

    @property
    def valid_rdf(self):
        return self.valid_rdf_

    @property
    def test_rdf(self):
        return self.test_rdf_


def sample_anchor_rdf(rdf_data, num=1):
    if num < len(rdf_data):
        return sample(rdf_data, num)
    else:
        return rdf_data


def construct_descendant(rdf_data):
    # take entity as h, map it to its r, t
    entity2desced = {}
    for rdf_ in rdf_data:
        h_, r_, t_ = parse_rdf(rdf_)
        if h_ not in entity2desced.keys():
            entity2desced[h_] = [(r_, t_)]
        else:
            entity2desced[h_].append((r_, t_))
    return entity2desced


def connected(entity2desced, head, tail):
    if head in entity2desced:
        decedents = entity2desced[head]
        for d in decedents:
            d_relation_, d_tail_ = d
            if d_tail_ == tail:
                return d_relation_
        return False
    else:
        return False


def search_closed_rel_paths(anchor_rdf, entity2desced, max_path_len=2):
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    visited = set()
    rules = []

    def dfs(current, rel_path):
        if len(rel_path) > max_path_len:  # max path length
            return
        if current == anchor_t and len(rel_path) == 1 and rel_path[-1] == anchor_r:  # remove directly connected
            return
        if current == anchor_t:
            rule = "|".join(rel_path)
            if rule not in rules:
                rules.append(rule)
        else:
            visited.add(current)
            if current in entity2desced:
                deced_list = entity2desced[current]
                for r, t in deced_list:
                    if t not in visited:
                        dfs(t, rel_path + [r])
            visited.remove(current)

    dfs(anchor_h, [])
    return rules


# def search_closed_rel_paths(anchor_rdf, entity2desced, max_path_len=2):
#     anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
#     possible_tails = []
#     for r, t in entity2desced[anchor_h]:
#         if r == anchor_r:
#             possible_tails.append(t)
#     stack = []
#     stack_print = []
#     records = []
#     # Init seeds from anchor_h
#     for r, t in entity2desced[anchor_h]:
#         if t not in possible_tails:
#             stack.append((anchor_h, r, t))
#             stack_print.append((anchor_h, f"{anchor_h}-{r}-{t}", t))
#     # Search
#     rule_seq, expended_node = [], [anchor_h]
#     while len(stack) > 0:
#         cur_h, cur_r, cur_t = stack.pop(-1)
#         record = stack_print.pop(-1)
#         deced_list = []
#
#         # Check rule
#         if cur_t in possible_tails:
#             if cur_r not in rule_seq:
#                 if cur_r not in rule_seq:
#                     rule_seq.append(cur_r)
#                 records.append(record[1])
#             continue
#
#         # Expand
#         if cur_t in entity2desced:
#             deced_list = entity2desced[cur_t]
#
#         if len(cur_r.split('|')) < max_path_len + 1 and len(deced_list) > 0 and cur_t not in expended_node:
#             for r_, t_ in deced_list:
#                 stack.append((cur_t, cur_r + '|' + r_, t_))
#                 stack_print.append((cur_t, record[1] + f" | {cur_t}-{r_}-{t_}", t_))
#         expended_node.append(cur_t)
#
#     return rule_seq


def body2idx(body_list, head_rdict):
    """
    Input a rule (string) and idx it
    """
    res = []
    for body in body_list:
        body_path = body.split('|')
        # indexs include body idx seq + notation + head idx
        indexs = []
        for rel in body_path:
            indexs.append(head_rdict.rel2idx[rel])
        res.append(indexs)
    return res


def inv_rel_idx(head_rdict):
    inv_rel_idx = []
    for i_ in range(len(head_rdict.idx2rel)):
        r_ = head_rdict.idx2rel[i_]
        if "inv_" in r_:
            inv_rel_idx.append(i_)
    return inv_rel_idx


def idx2body(index, head_rdict):
    body = "|".join([head_rdict.idx2rel[idx] for idx in index])
    return body


def rule2idx(rule, head_rdict):
    """
    Input a rule (string) and idx it
    """
    body, head = rule.split('-')
    body_path = body.split('|')
    # indexs include body idx seq + notation + head idx
    indexs = []
    for rel in body_path + [-1, head]:
        indexs.append(head_rdict.rel2idx[rel] if rel != -1 else -1)
    return indexs


def idx2rule(index, head_rdict):
    body_idx = index[0:-2]
    body = "|".join([head_rdict.idx2rel[b] for b in body_idx])
    rule = body + "-" + head_rdict.idx2rel[index[-1]]
    return rule


def enumerate_body(relation_num, body_len, rdict):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body
