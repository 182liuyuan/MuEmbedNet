import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all


def get_train_batch(x1, x2, train_set, k=5):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch


def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print('Left:\t',end='')
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)
    print('Right:\t',end='')
    for k in Hn_nums:
        pred_topk= S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)

    
def get_hits_stable(x1, x2, pair):
    pair_num = pair.size(0)
    S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    #index = S.flatten().argsort(descending=True)
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index//pair_num
    index_e2 = index%pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned/pair_num*100))
class L1_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L1_Loss, self).__init__()
        self.gamma = gamma

    def dis(self, x, y):
        return torch.sum(torch.abs(x - y), dim=-1)

    def forward(self, x1, x2, train_set, train_batch):
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))

        dis_x1_x2 = self.dis(x1_train, x2_train)
        loss11 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg1)))
        loss12 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg2)))
        loss21 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg1)))
        loss22 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg2)))
        loss = (loss11 + loss12 + loss21 + loss22) / 4
        return loss

    def forward_test(self, x1, x2, test_set):
        x1_test, x2_test = x1[test_set[:, 0]], x2[test_set[:, 1]]
        dis_x1_x2 = self.dis(x1_test, x2_test)
        loss = torch.mean(dis_x1_x2)
        return loss


import os
import json
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index


class DBP15K(InMemoryDataset):
    def __init__(self, root, pair, KG_num=1, rate=0.7, seed=1):
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.seed = seed
        torch.manual_seed(seed)
        super(DBP15K, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['zh_en', 'fr_en', 'ja_en']

    @property
    def processed_file_names(self):
        return '%s_%d_%.1f_%d.pt' % (self.pair, self.KG_num, self.rate, self.seed)

    def process(self):
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        emb_path = os.path.join(self.root, self.pair, self.pair[:2] + '_vectorList.json')

        x1, edge_index1, rel1, assoc1 = self.process_graph(g1_path, x1_path, emb_path)
        x2, edge_index2, rel2, assoc2 = self.process_graph(g2_path, x2_path, emb_path)

        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set = self.process_pair(pair_path, assoc1, assoc2)
        pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
        train_set = pair_set[:, :int(self.rate * pair_set.size(1))]
        test_set = pair_set[:, int(self.rate * pair_set.size(1)):]

        if self.KG_num == 1:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1,
                        x2=x2, edge_index2=edge_index2, rel2=rel2,
                        train_set=train_set.t(), test_set=test_set.t())
        else:
            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([edge_index1, edge_index2 + x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2 + rel1.max() + 1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel, train_set=train_set.t(), test_set=test_set.t())
        torch.save(self.collate([data]), self.processed_paths[0])

    def process_graph(self, triple_path, ent_path, emb_path):
        # g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        g = read_txt_array(triple_path, sep=',', dtype=torch.long)

        subj, rel, obj = g.t()

        assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel]

        idx = []
        with open(ent_path, 'r', encoding='utf-8') as f:
            for line in f:
                info = line.strip().split(',')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(emb_path, 'r', encoding='utf-8') as f:
            embedding_list = torch.tensor(json.load(f))
        x = embedding_list[idx]

        assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)
        return x, edge_index, rel, assoc

    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep=',', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)
