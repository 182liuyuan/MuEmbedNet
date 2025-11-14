import torch
import torch.nn as nn
import torch.nn.functional as F



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

# class L1_Loss(nn.Module):
#     def __init__(self, gamma=3):
#         super(L1_Loss, self).__init__()
#         self.gamma = gamma
#
#     def dis(self, x, y):
#         return torch.sum(torch.abs(x-y), dim=-1)
#
#     def forward(self, x1, x2, train_set, train_batch):
#         x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
#         x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
#         x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
#         x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
#         x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
#
#         dis_x1_x2 = self.dis(x1_train, x2_train)
#         loss11 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg1)))
#         loss12 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg2)))
#         loss21 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg1)))
#         loss22 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg2)))
#         loss = (loss11+loss12+loss21+loss22)/4
#         return loss