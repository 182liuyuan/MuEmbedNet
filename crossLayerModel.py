import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, degree

class MultiViewEmbedding(nn.Module):
    def __init__(self, e_hidden, view_hidden):
        super(MultiViewEmbedding, self).__init__()
        self.view1 = nn.Linear(e_hidden, view_hidden)
        self.view2 = nn.Linear(e_hidden, view_hidden)
        self.view3 = nn.Linear(e_hidden, view_hidden)
        # self.view4 = nn.Linear(e_hidden, view_hidden)
        self.attention = nn.Linear(3 * view_hidden, 1, bias=False)
        self.last = nn.Linear(3 * view_hidden, e_hidden)
        # self.bn= nn.BatchNorm1d(e_hidden)
        self.gelu = nn.GELU()

    def forward(self, x):
        v1 = self.view1(x)
        v2 = self.view2(x)
        v3 = self.view3(x)
        # v4 = self.view4(x)
        # 可以考虑在每个视图后添加非线性激活函数和 Dropout
        v1 = self.gelu(v1)
        v2 = self.gelu(v2)
        v3 = self.gelu(v3)
        # v4 = self.gelu(v4)
        # v1 = F.dropout(F.relu(self.view1(x)), p=0.2, training=self.training)
        # v2 = F.dropout(F.relu(self.view2(x)), p=0.2, training=self.training)
        # v3 = F.dropout(F.relu(self.view3(x)), p=0.2, training=self.training)
        # v4 = F.dropout(F.relu(self.view4(x)), p=0.2, training=self.training)

        combined = torch.cat([v1, v2, v3], dim=1)
        weights = torch.sigmoid(self.attention(combined))
        out = weights * combined+combined
        out = self.last(out)
        out = self.gelu(out)
        return out
class GCN(nn.Module):
    def __init__(self, hidden, dropout_prob=0.5):
        super(GCN, self).__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.gelu = nn.GELU()
        # self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j] * deg_inv_sqrt[edge_index_i]

        adj = torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0)))
        x = self.lin(x)
        out = torch.sparse.mm(adj, x)
        out=self.gelu(self.lin(out))
        out = F.dropout(out,p=0.1, training=self.training)
        return out
class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2) + torch.mul(1 - gate, x1)
        return x

class GAT_E_to_R(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_E_to_R, self).__init__()
        self.a_h1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(r_hidden, 1, bias=False)
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)
        self.r_h = nn.Linear(r_hidden, r_hidden)
        self.r_t = nn.Linear(r_hidden, r_hidden)
        self.bn = nn.BatchNorm1d(r_hidden)
        self.gelu = nn.GELU()
    def forward(self, x_e, edge_index, rel,rel_all):
        edge_index_h, edge_index_t = edge_index
        x_r_h = self.w_h(x_e)
        x_r_t = self.w_t(x_e)
        e1 = self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_h2(x_r_t).squeeze()[edge_index_t]
        e2 = self.a_t1(x_r_h).squeeze()[edge_index_h] + self.a_t2(x_r_t).squeeze()[edge_index_t]
        alpha_h = softmax(F.leaky_relu(e1).float(), rel)
        alpha_t = softmax(F.leaky_relu(e2).float(), rel)
        rel_index_h = torch.stack([rel, edge_index_h], dim=0)
        rel_index_t = torch.stack([rel, edge_index_t], dim=0)
        adj_h = torch.sparse_coo_tensor(rel_index_h, alpha_h, (rel.max().item() + 1, x_e.size(0)))
        adj_t = torch.sparse_coo_tensor(rel_index_t, alpha_t, (rel.max().item() + 1, x_e.size(0)))
        x_r_h = torch.sparse.mm(adj_h, self.r_h(x_r_h)+x_r_h)
        x_r_t = torch.sparse.mm(adj_t, self.r_t(x_r_t)+x_r_t)
        x_r = x_r_h + x_r_t
        return x_r

class GAT_R_to_E(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_R_to_E, self).__init__()
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)
        self.x_r1 = nn.Linear(r_hidden, 2*r_hidden)
        self.x_r2 = nn.Linear(2*r_hidden, r_hidden)
        self.r = nn.Linear(2*r_hidden, 2 * r_hidden)
        self.bn = nn.BatchNorm1d(2*r_hidden)
    def forward(self, x_e, x_r, edge_index, rel,rel_all):
        edge_index_h, edge_index_t = edge_index     #edge_index_h=70414, edge_index_t=70414
        e_h = self.a_h(x_e).squeeze()[edge_index_h] #e_h=70414
        e_t = self.a_t(x_e).squeeze()[edge_index_t]#e_t=70414
        e_r = self.a_r(x_r).squeeze()[rel]  #e_r=70414
        alpha_h = softmax(F.leaky_relu(e_h + e_r).float(), edge_index_h)#alpha_h=70414
        alpha_t = softmax(F.leaky_relu(e_t + e_r).float(), edge_index_t)#alpha_r=70414
        rel_index_h = torch.stack([edge_index_h, rel], dim=0)
        rel_index_t = torch.stack([edge_index_t, rel], dim=0)
        adj_h = torch.sparse_coo_tensor(rel_index_h, alpha_h, (x_e.size(0), x_r.size(0)))
        adj_t = torch.sparse_coo_tensor(rel_index_t, alpha_t, (x_e.size(0), x_r.size(0)))
        x_r1 = self.x_r1(x_r)
        x_r2 = self.x_r2(x_r1)
        x_e_h = torch.sparse.mm(adj_h, x_r+x_r2)
        x_e_t = torch.sparse.mm(adj_t, x_r+x_r2)
        x = torch.cat([x_e_h, x_e_t], dim=1)
        x=self.r(x)
        # x = self.bn(x)
        return x
class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.x_x = nn.Linear(hidden, hidden, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.gelu = nn.GELU()

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i + e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)

        adj = torch.sparse_coo_tensor(edge_index, alpha, (x.size(0), x.size(0)))

        x = torch.sparse.mm(adj, x+self.x_x(x))
        x = self.gelu(x)

        x = F.dropout(x,p=0.1, training=self.training)

        # x = self.bn(x)
        return x
# 跨层注意力整合多层特征
# x_e = self.cross_layer_attention(x_e, layer1_output, layer2_output)
class CrossLayerAttention(nn.Module):
    def __init__(self, hidden, num_layers=3):
        """
        跨层注意力机制，用于融合不同层的特征。
        :param hidden: 特征的维度
        :param num_layers: 层数
        """
        super(CrossLayerAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.zeros(num_layers, 1))  # 注意力权重
        self.projection = nn.Linear(hidden, hidden)  # 用于将多层特征投影到相同维度
        self.gelu = nn.GELU()

    def forward(self, *layer_outputs):
        """
        前向传播
        :param layer_outputs: 不同层的输出特征
        """
        layer_outputs = torch.stack(layer_outputs, dim=1)  # 将层特征堆叠为 [batch_size, num_layers, hidden]
        attention_scores = torch.softmax(self.attention_weights, dim=0)  # 计算权重
        weighted_sum = torch.sum(attention_scores * layer_outputs, dim=1)  # 按权重加权求和
        output = self.gelu(self.projection(weighted_sum))  # 投影并激活
        return output


class crossMuEmbedNet(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100, view_hidden=200):
        super(crossMuEmbedNet, self).__init__()
        self.multi_view = MultiViewEmbedding(e_hidden, view_hidden)
        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.gat_e_to_r = GAT_E_to_R(e_hidden, r_hidden)
        self.gat_r_to_e = GAT_R_to_E(e_hidden, r_hidden)
        self.gat = GAT(e_hidden + 2 * r_hidden)
        self.cross_layer_attention = CrossLayerAttention(e_hidden, num_layers=3)  # 跨层注意力

    def forward(self, x_e, edge_index, rel, edge_index_all, rel_all):
        x_e = self.multi_view(x_e)
        # 第一层 GCN 和 Highway
        layer1_output = self.highway1(x_e, self.gcn1(x_e, edge_index_all))

        # 第二层 GCN 和 Highway
        layer2_output = self.highway2(layer1_output, self.gcn2(layer1_output, edge_index_all))

        # 跨层注意力整合多层特征
        x_e = self.cross_layer_attention(x_e, layer1_output, layer2_output)

        # GAT 模块
        x_r = self.gat_e_to_r(x_e, edge_index, rel, rel_all)
        x_e = torch.cat([x_e, self.gat_r_to_e(x_e, x_r, edge_index, rel, rel_all)], dim=1)
        x_e = torch.cat([x_e, self.gat(x_e, edge_index_all)], dim=1)

        return x_e
