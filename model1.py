import torch as th
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter, Linear
from ChebnetII_pro import ChebnetII_prop


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret

# 新增BYOL 风格的非对称预测器 (防语义坍塌) 
class Predictor(nn.Module):
    def __init__(self, dim):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)
        

class Model(nn.Module):
    def __init__(self, in_dim, out_dim, K, dprate, dropout, is_bns, act_fn):
        super(Model, self).__init__()

        self.encoder = ChebNetII(num_features=in_dim, hidden=out_dim, K=K, dprate=dprate, dropout=dropout, is_bns=is_bns, act_fn=act_fn)
        
        self.act_fn = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # 替换原有的 Discriminator 为层级 Predictor
        self.node_predictor = Predictor(out_dim)
        self.graph_predictor = Predictor(out_dim)

    def get_embedding(self, edge_index, feat):
        h1 = self.encoder(x=feat, edge_index=edge_index, highpass=True)
        h2 = self.encoder(x=feat, edge_index=edge_index, highpass=False)

        h = torch.mul(self.alpha, h1) + torch.mul(self.beta, h2)

        return h.detach()

    def forward(self, edge_index, feat, batch=None):
        # PolyGCL 解耦表示
        Z_H = self.encoder(x=feat, edge_index=edge_index, highpass=True)
        Z_L = self.encoder(x=feat, edge_index=edge_index, highpass=False)
        Z = torch.mul(self.alpha, Z_H) + torch.mul(self.beta, Z_L)

        # 2. Node-level 预测
        P_node_H = self.node_predictor(Z_H)
        P_node_L = self.node_predictor(Z_L)

        # 3. Graph-level 
        if batch is None:
            batch = torch.zeros(feat.size(0), dtype=torch.long, device=feat.device)

        h_H = global_mean_pool(Z_H, batch)
        h_L = global_mean_pool(Z_L, batch)
        h = global_mean_pool(Z, batch)

        P_graph_H = self.graph_predictor(h_H)
        P_graph_L = self.graph_predictor(h_L)

        return Z_H, Z_L, Z, P_node_H, P_node_L, h_H, h_L, h, P_graph_H, P_graph_L


class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False, act_fn='relu'):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)

        self.prop1 = ChebnetII_prop(K=K)
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, highpass=True):

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, highpass=highpass)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, highpass=highpass)


        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x
