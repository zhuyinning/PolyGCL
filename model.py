import torch as th
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import global_mean_pool

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


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)


    def forward(self, h1, h2, h3, h4, c):
        c_x = c.expand_as(h1).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x).squeeze(1)
        sc_2 = self.fn(h1, c_x).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x).squeeze(1)
        sc_4 = self.fn(h3, c_x).squeeze(1)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, K, dprate, dropout, is_bns, act_fn):
        super(Model, self).__init__()

        self.encoder = ChebNetII(
            num_features=in_dim,
            hidden=out_dim,
            K=K,
            dprate=dprate,
            dropout=dropout,
            is_bns=is_bns,
            act_fn=act_fn
        )

        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x1, edge1, x2, edge2, batch):
        h1 = self.encoder(x1, edge1)
        h2 = self.encoder(x2, edge2)

        g1 = global_mean_pool(h1, batch)
        g2 = global_mean_pool(h2, batch)

        z1 = self.proj(g1)
        z2 = self.proj(g2)

        return z1, z2

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
