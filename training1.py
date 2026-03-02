import argparse
import warnings
import seaborn as sns

import torch
from alive_progress import alive_bar
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import random_splits

warnings.filterwarnings("ignore")

from model1 import LogReg, Model
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description="PolyGCL")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dev', type=int, default=0)
parser.add_argument("--dataname", type=str, default="cora")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.010)
parser.add_argument("--lr1", type=float, default=0.001)
parser.add_argument("--lr2", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--wd1", type=float, default=0.0)
parser.add_argument("--wd2", type=float, default=0.0)
parser.add_argument("--hid_dim", type=int, default=128)
parser.add_argument("--K", type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--dprate', type=float, default=0.5)
parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu')
# 新增权重
parser.add_argument('--lam_node', type=float, default=0.2)
parser.add_argument('--lam_graph', type=float, default=0.2)
parser.add_argument('--lam_ortho', type=float, default=0.05)
args = parser.parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)

from dataset_loader import DataLoader
import time

def get_feat(batch, n_feat, device):
    feat = batch.x
    if feat is None:
        row, col = batch.edge_index
        deg = torch.bincount(row, minlength=batch.num_nodes).float().unsqueeze(1)
        feat = deg.to(device)
    return feat

# 新增一致性对齐 Loss (BYOL 风格)
def byol_loss(p, z):
    p = F.normalize(p, dim=-1, eps=1e-8)
    z = F.normalize(z, dim=-1, eps=1e-8)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

# 新增正交解耦 Loss 
def ortho_loss(z1, z2):
    z1 = F.normalize(z1, dim=-1, eps=1e-8)
    z2 = F.normalize(z2, dim=-1, eps=1e-8)
    return (z1 * z2).sum(dim=-1).abs().mean()
    
if __name__ == "__main__":
    print(args)

    dataset = DataLoader(name=args.dataname)

    is_graph_dataset = len(dataset) > 1

    if not is_graph_dataset:
        data = dataset[0]
        feat = data.x
        label = data.y
        edge_index = data.edge_index

        n_feat = feat.shape[1]
        n_classes = np.unique(label).shape[0]

        edge_index = edge_index.to(args.device)
        feat = feat.to(args.device)

    else:
        n_feat = dataset.num_features
        if n_feat == 0:
            n_feat = 1
        n_classes = dataset.num_classes

        loader = GeoDataLoader(dataset, batch_size=64, shuffle=True)

        total_nodes = sum([d.num_nodes for d in dataset])
        lbl1 = th.ones(total_nodes * 2)
        lbl2 = th.zeros(total_nodes * 2)
        lbl = th.cat((lbl1, lbl2)).to(args.device)

    model = Model(in_dim=n_feat, out_dim=args.hid_dim, K=args.K,
                  dprate=args.dprate, dropout=args.dropout,
                  is_bns=args.is_bns, act_fn=args.act_fn,
                  n_classes=n_classes).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.node_predictor.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.graph_predictor.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
        {'params': model.classifier.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
        {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr}
    ])

    ce_loss = nn.CrossEntropyLoss()
    best = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = str(int(time.time()))

    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if not is_graph_dataset:
                batch_idx = torch.zeros(feat.size(0), dtype=torch.long, device=args.device)
                Z_H, Z_L, Z, P_node_H, P_node_L, h_H, h_L, h, P_graph_H, P_graph_L, logits = model(edge_index, feat, batch_idx)
                # 计算各种 Loss
                loss_node = byol_loss(P_node_H, Z.detach()) + byol_loss(P_node_L, Z.detach())
                loss_graph = byol_loss(P_graph_H, h.detach()) + byol_loss(P_graph_L, h.detach())
                loss_ortho = ortho_loss(Z_H, Z_L)
                loss_cls = ce_loss(logits, label)
                
                loss = loss_cls + args.lam_node * loss_node + args.lam_graph * loss_graph + args.lam_ortho * loss_ortho
                loss.backward()
                optimizer.step()
            else:
                loss_epoch = 0.0
                for batch in loader:
                    batch = batch.to(args.device)
                    feat = get_feat(batch, n_feat, args.device)
                    edge_index = batch.edge_index
                    
                    Z_H, Z_L, Z, P_node_H, P_node_L, h_H, h_L, h, P_graph_H, P_graph_L, logits = model(edge_index, feat, batch.batch)

                    loss_node = byol_loss(P_node_H, Z.detach()) + byol_loss(P_node_L, Z.detach())
                    loss_graph = byol_loss(P_graph_H, h.detach()) + byol_loss(P_graph_L, h.detach())
                    loss_ortho = ortho_loss(Z_H, Z_L)
                    loss_cls = ce_loss(logits, batch.y)

                    loss = loss_cls + args.lam_node * loss_node + args.lam_graph * loss_graph + args.lam_ortho * loss_ortho
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_epoch += loss.item()

                loss = loss_epoch / len(loader)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                th.save(model.state_dict(), 'pkl/best_model_' + args.dataname + tag + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                break
            bar()

    model.load_state_dict(th.load('pkl/best_model_' + args.dataname + tag + '.pkl'))
    model.eval()

    print("=== Evaluation ===")
    results = []

    if not is_graph_dataset:
        embeds = model.get_embedding(edge_index, feat)

        SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660,
                 2108550661, 1648766618, 629014539, 3212139042, 2424918363]

        train_rate = 0.6
        val_rate = 0.2
        percls_trn = int(round(train_rate * len(label) / n_classes))
        val_lb = int(round(val_rate * len(label)))

        for i in range(10):
            seed = SEEDS[i]
            train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

            train_mask = th.BoolTensor(train_mask).to(args.device)
            val_mask = th.BoolTensor(val_mask).to(args.device)
            test_mask = th.BoolTensor(test_mask).to(args.device)

            train_embs = embeds[train_mask]
            val_embs = embeds[val_mask]
            test_embs = embeds[test_mask]

            train_labels = label[train_mask]
            val_labels = label[val_mask]
            test_labels = label[test_mask]

            logreg = LogReg(args.hid_dim, n_classes).to(args.device)
            opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(2000):
                logreg.train()
                opt.zero_grad()
                logits = logreg(train_embs)
                loss = loss_fn(logits, train_labels)
                loss.backward()
                opt.step()

            logreg.eval()
            with th.no_grad():
                test_logits = logreg(test_embs)
                test_preds = th.argmax(test_logits, dim=1)
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

            results.append(test_acc.cpu().item())

    else:
        embeds_list = []
        labels_list = []

        loader = GeoDataLoader(dataset, batch_size=1, shuffle=False)

        for batch in loader:
            batch = batch.to(args.device)
            feat = get_feat(batch, n_feat, args.device)
            node_emb = model.get_embedding(batch.edge_index, feat)
            graph_emb = global_mean_pool(node_emb, batch.batch)
            embeds_list.append(graph_emb)
            labels_list.append(batch.y)

        embeds = torch.cat(embeds_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # ===== 10-fold Stratified Cross Validation =====
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

        for fold, (train_idx, test_idx) in enumerate(
                skf.split(embeds.cpu().numpy(), labels.cpu().numpy())):

            train_embs = embeds[train_idx]
            test_embs = embeds[test_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            logreg = LogReg(args.hid_dim, n_classes).to(args.device)
            opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(2000):
                logreg.train()
                opt.zero_grad()
                logits = logreg(train_embs)
                loss = loss_fn(logits, train_labels)
                loss.backward()
                opt.step()

            logreg.eval()
            with th.no_grad():
                test_logits = logreg(test_embs)
                test_preds = th.argmax(test_logits, dim=1)
                test_acc = (test_preds == test_labels).float().mean()

            results.append(test_acc.item())
            print(f"Fold {fold+1}: {test_acc.item()*100:.2f}")
        # =================================================
    mean = np.mean(results)
    std = np.std(results)
    print(f"\nFinal Acc: {mean*100:.2f} ± {std*100:.2f}")
