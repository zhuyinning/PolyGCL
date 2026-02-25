import argparse
import warnings
import seaborn as sns

import torch
from alive_progress import alive_bar
import random
import numpy as np
import torch as th
import torch.nn as nn
from utils import random_splits

warnings.filterwarnings("ignore")

from model import LogReg, Model
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_mean_pool

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
parser.add_argument("--hid_dim", type=int, default=512)
parser.add_argument("--K", type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--dprate', type=float, default=0.5)
parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu')
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
        feat = torch.ones((batch.num_nodes, n_feat), device=device)
    return feat
    
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

        n_node = feat.shape[0]
        lbl1 = th.ones(n_node * 2)
        lbl2 = th.zeros(n_node * 2)
        lbl = th.cat((lbl1, lbl2)).to(args.device)

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
                  is_bns=args.is_bns, act_fn=args.act_fn).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
        {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
        {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr}
    ])

    loss_fn = nn.BCEWithLogitsLoss()

    best = float("inf")
    cnt_wait = 0
    best_t = 0
    tag = str(int(time.time()))

    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if not is_graph_dataset:
                shuf_idx = np.random.permutation(feat.shape[0])
                shuf_feat = feat[shuf_idx, :]
                out = model(edge_index, feat, shuf_feat)
            else:
                outs = []
                for batch in loader:
                    batch = batch.to(args.device)
                    feat = batch.x
                    if feat is None:
                        feat = torch.ones((batch.num_nodes, n_feat), device=args.device)
                    edge_index = batch.edge_index
                    shuf_idx = torch.randperm(feat.shape[0])
                    shuf_feat = feat[shuf_idx]
                    out = model(edge_index, feat, shuf_feat)
                    outs.append(out)
                out = torch.cat(outs)

            loss = loss_fn(out, lbl)
            loss.backward()
            optimizer.step()

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

        # ===== 新增：随机打乱 =====
        perm = torch.randperm(len(labels))
        embeds = embeds[perm]
        labels = labels[perm]
        # ==========================

        train_size = int(0.8 * len(labels))
        train_embs = embeds[:train_size]
        test_embs = embeds[train_size:]
        train_labels = labels[:train_size]
        test_labels = labels[train_size:]

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

    print("Final Acc:", np.mean(results) * 100)
