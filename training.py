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
parser.add_argument("--hid_dim", type=int, default=128)
parser.add_argument("--K", type=int, default=3)
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

from torch_geometric.utils import dropout_edge
import torch.nn.functional as F

def graph_augment(x, edge_index, drop_edge_rate=0.1, drop_feat_rate=0.1):
    edge_index_aug, _ = dropout_edge(edge_index, p=drop_edge_rate)
    x_aug = F.dropout(x, p=drop_feat_rate, training=True)
    return x_aug, edge_index_aug

def info_nce(z1, z2, temperature=0.2):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    # mask self similarity
    mask = torch.eye(2*N, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # positive pairs
    pos = torch.cat([
        torch.diag(sim, N),
        torch.diag(sim, -N)
    ], dim=0)

    loss = -pos + torch.logsumexp(sim, dim=1)
    return loss.mean()

from dataset_loader import DataLoader
import time

def get_feat(batch, n_feat, device):
    feat = batch.x
    if feat is None:
        row, col = batch.edge_index
        deg = torch.bincount(row, minlength=batch.num_nodes).float().unsqueeze(1)
        feat = deg.to(device)
    return feat
    
if __name__ == "__main__":

    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    final_results = []

    for run_seed in SEEDS:
        print(f"\n========== Seed {run_seed} ==========")

        # 设置随机种子
        random.seed(run_seed)
        np.random.seed(run_seed)
        th.manual_seed(run_seed)
        th.cuda.manual_seed_all(run_seed)

        # 把 args.seed 同步成当前 seed
        args.seed = run_seed

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
            {'params': model.encoder.parameters(), 'lr': args.lr1},
            {'params': model.proj.parameters(), 'lr': args.lr1}
        ])

        best = float("inf")
        cnt_wait = 0

        # 用 seed 做 tag
        tag = f"_seed_{run_seed}"
        best_path = f"pkl/best_model_{args.dataname}{tag}.pkl"

        with alive_bar(args.epochs) as bar:
            for epoch in range(args.epochs):
                model.train()
                total_loss = 0

                for data in loader:
                    data = data.to(args.device)
                    feat = get_feat(data, n_feat, args.device)

                    x1, edge1 = graph_augment(feat, data.edge_index)
                    x2, edge2 = graph_augment(feat, data.edge_index)

                    z1, z2 = model(x1, edge1, x2, edge2, data.batch)
                    loss = info_nce(z1, z2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                loss = total_loss / len(loader)

                if loss < best:
                    best = loss
                    cnt_wait = 0
                    th.save(model.state_dict(), best_path)
                else:
                    cnt_wait += 1

                if cnt_wait == args.patience:
                    break
                bar()

        model.load_state_dict(th.load(best_path))
        model.eval()

        print("=== Evaluation ===")
        results = []

        if not is_graph_dataset:
            embeds = model.get_embedding(edge_index, feat)

            SEEDS_INNER = [1941488137, 4198936517, 983997847, 4023022221, 4019585660,
                           2108550661, 1648766618, 629014539, 3212139042, 2424918363]

            train_rate = 0.6
            val_rate = 0.2
            percls_trn = int(round(train_rate * len(label) / n_classes))
            val_lb = int(round(val_rate * len(label)))

            for i in range(10):
                seed = SEEDS_INNER[i]
                train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

                train_mask = th.BoolTensor(train_mask).to(args.device)
                val_mask = th.BoolTensor(val_mask).to(args.device)
                test_mask = th.BoolTensor(test_mask).to(args.device)

                train_embs = embeds[train_mask]
                test_embs = embeds[test_mask]
                train_labels = label[train_mask]
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
            graph_embs = []
            labels = []
            loader_eval = GeoDataLoader(dataset, batch_size=64, shuffle=False)

            with torch.no_grad():
                for data in loader_eval:
                    data = data.to(args.device)
                    feat = get_feat(data, n_feat, args.device)
                    h = model.encoder(feat, data.edge_index)
                    g = global_mean_pool(h, data.batch)
                    graph_embs.append(g)
                    labels.append(data.y)

            graph_emb = torch.cat(graph_embs, dim=0)
            labels = torch.cat(labels, dim=0)

            # 每次 split 都不同
            perm = torch.randperm(len(labels))
            graph_emb = graph_emb[perm]
            labels = labels[perm]

            train_size = int(0.8 * len(labels))
            train_embs = graph_emb[:train_size]
            test_embs = graph_emb[train_size:]
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

        acc = np.mean(results) * 100
        print("Final Acc:", acc)
        final_results.append(acc)

    mean_acc = np.mean(final_results)
    std_acc = np.std(final_results)

    print("\n======================================")
    print(f"Final Result over {len(SEEDS)} runs:")
    print(f"{mean_acc:.2f} ± {std_acc:.2f}")
    print("======================================")
