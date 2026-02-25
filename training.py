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
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from model import LogReg,Model

parser = argparse.ArgumentParser(description="PolyGCL")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # Default seed same as GCNII
parser.add_argument('--dev', type=int, default=0, help='device id')

parser.add_argument(
    "--dataname", type=str, default="cora", help="Name of dataset."
)
parser.add_argument(
    "--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu."
)
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Patient epochs to wait before early stopping.",
)
parser.add_argument(
    "--lr", type=float, default=0.010, help="Learning rate of prop."
)
parser.add_argument(
    "--lr1", type=float, default=0.001, help="Learning rate of PolyGCL."
)
parser.add_argument(
    "--lr2", type=float, default=0.01, help="Learning rate of linear evaluator."
)
parser.add_argument(
    "--wd", type=float, default=0.0, help="Weight decay of PolyGCL prop."
)
parser.add_argument(
    "--wd1", type=float, default=0.0, help="Weight decay of PolyGCL."
)
parser.add_argument(
    "--wd2", type=float, default=0.0, help="Weight decay of linear evaluator."
)

parser.add_argument(
    "--hid_dim", type=int, default=512, help="Hidden layer dim."
)

parser.add_argument(
    "--K", type=int, default=10, help="Layer of encoder."
)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu',
                    help='activation function')
parser.add_argument("--task", type=str, default="node", choices=["node", "graph"],
                    help="node: original (single-graph) SSL + linear eval; graph: TU graph classification.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--eval_epochs", type=int, default=300)  # 图分类训练轮数
parser.add_argument("--hidden_cls", type=int, default=0)     
args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
th.cuda.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)

from dataset_loader import DataLoader
import time

if __name__ == "__main__":
    print(args)
    # Step 1: Load data =================================================================== #
    dataset = DataLoader(name=args.dataname)
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
    lbl = th.cat((lbl1, lbl2))

    # Step 2: Create model =================================================================== #
    model = Model(in_dim=n_feat, out_dim=args.hid_dim, K=args.K, dprate=args.dprate, dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn)
    model = model.to(args.device)

    lbl = lbl.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam([{'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr}
                                  ])

    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    best_t = 0

    #generate a random number --> later use as a tag for saved model
    tag = str(int(time.time()))

    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            shuf_idx = np.random.permutation(n_node)
            shuf_feat = feat[shuf_idx, :]

            out = model(edge_index, feat, shuf_feat)
            loss = loss_fn(out, lbl)

            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss.item()))

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                th.save(model.state_dict(), 'pkl/best_model_'+ args.dataname + tag + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()

    print('Loading {}th epoch'.format(best_t + 1))

    model.load_state_dict(th.load('pkl/best_model_'+ args.dataname + tag + '.pkl'))
    model.eval()
    embeds = model.get_embedding(edge_index, feat)

    # Step 5:  Linear evaluation ========================================================== #
    print("=== Evaluation ===")
    ''' Linear Evaluation '''
    results = []
    # 10 fixed seeds for random splits from BernNet
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate*len(label)/n_classes))
    val_lb = int(round(val_rate*len(label)))
    for i in range(10):
        seed = SEEDS[i]
        assert label.shape[0] == n_node
        train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

        train_mask = th.BoolTensor(train_mask).to(args.device)
        val_mask = th.BoolTensor(val_mask).to(args.device)
        test_mask = th.BoolTensor(test_mask).to(args.device)

        train_embs = embeds[train_mask]
        val_embs = embeds[val_mask]
        test_embs = embeds[test_mask]

        label = label.to(args.device)

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_acc = 0
        eval_acc = 0
        bad_counter = 0

        logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(args.device)

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    bad_counter = 0
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc
                else:
                    bad_counter += 1

        print(i, 'Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        results.append(eval_acc.cpu().data)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')

else:
    # ============ Graph task: TU/ICL graph classification ============
    n = len(dataset)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_set = dataset[train_idx.tolist()]
    val_set = dataset[val_idx.tolist()]
    test_set = dataset[test_idx.tolist()]

    train_loader = GeoDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_dim = dataset.num_features
    # 没有 dataset.num_classes，用 y.max()+1 
    try:
        num_classes = dataset.num_classes
    except Exception:
        num_classes = int(dataset.data.y.max().item() + 1)

    model = Model(in_dim=in_dim, out_dim=args.hid_dim, K=args.K, dprate=args.dprate,
                  dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn).to(args.device)

    # 图分类 head
    if args.hidden_cls and args.hidden_cls > 0:
        cls_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hidden_cls),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_cls, num_classes),
        ).to(args.device)
    else:
        cls_head = nn.Linear(args.hid_dim, num_classes).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
        {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
        {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr},
        {'params': cls_head.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2},
    ])

    def run_epoch(loader, train: bool):
        if train:
            model.train()
            cls_head.train()
        else:
            model.eval()
            cls_head.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            batch = batch.to(args.device)

            # 复用 PolyGCL 的 node embedding
            h = model.get_embedding(batch.edge_index, batch.x)  # [N, hid_dim]
            g = global_mean_pool(h, batch.batch)                # [B, hid_dim]

            logits = cls_head(g)                                # [B, C]
            y = batch.y.view(-1)

            loss = F.cross_entropy(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch.num_graphs

        return total_loss / max(total, 1), correct / max(total, 1)

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, args.eval_epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        te_loss, te_acc = run_epoch(test_loader, train=False)

        if va_acc >= best_val:
            best_val = va_acc
            best_test = te_acc

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Graph-{args.dataname}] Epoch {epoch:03d}/{args.eval_epochs} | "
                  f"train {tr_acc*100:.2f}% loss {tr_loss:.4f} | "
                  f"val {va_acc*100:.2f}% | test {te_acc*100:.2f}% | best_test {best_test*100:.2f}%")

    print(f"[Graph-{args.dataname}] Final best_test={best_test*100:.2f}% (best_val={best_val*100:.2f}%)")
