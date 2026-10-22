"""Train + evaluate a CNN on EMNIST Letters. Saves model + metrics.

    python train.py --epochs 15
"""
import argparse, json
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Net, MEAN, STD, LETTERS

AUG = transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.1), shear=8)


class DS(Dataset):
    def __init__(self, x, y, aug):
        self.x, self.y, self.aug = x, y, aug
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        img = self.x[i].float().div(255).unsqueeze(0)
        if self.aug:
            img = AUG(img)
        return (img - MEAN) / STD, int(self.y[i])


def load_emnist(root):
    tr = datasets.EMNIST(root, split="letters", train=True, download=True)
    te = datasets.EMNIST(root, split="letters", train=False, download=True)
    # raw EMNIST images are stored transposed; labels are 1..26 -> shift to 0..25
    fix = lambda d: (d.data.transpose(1, 2).contiguous(), d.targets - 1)
    return fix(tr), fix(te)


@torch.no_grad()
def predict_all(model, loader, dev):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(dev)).cpu()); ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", dev)

    (xtr, ytr), (xte, yte) = load_emnist(a.data)
    perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(0))
    nval = len(ytr) // 10
    vi, ti = perm[:nval], perm[nval:]
    train_dl = DataLoader(DS(xtr[ti], ytr[ti], True), a.bs, shuffle=True, num_workers=a.workers)
    val_dl = DataLoader(DS(xtr[vi], ytr[vi], False), 512, num_workers=a.workers)
    test_dl = DataLoader(DS(xte, yte, False), 512, num_workers=a.workers)
    print(f"train {len(ti)} | val {nval} | test {len(yte)}")

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(train_dl))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, hist = 0.0, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y)
        lg, ys = predict_all(model, val_dl, dev)
        vacc = (lg.argmax(1) == ys).float().mean().item()
        hist.append({"epoch": ep, "train_loss": tot / len(ti), "val_acc": vacc})
        print(f"epoch {ep:2d}  loss {tot/len(ti):.4f}  val acc {vacc:.4f}")
        if vacc > best:
            best = vacc; torch.save(model.state_dict(), "model.pt")

    # ---- final test-set evaluation with the best checkpoint
    model.load_state_dict(torch.load("model.pt", map_location=dev))
    lg, ys = predict_all(model, test_dl, dev)
    pred = lg.argmax(1).numpy(); y = ys.numpy()
    top3 = (lg.topk(3, 1).indices.numpy() == y[:, None]).any(1).mean()
    metrics = {
        "dataset": "EMNIST Letters (test split, 20,800 images)",
        "test_accuracy": float((pred == y).mean()),
        "top3_accuracy": float(top3),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "best_val_accuracy": best,
        "per_class": classification_report(y, pred, target_names=list(LETTERS), output_dict=True),
        "history": hist,
    }
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(classification_report(y, pred, target_names=list(LETTERS), digits=4))
    print(f"TEST accuracy {metrics['test_accuracy']:.4f} | top-3 {top3:.4f} | macro-F1 {metrics['macro_f1']:.4f}")

    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(26)); ax.set_xticklabels(LETTERS, fontsize=7)
    ax.set_yticks(range(26)); ax.set_yticklabels(LETTERS, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("EMNIST Letters - normalized confusion matrix")
    fig.colorbar(im); fig.tight_layout(); fig.savefig("confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
