"""Adım 2: Sınıflandırıcı başlığı eğit, validation ile en iyi epoch'u seç, test setinde değerlendir.

Kullanım:  python src/train.py
Çıktı:     model/head.pt, results/metrics.json, results/*.png
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

sys.path.insert(0, os.path.dirname(__file__))
from common import Head, FullModel, CLASSES_EN, INPUT_SIZE

SEED = 42
HP = dict(optimizer="AdamW", lr=1e-3, weight_decay=1e-4, batch_size=256,
          epochs=30, scheduler="CosineAnnealingLR", dropout=0.3, hidden=256,
          loss="CrossEntropyLoss(label_smoothing=0.1)")

torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs("model", exist_ok=True); os.makedirs("results", exist_ok=True)

d = {s: torch.load(f"features/{s}.pt") for s in ["train", "val", "test"]}
Xtr, ytr = d["train"]["X"], d["train"]["y"]
Xva, yva = d["val"]["X"], d["val"]["y"]
Xte, yte = d["test"]["X"], d["test"]["y"]

head = Head(hidden=HP["hidden"], p_drop=HP["dropout"])
opt = torch.optim.AdamW(head.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


@torch.no_grad()
def evaluate(X, y):
    head.eval()
    logits = head(X)
    return crit(logits, y).item(), (logits.argmax(1) == y).float().mean().item(), logits


hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc, best_ep = 0, 0
t0 = time.time()
for ep in range(1, HP["epochs"] + 1):
    head.train()
    perm = torch.randperm(len(Xtr))
    tl, tc = 0.0, 0
    for i in range(0, len(Xtr), HP["batch_size"]):
        b = perm[i:i + HP["batch_size"]]
        logits = head(Xtr[b])
        loss = crit(logits, ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * len(b); tc += (logits.argmax(1) == ytr[b]).sum().item()
    sched.step()
    vl, va, _ = evaluate(Xva, yva)
    hist["train_loss"].append(tl / len(Xtr)); hist["train_acc"].append(tc / len(Xtr))
    hist["val_loss"].append(vl); hist["val_acc"].append(va)
    if va > best_acc:
        best_acc, best_ep = va, ep
        torch.save(head.state_dict(), "model/head.pt")
    print(f"epoch {ep:2d}  train_loss {tl/len(Xtr):.4f} acc {tc/len(Xtr):.4f} | val_loss {vl:.4f} acc {va:.4f}")
train_secs = time.time() - t0

# ---- Test değerlendirmesi (en iyi validation epoch'unun ağırlıkları) ----
head.load_state_dict(torch.load("model/head.pt"))
_, test_acc, logits = evaluate(Xte, yte)
pred = logits.argmax(1).numpy(); yt = yte.numpy()
P, R, F, _ = precision_recall_fscore_support(yt, pred, average="macro")
report = classification_report(yt, pred, target_names=CLASSES_EN, digits=4, output_dict=True)
print(classification_report(yt, pred, target_names=CLASSES_EN, digits=4))
cm = confusion_matrix(yt, pred)

# ---- Uçtan uca tahmin süresi (backbone + head, CPU, batch=1) ----
full = FullModel(head).eval()
x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
with torch.inference_mode():
    for _ in range(5): full(x)
    ts = []
    for _ in range(50):
        t = time.perf_counter(); full(x); ts.append(time.perf_counter() - t)
infer_ms = float(np.mean(ts) * 1000)

feat_secs = {s: d[s]["seconds"] for s in d}
metrics = {
    "hyperparameters": HP, "best_epoch": best_ep, "best_val_acc": best_acc,
    "test_accuracy": test_acc, "test_precision_macro": P, "test_recall_macro": R, "test_f1_macro": F,
    "per_class": {c: report[c] for c in CLASSES_EN},
    "head_training_seconds": train_secs,
    "feature_extraction_seconds": feat_secs,
    "total_training_seconds": train_secs + feat_secs["train"] + feat_secs["val"],
    "inference_ms_per_image_cpu": infer_ms,
    "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
    "trainable_params": sum(p.numel() for p in head.parameters()),
    "cpu_threads": torch.get_num_threads(),
    "history": hist,
}
json.dump(metrics, open("results/metrics.json", "w"), indent=2)

# ---- Grafikler ----
ep = range(1, HP["epochs"] + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, hist["train_acc"], label="train"); ax[1].plot(ep, hist["val_acc"], label="validation")
ax[1].axvline(best_ep, ls="--", c="gray", lw=1); ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10), CLASSES_EN, rotation=45, ha="right"); ax.set_yticks(range(10), CLASSES_EN)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title(f"Confusion Matrix (test, acc={test_acc:.3f})")
plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "history")}, indent=2))
