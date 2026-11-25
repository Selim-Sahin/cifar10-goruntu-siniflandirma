"""Adım 1: CIFAR-10'u indir, train/val/test böl, dondurulmuş ResNet18 ile öznitelik çıkar.

Kullanım:  python src/extract_features.py
Çıktı:     features/{train,val,test}.pt
"""
import io, os, sys, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from common import build_backbone, cifar_transform

HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
DATA_DIR = "data"
OUT_DIR = "features"
SEED = 42
BATCH = 128


def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(HF + f"{name}-00000-of-00001.parquet", path)
    df = pd.read_parquet(path)
    imgs = [Image.open(io.BytesIO(d["bytes"])).convert("RGB") for d in df["img"]]
    return imgs, df["label"].to_numpy()


@torch.inference_mode()
def extract(model, imgs, tag):
    feats = []
    t0 = time.time()
    for i in range(0, len(imgs), BATCH):
        x = torch.stack([cifar_transform(im) for im in imgs[i:i + BATCH]])
        dev = next(model.parameters()).device
        feats.append(model(x.to(dev)).cpu())
        if (i // BATCH) % 20 == 0:
            print(f"[{tag}] {i + len(x)}/{len(imgs)}  {time.time() - t0:.0f}s", flush=True)
    return torch.cat(feats), time.time() - t0


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    tr_imgs, tr_y = load_split("train")
    te_imgs, te_y = load_split("test")

    # 50.000 eğitim görüntüsü -> %90 train (45.000) / %10 validation (5.000), sınıf dengeli
    idx = np.arange(len(tr_imgs))
    i_tr, i_va = train_test_split(idx, test_size=0.1, stratify=tr_y, random_state=SEED)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_backbone().to(dev)
    splits = {
        "train": ([tr_imgs[i] for i in i_tr], tr_y[i_tr]),
        "val": ([tr_imgs[i] for i in i_va], tr_y[i_va]),
        "test": (te_imgs, te_y),
    }
    for name, (ims, ys) in splits.items():
        X, secs = extract(model, ims, name)
        torch.save({"X": X, "y": torch.tensor(ys), "seconds": secs}, f"{OUT_DIR}/{name}.pt")
        print(f"{name}: {tuple(X.shape)}  {secs:.0f}s")


if __name__ == "__main__":
    main()
