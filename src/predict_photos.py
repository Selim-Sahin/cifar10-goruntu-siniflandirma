"""Adım 3: Kendi telefon fotoğraflarını modelle test et ve görsel bir özet üret.

Kullanım:  python src/predict_photos.py
Girdi:     my_photos/ klasöründeki fotoğraflar + my_photos/labels.csv
           Sütunlar: dosya, gercek_sinif, kirpma (isteğe bağlı: "x0 y0 x1 y1", nesnenin etrafı)
           gercek_sinif: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
           ya da CIFAR-10'da olmayan bir nesne için 'yok'
Üç koşul test edilir:
  A) Tam fotoğraf, CIFAR benzeri ön işleme (32x32'ye küçült)  -- ilk tasarım
  B) Tam fotoğraf, doğrudan 224x224                              -- uygulamanın varsayılanı
  C) Nesnenin etrafından kırpılmış fotoğraf, doğrudan 224x224
Çıktı:     results/my_photos_results.png, results/my_photos_results.csv
"""
import csv, os, sys, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
from common import Head, FullModel, photo_transform, CLASSES_EN, CLASSES_TR

head = Head(); head.load_state_dict(torch.load("model/head.pt", map_location="cpu"))
model = FullModel(head).eval()


@torch.inference_mode()
def run(img, cifar_like):
    x, view = photo_transform(img, cifar_like=cifar_like)
    t = time.perf_counter(); p = torch.softmax(model(x.unsqueeze(0)), 1)[0]; ms = (time.perf_counter() - t) * 1000
    return p, view, ms


rows = [r for r in csv.DictReader(open("my_photos/labels.csv", encoding="utf-8"))]
out, n = [], len(rows)
fig, axes = plt.subplots(n, 5, figsize=(19, 3.3 * n), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1.1, 0.75, 1.1]))
tr = lambda i: f"{CLASSES_TR[i]} ({CLASSES_EN[i]})"


def bars(ax, p, true, title):
    top = p.topk(3)
    pred = CLASSES_EN[top.indices[0]]
    ok = pred == true
    col = "#2a9d8f" if ok else "#e76f51"
    ax.barh([tr(i) for i in top.indices][::-1], top.values.numpy()[::-1], color=["#9aa", "#9aa", col])
    ax.set_xlim(0, 1); ax.tick_params(labelsize=8)
    ax.set_title(f"{title}\n→ {pred} %{top.values[0] * 100:.0f}  {'DOĞRU' if ok else 'YANLIŞ'}", fontsize=9, color=col)
    return pred, float(top.values[0]), ok


for k, r in enumerate(rows):
    img = ImageOps.exif_transpose(Image.open(os.path.join("my_photos", r["dosya"]))).convert("RGB")
    true = r["gercek_sinif"]
    box = [int(v) for v in r.get("kirpma", "").split()] if r.get("kirpma") else None
    pA, _, _ = run(img, True)
    pB, _, msB = run(img, False)
    crop = img.crop(box) if box else img
    pC, _, _ = run(crop, False)

    a = axes[k]
    a[0].imshow(img); a[0].axis("off")
    gt = tr(CLASSES_EN.index(true)) if true in CLASSES_EN else "Sınıf dışı (bisiklet)"
    a[0].set_title(f"{r['dosya']} — Gerçek: {gt}", fontsize=9)
    if box:
        a[0].add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec="red", lw=2))
    predA, cA, okA = bars(a[1], pA, true, "A) Tam foto, 32×32 ön işleme")
    predB, cB, okB = bars(a[2], pB, true, "B) Tam foto, doğrudan (varsayılan)")
    a[3].imshow(crop.resize((224, 224))); a[3].axis("off"); a[3].set_title("Kırpılmış (kırmızı kutu)", fontsize=9)
    predC, cC, okC = bars(a[4], pC, true, "C) Nesneye kırpılmış, doğrudan")
    out.append([r["dosya"], true, predA, f"{cA:.3f}", int(okA), predB, f"{cB:.3f}", int(okB), predC, f"{cC:.3f}", int(okC), f"{msB:.1f}"])

plt.tight_layout(); plt.savefig("results/my_photos_results.png", dpi=110); plt.close()
with open("results/my_photos_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dosya", "gercek", "A_tahmin", "A_guven", "A_dogru", "B_tahmin", "B_guven", "B_dogru", "C_tahmin", "C_guven", "C_dogru", "B_sure_ms"])
    w.writerows(out)
for o in out:
    print(o)
for c, i in (("A", 4), ("B", 7), ("C", 10)):
    print(c, f"doğru: {sum(o[i] for o in out)}/{len(out)}")
