"""Gradio uygulaması: fotoğraf yükle -> CIFAR-10 sınıf tahmini (ilk 3 olasılık).

Çalıştırma:  python app.py        (yerel: http://127.0.0.1:7860)
Colab:       python app.py --share (herkese açık geçici link verir)
"""
import os, sys, time
import torch
import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import Head, FullModel, photo_transform, CLASSES_EN, CLASSES_TR, CONF_THRESHOLD

head = Head()
head.load_state_dict(torch.load(os.path.join(ROOT, "model", "head.pt"), map_location="cpu"))
model = FullModel(head).eval()
LABELS = [f"{tr} ({en})" for tr, en in zip(CLASSES_TR, CLASSES_EN)]


@torch.inference_mode()
def predict(img, cifar_like):
    if img is None:
        return None, None, ""
    x, view = photo_transform(img, cifar_like=cifar_like)
    t = time.perf_counter()
    probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0]
    ms = (time.perf_counter() - t) * 1000
    info = f"Tahmin süresi: {ms:.1f} ms"
    if probs.max() < CONF_THRESHOLD:
        info += (f"\n⚠️ Model emin değil (en yüksek olasılık %{probs.max() * 100:.0f}). "
                 "Nesne bu 10 sınıftan biri olmayabilir ya da kadrajda çok küçük olabilir.")
    preview = view.resize((224, 224), resample=0)
    return {LABELS[i]: float(probs[i]) for i in range(10)}, preview, info


ex_dir = os.path.join(ROOT, "examples")
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Fotoğraf yükle / kamera"),
            gr.Checkbox(value=False, label="CIFAR benzeri ön işleme (32×32'ye küçült)")],
    outputs=[gr.Label(num_top_classes=3, label="Tahmin"),
             gr.Image(label="Modelin gördüğü görüntü"),
             gr.Textbox(label="Bilgi", lines=3)],
    title="CIFAR-10 Görüntü Sınıflandırma — ResNet18 Transfer Learning",
    description="Sınıflar: uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi, kamyon. "
                "İpucu: nesne fotoğrafın ortasında olsun ve kadrajı doldursun; gerekirse fotoğrafı "
                "nesnenin etrafından kırpıp yükleyin. Fotoğrafın ortasından kare bir parça kullanılır.",
    examples=[[os.path.join(ex_dir, f), False] for f in sorted(os.listdir(ex_dir))] if os.path.isdir(ex_dir) else None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
