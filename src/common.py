"""Ortak tanımlar: sınıf adları, ön işleme ve model kurulumu."""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

CLASSES_EN = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
CLASSES_TR = ["Uçak", "Otomobil", "Kuş", "Kedi", "Geyik",
              "Köpek", "Kurbağa", "At", "Gemi", "Kamyon"]

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# CIFAR-10 görüntüleri (32x32) -> 224x224'e büyütülür, ImageNet normalizasyonu uygulanır.
cifar_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


CONF_THRESHOLD = 0.5  # en yüksek olasılık bunun altındaysa "emin değil" uyarısı


def photo_transform(img: Image.Image, cifar_like: bool = False):
    """Telefon fotoğrafı -> modele girdi. Döndürür: (tensor, modelin gördüğü görüntü)

    Her iki modda da fotoğraf önce merkezden kare kırpılır.
    - cifar_like=False (varsayılan): kare doğrudan 224x224'e küçültülür.
    - cifar_like=True : önce 32x32'ye küçültülür, sonra eğitimdeki gibi 224x224'e
      büyütülür (CIFAR-10'un bulanıklığını taklit eder).
    5 telefon fotoğrafı üzerindeki testte doğrudan mod daha iyi sonuç verdi
    (bkz. rapor, Bölüm 7), bu yüzden varsayılan odur.
    """
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if cifar_like:
        small = img.resize((32, 32), Image.BICUBIC)
        return cifar_transform(small), small
    view = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])(view), view


def build_backbone() -> nn.Module:
    """ImageNet ile önceden eğitilmiş ResNet18, son katman (fc) çıkarılmış -> 512-d öznitelik."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


class Head(nn.Module):
    """Eğitilen sınıflandırıcı başlık: 512 -> 256 -> 10."""

    def __init__(self, in_dim=512, hidden=256, n_classes=10, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(p_drop),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits; olasılık için softmax uygulanır


class FullModel(nn.Module):
    """Uygulamada kullanılan uçtan uca model: backbone + head."""

    def __init__(self, head: Head):
        super().__init__()
        self.backbone = build_backbone()
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))
