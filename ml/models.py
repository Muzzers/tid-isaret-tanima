"""Iki siniflandirici (BiGRU ve 1D-CNN + Transformer) ile kayitli agirligi yukleme yardimcilari.

Ikisi de ayni girdiyi alir: [B, 64, 100]. Cikis sinif sayisi kadar logit.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


class GRUNet(nn.Module):
    def __init__(self, n_cls, in_dim=100, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(hidden * 2, n_cls)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h.mean(1))


class PositionalEncoding(nn.Module):
    """Transformer zamansal sirayi kendisi gormedigi icin sinus tabanli konum ekliyoruz."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CNNTransformerNet(nn.Module):
    """1D-CNN ile yerel hareket ozellikleri, ardindan kucuk Transformer ile zaman iliskileri.

    GRU'ya gore fark: tekrarlayan katman yok, egitim paralellestirilebilir. Model yine de
    hafif kalsin diye d_model=256 ve 2 katman secildi (~1,5 M parametre).
    """

    def __init__(self, n_cls, in_dim=100, d_model=256, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, d_model, 3, padding=1, stride=2), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_cls)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T/2, d_model]
        h = self.transformer(self.pos(h))
        return self.out(self.norm(h.mean(1)))


def pick_device():
    # Mac'te MPS, egitim makinesinde CUDA; ikisi de yoksa CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(name):
    """checkpoints/<name>.pt dosyasini okur, dogru mimariyi kurup agirliklari yukler."""
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location="cpu")
    # model_type alani olmayan eski kayitlar (gru226.pt) BiGRU'dur.
    if ck.get("model_type") == "cnn_transformer":
        model = CNNTransformerNet(ck["n_classes"])
    else:
        model = GRUNet(ck["n_classes"], hidden=ck["hidden"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck
