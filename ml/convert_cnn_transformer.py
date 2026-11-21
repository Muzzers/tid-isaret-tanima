"""1B-CNN + Transformer modelini Keras'ta ayni katmanlarla kurar, agirliklari birebir tasir ve
LiteRT (.tflite) fp32 ile 8-bit (dinamik aralik) surumlerini uretir.

convert_tflite.py'nin (BiGRU) aynisi, bu kez CNNTransformerNet (ml/models.py) icin. Telefondaki
girdi/cikti ayni kaldigi icin ([1, 64, 100] -> [1, 226] logit) uygulamada yalnizca model dosyasi degisir.

Dikkat edilen farklar (PyTorch -> Keras):
  - Conv1d agirligi (cikis, giris, k) -> Conv1D (k, giris, cikis)
  - stride=2, padding=1: Keras 'same' sag tarafa fazla doldurdugu icin acik ZeroPadding1D(1) + 'valid'
  - BatchNorm / LayerNorm epsilon 1e-5 (Keras varsayilani 1e-3)
  - GELU tam (erf) surumu, PyTorch varsayilani
  - norm_first=True: x + MHA(LN(x)), x + FF(LN(x))

ml/.venv-convert (tensorflow) ile calisir. Girdi: export_weights.py ciktilari.
    python ml/export_weights.py --name cnn_transformer_all_s1          (ml/.venv)
    python ml/convert_cnn_transformer.py --name cnn_transformer_all_s1  (ml/.venv-convert)
"""
import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras
from keras import layers

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="cnn_transformer_all_s1")
a = ap.parse_args()

W = np.load(ROOT / "checkpoints" / f"{a.name}_weights.npz")
T, F, D, HEADS, LAYERS = 64, 100, 256, 4, 2
n_cls = W["out__weight"].shape[0]


def gelu(t):
    return keras.activations.gelu(t, approximate=False)


def konum_kodlamasi(n, d):
    pos = np.arange(n, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, d, 2, dtype=np.float32) * (-math.log(10000.0) / d))
    pe = np.zeros((n, d), np.float32)
    pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
    return pe


class SabitEkle(layers.Layer):
    """Sinus konum kodlamasini ekler (PyTorch'taki PositionalEncoding)."""

    def __init__(self, pe, **kw):
        super().__init__(**kw)
        self.pe = tf.constant(pe[None])

    def call(self, x):
        return x + self.pe


inp = keras.Input((T, F))
c1 = layers.Conv1D(128, 5, padding="same")
b1 = layers.BatchNormalization(epsilon=1e-5)
x = layers.Activation(gelu)(b1(c1(inp)))
x = layers.ZeroPadding1D(1)(x)
c2 = layers.Conv1D(D, 3, strides=2, padding="valid")
b2 = layers.BatchNormalization(epsilon=1e-5)
x = layers.Activation(gelu)(b2(c2(x)))
c3 = layers.Conv1D(D, 3, padding="same")
b3 = layers.BatchNormalization(epsilon=1e-5)
x = layers.Activation(gelu)(b3(c3(x)))
x = SabitEkle(konum_kodlamasi(T // 2, D))(x)

katman = []
for _ in range(LAYERS):
    n1, n2 = layers.LayerNormalization(epsilon=1e-5), layers.LayerNormalization(epsilon=1e-5)
    mha = layers.MultiHeadAttention(HEADS, D // HEADS)
    f1, f2 = layers.Dense(2 * D, activation=gelu), layers.Dense(D)
    y = n1(x)
    x = layers.Add()([x, mha(y, y)])
    x = layers.Add()([x, f2(f1(n2(x)))])
    katman.append((n1, n2, mha, f1, f2))
x = layers.GlobalAveragePooling1D()(x)
son_norm = layers.LayerNormalization(epsilon=1e-5)
cikis = layers.Dense(n_cls)
model = keras.Model(inp, cikis(son_norm(x)))


def conv(layer, k):
    layer.set_weights([W[f"conv__{k}__weight"].transpose(2, 1, 0), W[f"conv__{k}__bias"]])


def bn(layer, k):
    layer.set_weights([W[f"conv__{k}__{n}"] for n in ("weight", "bias", "running_mean", "running_var")])


conv(c1, 0); bn(b1, 1); conv(c2, 4); bn(b2, 5); conv(c3, 7); bn(b3, 8)
for l, (n1, n2, mha, f1, f2) in enumerate(katman):
    p = f"transformer__layers__{l}__"
    wq, wk, wv = np.split(W[p + "self_attn__in_proj_weight"], 3)
    bq, bk, bv = np.split(W[p + "self_attn__in_proj_bias"], 3)
    hd = D // HEADS
    mha.set_weights([
        wq.T.reshape(D, HEADS, hd), bq.reshape(HEADS, hd),
        wk.T.reshape(D, HEADS, hd), bk.reshape(HEADS, hd),
        wv.T.reshape(D, HEADS, hd), bv.reshape(HEADS, hd),
        W[p + "self_attn__out_proj__weight"].T.reshape(HEADS, hd, D), W[p + "self_attn__out_proj__bias"],
    ])
    n1.set_weights([W[p + "norm1__weight"], W[p + "norm1__bias"]])
    n2.set_weights([W[p + "norm2__weight"], W[p + "norm2__bias"]])
    f1.set_weights([W[p + "linear1__weight"].T, W[p + "linear1__bias"]])
    f2.set_weights([W[p + "linear2__weight"].T, W[p + "linear2__bias"]])
son_norm.set_weights([W["norm__weight"], W["norm__bias"]])
cikis.set_weights([W["out__weight"].T, W["out__bias"]])

Dt = np.load(ROOT / "data" / "export" / f"{a.name}_test.npz")
x_test, y, ref = Dt["x"], Dt["y"], Dt["ref"]
keras_out = model.predict(x_test[:256], verbose=0)
print("Keras-PyTorch en buyuk fark (256 ornek):", float(np.abs(keras_out - ref[:256]).max()))

out = ROOT / "ml" / "export"
out.mkdir(exist_ok=True)


def run_tflite(path):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    res = []
    for k in range(len(x_test)):
        it.set_tensor(i, x_test[k : k + 1])
        it.invoke()
        res.append(it.get_tensor(o)[0].copy())
    return np.stack(res)


# Sabit [1, 64, 100] girdi: telefondaki SignClassifier ile ayni. Keras 3'te agirliklarin dosyaya
# gomulmesi icin once SavedModel olarak disa aktarilip oradan donusturuluyor.
import tempfile
kayit = Path(tempfile.mkdtemp()) / "model"
model.export(str(kayit), input_signature=[tf.TensorSpec([1, T, F], tf.float32)], verbose=False)
for ad, opt in (("fp32", False), ("int8", True)):
    conv_ = tf.lite.TFLiteConverter.from_saved_model(str(kayit))
    if opt:
        conv_.optimizations = [tf.lite.Optimize.DEFAULT]  # agirliklar 8 bit (dinamik aralik)
    b = conv_.convert()
    p = out / f"{a.name}_{ad}.tflite"
    p.write_bytes(b)
    r = run_tflite(p)
    print(f"{ad}: {len(b) / 1e6:.2f} MB | test dogrulugu {(r.argmax(1) == y).mean():.4f} "
          f"(PyTorch {(ref.argmax(1) == y).mean():.4f}) | PyTorch ile ayni tahmin {(r.argmax(1) == ref.argmax(1)).mean():.4f}")
