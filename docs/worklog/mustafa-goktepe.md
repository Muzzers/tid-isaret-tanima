# Mustafa Göktepe — worklog

## 3 Kasım 2026 (Hafta 6)

**Ne yaptım?**
- `ml/train_gru.py`'ye `--model` seçeneği ekledim. Artık aynı betik hem BiGRU hem de
  1D-CNN + Transformer modelini eğitiyor. Veri, artırma ve değerlendirme kodu tek yerde
  kaldığı için iki modeli karşılaştırırken tek değişken mimari oluyor.
- `CNNTransformerNet`: 3 katmanlı 1D-CNN (100 → 128 → 256 → 256, ikinci katmanda stride 2),
  sinus konum kodlaması, 2 katmanlı küçük TransformerEncoder (d_model 256, 4 head,
  ff 512, GELU, norm_first). Çıkışta zaman ortalaması + LayerNorm.
- `ml/compare_models.py`: model × tohum çalıştırıp `experiments/` altına özet JSON ve
  markdown tablo yazıyor. Her koşu ayrı süreç, biri çökerse diğerleri devam ediyor.
- `ml/export_weights.py` yeni mimariyi de yükleyebiliyor (`model_type` alanı checkpoint'e yazılıyor).
- Eğitim cihazı seçimi: CUDA yoksa MPS (Mac), o da yoksa CPU. Aynı kod hem kendi Mac'imde
  hem ekibin NVIDIA makinesinde çalışsın diye.

**Ölçülen / doğrulanan (226 sınıf, 28.132 train / 4.418 val / 3.741 test, 60 epoch, seed 0)**
- `BiGRU-2x256` (2,11 M parametre): en iyi val **%89,25** | test **%88,24** | macro F1 **%88,09** (M4 MPS, 4658 sn)
- `1D-CNN+Transformer` (1,47 M parametre): en iyi val **%90,02** | test **%88,83** | macro F1 **%88,61** (M4 MPS, 1089 sn)
- `1D-CNN+Transformer`, BiGRU'ya göre %30 daha az parametreyle hem doğrulukta (+0,59 puan test) hem eğitim hızında (~4,3 kat daha hızlı) öne geçti.
- `ml/error_analysis.py` ile iki modelin de test kümesi hata analizi, en çok karışan 10 işaret çifti (`ağır↔hafif`, `sabun↔ayakkabı`, `okul↔çorba`, `dede↔dolu`), F1 skoru en düşük 10 sınıf tablosu, karışıklık matrisi (`cm_*.png`) ve eğitim eğrisi (`curve_*.png`) üretildi.

**Sırada**
- Ek tohumlarla (`seed 1,2`) standart sapma aralığının çıkarılması ve ara rapor / sunum tablolarının güncellenmesi.
