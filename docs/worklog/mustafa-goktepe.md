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

**Ölçülen / doğrulanan**
- Parametre sayısı: BiGRU-2x256 ≈ 2,11 M, 1D-CNN+Transformer ≈ 1,47 M (10 MB bütçesinin altında).
- 226 sınıflık gerçek nokta verisiyle (28.132 train / 4.418 val / 3.741 test) 5 epoch hızlı kontrol koşusu:
  - `BiGRU-2x256`: en iyi val %82,1 | test %80,4 (macro F1 %80,3)
  - `1D-CNN+Transformer`: en iyi val %81,7 | test %81,5 (macro F1 %81,2)
- `ml/error_analysis.py` eklendi: kayıtlı checkpoint üzerinden sınıf bazlı F1, en çok karışan 10 işaret çifti tablosu, karışıklık matrisi (`cm_*.png`) ve eğitim eğrisi (`curve_*.png`) üretiyor.

**Sırada**
- 60 epoch tam eğitim ve 3 tohumlu (`seed 0,1,2`) karşılaştırma tablosunun `experiments/` altına eklenmesi.
