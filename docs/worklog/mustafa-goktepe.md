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
- Sentetik 12 sınıflık küçük sette iki model de 2 epoch eğitildi; ileri/geri yayılım,
  checkpoint ve JSON yazımı çalışıyor (asıl veri gelince tekrar koşulacak).
- Parametre sayısı: BiGRU-2x256 ≈ 2,08 M, 1D-CNN+Transformer ≈ 1,45 M (10 MB bütçesinin altında).

**Blok / sırada**
- `data/keypoints` bende yok (Drive linki oturum istiyor). Nokta dosyaları gelince
  3 tohumla iki modeli koşup `experiments/` altına ekleyeceğim, ara rapora tablo girecek.
- Tam eğitimde önce 5 epoch'luk hızlı koşu ile yakınsama kontrolü, sonra 60 epoch.
