# ml/: nokta çıkarımı ve model eğitimi

## Dosyalar

| Dosya | İş |
|---|---|
| `extract_keypoints.py` | videodan MediaPipe ile el + gövde noktaları (`.npz`) |
| `features.py` | normalizasyon, 64 kareye örnekleme, artırma, batch |
| `models.py` | `GRUNet`, `CNNTransformerNet`, kayıtlı ağırlığı yükleme |
| `train_gru.py` | eğitim döngüsü (`--model gru` ya da `cnn_transformer`) |
| `compare_models.py` | model × tohum koşuları, özet tablo |
| `error_analysis.py` | sınıf bazlı F1, karışıklık matrisi, eğitim eğrisi |
| `demo.py` | hazır ağırlıkla örnek tahminler ve test doğruluğu |
| `calibrate_threshold.py`, `tune_segmentation.py` | güven eşiği ve segment kuralları |
| `export_weights.py`, `convert_tflite.py`, `export_selftest.py` | LiteRT dönüşümü (BiGRU) ve telefon öz-testi |
| `convert_cnn_transformer.py` | 1B-CNN + Transformer'ın LiteRT dönüşümü (uygulamadaki model) |
| `export_reference.py` | uygulamadaki işaret listesi için her kelimenin örnek hareketi |
| `live_analysis.py` | gönüllü testi kayıtlarını inceleme, AUTSL ile karşılaştırma |

## Ortam (Python 3.12)

```bash
uv venv --python 3.12 ml/.venv
uv pip install --python ml/.venv/Scripts/python.exe -r ml/requirements.txt
```

## MediaPipe modelleri (resmi Google adresi, repoya girmez)

```bash
mkdir -p ml/models
curl -L -o ml/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o ml/models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Nokta çıkarımı

Veri `data/archive (1)/` altında olmalıdır (Kaggle `sttaseen/autsl`, bkz. docs/data.md).

```bash
ml/.venv/Scripts/python.exe ml/extract_keypoints.py --split train --classes 30,175,65 --workers 4 --out data/keypoints_10
```

## Android uygulamasına model kopyalama

```bash
cp ml/models/hand_landmarker.task ml/models/pose_landmarker_lite.task app/android/app/src/main/assets/
```

## Tüm veri, birden çok makinede

Her makine kendi parçasını işler (0, 1, 2 / 3); çıktılar `data/keypoints/<split>/` altında birleştirilir:

```bash
ml/.venv/Scripts/python.exe ml/extract_keypoints.py --split train --shard 0/3 --workers 8 --out data/keypoints
```

`val` ve `test` için aynı komutu `--split` değiştirerek çalıştırın. Ölçülen hız: 10 işçi ile ~0,77 sn/video (toplam ~36.300 video ≈ 7,7 saat tek makinede, 3 makinede ~2,6 saat).

## Model karşılaştırma (İP3)

İki model aynı veri, artırma ve değerlendirme ile eğitilir; tek değişken mimaridir.

```bash
# tek koşu
ml/.venv/Scripts/python.exe ml/train_gru.py --model cnn_transformer --epochs 60 --name cnn226

# iki model x 3 tohum, sonunda markdown tablo + experiments/ altina ozet JSON
ml/.venv/Scripts/python.exe ml/compare_models.py --seeds 0,1,2 --epochs 60

# hata analizi, sinif bazli F1 ve karisiklik matrisi grafigi
ml/.venv/Scripts/python.exe ml/error_analysis.py --name gru226
```

Mac/Linux'ta aynı komutlar `ml/.venv/bin/python` ile. Eğitim cihazı sırayla CUDA → MPS → CPU seçilir.

