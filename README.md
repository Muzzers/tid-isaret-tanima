# TİD İşaret Tanıma

Telefon kamerasından alınan Türk İşaret Dili hareketini (AUTSL, 226 kelime) MediaPipe ile çıkarılan el ve
üst gövde noktalarından sınıflandırıp Türkçe kelime olarak gösteren, internetsiz çalışan Android uygulaması.
Erciyes Üniversitesi Bilgisayar Mühendisliği bitirme projesi.

Ekip: Ahmet Çelik, Mustafa Göktepe, Mustafa Küpeli.

## Durum (Kasım 2026)

| Ölçüt | Hedef | Ölçülen |
|---|---|---|
| Test doğruluğu, 1B-CNN + Transformer (3 tohum) | ≥ %85 | %89,42 ± 0,52 |
| Test doğruluğu, BiGRU (3 tohum) | ≥ %85 | %87,98 ± 0,23 |
| Uygulamadaki model (1B-CNN + Transformer, int8 LiteRT) | ≤ 10 MB | 1,61 MB, test %89,82 |
| Python ↔ telefon tahmin uyumu (BiGRU) | — | %99,5 |
| Uçtan uca hız (Redmi Note 10 Pro, önceki sürüm) | ≥ 15 FPS | ~9 FPS |
| Gerçek ortam doğruluğu (gönüllü testi) | ≥ %75 | ölçülecek |

Uygulama üç ekrandan oluşur: **Tanı** (kamera, altyazı, tek işaret / sürekli mod), **Metin** (biriken
kelimeler, seslendirme, kopyalama) ve **İşaretler** (226 kelime, arama, her kelimenin örnek hareketi
iskelet animasyonu olarak). İşaretler ekranındaki **Gönüllü testi** gerçek ortam doğruluğunu ölçer;
kayıtlar `ml/live_analysis.py` ile incelenir. Uygulamanın internet izni yoktur.

Bitirme tezi taslağı: [docs/tez/tez.pdf](docs/tez/tez.pdf). Henüz ölçülmemiş değerler PDF'te kırmızı
"ÖLÇÜLECEK" olarak işaretli.

## Klasörler

```
ml/            nokta çıkarımı, eğitim, değerlendirme, LiteRT dönüşümü (ml/README.md)
app/           Flutter + Kotlin uygulaması (CameraX, MediaPipe, LiteRT)
experiments/   deney kayıtları (JSON) ve grafikler
docs/          durum, kurulum, veri notları, ara rapor (docs/vize), bitirme tezi (docs/tez), worklog
PLAN.md        ayrıntılı plan, riskler, ölçüm tabloları
```

Veri (`data/`), eğitim ağırlıkları (`checkpoints/`) ve Flutter SDK repoya girmez.

## Hızlı başlangıç

Kurulum: [docs/KURULUM.md](docs/KURULUM.md). Ortam hazırsa:

```bash
# hazır ağırlıkla test kümesinde örnek tahminler ve doğruluk
ml/.venv/bin/python ml/demo.py --name cnn_transformer_all_s1

# 10 sınıfta birkaç dakikalık eğitim denemesi
ml/.venv/bin/python ml/train_gru.py --model cnn_transformer --classes 30,175,65,86,196,211,83,52,14,20 --epochs 5 --name deneme
```

VS Code kullananlar için aynı adımlar Çalıştır ve Hata Ayıkla menüsünde hazır (`.vscode/launch.json`).
Windows'ta Python yolu `ml/.venv/Scripts/python.exe`.

## Uygulamayı derleme ve telefona yükleme

```bash
cd app && ../flutter/bin/flutter build apk --release
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Flutter proje içindeki `flutter/` klasöründe 3.44.8 sürümüne sabitlidir (3.50 projeyi derlemiyor).
