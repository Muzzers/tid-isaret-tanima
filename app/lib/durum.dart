import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Kotlin tarafiyla konusan kanallar (MainActivity.kt).
const kontrol = MethodChannel('tid/control');
const _olaylar = EventChannel('tid/landmarks');

/// Kotlin'den gelen olaylar: yarim saniyede bir ozet (FPS, el sayisi) ya da {'segment': ...}.
/// Kanal tek dinleyici kabul ettigi icin uygulama boyunca tek bir abonelik paylasilir.
final Stream<Map> olayAkisi = _olaylar.receiveBroadcastStream().map((e) => e as Map).asBroadcastStream();

class Tahmin {
  final String ad;
  final double p;
  final int id;
  Tahmin(Map m)
      : ad = '${m['name']}',
        p = (m['p'] as num).toDouble(),
        id = (m['id'] as num?)?.toInt() ?? -1;

  String get yuzde => '%${(p * 100).toStringAsFixed(0)}';
}

/// Ekranlar arasinda paylasilan durum: biriken metin, guven esigi, kamera ayarlari, isaret listesi.
class UygulamaDurumu extends ChangeNotifier {
  final List<String> kelimeler = [];
  double esik = 0.6; // ml/calibrate_threshold.py: 0,6'da kabul edilenlerde test dogrulugu ~%97
  bool onKamera = true;
  bool surekli = false;
  int hizModu = 1; // 0 hepsi CPU, 1 el GPU + poz CPU, 2 hepsi GPU
  List<String> isaretler = const []; // sinif numarasi sirasiyla

  void ekle(String kelime) {
    kelimeler.add(kelime);
    notifyListeners();
  }

  void sil(int i) {
    kelimeler.removeAt(i);
    notifyListeners();
  }

  void temizle() {
    kelimeler.clear();
    notifyListeners();
  }

  void ayarla(VoidCallback degisiklik) {
    degisiklik();
    notifyListeners();
  }

  Future<void> isaretleriYukle() async {
    isaretler = await kontrol.invokeListMethod<String>('signs') ?? const [];
    notifyListeners();
  }
}

final durum = UygulamaDurumu();

const _harfler = 'aâbcçdefgğhıiîjklmnoöprsştuûüvyz';

/// Turkce alfabe sirasina gore siralama anahtari (c < ç, o < ö ...).
String siraAnahtari(String s) {
  final k = StringBuffer();
  for (final r in s.replaceAll('I', 'ı').replaceAll('İ', 'i').toLowerCase().runes) {
    final c = String.fromCharCode(r);
    final i = _harfler.indexOf(c);
    k.writeCharCode(i < 0 ? r : 0x100 + i);
  }
  return k.toString();
}

/// Aramada "corba" yazinca "çorba" da bulunsun diye harfleri sadelestirir.
String sadelestir(String s) {
  const a = 'çğıöşüâîûÇĞİÖŞÜI', b = 'cgiosuaiucgiosui';
  final k = StringBuffer();
  for (final r in s.runes) {
    final c = String.fromCharCode(r);
    final i = a.indexOf(c);
    k.write(i < 0 ? c : b[i]);
  }
  return k.toString().toLowerCase();
}
