import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// assets/reference.bin: her isaret icin egitim verisinden secilmis ornek hareket (ml/export_reference.py).
/// Degerler egitimdeki gibi normalize: omuz ortasi (0, 0), omuz genisligi 1; int16 = deger x 1000.
class Referans {
  static ByteData? _veri;
  static int sinifSayisi = 0, kareSayisi = 0, boyut = 0;

  static Future<void> yukle() async {
    final b = await rootBundle.load('assets/reference.bin');
    sinifSayisi = b.getInt32(4, Endian.little);
    kareSayisi = b.getInt32(8, Endian.little);
    boyut = b.getInt32(12, Endian.little);
    _veri = b;
  }

  static bool get hazir => _veri != null;

  /// [sinif] isaretinin [kare]. karesindeki 100 deger: 84 el, 14 govde, 2 el-var bayragi.
  static List<double> kare(int sinif, int kare) {
    final b = _veri!;
    final bas = 16 + ((sinif * kareSayisi + kare) * boyut) * 2;
    return List.generate(boyut, (i) => b.getInt16(bas + 2 * i, Endian.little) / 1000.0);
  }
}

// MediaPipe el iskeleti baglantilari (21 nokta)
const _elBaglanti = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
// Govde: 0 burun, 1-2 omuzlar, 3-4 dirsekler, 5-6 bilekler
const _govdeBaglanti = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6]];

class IskeletCizici extends CustomPainter {
  final List<double> v;
  final bool ayna;
  final Color govde, el1, el2;

  IskeletCizici(this.v, {required this.ayna, required this.govde, required this.el1, required this.el2});

  @override
  void paint(Canvas canvas, Size size) {
    // Gorunen alan (omuz genisligi biriminde): x -1,4..1,4, y -1,3..2,1
    const x0 = -1.4, x1 = 1.4, y0 = -1.3, y1 = 2.1;
    final olcek = (size.width / (x1 - x0)).clamp(0.0, size.height / (y1 - y0));
    final ox = size.width / 2, oy = (size.height - (y1 - y0) * olcek) / 2 - y0 * olcek;
    Offset nokta(double x, double y) => Offset(ox + (ayna ? -x : x) * olcek, oy + y * olcek);

    final kalem = Paint()
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final g = List.generate(7, (k) => nokta(v[84 + 2 * k], v[85 + 2 * k]));
    kalem
      ..color = govde
      ..strokeWidth = olcek * 0.07;
    for (final b in _govdeBaglanti) {
      canvas.drawLine(g[b[0]], g[b[1]], kalem);
    }
    canvas.drawCircle(g[0], olcek * 0.3, kalem);

    for (var s = 0; s < 2; s++) {
      if (v[98 + s] < 0.5) continue;
      final p = List.generate(21, (j) => nokta(v[s * 42 + 2 * j], v[s * 42 + 2 * j + 1]));
      kalem
        ..color = s == 0 ? el1 : el2
        ..strokeWidth = olcek * 0.03;
      for (final b in _elBaglanti) {
        canvas.drawLine(p[b[0]], p[b[1]], kalem);
      }
      final dolgu = Paint()..color = kalem.color;
      for (final q in p) {
        canvas.drawCircle(q, olcek * 0.022, dolgu);
      }
    }
  }

  @override
  bool shouldRepaint(IskeletCizici old) => old.v != v || old.ayna != ayna;
}

/// Bir isaretin ornek hareketini dongu halinde oynatir (2 sn hareket + 0,6 sn bekleme).
class IskeletAnimasyonu extends StatefulWidget {
  final int sinif;
  final bool ayna;
  final double hiz;

  const IskeletAnimasyonu({super.key, required this.sinif, this.ayna = false, this.hiz = 1.0});

  @override
  State<IskeletAnimasyonu> createState() => _IskeletAnimasyonuState();
}

class _IskeletAnimasyonuState extends State<IskeletAnimasyonu> with SingleTickerProviderStateMixin {
  static const _hareket = 2.0, _bekleme = 0.6;
  late final AnimationController _c = AnimationController(vsync: this);

  @override
  void initState() {
    super.initState();
    _baslat();
  }

  void _baslat() {
    _c.duration = Duration(milliseconds: ((_hareket + _bekleme) * 1000 / widget.hiz).round());
    _c.repeat();
  }

  @override
  void didUpdateWidget(IskeletAnimasyonu old) {
    super.didUpdateWidget(old);
    if (old.hiz != widget.hiz) _baslat();
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!Referans.hazir) return const Center(child: CircularProgressIndicator());
    final renk = Theme.of(context).colorScheme;
    return AnimatedBuilder(
      animation: _c,
      builder: (_, _) {
        final t = (_c.value * (_hareket + _bekleme) / _hareket).clamp(0.0, 1.0);
        final kare = (t * (Referans.kareSayisi - 1)).round();
        return CustomPaint(
          size: Size.infinite,
          painter: IskeletCizici(
            Referans.kare(widget.sinif, kare),
            ayna: widget.ayna,
            govde: renk.onSurfaceVariant,
            el1: renk.primary,
            el2: renk.tertiary,
          ),
        );
      },
    );
  }
}
