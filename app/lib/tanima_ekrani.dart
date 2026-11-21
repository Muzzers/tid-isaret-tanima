import 'dart:async';

import 'package:flutter/material.dart';

import 'durum.dart';
import 'kamera.dart';

/// Canli tanima. Ekranin ustu kamera, alti eski surumdeki gibi bilgi alani: mod secimi, metin ya da
/// son tahminin ilk 3 adayi, guven esigi ve "Hiz ayarlari ve olcumler". Kamera, MediaPipe ve model
/// Kotlin'de calisir; buraya yalnizca ozet bilgiler ve tahminler gelir (PLAN.md P-05).
class TanimaEkrani extends StatefulWidget {
  const TanimaEkrani({super.key});

  @override
  State<TanimaEkrani> createState() => _TanimaEkraniState();
}

class _TanimaEkraniState extends State<TanimaEkrani> {
  StreamSubscription<Map>? _abone;
  Map _istat = const {};
  bool _kayitta = false;
  Map? _elle; // tek isaret modunda son kaydin sonucu
  String _son = '';
  int _reddedilen = 0;
  String _ozTest = '';

  @override
  void initState() {
    super.initState();
    _abone = olayAkisi.listen((m) {
      if (!mounted) return;
      if (m['segment'] != null) {
        _segment(m['segment'] as Map);
      } else {
        setState(() => _istat = m);
      }
    });
  }

  @override
  void dispose() {
    _abone?.cancel();
    super.dispose();
  }

  String _bicim(Map m) => '${m['name']} %${((m['p'] as num) * 100).toStringAsFixed(0)}';

  /// Esigi gecen tahmin metne eklenir (Metin sekmesiyle ortak).
  bool _uygula(Map r) {
    final ilk = (r['top'] as List).cast<Map>().first;
    if ((ilk['p'] as num) >= durum.esik) {
      durum.ekle('${ilk['name']}');
      return true;
    }
    _reddedilen++;
    return false;
  }

  void _segment(Map r) {
    setState(() {
      if (r['error'] != null) {
        _son = '${r['error']}';
        return;
      }
      final ilk = (r['top'] as List).cast<Map>().first;
      _son = _uygula(r) ? _bicim(ilk) : 'Emin değilim (en yakın: ${_bicim(ilk)})';
    });
  }

  String _elleMetni() {
    if (_kayitta) return 'Kaydediliyor… ${_istat['recFrames'] ?? 0} kare. İşareti yapın, bitince "Bitir ve tanı"ya basın.';
    final r = _elle;
    if (r == null) return 'İşareti yapmadan önce kaydı başlatın.';
    if (r['error'] != null) return '${r['error']}';
    final top = (r['top'] as List).cast<Map>();
    final satirlar = top.map(_bicim).join('\n');
    final ayrinti = '\n${r['frames']} kare'
        '${r['handFrames'] != null ? ', elin göründüğü ${r['handFrames']}' : ''} · ${r['ms']} ms';
    return ((top.first['p'] as num) < durum.esik ? 'Emin değilim.\n' : '') + satirlar + ayrinti;
  }

  Future<void> _kayit() async {
    if (!_kayitta) {
      setState(() {
        _kayitta = true;
        _elle = null;
      });
      await kontrol.invokeMethod('record', true);
    } else {
      setState(() => _kayitta = false);
      final r = await kontrol.invokeMethod<Map>('record', false);
      if (!mounted) return;
      setState(() {
        _elle = r;
        if (r != null && r['error'] == null) _uygula(r);
      });
    }
  }

  void _surekli(bool s) {
    durum.ayarla(() => durum.surekli = s);
    kontrol.invokeMethod('setContinuous', s);
    setState(() {
      _kayitta = false;
      _elle = null;
    });
  }

  void _kameraDegistir() {
    durum.ayarla(() => durum.onKamera = !durum.onKamera);
    kontrol.invokeMethod('setLens', durum.onKamera);
    setState(() {});
  }

  Future<void> _ozTestCalistir() async {
    setState(() => _ozTest = 'Öz-test çalışıyor…');
    final r = await kontrol.invokeMethod<Map>('selftest');
    if (!mounted || r == null) return;
    setState(() => _ozTest = r['error'] != null
        ? '${r['error']}'
        : 'Öz-test: ${r['n']} örnek, doğruluk %${((r['accuracy'] as double) * 100).toStringAsFixed(1)}, '
            'Python ile aynı tahmin %${((r['agreePython'] as double) * 100).toStringAsFixed(1)}, '
            '${(r['msPerSample'] as double).toStringAsFixed(1)} ms/örnek');
  }

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    final renk = tema.colorScheme;
    final m = _istat;
    final hal = switch (m['state']) {
      'isaret' => 'İşaret algılanıyor…',
      'bekliyor' => 'Bekliyor (elleri kaldırın)',
      _ => '',
    };
    return Scaffold(
      appBar: AppBar(title: const Text('Tanı'), actions: [
        IconButton(tooltip: 'Öz-test', icon: const Icon(Icons.fact_check_outlined), onPressed: _ozTestCalistir),
        IconButton(
          tooltip: durum.onKamera ? 'Arka kameraya geç' : 'Ön kameraya geç',
          icon: Icon(durum.onKamera ? Icons.camera_rear_outlined : Icons.camera_front_outlined),
          onPressed: _kameraDegistir,
        ),
      ]),
      body: KameraIzni(
        child: Column(children: [
          Expanded(
            flex: 5,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 0),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Stack(fit: StackFit.expand, children: [
                  KameraGorunumu(surekli: durum.surekli),
                  Positioned(left: 10, top: 10, child: _Rozet(istat: m, kayitta: _kayitta)),
                ]),
              ),
            ),
          ),
          Expanded(
            flex: 5,
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                SegmentedButton<bool>(
                  segments: const [
                    ButtonSegment(value: false, label: Text('Elle kaydet'), icon: Icon(Icons.touch_app_outlined)),
                    ButtonSegment(value: true, label: Text('Sürekli'), icon: Icon(Icons.all_inclusive)),
                  ],
                  selected: {durum.surekli},
                  onSelectionChanged: (s) => _surekli(s.first),
                ),
                const SizedBox(height: 12),
                if (durum.surekli) ...[
                  _Kart(
                    baslik: 'Metin',
                    ikon: Icons.subject,
                    sag: hal.isEmpty ? null : _Etiket(hal, m['state'] == 'isaret' ? renk.primary : renk.outline),
                    child: ListenableBuilder(
                      listenable: durum,
                      builder: (_, _) => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(durum.kelimeler.isEmpty ? '(henüz yok)' : durum.kelimeler.join(' '),
                            style: tema.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600)),
                        const SizedBox(height: 10),
                        Row(children: [
                          Expanded(
                            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                              if (_son.isNotEmpty) Text('Son: $_son', style: tema.textTheme.bodyMedium),
                              Text('Reddedilen: $_reddedilen',
                                  style: tema.textTheme.bodySmall?.copyWith(color: renk.onSurfaceVariant)),
                            ]),
                          ),
                          TextButton.icon(
                            onPressed: () => setState(() {
                              durum.temizle();
                              _son = '';
                              _reddedilen = 0;
                            }),
                            icon: const Icon(Icons.delete_outline),
                            label: const Text('Temizle'),
                          ),
                        ]),
                      ]),
                    ),
                  ),
                ] else ...[
                  _Kart(
                    baslik: 'Sonuç',
                    ikon: Icons.insights_outlined,
                    sag: _elleEtiketi(renk),
                    child: _elleIcerik(tema),
                  ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: _kayit,
                    icon: Icon(_kayitta ? Icons.stop_rounded : Icons.fiber_manual_record),
                    label: Text(_kayitta ? 'Bitir ve tanı' : 'İşareti kaydet'),
                    style: FilledButton.styleFrom(
                      backgroundColor: _kayitta ? renk.error : null,
                      foregroundColor: _kayitta ? renk.onError : null,
                      minimumSize: const Size.fromHeight(54),
                      textStyle: tema.textTheme.titleMedium,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                  ),
                ],
                if (_ozTest.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _Kart(baslik: 'Öz-test', ikon: Icons.fact_check_outlined, child: Text(_ozTest)),
                ],
                const SizedBox(height: 12),
                _Kart(
                  baslik: 'Güven eşiği',
                  ikon: Icons.tune,
                  sag: _Etiket('%${(durum.esik * 100).round()}', renk.primary),
                  child: Slider(
                    min: 0.3,
                    max: 0.9,
                    divisions: 12,
                    value: durum.esik,
                    label: '%${(durum.esik * 100).round()}',
                    onChanged: (v) => setState(() => durum.ayarla(() => durum.esik = v)),
                  ),
                ),
                const SizedBox(height: 12),
                _OlcumKarti(istat: m),
              ]),
            ),
          ),
        ]),
      ),
    );
  }

  Widget? _elleEtiketi(ColorScheme renk) {
    if (_kayitta) return _Etiket('Kaydediliyor', renk.error);
    final r = _elle;
    if (r == null || r['error'] != null) return null;
    final p = ((r['top'] as List).first as Map)['p'] as num;
    return p >= durum.esik ? _Etiket('Kabul edildi', Colors.green) : _Etiket('Emin değilim', Colors.amber.shade700);
  }

  Widget _elleIcerik(ThemeData tema) {
    final r = _elle;
    if (_kayitta || r == null || r['error'] != null) {
      return Text(_elleMetni(), style: tema.textTheme.titleMedium);
    }
    final top = (r['top'] as List).cast<Map>();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      for (final (i, t) in top.indexed) _AdaySatiri(sira: i + 1, ad: '${t['name']}', p: (t['p'] as num).toDouble()),
      const SizedBox(height: 6),
      Text(
        '${r['frames']} kare'
        '${r['handFrames'] != null ? ' · elin göründüğü ${r['handFrames']}' : ''} · ${r['ms']} ms',
        style: tema.textTheme.bodySmall?.copyWith(color: tema.colorScheme.onSurfaceVariant),
      ),
    ]);
  }
}

/// Basligi ve ikonu olan yuvarlak kenarli bilgi karti.
class _Kart extends StatelessWidget {
  final String baslik;
  final IconData ikon;
  final Widget? sag;
  final Widget child;

  const _Kart({required this.baslik, required this.ikon, required this.child, this.sag});

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      decoration: BoxDecoration(
        color: tema.colorScheme.surfaceContainer,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: tema.colorScheme.outlineVariant.withValues(alpha: 0.5)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(ikon, size: 18, color: tema.colorScheme.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Text(baslik,
                style: tema.textTheme.labelLarge?.copyWith(color: tema.colorScheme.onSurfaceVariant)),
          ),
          ?sag,
        ]),
        const SizedBox(height: 10),
        child,
      ]),
    );
  }
}

class _Etiket extends StatelessWidget {
  final String metin;
  final Color renk;

  const _Etiket(this.metin, this.renk);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(color: renk.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(20)),
        child: Text(metin, style: TextStyle(color: renk, fontWeight: FontWeight.w600, fontSize: 12)),
      );
}

/// Bir tahmin adayi: sira, kelime, olasilik cubugu ve yuzde.
class _AdaySatiri extends StatelessWidget {
  final int sira;
  final String ad;
  final double p;

  const _AdaySatiri({required this.sira, required this.ad, required this.p});

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    final ilk = sira == 1;
    final renk = !ilk
        ? tema.colorScheme.outline
        : (p >= durum.esik ? Colors.green : Colors.amber.shade700);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(children: [
        SizedBox(
          width: 22,
          child: Text('$sira', style: tema.textTheme.bodySmall?.copyWith(color: tema.colorScheme.onSurfaceVariant)),
        ),
        Expanded(
          flex: 4,
          child: Text(ad,
              overflow: TextOverflow.ellipsis,
              style: ilk
                  ? tema.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600)
                  : tema.textTheme.bodyLarge),
        ),
        Expanded(
          flex: 5,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: p.clamp(0, 1),
              minHeight: ilk ? 8 : 6,
              color: renk,
              backgroundColor: tema.colorScheme.surfaceContainerHighest,
            ),
          ),
        ),
        SizedBox(
          width: 52,
          child: Text('%${(p * 100).toStringAsFixed(0)}',
              textAlign: TextAlign.right,
              style: tema.textTheme.bodyMedium?.copyWith(fontWeight: ilk ? FontWeight.w600 : null)),
        ),
      ]),
    );
  }
}

/// Kamera ustundeki kucuk durum rozeti (el sayisi ya da kayittaki kare sayisi).
class _Rozet extends StatelessWidget {
  final Map istat;
  final bool kayitta;

  const _Rozet({required this.istat, required this.kayitta});

  @override
  Widget build(BuildContext context) {
    final el = istat['hands'] ?? 0;
    final metin = kayitta ? 'Kaydediliyor · ${istat['recFrames'] ?? 0} kare' : (el == 0 ? 'El görünmüyor' : '$el el görünüyor');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: (kayitta ? Colors.red : Colors.black).withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(kayitta ? Icons.fiber_manual_record : Icons.back_hand_outlined, size: 14, color: Colors.white),
        const SizedBox(width: 6),
        Text(metin, style: const TextStyle(color: Colors.white)),
      ]),
    );
  }
}

/// "Hiz ayarlari ve olcumler": islemci secimi ve anlik olcumler.
class _OlcumKarti extends StatelessWidget {
  final Map istat;

  const _OlcumKarti({required this.istat});

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    final m = istat;
    String f0(String k) => (m[k] as num?)?.toStringAsFixed(0) ?? '-';
    final fps = (m['fps'] as num?)?.toStringAsFixed(1) ?? '-';
    final taraf = (m['sides'] as String?)?.isEmpty ?? true ? 'yok' : m['sides'];
    return Container(
      decoration: BoxDecoration(
        color: tema.colorScheme.surfaceContainer,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: tema.colorScheme.outlineVariant.withValues(alpha: 0.5)),
      ),
      clipBehavior: Clip.antiAlias,
      child: ExpansionTile(
        leading: Icon(Icons.speed, color: tema.colorScheme.primary),
        title: const Text('Hız ayarları ve ölçümler'),
        subtitle: Text('FPS $fps', style: tema.textTheme.bodySmall),
        shape: const Border(),
        collapsedShape: const Border(),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        expandedCrossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          StatefulBuilder(
            builder: (context, yenile) => SegmentedButton<int>(
              segments: const [
                ButtonSegment(value: 0, label: Text('CPU')),
                ButtonSegment(value: 1, label: Text('El GPU')),
                ButtonSegment(value: 2, label: Text('GPU')),
              ],
              selected: {durum.hizModu},
              onSelectionChanged: (s) {
                yenile(() => durum.ayarla(() => durum.hizModu = s.first));
                kontrol.invokeMethod('setMode', s.first);
              },
            ),
          ),
          const SizedBox(height: 12),
          Wrap(spacing: 8, runSpacing: 8, children: [
            _Olcu('FPS', fps),
            _Olcu('Çıkarım', '${f0('inferMs')} ms'),
            _Olcu('El', '${f0('handMs')} ms'),
            _Olcu('Poz', '${f0('poseMs')} ms'),
            _Olcu('Dönüşüm', '${f0('convMs')} ms'),
          ]),
          const SizedBox(height: 12),
          Text(
            '${m['delegate'] ?? '-'}\n'
            '${durum.onKamera ? "Ön" : "Arka"} kamera · ${m['w'] ?? '-'}x${m['h'] ?? '-'} · '
            'El: ${m['hands'] ?? '-'} ($taraf) · Gövde: ${m['pose'] == true ? 'var' : 'yok'}',
            style: tema.textTheme.bodySmall?.copyWith(color: tema.colorScheme.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}

class _Olcu extends StatelessWidget {
  final String ad, deger;

  const _Olcu(this.ad, this.deger);

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    return Container(
      width: 104,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: tema.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(ad, style: tema.textTheme.labelSmall?.copyWith(color: tema.colorScheme.onSurfaceVariant)),
        Text(deger, style: tema.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
      ]),
    );
  }
}
