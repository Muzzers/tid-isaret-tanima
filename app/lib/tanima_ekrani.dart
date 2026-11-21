import 'dart:async';

import 'package:flutter/material.dart';

import 'durum.dart';
import 'kamera.dart';

/// Canli tanima: kamera goruntusu, altyazi ve kayit dugmesi. Kamera, MediaPipe ve model Kotlin'de
/// calisir; buraya yalnizca ozet bilgiler ve tahminler gelir (PLAN.md P-05).
class TanimaEkrani extends StatefulWidget {
  const TanimaEkrani({super.key});

  @override
  State<TanimaEkrani> createState() => _TanimaEkraniState();
}

class _TanimaEkraniState extends State<TanimaEkrani> {
  final _istat = ValueNotifier<Map>(const {});
  StreamSubscription<Map>? _abone;
  bool _kayitta = false;
  String _altyazi = '';
  bool? _kabul; // son tahmin esigi gecti mi
  List<Tahmin> _adaylar = const [];

  @override
  void initState() {
    super.initState();
    _abone = olayAkisi.listen((m) {
      if (!mounted) return;
      if (m['segment'] != null) {
        _sonuc(m['segment'] as Map);
      } else {
        _istat.value = m;
      }
    });
  }

  @override
  void dispose() {
    _abone?.cancel();
    _istat.dispose();
    super.dispose();
  }

  /// Kayit ya da surekli moddaki bir segmentin sonucu: esigi gecerse metne eklenir.
  void _sonuc(Map? r) {
    setState(() {
      if (r == null) return;
      if (r['error'] != null) {
        _altyazi = '${r['error']}';
        _kabul = null;
        _adaylar = const [];
        return;
      }
      _adaylar = (r['top'] as List).map((e) => Tahmin(e as Map)).toList();
      final ilk = _adaylar.first;
      _kabul = ilk.p >= durum.esik;
      if (_kabul!) {
        durum.ekle(ilk.ad);
        _altyazi = ilk.ad;
      } else {
        _altyazi = 'Emin değilim';
      }
    });
  }

  Future<void> _kayit() async {
    if (!_kayitta) {
      setState(() {
        _kayitta = true;
        _altyazi = '';
        _kabul = null;
        _adaylar = const [];
      });
      await kontrol.invokeMethod('record', true);
    } else {
      setState(() => _kayitta = false);
      _sonuc(await kontrol.invokeMethod<Map>('record', false));
    }
  }

  void _surekli(bool s) {
    durum.ayarla(() => durum.surekli = s);
    kontrol.invokeMethod('setContinuous', s);
    setState(() {
      _kayitta = false;
      _altyazi = '';
      _kabul = null;
      _adaylar = const [];
    });
  }

  void _kameraDegistir() {
    durum.ayarla(() => durum.onKamera = !durum.onKamera);
    kontrol.invokeMethod('setLens', durum.onKamera);
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tanı'),
        actions: [
          IconButton(
            tooltip: durum.onKamera ? 'Arka kameraya geç' : 'Ön kameraya geç',
            icon: Icon(durum.onKamera ? Icons.camera_rear_outlined : Icons.camera_front_outlined),
            onPressed: _kameraDegistir,
          ),
          IconButton(
            tooltip: 'Ayarlar ve ölçümler',
            icon: const Icon(Icons.tune),
            onPressed: () => showModalBottomSheet(
              context: context,
              isScrollControlled: true,
              showDragHandle: true,
              builder: (_) => _AyarlarPaneli(istat: _istat),
            ),
          ),
        ],
      ),
      body: KameraIzni(
        child: Column(children: [
          Expanded(
            child: Stack(fit: StackFit.expand, children: [
              KameraGorunumu(surekli: durum.surekli),
              Positioned(left: 12, top: 12, child: _DurumRozeti(istat: _istat, kayitta: _kayitta)),
              if (_altyazi.isNotEmpty)
                Positioned(left: 16, right: 16, bottom: 16, child: _Altyazi(_altyazi, _kabul, _adaylar)),
            ]),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              SegmentedButton<bool>(
                segments: const [
                  ButtonSegment(value: false, label: Text('Tek işaret'), icon: Icon(Icons.touch_app_outlined)),
                  ButtonSegment(value: true, label: Text('Sürekli'), icon: Icon(Icons.all_inclusive)),
                ],
                selected: {durum.surekli},
                onSelectionChanged: (s) => _surekli(s.first),
              ),
              const SizedBox(height: 12),
              if (durum.surekli)
                Text('Ellerinizi kaldırınca işaret başlar, indirince tanınır.',
                    textAlign: TextAlign.center, style: tema.textTheme.bodyMedium)
              else
                FilledButton.icon(
                  onPressed: _kayit,
                  icon: Icon(_kayitta ? Icons.stop_rounded : Icons.fiber_manual_record),
                  label: Text(_kayitta ? 'Bitir ve tanı' : 'İşareti kaydet'),
                  style: FilledButton.styleFrom(
                    backgroundColor: _kayitta ? tema.colorScheme.error : null,
                    minimumSize: const Size.fromHeight(52),
                  ),
                ),
              const SizedBox(height: 8),
              ListenableBuilder(
                listenable: durum,
                builder: (_, _) => Text(
                  durum.kelimeler.isEmpty ? 'Tanınan kelimeler Metin sekmesinde birikir.' : durum.kelimeler.join(' '),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.center,
                  style: tema.textTheme.titleMedium,
                ),
              ),
            ]),
          ),
        ]),
      ),
    );
  }
}

class _Altyazi extends StatelessWidget {
  final String metin;
  final bool? kabul;
  final List<Tahmin> adaylar;

  const _Altyazi(this.metin, this.kabul, this.adaylar);

  @override
  Widget build(BuildContext context) {
    final alt = adaylar.isEmpty ? '' : adaylar.map((t) => '${t.ad} ${t.yuzde}').join('  ·  ');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.62),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: switch (kabul) { true => Colors.greenAccent, false => Colors.amberAccent, null => Colors.white24 },
          width: 2,
        ),
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Text(metin,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.w600)),
        if (alt.isNotEmpty) Text(alt, textAlign: TextAlign.center, style: const TextStyle(color: Colors.white70)),
      ]),
    );
  }
}

class _DurumRozeti extends StatelessWidget {
  final ValueNotifier<Map> istat;
  final bool kayitta;

  const _DurumRozeti({required this.istat, required this.kayitta});

  @override
  Widget build(BuildContext context) => ValueListenableBuilder<Map>(
        valueListenable: istat,
        builder: (_, m, _) {
          final el = m['hands'] ?? 0;
          final metin = kayitta
              ? 'Kaydediliyor · ${m['recFrames'] ?? 0} kare'
              : switch (m['state']) {
                  'isaret' => 'İşaret algılanıyor',
                  'bekliyor' => 'Bekliyor',
                  _ => el == 0 ? 'El görünmüyor' : '$el el görünüyor',
                };
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: (kayitta ? Colors.red : Colors.black).withValues(alpha: 0.6),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(metin, style: const TextStyle(color: Colors.white)),
          );
        },
      );
}

class _AyarlarPaneli extends StatefulWidget {
  final ValueNotifier<Map> istat;

  const _AyarlarPaneli({required this.istat});

  @override
  State<_AyarlarPaneli> createState() => _AyarlarPaneliState();
}

class _AyarlarPaneliState extends State<_AyarlarPaneli> {
  String _ozTest = '';

  Future<void> _ozTestCalistir() async {
    setState(() => _ozTest = 'Öz-test çalışıyor…');
    final r = await kontrol.invokeMethod<Map>('selftest');
    if (!mounted || r == null) return;
    setState(() => _ozTest = r['error'] != null
        ? '${r['error']}'
        : '${r['n']} kayıtlı test örneği: doğruluk %${((r['accuracy'] as double) * 100).toStringAsFixed(1)}, '
            'Python ile aynı tahmin %${((r['agreePython'] as double) * 100).toStringAsFixed(1)}, '
            '${(r['msPerSample'] as double).toStringAsFixed(1)} ms/örnek');
  }

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Güven eşiği: %${(durum.esik * 100).round()}', style: tema.textTheme.titleMedium),
        Text('Modelin bundan az emin olduğu tahminler metne eklenmez.', style: tema.textTheme.bodySmall),
        Slider(
          min: 0.3,
          max: 0.9,
          divisions: 12,
          value: durum.esik,
          onChanged: (v) => setState(() => durum.ayarla(() => durum.esik = v)),
        ),
        const SizedBox(height: 8),
        Text('İşlemci', style: tema.textTheme.titleMedium),
        const SizedBox(height: 8),
        SegmentedButton<int>(
          segments: const [
            ButtonSegment(value: 0, label: Text('CPU')),
            ButtonSegment(value: 1, label: Text('El GPU')),
            ButtonSegment(value: 2, label: Text('Hepsi GPU')),
          ],
          selected: {durum.hizModu},
          onSelectionChanged: (s) {
            setState(() => durum.ayarla(() => durum.hizModu = s.first));
            kontrol.invokeMethod('setMode', s.first);
          },
        ),
        const SizedBox(height: 12),
        ValueListenableBuilder<Map>(
          valueListenable: widget.istat,
          builder: (_, m, _) {
            String f(String k) => (m[k] as num?)?.toStringAsFixed(0) ?? '-';
            final fps = (m['fps'] as num?)?.toStringAsFixed(1) ?? '-';
            return Text(
              '${m['delegate'] ?? '-'}\n'
              'FPS $fps · kare başına ${f('inferMs')} ms (dönüşüm ${f('convMs')}, el ${f('handMs')}, gövde ${f('poseMs')})\n'
              'Görüntü ${m['w'] ?? '-'}×${m['h'] ?? '-'} · ${durum.onKamera ? 'ön' : 'arka'} kamera',
              style: tema.textTheme.bodySmall,
            );
          },
        ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          onPressed: _ozTestCalistir,
          icon: const Icon(Icons.fact_check_outlined),
          label: const Text('Öz-test (kayıtlı 200 örnek)'),
        ),
        if (_ozTest.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_ozTest)),
      ]),
    );
  }
}
