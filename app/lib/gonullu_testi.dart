import 'dart:math';

import 'package:flutter/material.dart';

import 'durum.dart';
import 'iskelet.dart';
import 'kamera.dart';

/// Gonullu testi (tez onerisi Tablo 2: gercek ortam dogrulugu, 30 isaret, hedef >= %75).
///
/// Katilimciya sirayla isaret gosterilir, katilimci isareti kameraya yapar, tahmin kaydedilir.
/// Her deneme yalnizca nokta koordinatlari ve tahminle birlikte JSON olarak telefona yazilir
/// (goruntu saklanmaz); bilgisayarda ml/live_analysis.py ile incelenir.
class GonulluTesti extends StatefulWidget {
  const GonulluTesti({super.key});

  @override
  State<GonulluTesti> createState() => _GonulluTestiState();
}

class _Deneme {
  final int hedef;
  final List<Tahmin> tahmin;
  _Deneme(this.hedef, this.tahmin);
  bool get dogru => tahmin.isNotEmpty && tahmin.first.id == hedef;
  bool get ilk3 => tahmin.any((t) => t.id == hedef);
}

class _GonulluTestiState extends State<GonulluTesti> {
  final _kod = TextEditingController(text: 'K01');
  int _adet = 30;
  List<int> _sira = const [];
  int _i = -1; // -1: giris ekrani, _sira.length: sonuc ekrani
  final List<_Deneme> _denemeler = [];
  bool _kayitta = false;
  bool _bekliyor = false;
  List<Tahmin>? _son;
  String? _hata;
  String _oturum = '';

  @override
  void dispose() {
    _kod.dispose();
    super.dispose();
  }

  void _basla() {
    final r = Random();
    _sira = (List.generate(durum.isaretler.length, (i) => i)..shuffle(r)).take(_adet).toList();
    _denemeler.clear();
    _oturum = DateTime.now().millisecondsSinceEpoch.toString();
    setState(() {
      _i = 0;
      _son = null;
    });
  }

  Future<void> _kayit() async {
    if (!_kayitta) {
      setState(() {
        _kayitta = true;
        _son = null;
        _hata = null;
      });
      await kontrol.invokeMethod('record', true);
      return;
    }
    setState(() {
      _kayitta = false;
      _bekliyor = true;
    });
    final r = await kontrol.invokeMethod<Map>('record', false);
    if (!mounted) return;
    setState(() {
      _bekliyor = false;
      if (r == null || r['error'] != null) {
        _hata = '${r?['error'] ?? 'Kayıt alınamadı'}';
      } else {
        _son = (r['top'] as List).map((e) => Tahmin(e as Map)).toList();
      }
    });
    // Kayit "Bitir"de hemen yazilir; ekrandan cikilsa da kaybolmaz. Yeniden kayit ayni dosyanin uzerine yazar.
    if (_son != null) await _kaydet(_sira[_i], _son!);
  }

  Future<void> _kaydet(int hedef, List<Tahmin> t) => kontrol.invokeMethod('saveLast', {
        'katilimci': _kod.text.trim(),
        'oturum': _oturum,
        'sira': _i + 1,
        'hedef_id': hedef,
        'hedef_ad': durum.isaretler[hedef],
        'tahmin': [for (final x in t) {'id': x.id, 'ad': x.ad, 'p': x.p}],
        'esik': durum.esik,
        'zaman': DateTime.now().toIso8601String(),
      });

  Future<void> _sonraki() async {
    final hedef = _sira[_i];
    final t = _son!;
    _denemeler.add(_Deneme(hedef, t));
    setState(() {
      _i++;
      _son = null;
      _hata = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final govde = _i < 0
        ? _giris(context)
        : _i >= _sira.length
            ? _ozet(context)
            : _test(context);
    return Scaffold(appBar: AppBar(title: const Text('Gönüllü testi')), body: govde);
  }

  Widget _giris(BuildContext context) {
    final tema = Theme.of(context);
    return ListView(padding: const EdgeInsets.all(20), children: [
      Text('Nasıl yapılır?', style: tema.textTheme.titleLarge),
      const SizedBox(height: 8),
      const Text('1. Telefonu sabit bir yere koyun; baş, omuzlar ve eller görüntüde olsun.\n'
          '2. Ekranda bir kelime ve örnek hareketi çıkar. Hareketi birkaç kez izleyin.\n'
          '3. "Kaydet"e basın, işareti yapın, "Bitir"e basın.\n'
          '4. Sonucu görün ve sonraki kelimeye geçin.\n\n'
          'Görüntü kaydedilmez; yalnızca el ve gövde noktaları ile tahmin telefona yazılır.'),
      const SizedBox(height: 20),
      TextField(
        controller: _kod,
        decoration: const InputDecoration(labelText: 'Katılımcı kodu (ad yazmayın)', border: OutlineInputBorder()),
      ),
      const SizedBox(height: 16),
      Text('İşaret sayısı', style: tema.textTheme.titleSmall),
      const SizedBox(height: 8),
      SegmentedButton<int>(
        segments: const [
          ButtonSegment(value: 10, label: Text('10')),
          ButtonSegment(value: 20, label: Text('20')),
          ButtonSegment(value: 30, label: Text('30')),
        ],
        selected: {_adet},
        onSelectionChanged: (s) => setState(() => _adet = s.first),
      ),
      const SizedBox(height: 24),
      FilledButton.icon(
        onPressed: durum.isaretler.isEmpty ? null : _basla,
        icon: const Icon(Icons.play_arrow),
        label: const Text('Teste başla'),
        style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
      ),
    ]);
  }

  Widget _test(BuildContext context) {
    final tema = Theme.of(context);
    final hedef = _sira[_i];
    final son = _son;
    return KameraIzni(
      child: Column(children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
          child: Row(children: [
            SizedBox(
              width: 110,
              height: 110,
              child: Card(child: IskeletAnimasyonu(sinif: hedef)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('${_i + 1} / ${_sira.length}', style: tema.textTheme.labelLarge),
                Text(durum.isaretler[hedef], style: tema.textTheme.headlineMedium),
                LinearProgressIndicator(value: _i / _sira.length),
              ]),
            ),
          ]),
        ),
        const Expanded(child: KameraGorunumu()),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            if (_hata != null) Text(_hata!, style: TextStyle(color: tema.colorScheme.error)),
            if (son != null) ...[
              Text(
                son.first.id == hedef ? 'Doğru: ${son.first.ad} ${son.first.yuzde}' : 'Tahmin: ${son.first.ad} ${son.first.yuzde}',
                style: tema.textTheme.titleLarge?.copyWith(
                    color: son.first.id == hedef ? Colors.green.shade700 : tema.colorScheme.error),
              ),
              Text('Diğer adaylar: ${son.skip(1).map((t) => '${t.ad} ${t.yuzde}').join(', ')}'),
              const SizedBox(height: 8),
            ],
            Row(children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: _bekliyor ? null : _kayit,
                  icon: Icon(_kayitta ? Icons.stop_rounded : Icons.fiber_manual_record),
                  label: Text(_kayitta ? 'Bitir' : (son == null ? 'Kaydet' : 'Yeniden kaydet')),
                  style: FilledButton.styleFrom(
                    backgroundColor: _kayitta ? tema.colorScheme.error : null,
                    minimumSize: const Size.fromHeight(52),
                  ),
                ),
              ),
              if (son != null && !_kayitta) ...[
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _sonraki,
                    icon: const Icon(Icons.arrow_forward),
                    label: const Text('Sonraki'),
                    style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
                  ),
                ),
              ],
            ]),
          ]),
        ),
      ]),
    );
  }

  Widget _ozet(BuildContext context) {
    final tema = Theme.of(context);
    final n = _denemeler.length;
    final dogru = _denemeler.where((d) => d.dogru).length;
    final ilk3 = _denemeler.where((d) => d.ilk3).length;
    final kabul = _denemeler.where((d) => d.tahmin.first.p >= durum.esik).toList();
    final kabulDogru = kabul.where((d) => d.dogru).length;
    String yuzde(int a, int b) => b == 0 ? '-' : '%${(a * 100 / b).toStringAsFixed(1)}';
    return ListView(padding: const EdgeInsets.all(20), children: [
      Text('Katılımcı ${_kod.text}', style: tema.textTheme.titleLarge),
      const SizedBox(height: 12),
      Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Doğruluk (ilk tahmin): ${yuzde(dogru, n)}  ($dogru / $n)', style: tema.textTheme.titleMedium),
            Text('İlk 3 tahmin içinde: ${yuzde(ilk3, n)}'),
            Text('Eşiği geçenler: ${kabul.length} / $n, bunlarda doğruluk ${yuzde(kabulDogru, kabul.length)}'),
            const SizedBox(height: 4),
            Text('Hedef (tez önerisi): ≥ %75', style: tema.textTheme.bodySmall),
          ]),
        ),
      ),
      const SizedBox(height: 12),
      for (final d in _denemeler)
        ListTile(
          dense: true,
          leading: Icon(d.dogru ? Icons.check_circle : Icons.cancel,
              color: d.dogru ? Colors.green : tema.colorScheme.error),
          title: Text(durum.isaretler[d.hedef]),
          subtitle: Text(d.tahmin.map((t) => '${t.ad} ${t.yuzde}').join(' · ')),
        ),
      const SizedBox(height: 12),
      Text('Kayıtlar telefonda: Android/data/tr.edu.erciyes.tid_app/files/gonullu_testi',
          style: tema.textTheme.bodySmall),
      const SizedBox(height: 12),
      FilledButton(onPressed: () => setState(() => _i = -1), child: const Text('Yeni katılımcı')),
    ]);
  }
}
