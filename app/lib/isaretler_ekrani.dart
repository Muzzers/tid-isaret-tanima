import 'package:flutter/material.dart';

import 'durum.dart';
import 'gonullu_testi.dart';
import 'iskelet.dart';

/// Modelin tanidigi 226 isaret: arama ve her isaretin ornek hareketi.
class IsaretlerEkrani extends StatefulWidget {
  const IsaretlerEkrani({super.key});

  @override
  State<IsaretlerEkrani> createState() => _IsaretlerEkraniState();
}

class _IsaretlerEkraniState extends State<IsaretlerEkrani> {
  String _ara = '';

  void _hakkinda() => showAboutDialog(
        context: context,
        applicationName: 'TİD Tanıma',
        applicationVersion: '1.0',
        applicationIcon: const Icon(Icons.sign_language, size: 40),
        children: const [
          Text('Telefon kamerasından Türk İşaret Dili kelimelerini tanıyıp yazıya çeviren, internetsiz çalışan '
              'uygulama. Erciyes Üniversitesi Bilgisayar Mühendisliği bitirme projesi.\n\n'
              'El ve gövde noktaları MediaPipe ile çıkarılır, 1B-CNN + Transformer modeli (LiteRT, int8, 1,6 MB, ayna görüntüsüne dayanıklı) AUTSL veri setindeki '
              '226 işareti tanır. Kamera görüntüsü cihazdan çıkmaz.'),
        ],
      );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('İşaretler'), actions: [
        IconButton(
          tooltip: 'Gönüllü testi',
          icon: const Icon(Icons.assignment_outlined),
          onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const GonulluTesti())),
        ),
        IconButton(tooltip: 'Hakkında', icon: const Icon(Icons.info_outline), onPressed: _hakkinda),
      ]),
      body: ListenableBuilder(
        listenable: durum,
        builder: (_, _) {
          final q = sadelestir(_ara.trim());
          final liste = [
            for (var i = 0; i < durum.isaretler.length; i++)
              if (q.isEmpty || sadelestir(durum.isaretler[i]).contains(q)) i,
          ]..sort((a, b) => siraAnahtari(durum.isaretler[a]).compareTo(siraAnahtari(durum.isaretler[b])));
          return Column(children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
              child: SearchBar(
                hintText: '${durum.isaretler.length} işaret içinde ara',
                leading: const Icon(Icons.search),
                onChanged: (s) => setState(() => _ara = s),
              ),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: liste.length,
                itemBuilder: (_, k) {
                  final i = liste[k];
                  return ListTile(
                    title: Text(durum.isaretler[i]),
                    trailing: const Icon(Icons.play_circle_outline),
                    onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => IsaretDetayi(sinif: i))),
                  );
                },
              ),
            ),
          ]);
        },
      ),
    );
  }
}

class IsaretDetayi extends StatefulWidget {
  final int sinif;

  const IsaretDetayi({super.key, required this.sinif});

  @override
  State<IsaretDetayi> createState() => _IsaretDetayiState();
}

class _IsaretDetayiState extends State<IsaretDetayi> {
  bool _ayna = true;
  bool _yavas = false;

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(durum.isaretler[widget.sinif])),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        // Yatay tablette kare alan ekrandan tasmasin diye yukseklik ekranin ~%60'i ile sinirli.
        Center(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).height * 0.6),
            child: Card(
              clipBehavior: Clip.antiAlias,
              child: AspectRatio(
                aspectRatio: 1,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: IskeletAnimasyonu(sinif: widget.sinif, cevir: !_ayna, hiz: _yavas ? 0.5 : 1.0),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        SwitchListTile(
          title: const Text('Ayna görüntüsü'),
          subtitle: const Text('Açıkken hareketi aynaya bakar gibi görürsünüz; taklit etmesi daha kolaydır.'),
          value: _ayna,
          onChanged: (v) => setState(() => _ayna = v),
        ),
        SwitchListTile(
          title: const Text('Yavaş oynat'),
          value: _yavas,
          onChanged: (v) => setState(() => _yavas = v),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Text(
            'Hareket, AUTSL veri setindeki bir işaretçinin kaydından çıkarılan el ve gövde noktalarıdır. '
            'Renkli çizgiler iki eli, gri çizgiler baş, omuz ve kolları gösterir.',
            style: tema.textTheme.bodySmall,
          ),
        ),
      ]),
    );
  }
}
