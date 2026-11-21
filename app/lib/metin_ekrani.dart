import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'durum.dart';

/// Taninan kelimelerin biriktigi metin: kopyalama, seslendirme, kelime silme.
class MetinEkrani extends StatelessWidget {
  const MetinEkrani({super.key});

  Future<void> _seslendir(BuildContext context) async {
    final ok = await kontrol.invokeMethod<bool>('speak', durum.kelimeler.join(' ')) ?? false;
    if (!ok && context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Türkçe ses paketi bulunamadı. Ayarlar > Metin okuma çıkışı bölümünden yükleyebilirsiniz.')));
    }
  }

  void _kopyala(BuildContext context) {
    Clipboard.setData(ClipboardData(text: durum.kelimeler.join(' ')));
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Metin kopyalandı.')));
  }

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Metin'), actions: [
        ListenableBuilder(
          listenable: durum,
          builder: (_, _) => IconButton(
            tooltip: 'Hepsini sil',
            icon: const Icon(Icons.delete_sweep_outlined),
            onPressed: durum.kelimeler.isEmpty ? null : durum.temizle,
          ),
        ),
      ]),
      body: ListenableBuilder(
        listenable: durum,
        builder: (_, _) {
          if (durum.kelimeler.isEmpty) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.notes, size: 56, color: tema.colorScheme.outline),
                  const SizedBox(height: 12),
                  Text('Henüz kelime yok', style: tema.textTheme.titleLarge),
                  const SizedBox(height: 4),
                  const Text('Tanı sekmesinde tanınan işaretler burada sırayla birikir.', textAlign: TextAlign.center),
                ]),
              ),
            );
          }
          return ListView(padding: const EdgeInsets.all(16), children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: SelectableText(durum.kelimeler.join(' '), style: tema.textTheme.headlineMedium),
              ),
            ),
            const SizedBox(height: 16),
            Row(children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => _seslendir(context),
                  icon: const Icon(Icons.volume_up_outlined),
                  label: const Text('Seslendir'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => _kopyala(context),
                  icon: const Icon(Icons.copy),
                  label: const Text('Kopyala'),
                ),
              ),
            ]),
            const SizedBox(height: 24),
            Text('Kelimeler (silmek için ×)', style: tema.textTheme.titleSmall),
            const SizedBox(height: 8),
            Wrap(spacing: 8, runSpacing: 8, children: [
              for (var i = 0; i < durum.kelimeler.length; i++)
                InputChip(label: Text(durum.kelimeler[i]), onDeleted: () => durum.sil(i)),
            ]),
          ]);
        },
      ),
    );
  }
}
