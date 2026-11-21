import 'package:flutter/material.dart';

import 'durum.dart';
import 'isaretler_ekrani.dart';
import 'iskelet.dart';
import 'metin_ekrani.dart';
import 'tanima_ekrani.dart';

void main() => runApp(const TidApp());

class TidApp extends StatelessWidget {
  const TidApp({super.key});

  ThemeData _tema(Brightness b) => ThemeData(colorSchemeSeed: Colors.teal, brightness: b, useMaterial3: true);

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'TİD Tanıma',
        debugShowCheckedModeBanner: false,
        theme: _tema(Brightness.light),
        darkTheme: _tema(Brightness.dark),
        home: const AnaSayfa(),
      );
}

/// Uc ekran (tez onerisi Bolum 4): canli tanima, metin, desteklenen isaretler.
class AnaSayfa extends StatefulWidget {
  const AnaSayfa({super.key});

  @override
  State<AnaSayfa> createState() => _AnaSayfaState();
}

class _AnaSayfaState extends State<AnaSayfa> {
  int _sekme = 0;

  @override
  void initState() {
    super.initState();
    durum.isaretleriYukle();
    Referans.yukle().then((_) => mounted ? setState(() {}) : null);
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        // Sekme degisince kamera gorunumu kapanir; kamera yalnizca Tani sekmesinde calisir.
        body: switch (_sekme) {
          0 => const TanimaEkrani(),
          1 => const MetinEkrani(),
          _ => const IsaretlerEkrani(),
        },
        bottomNavigationBar: ListenableBuilder(
          listenable: durum,
          builder: (_, _) => NavigationBar(
            selectedIndex: _sekme,
            onDestinationSelected: (i) => setState(() => _sekme = i),
            destinations: [
              const NavigationDestination(icon: Icon(Icons.videocam_outlined), selectedIcon: Icon(Icons.videocam), label: 'Tanı'),
              NavigationDestination(
                icon: Badge(
                  isLabelVisible: durum.kelimeler.isNotEmpty,
                  label: Text('${durum.kelimeler.length}'),
                  child: const Icon(Icons.notes),
                ),
                label: 'Metin',
              ),
              const NavigationDestination(
                  icon: Icon(Icons.menu_book_outlined), selectedIcon: Icon(Icons.menu_book), label: 'İşaretler'),
            ],
          ),
        ),
      );
}
