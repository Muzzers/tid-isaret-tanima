import 'package:flutter/material.dart';

import 'durum.dart';

/// Kotlin'deki kamera gorunumu (CameraX onizleme + nokta kaplamasi). Gorunum her olusturuldugunda
/// Kotlin tarafi varsayilana doner; bu yuzden kayitli ayarlar burada yeniden gonderilir.
class KameraGorunumu extends StatelessWidget {
  final bool surekli;

  const KameraGorunumu({super.key, this.surekli = false});

  @override
  Widget build(BuildContext context) => AndroidView(
        viewType: 'tid/camera_view',
        onPlatformViewCreated: (_) {
          kontrol.invokeMethod('setLens', durum.onKamera);
          kontrol.invokeMethod('setMode', durum.hizModu);
          kontrol.invokeMethod('setContinuous', surekli);
        },
      );
}

/// Kamera izni alinana kadar bekler; izin yoksa aciklama gosterir.
class KameraIzni extends StatefulWidget {
  final Widget child;

  const KameraIzni({super.key, required this.child});

  @override
  State<KameraIzni> createState() => _KameraIzniState();
}

class _KameraIzniState extends State<KameraIzni> {
  static bool? _izin;

  @override
  void initState() {
    super.initState();
    if (_izin != true) _iste();
  }

  Future<void> _iste() async {
    final ok = await kontrol.invokeMethod<bool>('ensureCamera') ?? false;
    if (mounted) setState(() => _izin = ok);
  }

  @override
  Widget build(BuildContext context) => switch (_izin) {
        null => const Center(child: CircularProgressIndicator()),
        false => Center(
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              const Icon(Icons.no_photography_outlined, size: 48),
              const SizedBox(height: 12),
              const Text('İşaretleri tanımak için kamera izni gerekli.'),
              const SizedBox(height: 12),
              FilledButton(onPressed: _iste, child: const Text('İzin ver')),
            ]),
          ),
        true => widget.child,
      };
}
