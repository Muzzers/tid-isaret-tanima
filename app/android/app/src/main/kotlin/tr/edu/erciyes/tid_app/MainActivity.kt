package tr.edu.erciyes.tid_app

import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : FlutterActivity() {
    private var sink: EventChannel.EventSink? = null
    private var pendingPermission: MethodChannel.Result? = null
    private var cameraView: CameraPlatformView? = null
    private var tts: android.speech.tts.TextToSpeech? = null
    private var ttsReady = false

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger

        EventChannel(messenger, "tid/landmarks").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(args: Any?, events: EventChannel.EventSink?) { sink = events }
            override fun onCancel(args: Any?) { sink = null }
        })

        MethodChannel(messenger, "tid/control").setMethodCallHandler { call, result ->
            if (call.method == "ensureCamera") {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    result.success(true)
                } else {
                    pendingPermission = result
                    ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 42)
                }
            } else if (call.method == "setLens") {
                cameraView?.setFront(call.arguments as? Boolean ?: false)
                result.success(null)
            } else if (call.method == "setMode") {
                cameraView?.setMode(call.arguments as? Int ?: 1)
                result.success(null)
            } else if (call.method == "setContinuous") {
                cameraView?.setContinuous(call.arguments as? Boolean ?: false)
                result.success(null)
            } else if (call.method == "selftest") {
                runSelfTest { result.success(it) }
            } else if (call.method == "signs") {
                // sinif no sirasiyla Turkce yazim (signs.csv 4. sutun)
                result.success(assets.open("signs.csv").bufferedReader().readLines().drop(1)
                    .map { line -> line.split(",").let { c -> c.getOrNull(3) ?: c[1].replace('_', ' ') } })
            } else if (call.method == "saveLast") {
                cameraView?.saveLast(call.arguments as? Map<*, *> ?: emptyMap<String, Any>(), result) ?: result.success(null)
            } else if (call.method == "speak") {
                speak(call.arguments as? String ?: "", result)
            } else if (call.method == "record") {
                cameraView?.record(call.arguments as? Boolean ?: false, result) ?: result.success(null)
            } else result.notImplemented()
        }

        // adb ile: am start ... --es videotest <mod> -> sonuc logcat'te (etiket TIDVTEST)
        intent.getStringExtra("videotest")?.let { runVideoTest(it.toIntOrNull() ?: 1) }
        // adb ile: am start ... --ez selftest true  -> sonuc logcat'te (etiket TIDSELFTEST)
        if (intent.getBooleanExtra("selftest", false)) runSelfTest { }

        flutterEngine.platformViewsController.registry
            .registerViewFactory("tid/camera_view", CameraViewFactory(this, { sink },
                onCreated = { cameraView = it },
                onDisposed = { if (cameraView === it) cameraView = null }))
    }

    /** Kotlin on isleme + LiteRT'in Python ile ayni sonucu verdigini gomulu 200 test ornegiyle dogrular. */
    private fun runSelfTest(done: (Map<String, Any>) -> Unit) {
        Thread {
            val reply: Map<String, Any> = try {
                val clf = SignClassifier(this)
                val buf = ByteBuffer.wrap(assets.open("selftest.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
                var n = 0; var correct = 0; var agree = 0
                val t0 = System.nanoTime()
                while (buf.hasRemaining()) {
                    val t = buf.int; val label = buf.int; val pyPred = buf.int
                    val frames = ArrayList<SignClassifier.Sample>(t)
                    repeat(t) {
                        val hand = FloatArray(84) { buf.float }
                        val pose = FloatArray(14) { buf.float }
                        val hok = floatArrayOf(buf.float, buf.float)
                        val pok = buf.float > 0.5f
                        frames.add(SignClassifier.Sample(
                            Array(2) { s -> if (hok[s] > 0.5f) hand.copyOfRange(s * 42, s * 42 + 42) else null },
                            if (pok) pose else null, 512, 512))
                    }
                    val top = clf.classify(frames, 1)[0].index
                    n++; if (top == label) correct++; if (top == pyPred) agree++
                }
                clf.close()
                mapOf("n" to n, "accuracy" to correct.toDouble() / n, "agreePython" to agree.toDouble() / n,
                    "msPerSample" to (System.nanoTime() - t0) / 1_000_000.0 / n)
            } catch (e: Exception) { mapOf("error" to (e.message ?: e.toString())) }
            Log.i("TIDSELFTEST", reply.toString())
            runOnUiThread { done(reply) }
        }.start()
    }

    /**
     * Video dosyasini kamerasiz, telefondaki MediaPipe + Kotlin on isleme + LiteRT zincirinden gecirir.
     * Dosyalar: /sdcard/Android/data/<paket>/files/vtest/<ad>__<sinif no>.mp4 (adb push ile).
     * Python cikarimi + ayni model sonucuyla karsilastirmak icin: ayni kliplerin Python tahmini ayri hesaplanir.
     */
    private fun runVideoTest(mode: Int) {
        Thread {
            try {
                val dir = java.io.File(getExternalFilesDir(null), "vtest")
                val files = dir.listFiles { f -> f.name.endsWith(".mp4") }?.sortedBy { it.name } ?: emptyList()
                Log.i("TIDVTEST", "mod=$mode, ${files.size} video")
                val clf = SignClassifier(this)
                var correct = 0; var n = 0
                for (f in files) {
                    val label = f.name.substringAfter("__").substringBefore(".").toInt()
                    val r = android.media.MediaMetadataRetriever().apply { setDataSource(f.absolutePath) }
                    val count = r.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
                    val pl2 = LandmarkPipeline(this, mode)  // her video icin yeni izleme durumu (Python ile ayni)
                    val frames = ArrayList<SignClassifier.Sample>(count)
                    for (i in 0 until count) {
                        val bmp = r.getFrameAtIndex(i) ?: continue
                        val fr = pl2.processBitmap(bmp.copy(android.graphics.Bitmap.Config.ARGB_8888, false), i * 1000L / 30)
                        frames.add(SignClassifier.Sample(fr.hands.toTypedArray(), fr.pose, fr.width, fr.height))
                    }
                    Log.i("TIDVTEST", "delegate=${pl2.delegate}"); pl2.close(); r.release()
                    val top = clf.classify(frames, 3)
                    n++; if (top[0].index == label) correct++
                    Log.i("TIDVTEST", "${f.name}: kare=${frames.size} el_var=${frames.count { s -> s.hands.any { it != null } }} " +
                        "tahmin=${top.joinToString(" | ") { "${it.name} %${(it.prob * 100).toInt()}" }} dogru=${top[0].index == label}")
                }
                clf.close()
                Log.i("TIDVTEST", "BITTI mod=$mode dogruluk=$correct/$n")
            } catch (e: Exception) { Log.e("TIDVTEST", "hata: $e") }
        }.start()
    }

    /** Metni cihazin Turkce ses motoruyla okur (internet gerekmez; ses paketi yuklu olmalidir). */
    private fun speak(text: String, result: MethodChannel.Result) {
        val say = {
            val ok = ttsReady && tts!!.speak(text, android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "tid") ==
                android.speech.tts.TextToSpeech.SUCCESS
            result.success(ok)
        }
        if (tts != null) { say(); return }
        tts = android.speech.tts.TextToSpeech(this) { status ->
            ttsReady = status == android.speech.tts.TextToSpeech.SUCCESS &&
                tts!!.setLanguage(java.util.Locale.forLanguageTag("tr-TR")) >= android.speech.tts.TextToSpeech.LANG_AVAILABLE
            runOnUiThread { say() }
        }
    }

    override fun onDestroy() {
        tts?.shutdown()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, grants: IntArray) {
        super.onRequestPermissionsResult(code, permissions, grants)
        if (code == 42) {
            pendingPermission?.success(grants.isNotEmpty() && grants[0] == PackageManager.PERMISSION_GRANTED)
            pendingPermission = null
        }
    }
}
