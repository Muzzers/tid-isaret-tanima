package tr.edu.erciyes.tid_app

import android.content.Context
import android.view.View
import android.widget.FrameLayout
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.Executors

class CameraViewFactory(
    private val owner: LifecycleOwner,
    private val events: () -> EventChannel.EventSink?,
    private val onCreated: (CameraPlatformView) -> Unit,
    private val onDisposed: (CameraPlatformView) -> Unit,
) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {
    override fun create(context: Context, viewId: Int, args: Any?): PlatformView =
        CameraPlatformView(context, owner, events, onDisposed).also { onCreated(it) }
}

/** CameraX onizleme + MediaPipe cikarimi + nokta kaplamasi. Kare akisi Dart'a gitmez, yalnizca ozet (FPS, el sayisi) gider. */
class CameraPlatformView(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val events: () -> EventChannel.EventSink?,
    private val onDisposed: (CameraPlatformView) -> Unit,
) : PlatformView {
    private val root = FrameLayout(context)
    private val preview = PreviewView(context).apply { implementationMode = PreviewView.ImplementationMode.COMPATIBLE }
    private val overlay = LandmarkOverlay(context)
    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var wantMode = 1
    private val recLock = Any()
    @Volatile private var recording = false
    private val rec = ArrayList<SignClassifier.Sample>()
    private var lastRec: List<SignClassifier.Sample> = emptyList()  // gonullu testinde dosyaya yazilir
    private var classifier: SignClassifier? = null
    private val classifyExec = Executors.newSingleThreadExecutor()
    @Volatile private var continuous = false
    private val tracker = SegmentTracker { frames -> classifySegment(frames) }
    private var pipeline: LandmarkPipeline? = null
    private var provider: ProcessCameraProvider? = null
    private var front = false
    private var prev: Preview? = null
    private var analysis: ImageAnalysis? = null
    private var lastReport = System.nanoTime()
    private var frames = 0
    private var inferSum = 0L
    private var convSum = 0L
    private var handSum = 0L
    private var poseSum = 0L

    init {
        root.addView(preview, FrameLayout.LayoutParams(-1, -1))
        root.addView(overlay, FrameLayout.LayoutParams(-1, -1))
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            provider = future.get()
            prev = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis!!.setAnalyzer(executor) { img ->
                try {
                    var pl = pipeline
                    if (pl == null || pl.mode != wantMode) {
                        pl?.close()
                        pl = LandmarkPipeline(context, wantMode).also { pipeline = it }
                    }
                    val f = pl.process(img)
                    if (recording) synchronized(recLock) {
                        rec.add(SignClassifier.Sample(f.hands.toTypedArray(), f.pose, f.width, f.height))
                    }
                    if (continuous) tracker.add(
                        SignClassifier.Sample(f.hands.toTypedArray(), f.pose, f.width, f.height), System.nanoTime())
                    overlay.update(f.hands, f.pose, f.width, f.height)
                    report(f)
                } finally {
                    img.close()
                }
            }
            bind()
        }, ContextCompat.getMainExecutor(context))
    }

    /** Arka/on kamera degistir. Analiz goruntusu aynalanmaz (AUTSL videolari gibi); yalnizca kaplama onizlemeyle hizalanir. */
    fun setFront(f: Boolean) {
        front = f
        ContextCompat.getMainExecutor(context).execute { bind() }
    }

    fun setMode(m: Int) { wantMode = m }

    fun setContinuous(c: Boolean) { synchronized(recLock) { tracker.reset() }; continuous = c }

    /** Surekli modda bir segment kapaninca siniflandirir ve olay akisiyla Dart'a gonderir. */
    private fun classifySegment(frames: List<SignClassifier.Sample>) {
        classifyExec.execute {
            try {
                val c = classifier ?: SignClassifier(context).also { classifier = it }
                val t0 = System.nanoTime()
                val top = c.classify(frames)
                android.util.Log.i("TIDSEG", "tahmin: " + top.joinToString(" | ") { "${it.name} %${(it.prob * 100).toInt()}" } + " (kare=${frames.size})")
                val seg = mapOf(
                    "frames" to frames.size,
                    "ms" to (System.nanoTime() - t0) / 1_000_000,
                    "top" to top.map { mapOf("name" to it.name, "p" to it.prob.toDouble(), "id" to it.index) },
                )
                ContextCompat.getMainExecutor(context).execute { events()?.success(mapOf("segment" to seg)) }
            } catch (e: Exception) {
                ContextCompat.getMainExecutor(context).execute {
                    events()?.success(mapOf("segment" to mapOf("error" to (e.message ?: e.toString()))))
                }
            }
        }
    }

    /** Kaydi baslat / bitir; bitirince kare dizisini siniflandirip ilk 3 tahmini dondurur. */
    fun record(start: Boolean, result: MethodChannel.Result) {
        if (start) {
            synchronized(recLock) { rec.clear() }
            recording = true
            result.success(null)
            return
        }
        recording = false
        val frames = synchronized(recLock) { ArrayList(rec) }
        lastRec = frames
        if (frames.size < 8) {
            result.success(mapOf("error" to "Çok az kare (${frames.size}); işareti kamerada biraz daha uzun yapın."))
            return
        }
        // El neredeyse hic gorunmediyse modele vermiyoruz: model "eller yok" girdilerinde de bazen
        // yuksek guvenle bir kelime seciyor (ml/calibrate_threshold.py, negatif ornekler).
        val handFrames = frames.count { f -> f.hands.any { it != null } }
        if (handFrames < frames.size / 5) {
            result.success(mapOf("error" to "El görünmedi. Ellerinizin kamerada göründüğünden emin olun."))
            return
        }
        classifyExec.execute {
            val reply: Map<String, Any> = try {
                val c = classifier ?: SignClassifier(context).also { classifier = it }
                val t0 = System.nanoTime()
                val top = c.classify(frames)
                mapOf(
                    "frames" to frames.size,
                    "ms" to (System.nanoTime() - t0) / 1_000_000,
                    "top" to top.map { mapOf("name" to it.name, "p" to it.prob.toDouble(), "id" to it.index) },
                )
            } catch (e: Exception) {
                mapOf("error" to "Siniflandirma hatasi: ${e.message}")
            }
            ContextCompat.getMainExecutor(context).execute { result.success(reply) }
        }
    }

    /**
     * Gonullu testi: son kaydedilen hareketi (yalnizca nokta koordinatlari, goruntu yok) ve tahmini
     * JSON olarak uygulamanin dosya klasorune yazar. Bilgisayara almak icin:
     *   adb pull /sdcard/Android/data/tr.edu.erciyes.tid_app/files/gonullu_testi
     * Python tarafinda ml/live_analysis.py okur.
     */
    fun saveLast(info: Map<*, *>, result: MethodChannel.Result) {
        val frames = lastRec
        if (frames.isEmpty()) { result.success(null); return }
        classifyExec.execute {
            val path = try {
                val dir = java.io.File(context.getExternalFilesDir(null), "gonullu_testi").apply { mkdirs() }
                val who = (info["katilimci"] as? String ?: "anonim").replace(Regex("[^A-Za-z0-9_-]"), "_")
                val f = java.io.File(dir, "${who}_${System.currentTimeMillis()}.json")
                val js = org.json.JSONObject()
                info.forEach { (k, v) -> js.put(k.toString(), org.json.JSONObject.wrap(v)) }
                js.put("w", frames[0].w); js.put("h", frames[0].h); js.put("on_kamera", front)
                val arr = org.json.JSONArray()
                for (fr in frames) {
                    val o = org.json.JSONObject()
                    o.put("el", org.json.JSONArray(fr.hands.map { h -> h?.let { org.json.JSONArray(it.map(Float::toDouble)) } ?: org.json.JSONObject.NULL }))
                    o.put("govde", fr.pose?.let { org.json.JSONArray(it.map(Float::toDouble)) } ?: org.json.JSONObject.NULL)
                    arr.put(o)
                }
                js.put("kareler", arr)
                f.writeText(js.toString())
                f.absolutePath
            } catch (e: Exception) { null }
            ContextCompat.getMainExecutor(context).execute { result.success(path) }
        }
    }

    private fun bind() {
        val p = provider ?: return
        overlay.mirror = front
        p.unbindAll()
        val sel = if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        p.bindToLifecycle(owner, sel, prev!!, analysis!!)
    }

    private fun report(f: LandmarkPipeline.Frame) {
        frames++; inferSum += f.inferMs; convSum += f.convMs; handSum += f.handMs; poseSum += f.poseMs
        val now = System.nanoTime()
        if (now - lastReport >= 500_000_000L) {
            val fps = frames * 1e9 / (now - lastReport)
            val msg = mapOf(
                "fps" to fps, "inferMs" to inferSum.toDouble() / frames,
                "hands" to f.hands.count { it != null }, "pose" to (f.pose != null),
                // MediaPipe'in ham etiketi (slot 0 = Left, slot 1 = Right); gercek elle karsilastirmak icin
                "sides" to f.hands.withIndex().filter { it.value != null }.joinToString("+") { if (it.index == 0) "Left" else "Right" },
                "w" to f.width, "h" to f.height, "front" to front, "recFrames" to synchronized(recLock) { rec.size }, "recording" to recording, "state" to (if (!continuous) "-" else if (tracker.inSegment) "isaret" else "bekliyor"), "delegate" to (pipeline?.delegate ?: "-"),
                "convMs" to convSum.toDouble() / frames, "handMs" to handSum.toDouble() / frames, "poseMs" to poseSum.toDouble() / frames,
            )
            ContextCompat.getMainExecutor(context).execute { events()?.success(msg) }
            frames = 0; inferSum = 0; convSum = 0; handSum = 0; poseSum = 0; lastReport = now
        }
    }

    override fun getView(): View = root

    override fun dispose() {
        provider?.unbindAll()
        executor.execute { pipeline?.close() }
        classifyExec.execute { classifier?.close() }
        classifyExec.shutdown()
        executor.shutdown() // kuyruktaki close() calisir
        onDisposed(this)
    }
}
