package tr.edu.erciyes.tid_app

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Kaydedilen kare dizisini egitimle ayni sekilde (ml/train_gru.py: normalize + fix_length + to_vec)
 * isleyip LiteRT modeliyle siniflandirir. Python ile birebir ayni olmasi ZORUNLU (PLAN.md P-01).
 *
 * Koordinatlar piksel birimine cevrilir; omuz genisligine bolundugu icin egitimdeki kare (512x512) ile
 * telefondaki dikdortgen goruntu arasindaki olcek farki ortadan kalkar.
 */
class SignClassifier(context: Context) {
    class Sample(val hands: Array<FloatArray?>, val pose: FloatArray?, val w: Int, val h: Int)
    class Prediction(val name: String, val prob: Float, val index: Int)

    private val interpreter: Interpreter
    private val names: List<String>

    init {
        val fd = context.assets.openFd("sign_model.tflite")
        val map: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(map, Interpreter.Options().setNumThreads(2))
        names = context.assets.open("signs.csv").bufferedReader().readLines().drop(1)
            .map { line -> line.split(",").let { c -> c.getOrNull(3) ?: c[1].replace('_', ' ') } }  // 4. sutun: Turkce yazim
    }

    fun classify(frames: List<Sample>, topK: Int = 3): List<Prediction> {
        val t = frames.size
        // [T, 2, 21, 2] el, [T, 7, 2] poz, [T, 2] el var mi
        val hands = Array(t) { FloatArray(84) }
        val pose = Array(t) { FloatArray(14) }
        val hok = Array(t) { FloatArray(2) }
        val center = Array(t) { FloatArray(2) }
        val scale = FloatArray(t)
        val good = BooleanArray(t)

        for (i in 0 until t) {
            val f = frames[i]
            f.hands.forEachIndexed { s, h ->
                if (h != null) {
                    hok[i][s] = 1f
                    for (j in 0 until 21) {
                        hands[i][s * 42 + 2 * j] = h[2 * j] * f.w
                        hands[i][s * 42 + 2 * j + 1] = h[2 * j + 1] * f.h
                    }
                }
            }
            f.pose?.let { p -> for (k in 0 until 7) { pose[i][2 * k] = p[2 * k] * f.w; pose[i][2 * k + 1] = p[2 * k + 1] * f.h } }
            val lx = pose[i][2]; val ly = pose[i][3]; val rx = pose[i][4]; val ry = pose[i][5] // idx1 sol omuz, idx2 sag omuz
            center[i][0] = (lx + rx) / 2; center[i][1] = (ly + ry) / 2
            scale[i] = sqrt((lx - rx) * (lx - rx) + (ly - ry) * (ly - ry))
            good[i] = f.pose != null && scale[i] > 1e-4f
        }
        // omuz bulunamayan karelerde son gecerli degeri tasi (ilk gecerli kareye geri doldur)
        val first = good.indexOfFirst { it }
        if (first < 0) {
            val f = frames[0]
            for (i in 0 until t) { center[i][0] = 0.5f * f.w; center[i][1] = 0.5f * f.h; scale[i] = 0.3f * max(f.w, f.h) }
        } else {
            val idx = IntArray(t)
            var last = first
            for (i in 0 until t) { if (good[i]) last = i; idx[i] = if (i <= first) first else last }
            val c = Array(t) { center[idx[it]].copyOf() }
            val s = FloatArray(t) { scale[idx[it]] }
            for (i in 0 until t) { center[i] = c[i]; scale[i] = s[i] }
        }

        // normalize: (xy - merkez) / olcek; eksik el = 0
        for (i in 0 until t) {
            for (s in 0 until 2) for (j in 0 until 21) for (c in 0 until 2) {
                val k = s * 42 + 2 * j + c
                hands[i][k] = (hands[i][k] - center[i][c]) / scale[i] * hok[i][s]
            }
            for (k in 0 until 7) for (c in 0 until 2) pose[i][2 * k + c] = (pose[i][2 * k + c] - center[i][c]) / scale[i]
        }

        // sabit 64 kareye dogrusal yeniden ornekleme (egitimdeki resample ile ayni)
        val T = 64
        val input = Array(1) { Array(T) { FloatArray(100) } }
        for (o in 0 until T) {
            val pos = if (t == 1) 0f else o * (t - 1f) / (T - 1f)
            val lo = floor(pos).toInt()
            val hi = min(lo + 1, t - 1)
            val w = pos - lo
            val row = input[0][o]
            for (k in 0 until 84) row[k] = hands[lo][k] * (1 - w) + hands[hi][k] * w
            for (k in 0 until 14) row[84 + k] = pose[lo][k] * (1 - w) + pose[hi][k] * w
            for (s in 0 until 2) row[98 + s] = if (hok[lo][s] * (1 - w) + hok[hi][s] * w > 0.5f) 1f else 0f
        }

        val out = Array(1) { FloatArray(names.size) }
        interpreter.run(input, out)
        val logits = out[0]
        val mx = logits.max()
        val ex = FloatArray(logits.size) { exp(logits[it] - mx) }
        val sum = ex.sum()
        return ex.indices.sortedByDescending { ex[it] }.take(topK).map { Prediction(names[it], ex[it] / sum, it) }
    }

    fun close() = interpreter.close()
}
