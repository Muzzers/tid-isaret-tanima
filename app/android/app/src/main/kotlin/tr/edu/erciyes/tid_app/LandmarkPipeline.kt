package tr.edu.erciyes.tid_app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * Egitimle ayni modeller (hand_landmarker, pose_landmarker_lite) ve ayni VIDEO modu.
 * El ve poz ayri is parcaciklarinda paralel kosar. GPU delegesi olusturuldugu is parcaciginda
 * kullanilmak zorunda oldugundan her landmarker kendi tek is parcacikli yurutucusunda yasar.
 */
class LandmarkPipeline(private val context: Context, val mode: Int) {
    // mode 0 = hepsi CPU, 1 = el GPU + poz CPU, 2 = hepsi GPU
    private val handGpu = mode >= 1
    private val poseGpu = mode >= 2

    class Frame(
        val hands: List<FloatArray?>,   // [slot 0 = Left, slot 1 = Right] -> x0,y0,x1,y1...
        val pose: FloatArray?,          // 7 nokta: burun, omuzlar, dirsekler, bilekler
        val width: Int, val height: Int,
        val inferMs: Long,
        val convMs: Long, val handMs: Long, val poseMs: Long,
    )

    private val handExec = Executors.newSingleThreadExecutor()
    private val poseExec = Executors.newSingleThreadExecutor()
    private val poseIdx = intArrayOf(0, 11, 12, 13, 14, 15, 16)
    private var lastTs = -1L

    /**
     * Govde noktalari her POSE_EVERY karede bir cikarilir, aradaki karelerde son deger kullanilir.
     * Govde yavas hareket ettigi icin dogruluk degismiyor (10 FPS'e seyreltilmis AUTSL testinde
     * her karede %87,22, iki karede bir %87,22); el modeli her karede calismaya devam eder.
     */
    private var frameNo = 0L
    private var lastPose: FloatArray? = null

    /** GPU kurulamazsa CPU'ya duser; hangisinin calistigi `delegate` ile bildirilir. */
    var delegate = when (mode) { 0 -> "El CPU / Poz CPU"; 1 -> "El GPU / Poz CPU"; else -> "El GPU / Poz GPU" }
        private set

    private fun base(model: String, useGpu: Boolean) = BaseOptions.builder()
        .setModelAssetPath(model)
        .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
        .build()

    private fun createHand(useGpu: Boolean) = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base("hand_landmarker.task", useGpu))
            .setRunningMode(RunningMode.VIDEO).setNumHands(2).build()
    )

    private fun createPose(useGpu: Boolean) = PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base("pose_landmarker_lite.task", useGpu))
            .setRunningMode(RunningMode.VIDEO).build()
    )

    private val hand: HandLandmarker
    private val pose: PoseLandmarker

    init {
        var h: HandLandmarker
        var p: PoseLandmarker
        try {
            h = handExec.submit(Callable { createHand(handGpu) }).get()
            p = poseExec.submit(Callable { createPose(poseGpu) }).get()
        } catch (e: Exception) {
            if (mode == 0) throw e
            delegate = "CPU (GPU kurulamadi)"
            h = handExec.submit(Callable { createHand(false) }).get()
            p = poseExec.submit(Callable { createPose(false) }).get()
        }
        hand = h; pose = p
    }

    private fun toUpright(image: ImageProxy): Bitmap {
        val bmp = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        image.planes[0].buffer.rewind()
        bmp.copyPixelsFromBuffer(image.planes[0].buffer)
        val rot = image.imageInfo.rotationDegrees
        return if (rot == 0) bmp else
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(rot.toFloat()) }, true)
    }

    fun process(image: ImageProxy): Frame {
        val t0 = System.nanoTime()
        val upright = toUpright(image)
        return processBitmap(upright, image.imageInfo.timestamp / 1_000_000L, t0)
    }

    /** Dik duran bir kareyi isler (kamera ve video dosyasi testi ayni yolu kullanir). */
    fun processBitmap(upright: Bitmap, tsMs: Long, t0: Long = System.nanoTime()): Frame {
        val mpImage: MPImage = BitmapImageBuilder(upright).build()
        val tConv = System.nanoTime()

        var ts = tsMs
        if (ts <= lastTs) ts = lastTs + 1
        lastTs = ts

        val handF = handExec.submit(Callable {
            val t = System.nanoTime()
            val res = hand.detectForVideo(mpImage, ts)
            val slots = arrayOfNulls<FloatArray>(2)
            for (i in res.landmarks().indices) {
                val slot = if (res.handednesses()[i][0].categoryName() == "Left") 0 else 1
                slots[slot] = FloatArray(42).also { a ->
                    res.landmarks()[i].forEachIndexed { j, p -> a[2 * j] = p.x(); a[2 * j + 1] = p.y() }
                }
            }
            Pair(slots.toList(), (System.nanoTime() - t) / 1_000_000)
        })
        val runPose = frameNo++ % POSE_EVERY == 0L || lastPose == null
        val poseF = if (!runPose) null else poseExec.submit(Callable {
            val t = System.nanoTime()
            val res = pose.detectForVideo(mpImage, ts)
            val arr = res.landmarks().firstOrNull()?.let { lms ->
                FloatArray(14).also { a -> poseIdx.forEachIndexed { k, j -> a[2 * k] = lms[j].x(); a[2 * k + 1] = lms[j].y() } }
            }
            Pair(arr, (System.nanoTime() - t) / 1_000_000)
        })
        val (hands, handMs) = handF.get()
        val (poseArr, poseMs) = poseF?.get()?.also { lastPose = it.first } ?: Pair(lastPose, 0L)
        val total = (System.nanoTime() - t0) / 1_000_000
        return Frame(hands, poseArr, upright.width, upright.height, total, (tConv - t0) / 1_000_000, handMs, poseMs)
    }

    companion object { const val POSE_EVERY = 2L }

    fun close() {
        handExec.submit { hand.close() }.get(); poseExec.submit { pose.close() }.get()
        handExec.shutdown(); poseExec.shutdown()
    }
}
