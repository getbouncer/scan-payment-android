package com.getbouncer.scan.payment.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Size
import com.getbouncer.scan.framework.Loader
import com.getbouncer.scan.framework.ResourceLoader
import com.getbouncer.scan.framework.hasOpenGl31
import com.getbouncer.scan.framework.ml.TFLAnalyzerFactory
import com.getbouncer.scan.framework.ml.TensorFlowLiteAnalyzer
import com.getbouncer.scan.framework.scale
import com.getbouncer.scan.framework.toRGBByteBuffer
import com.getbouncer.scan.framework.util.indexOfMax
import com.getbouncer.scan.payment.R
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import org.tensorflow.lite.Interpreter

private val TRAINED_IMAGE_SIZE = Size(48, 48)

/** model returns whether or not there is a screen present */
private const val NUM_CLASS = 27
class AlphabetDetect private constructor(interpreter: Interpreter) :
    TensorFlowLiteAnalyzer<AlphabetDetect.Input, ByteBuffer,
            AlphabetDetect.Prediction,
            Array<FloatArray>>(interpreter) {

    data class Input(val objDetectionImage: Bitmap)

    data class Prediction(val character: Char, val confidence: Float, val probabilities: FloatArray)

    override val name: String = Factory.NAME

    override fun buildEmptyMLOutput() = arrayOf(FloatArray(NUM_CLASS))

    override fun interpretMLOutput(data: Input, mlOutput: Array<FloatArray>): Prediction {
        val prediction = mlOutput[0]
        val index = prediction.indexOfMax()
        val character = if (index != null && index > 0) {
            ('A'.toInt() - 1 + index).toChar()
        } else {
            ' '
        }
        val confidence = if (index != null) prediction[index] else 0F
        return Prediction(
            character, confidence, prediction
        )
    }

    override fun transformData(data: Input): ByteBuffer = data.objDetectionImage
        .scale(TRAINED_IMAGE_SIZE)
        .toRGBByteBuffer()

    override fun executeInference(
        tfInterpreter: Interpreter,
        data: ByteBuffer,
        mlOutput: Array<FloatArray>
    ) = tfInterpreter.run(data, mlOutput)

    /**
     * A factory for creating instances of the [ScreenDetect]. This downloads the model from the
     * web. If unable to download from the web, this will throw a [FileNotFoundException].
     */
    class Factory(context: Context, loader: Loader) : TFLAnalyzerFactory<AlphabetDetect>(loader) {
        companion object {
            private const val USE_GPU = false
            private const val NUM_THREADS = 2
            const val IS_THREAD_SAFE = true

            const val NAME = "alphabet_detect"
        }

        override val isThreadSafe: Boolean = IS_THREAD_SAFE

        override val tfOptions: Interpreter.Options = Interpreter
            .Options()
            .setUseNNAPI(USE_GPU && hasOpenGl31(context))
            .setNumThreads(NUM_THREADS)

        override suspend fun newInstance(): AlphabetDetect? = createInterpreter()?.let { AlphabetDetect(it) }
    }

    /**
     * A loader for downloading and loading into memory instances of the [ScreenDetect] model.
     */
    class ModelLoader(context: Context) : ResourceLoader(context) {
        override val resource: Int = R.raw.s48_a_50_char_v4_147_0_94_16
    }
}
