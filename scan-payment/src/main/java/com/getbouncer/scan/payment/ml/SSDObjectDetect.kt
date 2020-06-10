package com.getbouncer.scan.payment.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import android.util.Size
import com.getbouncer.scan.framework.Loader
import com.getbouncer.scan.framework.ModelWebLoader
import com.getbouncer.scan.payment.hasOpenGl31
import com.getbouncer.scan.framework.ml.TFLAnalyzerFactory
import com.getbouncer.scan.framework.ml.TensorFlowLiteAnalyzer
import com.getbouncer.scan.framework.ml.ssd.adjustLocations
import com.getbouncer.scan.framework.ml.ssd.softMax2D
import com.getbouncer.scan.framework.ml.ssd.toRectForm
import com.getbouncer.scan.framework.util.maxAspectRatioInSize
import com.getbouncer.scan.framework.util.reshape
import com.getbouncer.scan.framework.util.scaleAndCenterWithin
import com.getbouncer.scan.payment.crop
import com.getbouncer.scan.payment.ml.ssd.DetectionBox
import com.getbouncer.scan.payment.ml.ssd.ObjectPriorsGen
import com.getbouncer.scan.payment.ml.ssd.extractPredictions
import com.getbouncer.scan.payment.ml.ssd.rearrangeObjDetectionArray
import com.getbouncer.scan.payment.scale
import com.getbouncer.scan.payment.size
import com.getbouncer.scan.payment.toRGBByteBuffer
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Training images are normalized with mean 127.5. */
private const val IMAGE_MEAN = 127.5f

/** Training images are normalized with std 128.5. */
private const val IMAGE_STD = 128.5f

/**
 * We use the output from last two layers with feature maps 19x19 and 10x10
 * and for each feature map activation we have 6 priors, so total priors are
 * 19x19x6 + 10x10x6 = 2766
 */
private const val NUM_OF_PRIORS = 2766

/**
 * For each activation in our feature map, we have predictions for 6 bounding boxes
 * of different aspect ratios
 */
private const val NUM_OF_PRIORS_PER_ACTIVATION = 6

/**
 * We can detect a total of 13 objects plus the background class
 */
private const val NUM_OF_CLASSES = 14

/**
 * Each prior or bounding box can be represented by 4 coordinates XMin, YMin, XMax, YMax.
 */
private const val NUM_OF_COORDINATES = 4

/**
 * Represents the total number of data points for locations
 */
private const val NUM_LOC = NUM_OF_COORDINATES * NUM_OF_PRIORS

/**
 * Represents the total number of data points for classes
 */
private const val NUM_CLASS = NUM_OF_CLASSES * NUM_OF_PRIORS

private const val PROB_THRESHOLD = 0.3f
private const val IOU_THRESHOLD = 0.45f
private const val CENTER_VARIANCE = 0.1f
private const val SIZE_VARIANCE = 0.2f
private const val LIMIT = 10

private val TRAINED_IMAGE_SIZE = Size(300, 300)

private val FEATURE_MAP_SIZES = intArrayOf(19, 10)

/**
 * This value should never change, and is thread safe.
 */
private val PRIORS by lazy { ObjectPriorsGen.combinePriors() }

/**
 * Convert a [Rect] to a [Size].
 */
private fun Rect.size() = Size(width(), height())

fun RectF.scaled(scaledSize: Size): RectF {
    return RectF(
        this.left * scaledSize.width,
        this.top * scaledSize.height,
        this.right * scaledSize.width,
        this.bottom * scaledSize.height
    )
}

fun calculateCardFinderCoordinatesFromObjectDetection(rect: RectF, previewImage: Size, cardFinder: Rect): RectF {
    val objectDetection = SSDObjectDetect.calculateObjectDetectionFromCardFinder(
        previewImage,
        cardFinder
    )
    val scaled = rect.scaled(objectDetection.size())
    return RectF(
        /* left */ (scaled.left - (objectDetection.width() / 2 - cardFinder.width() / 2)) / cardFinder.width(),
        /* top */ (scaled.top - (objectDetection.height() / 2 - cardFinder.height() / 2)) / cardFinder.height(),
        /* right */ (scaled.right - (objectDetection.width() / 2 - cardFinder.width() / 2)) / cardFinder.width(),
        /* bottom */ (scaled.bottom - (objectDetection.height() / 2 - cardFinder.height() / 2)) / cardFinder.height()
    )
}

class SSDObjectDetect private constructor(interpreter: Interpreter) :
    TensorFlowLiteAnalyzer<SSDObjectDetect.Input, Array<ByteBuffer>, SSDObjectDetect.Prediction, Map<Int, Array<FloatArray>>>(interpreter) {

    enum class Labels {
        BACKGROUND,
        AMERICAN_EXPRESS_LOGO,
        BANK_OF_AMERICA_TEXT,
        CARD,
        CHASE_LOGO,
        CHIP,
        DEBIT_TEXT,
        DOVE_LOGO_HOLO,
        MASTERCARD,
        NAME,
        PAN,
        VISA,
        WELLS_FARGO_LOGO
    }

    companion object {
        /**
         * Given a card finder region of a preview image, calculate the associated object detection
         * square.
         */
        internal fun calculateObjectDetectionFromCardFinder(previewImage: Size, cardFinder: Rect): Rect {
            val objectDetectionSquareSize = maxAspectRatioInSize(previewImage, 1F)
            return Rect(
                /* left */ max(0, cardFinder.centerX() - objectDetectionSquareSize.width / 2),
                /* top */ max(0, cardFinder.centerY() - objectDetectionSquareSize.height / 2),
                /* right */ min(previewImage.width, cardFinder.centerX() + objectDetectionSquareSize.width / 2),
                /* bottom */ min(previewImage.height, cardFinder.centerY() + objectDetectionSquareSize.height / 2)
            )
        }

        /**
         * Calculate what portion of the full image should be cropped for object detection based on
         * the position of card finder within the preview image.
         */
        private fun calculateImageCrop(data: Input): Rect {
            require(
                data.cardFinder.left >= 0 &&
                        data.cardFinder.right <= data.previewSize.width &&
                        data.cardFinder.top >= 0 &&
                        data.cardFinder.bottom <= data.previewSize.height
            ) { "Card finder is outside preview image bounds" }

            // Calculate the object detection square based on the card finder, limited by the preview
            val objectDetectionSquare =
                calculateObjectDetectionFromCardFinder(
                    data.previewSize,
                    data.cardFinder
                )

            val scaledPreviewImage = data.previewSize.scaleAndCenterWithin(data.fullImage.size())
            val previewScale = scaledPreviewImage.width().toFloat() / data.previewSize.width

            // Scale the objectDetectionSquare to match the scaledPreviewImage
            val scaledObjectDetectionSquare = Rect(
                (objectDetectionSquare.left * previewScale).roundToInt(),
                (objectDetectionSquare.top * previewScale).roundToInt(),
                (objectDetectionSquare.right * previewScale).roundToInt(),
                (objectDetectionSquare.bottom * previewScale).roundToInt()
            )

            // Position the scaledObjectDetectionSquare on the fullImage
            return Rect(
                max(0, scaledObjectDetectionSquare.left + scaledPreviewImage.left),
                max(0, scaledObjectDetectionSquare.top + scaledPreviewImage.top),
                min(data.fullImage.width, scaledObjectDetectionSquare.right + scaledPreviewImage.left),
                min(data.fullImage.height, scaledObjectDetectionSquare.bottom + scaledPreviewImage.top)
            )
        }

        /**
         * Calculate what portion of the full image should be cropped for object detection based on
         * the position of card finder within the preview image.
         */
        fun cropImage(data: Input): Bitmap = data.fullImage.crop(calculateImageCrop(data))
    }

    /**
     * Data used by this analyzer
     */
    data class Input(
        val fullImage: Bitmap,
        val previewSize: Size,
        val cardFinder: Rect,
        val iin: String?
    )

    /**
     * The result of this analyzer
     */
    data class Prediction(
        val detectionBoxes: List<DetectionBox>,
        val objectDetectionImageSize: Size,
        val iin: String?
    )

    override val name: String = Factory.NAME

    /**
     * The model reshapes all the data to 1 x [All Data Points]
     */
    override fun buildEmptyMLOutput(): Map<Int, Array<FloatArray>> = mapOf(
        0 to arrayOf(FloatArray(NUM_CLASS)),
        1 to arrayOf(FloatArray(NUM_LOC))
    )

    override fun transformData(data: Input): Array<ByteBuffer> = arrayOf(
        cropImage(data)
            .scale(TRAINED_IMAGE_SIZE)
            .toRGBByteBuffer(mean = IMAGE_MEAN, std = IMAGE_STD)
    )

    override fun interpretMLOutput(
        data: Input,
        mlOutput: Map<Int, Array<FloatArray>>
    ): Prediction {
        val outputClasses = mlOutput[0] ?: arrayOf(FloatArray(NUM_CLASS))
        val outputLocations = mlOutput[1] ?: arrayOf(FloatArray(NUM_LOC))

        val boxes = rearrangeObjDetectionArray(
            locations = outputLocations,
            featureMapSizes = FEATURE_MAP_SIZES,
            numberOfPriors = NUM_OF_PRIORS_PER_ACTIVATION,
            locationsPerPrior = NUM_OF_COORDINATES
        ).reshape(NUM_OF_COORDINATES)
        boxes.adjustLocations(
            priors = PRIORS,
            centerVariance = CENTER_VARIANCE,
            sizeVariance = SIZE_VARIANCE
        )
        boxes.forEach { it.toRectForm() }

        val scores = rearrangeObjDetectionArray(
            locations = outputClasses,
            featureMapSizes = FEATURE_MAP_SIZES,
            numberOfPriors = NUM_OF_PRIORS_PER_ACTIVATION,
            locationsPerPrior = NUM_OF_CLASSES
        ).reshape(NUM_OF_CLASSES)
        scores.forEach { it.softMax2D() }

        val detectionBoxes = extractPredictions(
            scores = scores,
            boxes = boxes,
            probabilityThreshold = PROB_THRESHOLD,
            intersectionOverUnionThreshold = IOU_THRESHOLD,
            limit = LIMIT,
            classifierToLabel = { it - 1 }
        )

        return Prediction(
            detectionBoxes = detectionBoxes,
            objectDetectionImageSize = calculateImageCrop(data).size(),
            iin = data.iin
        )
    }

    override fun executeInference(
        tfInterpreter: Interpreter,
        data: Array<ByteBuffer>,
        mlOutput: Map<Int, Array<FloatArray>>
    ) = tfInterpreter.runForMultipleInputsOutputs(data, mlOutput)

    /**
     * A factory for creating instances of the [SSDObjectDetect]. This downloads the model from the
     * web. If unable to download from the web, this will return null.
     */
    class Factory(
        context: Context,
        loader: Loader,
        threads: Int = DEFAULT_THREADS
    ) : TFLAnalyzerFactory<SSDObjectDetect>(loader) {
        companion object {
            private const val USE_GPU = false
            private const val DEFAULT_THREADS = 2

            const val NAME = "ssd_object_detect"
        }

        override val tfOptions: Interpreter.Options = Interpreter
            .Options()
            .setUseNNAPI(USE_GPU && hasOpenGl31(context))
            .setNumThreads(threads)

        override suspend fun newInstance(): SSDObjectDetect? = createInterpreter()?.let { SSDObjectDetect(it) }
    }

    /**
     * A loader for downloading and loading into memory instances of [SSDObjectDetect].
     */
    class ModelLoader(context: Context) : ModelWebLoader(context) {
        companion object {
            const val VERSION = "v0.0.3"
        }

        override val modelClass: String = "object_detection"
        override val modelVersion: String = VERSION
        override val modelFileName: String = "ssd.tflite"
        override val hash: String = "7c5a294ff9a1e665f07d3e64d898062e17a2348f01b0be75b2d5295988ce6a4c"
    }
}
