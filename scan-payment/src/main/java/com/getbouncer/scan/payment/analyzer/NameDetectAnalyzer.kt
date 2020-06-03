package com.getbouncer.scan.payment.analyzer

import android.graphics.Bitmap
import android.graphics.RectF
import com.getbouncer.scan.framework.Analyzer
import com.getbouncer.scan.framework.AnalyzerFactory
import com.getbouncer.scan.framework.ml.ssd.hardNonMaximumSuppression
import com.getbouncer.scan.framework.ml.ssd.rectForm
import com.getbouncer.scan.framework.size
import com.getbouncer.scan.payment.ml.AlphabetDetect
import com.getbouncer.scan.payment.ml.SSDObjectDetect
import com.getbouncer.scan.payment.ml.SSDOcr
import com.getbouncer.scan.payment.ml.scaled
import com.getbouncer.scan.payment.ml.ssd.DetectionBox
import com.getbouncer.scan.payment.ml.toObjectDetectionCroppedImage

// Some params for how we post process our name detector
// Number of predictions per predicted character box
private const val NUM_PREDICTION_STRIDES = 10
private const val NMS_THRESHOLD = 0.75F
private const val CHAR_CONFIDENCE_THRESHOLD = 0.7

class NameDetectAnalyzer private constructor(
    private val ssdObjectDetect: SSDObjectDetect?,
    private val alphabetDetect: AlphabetDetect?
) : Analyzer<SSDOcr.Input, PaymentCardOcrState, NameDetectAnalyzer.Output> {

    data class Output(
        val name: String?,
        val boxes: List<DetectionBox>?
    )

    fun isAvailable() = ssdObjectDetect != null

    override val name: String = "name_detect_analyzer"

    override suspend fun analyze(
        data: SSDOcr.Input,
        state: PaymentCardOcrState
    ) = if (!state.runNameExtraction || ssdObjectDetect == null || alphabetDetect == null) {
        Output("", null)
    } else {
        val objDetectInput = SSDObjectDetect.Input(
            fullImage = data.fullImage,
            previewSize = data.previewSize,
            cardFinder = data.cardFinder,
            iin = null
        )

        val output = ssdObjectDetect.analyze(objDetectInput, Unit)
        var nameRect: RectF? = null

        // check to see if we detected a name. If so, run our name extraction model, serially for now
        for (box in output.detectionBoxes) {
            if (box.label == SSDObjectDetect.Labels.NAME.ordinal) {
                nameRect = box.rect
                break
            }
        }

        val name = if (nameRect != null) {
            processPredictions(nameRect, objDetectInput.toObjectDetectionCroppedImage())
        } else {
            null
        }
        Output(name, output.detectionBoxes)
    }

    internal class CharPredictionWithBox(
        val characterPrediction: AlphabetDetect.Prediction,
        val box: RectF
    ) {
        fun getNormalizedRectform(width: Int, height: Int) = rectForm(
            left = box.left / width,
            top = box.top / height,
            right = box.right / width,
            bottom = box.bottom / height
        )
    }

    private suspend fun processPredictions(
        nameRect: RectF,
        bitmapForObjectDetection: Bitmap
    ): String? {
        if (alphabetDetect == null) {
            return null
        }

        val scaledNameRect = nameRect.scaled(bitmapForObjectDetection.size())
        val x = scaledNameRect.left.toInt()
        val y = scaledNameRect.top.toInt()
        val width = scaledNameRect.width().toInt()
        val height = scaledNameRect.height().toInt()

        // We use a square aspect ratio for our character recognizer, so we assume that the height
        // of the name bounding box is the height and width of the square
        val charWidth = height

        // We adjust the start and end of the name bounding box to better capture the first char
        val xStart = Math.max(0, x - charWidth / 2)
        val nameWidth = Math.min(bitmapForObjectDetection.width - xStart, width + charWidth)

        val nameBitmap = Bitmap.createBitmap(bitmapForObjectDetection, xStart, y, nameWidth, height)
        val predictions: MutableList<CharPredictionWithBox> = ArrayList()

        // iterate through each stride, making a prediction per stride
        var nameX = 0
        while (nameX < nameWidth - charWidth) {
            val firstLetterBitmap = Bitmap.createBitmap(nameBitmap, nameX, 0, height, height)
            predictions.add(
                CharPredictionWithBox(
                    characterPrediction = alphabetDetect.analyze(AlphabetDetect.Input(firstLetterBitmap), Unit),
                    box = RectF(nameX.toFloat(), 0F, height.toFloat(), height.toFloat())
                )
            )
            nameX += charWidth / NUM_PREDICTION_STRIDES
        }

        val (boxes, probabilities) = predictions.map {
            it.getNormalizedRectform(
                width = bitmapForObjectDetection.width,
                height = bitmapForObjectDetection.height
            ) to it.characterPrediction.confidence
        }.unzip()

        val indices: List<Int> = hardNonMaximumSuppression(
            boxes.toTypedArray(),
            probabilities.toFloatArray(),
            NMS_THRESHOLD,
            limit = 0
        )
        return processNMSResults(predictions.filterIndexed { index, _ -> indices.contains(index) })
    }

    /**
     * Processes each cluster of letters from NMS, doing a simple voting algorithm and
     * tie-breaking with confidence
     */
    private fun processNMSCluster(charClusters: List<AlphabetDetect.Prediction>): Char {
        var candidateLetter = 0.toChar()
        var candidateLetterConfidence = 0f
        var candidateConsecutiveCount = 0
        var currentConsecutiveCount = 0
        var currentLetterMaxConfidence = 0f
        var lastSeenLetter = 0.toChar()
        charClusters.forEach { characterPrediction ->
            if (lastSeenLetter == characterPrediction.character) {
                currentConsecutiveCount += 1
                currentLetterMaxConfidence =
                    Math.max(currentLetterMaxConfidence, characterPrediction.confidence)
            } else {
                currentConsecutiveCount = 1
                currentLetterMaxConfidence = characterPrediction.confidence
                lastSeenLetter = characterPrediction.character
            }
            if (currentConsecutiveCount == candidateConsecutiveCount && currentLetterMaxConfidence > candidateLetterConfidence ||
                currentConsecutiveCount > candidateConsecutiveCount
            ) {
                candidateLetterConfidence = currentLetterMaxConfidence
                candidateLetter = characterPrediction.character
                candidateConsecutiveCount = currentConsecutiveCount
            }
        }
        return if (candidateLetterConfidence > CHAR_CONFIDENCE_THRESHOLD) {
            candidateLetter
        } else ' '
    }

    /**
     * Black magic to calculate the "width" (in terms of prediction index) for each space
     *   Compares the two widest (by index) spaces with the p25 of all of the background
     *   predictions, then, checks to see if difference between the widest and the second
     *   widest is sufficiently close compared to the p25
     * For example: [1,1,1,1,1,5,6] would yield 5, since p25 is 1, pMax is 6, pMax2 is 5,
     * and pMax - pMax2 << pMax2 - p25
     */
    private fun getSpacesWidth(spaces: List<Int>): Int {
        if (spaces.size <= 2) {
            return 10
        }
        val slice = spaces.subList(1, spaces.size - 1).sorted()
        val p25Index = slice.size * 25 / 100
        val p25 = slice[p25Index]
        val pmax = slice[slice.size - 1]
        val pmax2 = if (slice.size >= 2) slice[slice.size - 2] else pmax
        return if (pmax == pmax2 && pmax == p25) {
            pmax + 1
        } else if ((pmax - pmax2) * 2 <= pmax2 - p25) {
            pmax2
        } else {
            pmax
        }
    }

    /**
     * Accepts the output from hard NMS and produces the predicted word
     */
    private fun processNMSResults(
        predictions: List<CharPredictionWithBox>
    ): String? {
        val intermediateChars: MutableList<Char> = ArrayList()
        val charClusters: MutableList<AlphabetDetect.Prediction> = ArrayList()
        val spaces: MutableList<Int> = ArrayList()

        var currentSpaceClusterSize = 0
        val debugWord = StringBuilder()
        for (prediction in predictions) {
            debugWord.append(prediction.characterPrediction.character)
            if (prediction.characterPrediction.character == ' ') {
                if (charClusters.size > 0) {
                    // process the cluster
                    intermediateChars.add(processNMSCluster(charClusters))
                    charClusters.clear()
                }
                currentSpaceClusterSize += 1
                intermediateChars.add(' ')
            } else {
                charClusters.add(prediction.characterPrediction)
                if (currentSpaceClusterSize > 0) {
                    // add space cluster, reset counter
                    spaces.add(currentSpaceClusterSize)
                    currentSpaceClusterSize = 0
                }
            }
        }

        // do one last process if we ended w/ a char
        if (charClusters.size > 0) {
            // process the cluster
            intermediateChars.add(processNMSCluster(charClusters))
            charClusters.clear()
        }

        val word = StringBuilder()
        val spaceWidth = getSpacesWidth(spaces)
        var numConsecSpaces = 0
        for (c in intermediateChars) {
            if (c == ' ') {
                numConsecSpaces += 1
                if (numConsecSpaces == spaceWidth) {
                    word.append(' ')
                }
            } else {
                word.append(c)
                numConsecSpaces = 0
            }
        }
        return word.toString().trim { it <= ' ' }
    }

    class Factory(
        private val ssdObjectDetectFactory: SSDObjectDetect.Factory,
        private val alphabetDetectFactory: AlphabetDetect.Factory
    ) : AnalyzerFactory<NameDetectAnalyzer> {
        override val isThreadSafe: Boolean = ssdObjectDetectFactory.isThreadSafe && alphabetDetectFactory.isThreadSafe

        override suspend fun newInstance(): NameDetectAnalyzer? {
            return NameDetectAnalyzer(
                ssdObjectDetectFactory.newInstance(),
                alphabetDetectFactory.newInstance()
            )
        }
    }
}
