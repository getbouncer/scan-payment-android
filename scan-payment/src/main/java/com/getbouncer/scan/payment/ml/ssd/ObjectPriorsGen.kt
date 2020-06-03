package com.getbouncer.scan.payment.ml.ssd

import com.getbouncer.scan.framework.util.clamp

/**
 * A utility class used to generate priors for initializing SSD. Since we use the output feature
 * maps of only two layers, we call genPriors twice and combine the information. The specification
 * is followed as in the original paper https://arxiv.org/abs/1512.02325 by Wei Liu Et al.
 */
internal object ObjectPriorsGen {

    fun combinePriors(): Array<FloatArray> {
        val priorsCombined: Array<FloatArray>
        val priorsOne: Array<FloatArray> = genPriors(19, 16, 60, 105, 2, 3, 6)
        val priorsTwo: Array<FloatArray> = genPriors(10, 32, 105, 150, 2, 3, 6)
        priorsCombined = Array(
            priorsOne.size + priorsTwo.size
        ) { FloatArray(4) }
        for (i in priorsOne.indices) {
            for (j in priorsOne[0].indices) {
                priorsCombined[i][j] = clamp(
                    priorsOne[i][j],
                    0.0f,
                    1.0f
                )
            }
        }
        for (i in priorsTwo.indices) {
            for (j in priorsTwo[0].indices) {
                priorsCombined[i + priorsOne.size][j] =
                    clamp(
                        priorsTwo[i][j],
                        0.0f,
                        1.0f
                    )
            }
        }
        return priorsCombined
    }

    private fun genPriors(
        featureMapSize: Int,
        shrinkage: Int,
        boxSizeMin: Int,
        boxSizeMax: Int,
        aspecRatioOne: Int,
        aspectRatioTwo: Int,
        noOfPriors: Int
    ): Array<FloatArray> {
        val boxes = Array(
            featureMapSize * featureMapSize * noOfPriors
        ) { FloatArray(4) }
        var xCenter: Float
        var yCenter: Float
        val imageSize = 300
        var size: Float
        val scale = imageSize / shrinkage.toFloat()
        var h: Float
        var w: Float
        var priorIndex = 0
        var ratioOne: Float
        var ratioTwo: Float
        for (j in 0 until featureMapSize) {
            for (i in 0 until featureMapSize) {
                xCenter = (i + 0.5).toFloat() / scale
                yCenter = (j + 0.5).toFloat() / scale
                size = boxSizeMin.toFloat()
                w = size / imageSize
                h = w
                boxes[priorIndex][0] = xCenter
                boxes[priorIndex][1] = yCenter
                boxes[priorIndex][2] = h
                boxes[priorIndex][3] = w
                priorIndex++
                size = Math.sqrt(boxSizeMax * boxSizeMin.toDouble()).toFloat()
                w = size / imageSize
                h = w
                boxes[priorIndex][0] = xCenter
                boxes[priorIndex][1] = yCenter
                boxes[priorIndex][2] = h
                boxes[priorIndex][3] = w
                priorIndex++
                size = boxSizeMin.toFloat()
                w = size / imageSize
                h = w
                ratioOne = Math.sqrt(aspecRatioOne.toDouble()).toFloat()
                ratioTwo = Math.sqrt(aspectRatioTwo.toDouble()).toFloat()
                boxes[priorIndex][0] = xCenter
                boxes[priorIndex][1] = yCenter
                boxes[priorIndex][2] = h * ratioOne
                boxes[priorIndex][3] = w / ratioOne
                priorIndex++
                boxes[priorIndex][0] = xCenter
                boxes[priorIndex][1] = yCenter
                boxes[priorIndex][2] = h / ratioOne
                boxes[priorIndex][3] = w * ratioOne
                priorIndex++
                boxes[priorIndex][0] = xCenter
                boxes[priorIndex][1] = yCenter
                boxes[priorIndex][2] = h * ratioTwo
                boxes[priorIndex][3] = w / ratioTwo
                priorIndex++
                boxes[priorIndex][0] = xCenter
                boxes[priorIndex][1] = yCenter
                boxes[priorIndex][2] = h / ratioTwo
                boxes[priorIndex][3] = w * ratioTwo
                priorIndex++
            }
        }
        return boxes
    }
}
