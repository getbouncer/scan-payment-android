package com.getbouncer.scan.payment.analyzer

import android.util.Log
import com.getbouncer.scan.framework.Analyzer
import com.getbouncer.scan.framework.AnalyzerFactory
import com.getbouncer.scan.payment.ml.SSDOcr
import com.getbouncer.scan.payment.ml.ssd.DetectionBox
import kotlinx.coroutines.async
import kotlinx.coroutines.supervisorScope

class PaymentCardOcrAnalyzer private constructor(
    private val ssdOcr: SSDOcr?,
    private val nameDetect: NameDetectAnalyzer?
) : Analyzer<SSDOcr.Input, PaymentCardOcrState, PaymentCardOcrAnalyzer.Prediction> {

    data class Prediction(
        val pan: String?,
        val panDetectionBoxes: List<DetectionBox>?,
        val name: String?,
        val expiry: String?,
        val objDetectionBoxes: List<DetectionBox>?,
        val isNameExtractionAvailable: Boolean
    )

    override val name: String = "payment_card_ocr_analyzer"

    override suspend fun analyze(data: SSDOcr.Input, state: PaymentCardOcrState): Prediction {
        Log.d("DEBUG", "running name detect? ${nameDetect != null} ${state.runNameExtraction}")
        val cardDetectFuture = if (state.runNameExtraction && nameDetect != null) {
            nameDetect.analyze(data, state)
        } else {
            null
        }

        val ocrFuture = if (state.runOcr && ssdOcr != null) {

                ssdOcr.analyze(data, Unit)

        } else {
            null
        }

        return Prediction(
            pan = ocrFuture?.pan,
            panDetectionBoxes = ocrFuture?.detectedBoxes,
            name = cardDetectFuture?.name,
            expiry = null,
            objDetectionBoxes = cardDetectFuture?.boxes,
            isNameExtractionAvailable = nameDetect?.isAvailable() ?: false
        )
    }

    class Factory(
        private val ssdOcrFactory: SSDOcr.Factory,
        private val nameDetectFactory: NameDetectAnalyzer.Factory?
    ) : AnalyzerFactory<PaymentCardOcrAnalyzer> {
        override val isThreadSafe: Boolean = ssdOcrFactory.isThreadSafe && nameDetectFactory?.isThreadSafe ?: true

        override suspend fun newInstance(): PaymentCardOcrAnalyzer? = PaymentCardOcrAnalyzer(
            ssdOcrFactory.newInstance(),
            nameDetectFactory?.newInstance()
        )
    }
}
