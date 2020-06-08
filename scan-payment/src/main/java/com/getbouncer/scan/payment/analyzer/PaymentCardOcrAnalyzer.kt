package com.getbouncer.scan.payment.analyzer

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

    override suspend fun analyze(data: SSDOcr.Input, state: PaymentCardOcrState) = supervisorScope {
        val cardDetectFuture = if (state.runNameExtraction && nameDetect != null) {
            this.async {
                nameDetect.analyze(data, state)
            }
        } else {
            null
        }

        val ocrFuture = if (state.runOcr && ssdOcr != null) {
            this.async {
                ssdOcr.analyze(data, Unit)
            }
        } else {
            null
        }

        Prediction(
            pan = ocrFuture?.await()?.pan,
            panDetectionBoxes = ocrFuture?.await()?.detectedBoxes,
            name = cardDetectFuture?.await()?.name,
            expiry = null,
            objDetectionBoxes = cardDetectFuture?.await()?.boxes,
            isNameExtractionAvailable = nameDetect?.isAvailable() ?: false
        )
    }

    class Factory(
        private val ssdOcrFactory: SSDOcr.Factory,
        private val nameDetectFactory: NameDetectAnalyzer.Factory?
    ) : AnalyzerFactory<PaymentCardOcrAnalyzer> {
        override suspend fun newInstance(): PaymentCardOcrAnalyzer? = PaymentCardOcrAnalyzer(
            ssdOcrFactory.newInstance(),
            nameDetectFactory?.newInstance()
        )
    }
}
