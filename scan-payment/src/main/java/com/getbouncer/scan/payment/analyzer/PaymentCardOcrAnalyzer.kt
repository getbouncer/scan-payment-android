package com.getbouncer.scan.payment.analyzer

import com.getbouncer.scan.framework.Analyzer
import com.getbouncer.scan.framework.AnalyzerFactory
import com.getbouncer.scan.payment.ml.ExpiryDetect
import com.getbouncer.scan.payment.ml.SSDOcr
import com.getbouncer.scan.payment.ml.ssd.DetectionBox
import kotlinx.coroutines.async
import kotlinx.coroutines.supervisorScope

class PaymentCardOcrAnalyzer private constructor(
    private val ssdOcr: SSDOcr?,
    private val nameAndExpiryAnalyzer: NameAndExpiryAnalyzer?
) : Analyzer<SSDOcr.Input, PaymentCardOcrState, PaymentCardOcrAnalyzer.Prediction> {

    data class Prediction(
        val pan: String?,
        val panDetectionBoxes: List<DetectionBox>?,
        val name: String?,
        val expiry: ExpiryDetect.Expiry?,
        val objDetectionBoxes: List<DetectionBox>?,
        val isNameAndExpiryExtractionAvailable: Boolean
    )

    override val name: String = "payment_card_ocr_analyzer"

    override suspend fun analyze(data: SSDOcr.Input, state: PaymentCardOcrState) = supervisorScope {
        val cardDetectDeferred = if ((state.runNameExtraction || state.runExpiryExtraction) && nameAndExpiryAnalyzer != null) {
            this.async {
                nameAndExpiryAnalyzer.analyze(data, state)
            }
        } else {
            null
        }

        val ocrDeferred = if (state.runOcr && ssdOcr != null) {
            this.async {
                ssdOcr.analyze(data, Unit)
            }
        } else {
            null
        }

        Prediction(
            pan = ocrDeferred?.await()?.pan,
            panDetectionBoxes = ocrDeferred?.await()?.detectedBoxes,
            name = cardDetectDeferred?.await()?.name,
            expiry = cardDetectDeferred?.await()?.expiry,
            objDetectionBoxes = cardDetectDeferred?.await()?.boxes,
            isNameAndExpiryExtractionAvailable = nameAndExpiryAnalyzer?.isAvailable() ?: false
        )
    }

    class Factory(
        private val ssdOcrFactory: SSDOcr.Factory,
        private val nameDetectFactory: NameAndExpiryAnalyzer.Factory?
    ) : AnalyzerFactory<PaymentCardOcrAnalyzer> {
        override suspend fun newInstance(): PaymentCardOcrAnalyzer? = PaymentCardOcrAnalyzer(
            ssdOcrFactory.newInstance(),
            nameDetectFactory?.newInstance()
        )
    }
}
