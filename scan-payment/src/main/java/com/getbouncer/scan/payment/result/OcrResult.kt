package com.getbouncer.scan.payment.result

import com.getbouncer.scan.framework.AggregateResultListener
import com.getbouncer.scan.framework.ResultAggregator
import com.getbouncer.scan.framework.ResultAggregatorConfig
import com.getbouncer.scan.payment.card.isValidPan
import com.getbouncer.scan.payment.ml.PaymentCardPanOcr
import com.getbouncer.scan.payment.ml.SSDOcr
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Keep track of the results from the [AnalyzerLoop]. Count the number of times the loop sends each
 * PAN as a result, and when the first result is received.
 *
 * The [listener] will be notified of a result once [requiredAgreementCount] matching results are
 * received or the time since the first result exceeds the
 * [ResultAggregatorConfig.maxTotalAggregationTime].
 */
abstract class PaymentCardResultAggregator<ImageFormat>(
    config: ResultAggregatorConfig,
    listener: AggregateResultListener<ImageFormat, Unit, PaymentCardPanOcr, String>,
    name: String,
    private val requiredAgreementCount: Int? = null
) : ResultAggregator<ImageFormat, Unit, PaymentCardPanOcr, String>(config, listener, name) {

    private val storeFieldMutex = Mutex()
    private val panResults = mutableMapOf<String, Int>()

    override fun resetAndPause() {
        super.resetAndPause()
        panResults.clear()
    }

    override suspend fun aggregateResult(
        result: PaymentCardPanOcr,
        state: Unit,
        mustReturn: Boolean,
        updateState: (Unit) -> Unit
    ): String? {
        val numberCount = if (isValidResult(result)) {
            storeField(result.pan, panResults) // This must be last so numberCount is assigned.
        } else 0

        val hasMetRequiredAgreementCount =
            if (requiredAgreementCount != null) numberCount >= requiredAgreementCount else false

        return if (mustReturn || hasMetRequiredAgreementCount) {
            getMostLikelyField(panResults)
        } else {
            null
        }
    }

    private fun <T> getMostLikelyField(storage: Map<T, Int>): T? = storage.maxBy { it.value }?.key

    private suspend fun <T> storeField(field: T?, storage: MutableMap<T, Int>): Int = storeFieldMutex.withLock {
        if (field != null) {
            val count = 1 + (storage[field] ?: 0)
            storage[field] = count
            count
        } else {
            0
        }
    }
}

/**
 * Identify valid cards to be those with valid numbers. If a [requiredCardNumber] is provided, only matching cards are
 * considered valid.
 *
 * The [listener] will be notified of a result once [requiredAgreementCount] matching results are received or the time
 * since the first result exceeds the [ResultAggregatorConfig.maxTotalAggregationTime].
 */
class PaymentCardImageResultAggregator(
    config: ResultAggregatorConfig,
    listener: AggregateResultListener<SSDOcr.SSDOcrInput, Unit, PaymentCardPanOcr, String>,
    name: String,
    requiredAgreementCount: Int? = null,
    private val requiredCardNumber: String? = null
) : PaymentCardResultAggregator<SSDOcr.SSDOcrInput>(config, listener, name, requiredAgreementCount) {

    companion object {
        const val FRAME_TYPE_VALID_NUMBER = "valid_number"
        const val FRAME_TYPE_INVALID_NUMBER = "invalid_number"
    }

    init {
        assert(requiredCardNumber == null || isValidPan(requiredCardNumber)) {
            "Invalid required payment card number"
        }
    }

    override fun isValidResult(result: PaymentCardPanOcr): Boolean =
        if (isValidPan(requiredCardNumber)) {
            requiredCardNumber == result.pan
        } else {
            isValidPan(result.pan)
        }

    override fun getFrameSizeBytes(frame: SSDOcr.SSDOcrInput): Int = frame.fullImage.byteCount

    // TODO: This should store the least blurry images available
    override fun getSaveFrameIdentifier(result: PaymentCardPanOcr, frame: SSDOcr.SSDOcrInput): String? =
        if (isValidResult(result)) {
            FRAME_TYPE_VALID_NUMBER
        } else {
            FRAME_TYPE_INVALID_NUMBER
        }
}
