package com.getbouncer.scan.payment.analyzer

import kotlinx.coroutines.CoroutineScope

data class PaymentCardOcrState(
    val runOcr: Boolean,
    val runNameExtraction: Boolean
)
