package com.getbouncer.scan.payment.analyzer

data class PaymentCardOcrState(
    val runOcr: Boolean,
    val runNameExtraction: Boolean,
    val runExpiryExtraction: Boolean
)
