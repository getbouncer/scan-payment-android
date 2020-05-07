package com.getbouncer.scan.payment.image

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Size
import androidx.annotation.RestrictTo

/**
 * An image scanned and ready for image processing
 */
@RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
data class CardPreviewImage(
    val fullImage: Bitmap,
    val previewSize: Size,
    val cardFinder: Rect
) {
    val sizeInBytes = fullImage.byteCount
}
