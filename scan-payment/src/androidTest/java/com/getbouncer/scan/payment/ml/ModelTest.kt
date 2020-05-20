package com.getbouncer.scan.payment.ml

import android.graphics.Rect
import android.util.Size
import androidx.core.graphics.drawable.toBitmap
import androidx.test.filters.MediumTest
import androidx.test.platform.app.InstrumentationRegistry
import com.getbouncer.scan.framework.image.size
import com.getbouncer.scan.payment.test.R
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlinx.coroutines.test.runBlockingTest
import org.junit.Test

class ModelTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    @Test
    @MediumTest
    @ExperimentalCoroutinesApi
    fun resourceModelExecution_works() = runBlockingTest {
        val bitmap = testContext.resources.getDrawable(R.drawable.ocr_card_numbers_clear, null).toBitmap()
        val model = SSDOcr.Factory(appContext, SSDOcr.ModelLoader(appContext)).newInstance()
        assertNotNull(model)

        val prediction = model.analyze(SSDOcr.SSDOcrInput(bitmap, bitmap.size(), bitmap.size().toRect()), Unit)
        assertNotNull(prediction)
        assertEquals("4557095462268383", prediction.pan)
    }

    @Test
    @MediumTest
    @ExperimentalCoroutinesApi
    fun resourceModelExecution_worksRepeatedly() = runBlockingTest {
        val bitmap = testContext.resources.getDrawable(R.drawable.ocr_card_numbers_clear, null).toBitmap()
        val model = SSDOcr.Factory(appContext, SSDOcr.ModelLoader(appContext)).newInstance()
        assertNotNull(model)

        val prediction1 = model.analyze(SSDOcr.SSDOcrInput(bitmap, bitmap.size(), bitmap.size().toRect()), Unit)
        val prediction2 = model.analyze(SSDOcr.SSDOcrInput(bitmap, bitmap.size(), bitmap.size().toRect()), Unit)
        assertNotNull(prediction1)
        assertEquals("4557095462268383", prediction1.pan)

        assertNotNull(prediction2)
        assertEquals("4557095462268383", prediction2.pan)
    }
}

fun Size.toRect(): Rect = Rect(0, 0, width, height)
