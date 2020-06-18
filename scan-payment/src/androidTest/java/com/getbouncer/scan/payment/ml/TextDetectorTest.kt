package com.getbouncer.scan.payment.ml

import androidx.test.filters.SmallTest
import androidx.test.platform.app.InstrumentationRegistry
import com.getbouncer.scan.framework.Config
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.test.assertNotNull

class TextDetectorTest {
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    @Before
    fun before() {
        Config.apiKey = "uXDc2sbugrkmvj1Bm3xOTXBw7NW4llgn"
    }

    @After
    fun after() {
        Config.apiKey = null
    }

    /**
     * TODO: this method should use runBlockingTest instead of runBlocking. However, an issue with
     * runBlockingTest currently fails when functions under test use withContext(Dispatchers.IO) or
     * withContext(Dispatchers.Default).
     *
     * See https://github.com/Kotlin/kotlinx.coroutines/issues/1204 for details.
     */
    @Test
    @SmallTest
    fun objectDetect_createsInterpreter() = runBlocking {
        val loader = TextDetector.ModelLoader(testContext)
        val factory = TextDetector.Factory(testContext, loader)

        val localFileName = loader.url.path.replace('/', '_')
        val localFile = File(testContext.cacheDir, localFileName)
        if (localFile.exists()) {
            localFile.delete()
        }

        assertNotNull(factory.newInstance())
    }.let { Unit }

    @Test
    @SmallTest
    fun objectDetect_findsObjects() {
        // TODO: add resources and test the object detector
    }
}
