package com.valkeryne.multimodal

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

data class ModelFileInfo(
    val relPath: String,
    val fileName: String,
    val expectedBytes: Long,
    val isOnnxModel: Boolean
)

object ModelDownloader {

    private const val HF_BASE_URL = "https://huggingface.co/Nihilux/BYVV-onnx/resolve/main"

    // Queue of models to download sequentially in order:
    // 1. Vision Encoder (~108 MB)
    // 2. TTS Voice (~27 MB) + Config (~5 KB)
    // 3. STT Encoder (~32 MB) + Decoder (~118 MB)
    // 4. Main Language Model (~762 MB)
    // 5. Text Embeddings (~970 MB)
    val REQUIRED_MODELS = listOf(
        ModelFileInfo("valkeryne_onnx_int8/vision_encoder_int8.onnx", "vision_encoder_int8.onnx", 113420269L, true),
        ModelFileInfo("tts/vi_VN-vivos-x_low.onnx", "vi_VN-vivos-x_low.onnx", 27789545L, true),
        ModelFileInfo("tts/vi_VN-vivos-x_low.onnx.json", "vi_VN-vivos-x_low.onnx.json", 4966L, false),
        ModelFileInfo("stt/whisper_tiny/onnx/encoder_model.onnx", "encoder_model.onnx", 32904992L, true),
        ModelFileInfo("stt/whisper_tiny/onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx", 118553827L, true),
        ModelFileInfo("valkeryne_onnx_int8/main_language_model_int8.onnx", "main_language_model_int8.onnx", 798829186L, true),
        ModelFileInfo("valkeryne_onnx_int8/text_encoder_embed.onnx", "text_encoder_embed.onnx", 1017118959L, true)
    )

    // Download lock to prevent duplicate concurrent downloads
    private val isDownloadActive = AtomicBoolean(false)

    fun isDownloading(): Boolean = isDownloadActive.get()

    /**
     * Checks whether all required models exist and are valid.
     * Does NOT delete files during inspection to prevent loop oscillations.
     */
    fun areModelsDownloadedAndValid(targetDir: File): Boolean {
        for (info in REQUIRED_MODELS) {
            val localFile = File(targetDir, info.fileName)
            if (!isModelValid(localFile, info)) {
                return false
            }
        }
        return true
    }

    /**
     * Validates file existence and byte length.
     */
    fun isModelValid(file: File, info: ModelFileInfo): Boolean {
        if (!file.exists()) return false
        val currentSize = file.length()
        if (info.expectedBytes > 0 && currentSize != info.expectedBytes) {
            return false
        }
        return true
    }

    /**
     * Downloads models sequentially in a strict queue.
     */
    suspend fun downloadAllModels(
        targetDir: File,
        onProgress: (fileName: String, percent: Int, status: String) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        if (!isDownloadActive.compareAndSet(false, true)) {
            // Download already in progress in another thread/service
            return@withContext true
        }

        try {
            if (!targetDir.exists()) {
                targetDir.mkdirs()
            }

            for ((idx, info) in REQUIRED_MODELS.withIndex()) {
                val localFile = File(targetDir, info.fileName)
                val partFile = File(targetDir, "${info.fileName}.part")

                // If already completely downloaded, verify once and continue
                if (isModelValid(localFile, info)) {
                    onProgress(info.fileName, 100, "Ready (${idx + 1}/${REQUIRED_MODELS.size}) ${info.fileName}")
                    continue
                }

                var downloadSuccess = false
                var retryCount = 0
                val maxRetries = 10

                while (!downloadSuccess && retryCount < maxRetries) {
                    try {
                        val fileUrl = "$HF_BASE_URL/${info.relPath}"
                        onProgress(
                            info.fileName,
                            0,
                            "Downloading [${idx + 1}/${REQUIRED_MODELS.size}] ${info.fileName}..."
                        )

                        downloadWithResume(fileUrl, partFile, info.expectedBytes) { percent ->
                            onProgress(info.fileName, percent, "[${idx + 1}/${REQUIRED_MODELS.size}] ${info.fileName}: $percent%")
                        }

                        // Verify partFile size before renaming
                        if (partFile.exists() && partFile.length() == info.expectedBytes) {
                            if (localFile.exists()) localFile.delete()
                            partFile.renameTo(localFile)

                            // Quick ONNX graph test if applicable
                            if (info.isOnnxModel) {
                                try {
                                    val ortEnv = OrtEnvironment.getEnvironment()
                                    val testSession = ortEnv.createSession(localFile.absolutePath)
                                    testSession.close()
                                } catch (e: Exception) {
                                    // Corrupted, remove and retry
                                    localFile.delete()
                                    retryCount++
                                    continue
                                }
                            }

                            downloadSuccess = true
                            onProgress(info.fileName, 100, "[${idx + 1}/${REQUIRED_MODELS.size}] ${info.fileName} Complete")
                        } else {
                            retryCount++
                        }
                    } catch (e: Exception) {
                        retryCount++
                        kotlinx.coroutines.delay(2000L)
                        if (retryCount >= maxRetries) {
                            onProgress(info.fileName, 0, "Error downloading ${info.fileName}: ${e.message}")
                            return@withContext false
                        }
                    }
                }

                if (!downloadSuccess) {
                    return@withContext false
                }
            }

            onProgress("Done", 100, "All models downloaded and verified successfully!")
            return@withContext true
        } finally {
            isDownloadActive.set(false)
        }
    }

    private fun downloadWithResume(
        urlStr: String,
        destPartFile: File,
        expectedTotal: Long,
        onProgress: (Int) -> Unit
    ) {
        var existingBytes = if (destPartFile.exists()) destPartFile.length() else 0L

        // If part file already exceeded expected total, delete and restart
        if (existingBytes > expectedTotal && expectedTotal > 0) {
            destPartFile.delete()
            existingBytes = 0L
        } else if (existingBytes == expectedTotal && expectedTotal > 0) {
            onProgress(100)
            return
        }

        val url = URL(urlStr)
        val conn = url.openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        conn.connectTimeout = 30000
        conn.readTimeout = 30000

        val requestRange = (existingBytes > 0)
        if (requestRange) {
            conn.setRequestProperty("Range", "bytes=$existingBytes-")
        }
        conn.connect()

        val responseCode = conn.responseCode
        val isPartial = (responseCode == HttpURLConnection.HTTP_PARTIAL)
        val isOk = (responseCode == HttpURLConnection.HTTP_OK)

        if (!isPartial && !isOk) {
            throw Exception("HTTP server error code: $responseCode")
        }

        // If server returned 200 OK instead of 206 Partial, it doesn't support resuming this range
        val appendMode = (requestRange && isPartial)
        var downloadedBytes = if (appendMode) existingBytes else 0L
        val totalBytes = if (expectedTotal > 0) expectedTotal else (conn.contentLengthLong + downloadedBytes)

        val inputStream: InputStream = conn.inputStream
        val outputStream = FileOutputStream(destPartFile, appendMode)

        var lastReportedPercent = -1

        inputStream.use { input ->
            outputStream.use { output ->
                val buffer = ByteArray(256 * 1024)
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    downloadedBytes += bytesRead
                    if (totalBytes > 0) {
                        val percent = ((downloadedBytes * 100) / totalBytes).toInt().coerceIn(0, 100)
                        if (percent != lastReportedPercent) {
                            lastReportedPercent = percent
                            onProgress(percent)
                        }
                    }
                }
            }
        }
    }
}
