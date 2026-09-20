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

data class ModelFileInfo(
    val relPath: String,
    val fileName: String,
    val expectedBytes: Long,
    val isOnnxModel: Boolean
)

object ModelDownloader {

    private const val HF_BASE_URL = "https://huggingface.co/Nihilux/BYVV-onnx/resolve/main"

    val REQUIRED_MODELS = listOf(
        ModelFileInfo("valkeryne_onnx_int8/vision_encoder_int8.onnx", "vision_encoder_int8.onnx", 113420269L, true),
        ModelFileInfo("valkeryne_onnx_int8/text_encoder_embed.onnx", "text_encoder_embed.onnx", 1017118959L, true),
        ModelFileInfo("valkeryne_onnx_int8/main_language_model_int8.onnx", "main_language_model_int8.onnx", 798829186L, true),
        ModelFileInfo("tts/vi_VN-vivos-x_low.onnx", "vi_VN-vivos-x_low.onnx", 27789545L, true),
        ModelFileInfo("tts/vi_VN-vivos-x_low.onnx.json", "vi_VN-vivos-x_low.onnx.json", 4966L, false),
        ModelFileInfo("stt/whisper_tiny/onnx/encoder_model.onnx", "encoder_model.onnx", 32904992L, true),
        ModelFileInfo("stt/whisper_tiny/onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx", 118553827L, true)
    )

    /**
     * Checks whether all required models exist and pass validation.
     * Corrupted or incomplete models are deleted.
     */
    fun areModelsDownloadedAndValid(targetDir: File): Boolean {
        for (info in REQUIRED_MODELS) {
            val localFile = File(targetDir, info.fileName)
            if (!isModelValid(localFile, info)) {
                if (localFile.exists()) {
                    localFile.delete()
                }
                return false
            }
        }
        return true
    }

    /**
     * Validates file existence, exact byte size, and verifies ONNX protobuf parseability.
     */
    fun isModelValid(file: File, info: ModelFileInfo): Boolean {
        if (!file.exists()) return false
        val currentSize = file.length()
        // Size must match expected byte length (or within reasonable tolerance for json)
        if (info.expectedBytes > 0 && currentSize != info.expectedBytes) {
            return false
        }
        // Test ONNX protobuf graph parseability to prevent "Protobuf parsing failed"
        if (info.isOnnxModel) {
            try {
                val ortEnv = OrtEnvironment.getEnvironment()
                val testSession = ortEnv.createSession(file.absolutePath)
                testSession.close()
            } catch (e: Exception) {
                // Protobuf parsing failed or corrupted graph
                return false
            }
        }
        return true
    }

    /**
     * Downloads models with auto-resume, retry logic, and validation.
     */
    suspend fun downloadAllModels(
        targetDir: File,
        onProgress: (fileName: String, percent: Int, status: String) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }

        for ((idx, info) in REQUIRED_MODELS.withIndex()) {
            val localFile = File(targetDir, info.fileName)
            val partFile = File(targetDir, "${info.fileName}.part")

            // If already complete and valid, skip
            if (isModelValid(localFile, info)) {
                onProgress(info.fileName, 100, "Validated ($idx/${REQUIRED_MODELS.size}) ${info.fileName}")
                continue
            } else if (localFile.exists()) {
                localFile.delete()
            }

            var downloadSuccess = false
            var retryCount = 0
            val maxRetries = 5

            while (!downloadSuccess && retryCount < maxRetries) {
                try {
                    val fileUrl = "$HF_BASE_URL/${info.relPath}"
                    onProgress(
                        info.fileName,
                        0,
                        "Downloading [${idx + 1}/${REQUIRED_MODELS.size}] ${info.fileName} (Attempt ${retryCount + 1})..."
                    )

                    downloadWithResume(fileUrl, partFile, info.expectedBytes) { percent ->
                        onProgress(info.fileName, percent, "Downloading ${info.fileName}: $percent%")
                    }

                    // Rename .part to final
                    if (partFile.exists()) {
                        if (localFile.exists()) localFile.delete()
                        partFile.renameTo(localFile)
                    }

                    // Verify downloaded file integrity
                    if (isModelValid(localFile, info)) {
                        downloadSuccess = true
                        onProgress(info.fileName, 100, "Verified ${info.fileName} successfully!")
                    } else {
                        // Corrupted download, delete and retry
                        if (localFile.exists()) localFile.delete()
                        if (partFile.exists()) partFile.delete()
                        retryCount++
                    }
                } catch (e: Exception) {
                    retryCount++
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
    }

    private fun downloadWithResume(
        urlStr: String,
        destPartFile: File,
        expectedTotal: Long,
        onProgress: (Int) -> Unit
    ) {
        val existingBytes = if (destPartFile.exists()) destPartFile.length() else 0L

        // If part file is already oversized, reset it
        if (existingBytes >= expectedTotal && expectedTotal > 0) {
            destPartFile.delete()
        }

        val url = URL(urlStr)
        val conn = url.openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        conn.connectTimeout = 30000
        conn.readTimeout = 30000

        val startByte = if (destPartFile.exists()) destPartFile.length() else 0L
        if (startByte > 0) {
            conn.setRequestProperty("Range", "bytes=$startByte-")
        }
        conn.connect()

        val responseCode = conn.responseCode
        val isRangeResponse = (responseCode == HttpURLConnection.HTTP_PARTIAL)
        val appendMode = (startByte > 0 && isRangeResponse)

        val inputStream: InputStream = conn.inputStream
        val outputStream = FileOutputStream(destPartFile, appendMode)

        var downloadedBytes = if (appendMode) startByte else 0L
        val totalBytes = if (expectedTotal > 0) expectedTotal else (conn.contentLengthLong + downloadedBytes)

        inputStream.use { input ->
            outputStream.use { output ->
                val buffer = ByteArray(128 * 1024)
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    downloadedBytes += bytesRead
                    if (totalBytes > 0) {
                        val percent = ((downloadedBytes * 100) / totalBytes).toInt().coerceIn(0, 100)
                        onProgress(percent)
                    }
                }
            }
        }
    }
}
