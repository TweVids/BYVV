package com.valkeryne.multimodal

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelDownloader {

    private const val HF_BASE_URL = "https://huggingface.co/Nihilux/BYVV-onnx/resolve/main"

    // Key models to download
    private val MODEL_FILES = listOf(
        "valkeryne_onnx_int8/vision_encoder_int8.onnx",
        "valkeryne_onnx_int8/text_encoder_embed.onnx",
        "valkeryne_onnx_int8/main_language_model_int8.onnx",
        "tts/vi_VN-vivos-x_low.onnx",
        "tts/vi_VN-vivos-x_low.onnx.json",
        "stt/whisper_tiny/onnx/encoder_model.onnx",
        "stt/whisper_tiny/onnx/decoder_model_merged.onnx"
    )

    fun areModelsDownloaded(targetDir: File): Boolean {
        for (relPath in MODEL_FILES) {
            val fileName = File(relPath).name
            val localFile = File(targetDir, fileName)
            if (!localFile.exists() || localFile.length() == 0L) {
                return false
            }
        }
        return true
    }

    suspend fun downloadAllModels(
        targetDir: File,
        onProgress: (fileName: String, progressPercent: Int, status: String) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }

        for ((idx, relPath) in MODEL_FILES.withIndex()) {
            val fileName = File(relPath).name
            val localFile = File(targetDir, fileName)
            if (localFile.exists() && localFile.length() > 1024) {
                continue
            }

            val fileUrl = "$HF_BASE_URL/$relPath"
            try {
                onProgress(fileName, 0, "Downloading ($idx/${MODEL_FILES.size}) $fileName...")
                downloadFile(fileUrl, localFile) { percent ->
                    onProgress(fileName, percent, "Downloading $fileName ($percent%)")
                }
            } catch (e: Exception) {
                onProgress(fileName, 0, "Error downloading $fileName: ${e.message}")
                return@withContext false
            }
        }
        onProgress("Done", 100, "All models downloaded successfully!")
        return@withContext true
    }

    private fun downloadFile(urlStr: String, destFile: File, onProgress: (Int) -> Unit) {
        val url = URL(urlStr)
        val conn = url.openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        conn.connectTimeout = 30000
        conn.readTimeout = 30000
        conn.connect()

        val totalBytes = conn.contentLengthLong
        var downloadedBytes = 0L

        conn.inputStream.use { input ->
            FileOutputStream(destFile).use { output ->
                val buffer = ByteArray(64 * 1024)
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    downloadedBytes += bytesRead
                    if (totalBytes > 0) {
                        val percent = ((downloadedBytes * 100) / totalBytes).toInt()
                        onProgress(percent)
                    }
                }
            }
        }
    }
}
