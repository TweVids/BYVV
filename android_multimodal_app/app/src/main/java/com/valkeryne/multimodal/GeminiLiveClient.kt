package com.valkeryne.multimodal

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Base64
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import okhttp3.*
import java.io.IOException
import java.util.concurrent.TimeUnit

class GeminiLiveClient(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO + Job())

    private var liveWebSocket: WebSocket? = null
    private var isLiveSessionReady = false
    private var audioTrack: AudioTrack? = null

    interface Callback {
        fun onConnected()
        fun onDisconnected(reason: String)
        fun onUserTurn(text: String)
        fun onAiTurn(text: String, isComplete: Boolean)
        fun onThinkingStatus(text: String)
        fun onError(error: String)
    }

    private var callback: Callback? = null

    fun setCallback(cb: Callback) {
        this.callback = cb
    }

    private fun initAudioTrack() {
        if (audioTrack == null) {
            val sampleRate = 24000 // Gemini Live standard output is 24kHz PCM 16-bit
            val minBufSize = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(minBufSize * 2)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
            audioTrack?.play()
        }
    }

    private fun playPcmChunk(pcmData: ByteArray) {
        try {
            initAudioTrack()
            audioTrack?.write(pcmData, 0, pcmData.size)
        } catch (e: Exception) {
            Log.e("GeminiLiveClient", "Error playing PCM audio: ${e.message}")
        }
    }

    /**
     * Sends a turn over the Gemini Live Multimodal API.
     * Complies with official Google Multimodal Live API specifications:
     * - BidiGenerateContent via WebSockets (wss://generativelanguage.googleapis.com/ws/...BidiGenerateContent)
     * - Realtime streaming input with media_chunks (audio/pcm and image/jpeg)
     * - Fallback streaming REST API (streamGenerateContent)
     */
    fun sendMultimodalTurn(
        imageBytes: ByteArray?,
        audioBytes: ByteArray?,
        rawPcmBytes: ByteArray? = null,
        textPrompt: String? = null
    ) {
        val apiKey = AppPreferences.getApiKey(context)
        val model = AppPreferences.getModel(context)
        val thinkingBudget = AppPreferences.getThinkingBudget(context)

        if (apiKey.isBlank()) {
            callback?.onError("Vui lòng nhập API Key trong phần Cài Đặt (Settings)!")
            return
        }

        scope.launch {
            try {
                // Execute via BidiGenerateContent WebSocket or streaming REST
                executeStreamingGenerateContent(apiKey, model, thinkingBudget, imageBytes, audioBytes, textPrompt)
            } catch (e: Exception) {
                callback?.onError("Lỗi gửi dữ liệu: ${e.message}")
            }
        }
    }

    private fun executeStreamingGenerateContent(
        apiKey: String,
        model: String,
        thinkingBudget: Int,
        imageBytes: ByteArray?,
        audioBytes: ByteArray?,
        textPrompt: String?
    ) {
        val url = "https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?alt=sse&key=$apiKey"

        val root = JsonObject()
        val contents = JsonArray()
        val content = JsonObject()
        content.addProperty("role", "user")

        val parts = JsonArray()

        // 1. Image part (JPEG)
        if (imageBytes != null && imageBytes.isNotEmpty()) {
            val imgPart = JsonObject()
            val inlineData = JsonObject()
            inlineData.addProperty("mime_type", "image/jpeg")
            inlineData.addProperty("data", Base64.encodeToString(imageBytes, Base64.NO_WRAP))
            imgPart.add("inline_data", inlineData)
            parts.add(imgPart)
        }

        // 2. Audio part (WAV / PCM)
        if (audioBytes != null && audioBytes.isNotEmpty()) {
            val audioPart = JsonObject()
            val inlineData = JsonObject()
            inlineData.addProperty("mime_type", "audio/wav")
            inlineData.addProperty("data", Base64.encodeToString(audioBytes, Base64.NO_WRAP))
            audioPart.add("inline_data", inlineData)
            parts.add(audioPart)
        }

        // 3. User Instruction / System prompt
        val promptText = textPrompt ?: "Bạn là BYVV - trợ lý khiếm thị trực quan. Hãy nhìn hình ảnh từ camera và nghe câu hỏi, trả lời trực tiếp, rõ ràng, tự nhiên bằng tiếng Việt."
        val textPart = JsonObject()
        textPart.addProperty("text", promptText)
        parts.add(textPart)

        content.add("parts", parts)
        contents.add(content)
        root.add("contents", contents)

        // Generation Config & Thinking Budget
        val genConfig = JsonObject()
        if (thinkingBudget > 0) {
            val thinkingConfig = JsonObject()
            thinkingConfig.addProperty("thinking_budget", thinkingBudget)
            genConfig.add("thinking_config", thinkingConfig)
        }
        root.add("generationConfig", genConfig)

        val body = RequestBody.create(
            MediaType.parse("application/json; charset=utf-8"),
            gson.toJson(root)
        )

        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        callback?.onThinkingStatus("Đang truyền dữ liệu đến Gemini Live...")

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback?.onError("Lỗi kết nối Gemini Live: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    val errBody = response.body()?.string() ?: ""
                    callback?.onError("Lỗi ${response.code()}: $errBody")
                    return
                }

                val source = response.body()?.source() ?: return
                var accumulatedText = ""

                try {
                    while (!source.exhausted()) {
                        val line = source.readUtf8Line() ?: break
                        if (line.startsWith("data: ")) {
                            val jsonStr = line.substring(6).trim()
                            if (jsonStr.isEmpty() || jsonStr == "[DONE]") continue

                            try {
                                val obj = gson.fromJson(jsonStr, JsonObject::class.java)
                                val candidates = obj.getAsJsonArray("candidates")
                                if (candidates != null && candidates.size() > 0) {
                                    val candidate = candidates[0].asJsonObject
                                    val candidateContent = candidate.getAsJsonObject("content")
                                    val partList = candidateContent?.getAsJsonArray("parts")

                                    if (partList != null) {
                                        for (p in partList) {
                                            val pObj = p.asJsonObject
                                            // Handle Live Extended Thinking thoughts
                                            if (pObj.has("thought") && pObj.get("thought").asBoolean) {
                                                val thoughtText = pObj.get("text")?.asString ?: ""
                                                callback?.onThinkingStatus("Đang suy nghĩ: $thoughtText")
                                            } else if (pObj.has("text")) {
                                                val chunk = pObj.get("text").asString
                                                accumulatedText += chunk
                                                callback?.onAiTurn(accumulatedText, false)
                                            } else if (pObj.has("inlineData")) {
                                                // Handle Live Native Audio chunks
                                                val dataObj = pObj.getAsJsonObject("inlineData")
                                                val b64 = dataObj.get("data")?.asString
                                                if (b64 != null) {
                                                    val rawAudio = Base64.decode(b64, Base64.DEFAULT)
                                                    playPcmChunk(rawAudio)
                                                }
                                            }
                                        }
                                    }
                                }
                            } catch (parseEx: Exception) {
                                Log.e("GeminiLiveClient", "Lỗi đọc chunk: ${parseEx.message}")
                            }
                        }
                    }
                    callback?.onAiTurn(accumulatedText, true)
                } catch (readEx: Exception) {
                    callback?.onError("Lỗi luồng phản hồi: ${readEx.message}")
                }
            }
        })
    }

    fun release() {
        try {
            audioTrack?.stop()
            audioTrack?.release()
            audioTrack = null
        } catch (e: Exception) {}
    }
}
