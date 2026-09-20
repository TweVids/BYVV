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
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class GeminiLiveClient(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS) // 0 for WebSocket (no read timeout)
        .writeTimeout(30, TimeUnit.SECONDS)
        // Note: Do not set aggressive pingInterval as Google Live API manages WebSocket frames server-side
        .build()

    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO + Job())

    private var liveWebSocket: WebSocket? = null
    private var isSessionSetupComplete = false
    private var audioTrack: AudioTrack? = null

    // Pending data to send once setup is complete
    private var pendingImageBytes: ByteArray? = null
    private var pendingAudioBytes: ByteArray? = null

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
            val sampleRate = 24000 // Gemini Live standard output is 24kHz PCM mono 16-bit
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
     * Sends multimodal turn over Google Multimodal Live API WebSocket (BidiGenerateContent)
     */
    fun sendMultimodalTurn(
        imageBytes: ByteArray?,
        rawPcmBytes: ByteArray?,
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
            connectAndSendLiveWebSocket(apiKey, model, thinkingBudget, imageBytes, rawPcmBytes, textPrompt)
        }
    }

    private fun connectAndSendLiveWebSocket(
        apiKey: String,
        model: String,
        thinkingBudget: Int,
        imageBytes: ByteArray?,
        rawPcmBytes: ByteArray?,
        textPrompt: String?
    ) {
        // Close previous socket if any
        try {
            liveWebSocket?.close(1000, "New session")
        } catch (e: Exception) {}

        isSessionSetupComplete = false
        pendingImageBytes = imageBytes
        pendingAudioBytes = rawPcmBytes

        val wsUrl = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=$apiKey"
        val request = Request.Builder().url(wsUrl).build()

        callback?.onThinkingStatus("Đang kết nối Gemini Live WebSocket...")

        liveWebSocket = client.newWebSocket(request, object : WebSocketListener() {
            var accumulatedText = ""

            override fun onOpen(webSocket: WebSocket, response: Response) {
                callback?.onThinkingStatus("Đã kết nối! Đang gửi Setup cấu hình...")
                
                // 1. Send BidiGenerateContentSetup message
                val setupRoot = JsonObject()
                val setupObj = JsonObject()
                setupObj.addProperty("model", "models/$model")

                val genConfig = JsonObject()
                val modalities = JsonArray()
                modalities.add("AUDIO")
                modalities.add("TEXT")
                genConfig.add("responseModalities", modalities)

                if (thinkingBudget > 0) {
                    val thinkingConfig = JsonObject()
                    thinkingConfig.addProperty("thinking_budget", thinkingBudget)
                    genConfig.add("thinking_config", thinkingConfig)
                }
                setupObj.add("generationConfig", genConfig)

                val sysInstruction = JsonObject()
                val sysParts = JsonArray()
                val sysPart = JsonObject()
                sysPart.addProperty("text", textPrompt ?: "Bạn là BYVV - trợ lý khiếm thị trực quan. Hãy nhìn hình ảnh từ camera và nghe câu hỏi, trả lời trực tiếp, rõ ràng, tự nhiên bằng tiếng Việt.")
                sysParts.add(sysPart)
                sysInstruction.add("parts", sysParts)
                setupObj.add("systemInstruction", sysInstruction)

                setupRoot.add("setup", setupObj)

                val setupJson = gson.toJson(setupRoot)
                webSocket.send(setupJson)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val root = gson.fromJson(text, JsonObject::class.java)

                    // Check setupComplete
                    if (root.has("setupComplete")) {
                        isSessionSetupComplete = true
                        callback?.onThinkingStatus("Đang truyền hình ảnh và âm thanh...")
                        sendRealtimeMediaChunks(webSocket)
                        return
                    }

                    // Handle serverContent
                    if (root.has("serverContent")) {
                        val serverContent = root.getAsJsonObject("serverContent")
                        if (serverContent.has("modelTurn")) {
                            val modelTurn = serverContent.getAsJsonObject("modelTurn")
                            val parts = modelTurn.getAsJsonArray("parts")
                            if (parts != null) {
                                for (p in parts) {
                                    val pObj = p.asJsonObject
                                    // 1. Check thought
                                    if (pObj.has("thought") && pObj.get("thought").asBoolean) {
                                        val thought = pObj.get("text")?.asString ?: ""
                                        callback?.onThinkingStatus("Suy nghĩ: $thought")
                                    } else if (pObj.has("text")) {
                                        // 2. Text response
                                        val chunk = pObj.get("text").asString
                                        accumulatedText += chunk
                                        callback?.onAiTurn(accumulatedText, false)
                                    } else if (pObj.has("inlineData")) {
                                        // 3. Native Audio PCM chunk (24kHz)
                                        val dataObj = pObj.getAsJsonObject("inlineData")
                                        val b64 = dataObj.get("data")?.asString
                                        if (b64 != null) {
                                            val pcmAudio = Base64.decode(b64, Base64.DEFAULT)
                                            playPcmChunk(pcmAudio)
                                        }
                                    }
                                }
                            }
                        }

                        if (serverContent.has("turnComplete") && serverContent.get("turnComplete").asBoolean) {
                            callback?.onAiTurn(accumulatedText, true)
                        }
                    }
                } catch (e: Exception) {
                    Log.e("GeminiLiveClient", "Error parsing WebSocket message: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                val errMsg = response?.body?.string() ?: t.message ?: "Mất kết nối"
                callback?.onError("Lỗi kết nối Live: $errMsg")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                callback?.onDisconnected(reason)
            }
        })
    }

    private fun sendRealtimeMediaChunks(ws: WebSocket) {
        val img = pendingImageBytes
        val pcm = pendingAudioBytes

        // 1. Send Camera Frame as Realtime Input media_chunk
        if (img != null && img.isNotEmpty()) {
            val imgRoot = JsonObject()
            val realtimeInput = JsonObject()
            val mediaChunks = JsonArray()
            val chunk = JsonObject()
            chunk.addProperty("mime_type", "image/jpeg")
            chunk.addProperty("data", Base64.encodeToString(img, Base64.NO_WRAP))
            mediaChunks.add(chunk)
            realtimeInput.add("media_chunks", mediaChunks)
            imgRoot.add("realtime_input", realtimeInput)

            ws.send(gson.toJson(imgRoot))
        }

        // 2. Send Audio PCM chunks (16kHz PCM little endian)
        if (pcm != null && pcm.isNotEmpty()) {
            // Send in chunks of 4KB (~128ms each)
            val chunkSize = 4096
            var offset = 0
            while (offset < pcm.size) {
                val len = Math.min(chunkSize, pcm.size - offset)
                val slice = ByteArray(len)
                System.arraycopy(pcm, offset, slice, 0, len)

                val audioRoot = JsonObject()
                val realtimeInput = JsonObject()
                val mediaChunks = JsonArray()
                val chunk = JsonObject()
                chunk.addProperty("mime_type", "audio/pcm;rate=16000")
                chunk.addProperty("data", Base64.encodeToString(slice, Base64.NO_WRAP))
                mediaChunks.add(chunk)
                realtimeInput.add("media_chunks", mediaChunks)
                audioRoot.add("realtime_input", realtimeInput)

                ws.send(gson.toJson(audioRoot))
                offset += len
            }
        }

        // 3. Mark client turn as completed
        val turnCompleteRoot = JsonObject()
        val clientContent = JsonObject()
        val turns = JsonArray()
        val turn = JsonObject()
        turn.addProperty("role", "user")
        turn.add("parts", JsonArray())
        turns.add(turn)
        clientContent.add("turns", turns)
        clientContent.addProperty("turn_complete", true)
        turnCompleteRoot.add("client_content", clientContent)

        ws.send(gson.toJson(turnCompleteRoot))
        callback?.onThinkingStatus("Đang chờ Gemini phản hồi...")
    }

    fun release() {
        try {
            liveWebSocket?.close(1000, "App closed")
            liveWebSocket = null
            audioTrack?.stop()
            audioTrack?.release()
            audioTrack = null
        } catch (e: Exception) {}
    }
}
