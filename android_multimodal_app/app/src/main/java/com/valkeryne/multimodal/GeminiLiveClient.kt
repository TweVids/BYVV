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
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO + Job())

    private var liveWebSocket: WebSocket? = null
    private var isSessionSetupComplete = false
    private var isConnected = false
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
            val sampleRate = 24000 // Gemini Live outputs 24kHz PCM mono 16-bit
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
                .setBufferSizeInBytes(minBufSize * 4)
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
     * Start/ensure active persistent live session before speaking
     */
    fun ensureConnected(onReady: () -> Unit) {
        if (isConnected && isSessionSetupComplete && liveWebSocket != null) {
            onReady()
            return
        }

        val apiKey = AppPreferences.getApiKey(context)
        val model = AppPreferences.getModel(context)
        val thinkingBudget = AppPreferences.getThinkingBudget(context)

        if (apiKey.isBlank()) {
            callback?.onError("Vui lòng nhập API Key trong phần Cài Đặt!")
            return
        }

        connectLiveWebSocket(apiKey, model, thinkingBudget, onReady)
    }

    private fun connectLiveWebSocket(
        apiKey: String,
        model: String,
        thinkingBudget: Int,
        onReady: (() -> Unit)?
    ) {
        try {
            liveWebSocket?.close(1000, "Reconnecting")
        } catch (e: Exception) {}

        isConnected = false
        isSessionSetupComplete = false

        val wsUrl = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=$apiKey"
        val request = Request.Builder().url(wsUrl).build()

        callback?.onThinkingStatus("Đang kết nối WebSocket Gemini Live...")

        liveWebSocket = client.newWebSocket(request, object : WebSocketListener() {
            var currentTurnText = ""

            override fun onOpen(webSocket: WebSocket, response: Response) {
                isConnected = true
                callback?.onConnected()
                callback?.onThinkingStatus("Đã kết nối! Đang gửi Setup cấu hình...")

                // Send BidiGenerateContentSetup message
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
                sysPart.addProperty("text", "Bạn là BYVV - trợ lý trực quan cho người khiếm thị. Hãy nhìn camera và lắng nghe người dùng, trả lời súc tích, tự nhiên bằng tiếng Việt.")
                sysParts.add(sysPart)
                sysInstruction.add("parts", sysParts)
                setupObj.add("systemInstruction", sysInstruction)

                setupRoot.add("setup", setupObj)
                webSocket.send(gson.toJson(setupRoot))
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val root = gson.fromJson(text, JsonObject::class.java)

                    // 1. Setup Complete event
                    if (root.has("setupComplete")) {
                        isSessionSetupComplete = true
                        callback?.onThinkingStatus("Sẵn sàng! Giữ nút để nói.")
                        onReady?.invoke()
                        return
                    }

                    // 2. Server Content stream
                    if (root.has("serverContent")) {
                        val serverContent = root.getAsJsonObject("serverContent")
                        if (serverContent.has("modelTurn")) {
                            val modelTurn = serverContent.getAsJsonObject("modelTurn")
                            val parts = modelTurn.getAsJsonArray("parts")
                            if (parts != null) {
                                for (p in parts) {
                                    val pObj = p.asJsonObject
                                    // Extended thinking
                                    if (pObj.has("thought") && pObj.get("thought").asBoolean) {
                                        val thought = pObj.get("text")?.asString ?: ""
                                        callback?.onThinkingStatus("Đang nghĩ: $thought")
                                    } else if (pObj.has("text")) {
                                        // Text stream
                                        val chunk = pObj.get("text").asString
                                        currentTurnText += chunk
                                        callback?.onAiTurn(currentTurnText, false)
                                    } else if (pObj.has("inlineData")) {
                                        // Live PCM audio stream (24kHz 16-bit mono)
                                        val dataObj = pObj.getAsJsonObject("inlineData")
                                        val b64 = dataObj.get("data")?.asString
                                        if (b64 != null) {
                                            val pcmBytes = Base64.decode(b64, Base64.DEFAULT)
                                            playPcmChunk(pcmBytes)
                                        }
                                    }
                                }
                            }
                        }

                        // Check turn complete
                        if (serverContent.has("turnComplete") && serverContent.get("turnComplete").asBoolean) {
                            val finishedText = currentTurnText
                            currentTurnText = ""
                            callback?.onAiTurn(finishedText, true)
                        }
                    }
                } catch (e: Exception) {
                    Log.e("GeminiLiveClient", "Error parsing WebSocket message: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                isConnected = false
                isSessionSetupComplete = false
                val errMsg = response?.body?.string() ?: t.message ?: "Mất kết nối"
                callback?.onError("Lỗi Live: $errMsg")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                isConnected = false
                isSessionSetupComplete = false
                callback?.onDisconnected(reason)
            }
        })
    }

    /**
     * Send camera image frame in real-time over WebSocket
     */
    fun sendCameraFrame(imageBytes: ByteArray) {
        val ws = liveWebSocket ?: return
        if (!isConnected || !isSessionSetupComplete) return

        scope.launch {
            val root = JsonObject()
            val realtimeInput = JsonObject()
            val mediaChunks = JsonArray()
            val chunk = JsonObject()
            chunk.addProperty("mimeType", "image/jpeg")
            chunk.addProperty("data", Base64.encodeToString(imageBytes, Base64.NO_WRAP))
            mediaChunks.add(chunk)
            realtimeInput.add("mediaChunks", mediaChunks)
            root.add("realtimeInput", realtimeInput)

            ws.send(gson.toJson(root))
        }
    }

    /**
     * Stream real-time microphone PCM audio chunk (16kHz 16-bit little endian)
     */
    fun sendRealtimeAudioChunk(pcmChunk: ByteArray) {
        val ws = liveWebSocket ?: return
        if (!isConnected || !isSessionSetupComplete) return

        val root = JsonObject()
        val realtimeInput = JsonObject()
        val mediaChunks = JsonArray()
        val chunk = JsonObject()
        chunk.addProperty("mimeType", "audio/pcm;rate=16000")
        chunk.addProperty("data", Base64.encodeToString(pcmChunk, Base64.NO_WRAP))
        mediaChunks.add(chunk)
        realtimeInput.add("mediaChunks", mediaChunks)
        root.add("realtimeInput", realtimeInput)

        ws.send(gson.toJson(root))
    }

    /**
     * Notify Gemini that user has finished speaking
     */
    fun finishUserTurn() {
        val ws = liveWebSocket ?: return
        if (!isConnected || !isSessionSetupComplete) return

        scope.launch {
            val root = JsonObject()
            val clientContent = JsonObject()
            val turns = JsonArray()
            val turn = JsonObject()
            turn.addProperty("role", "user")
            turn.add("parts", JsonArray())
            turns.add(turn)
            clientContent.add("turns", turns)
            clientContent.addProperty("turnComplete", true)
            root.add("clientContent", clientContent)

            ws.send(gson.toJson(root))
            callback?.onThinkingStatus("Đang chờ Gemini phản hồi...")
        }
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
