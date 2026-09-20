package com.valkeryne.multimodal

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var viewFinder: PreviewView
    private lateinit var cameraStatusText: TextView
    private lateinit var btnSettings: Button
    private lateinit var userTurnText: TextView
    private lateinit var aiTurnText: TextView
    private lateinit var holdToSpeakBtn: Button

    private var tts: TextToSpeech? = null
    private lateinit var geminiClient: GeminiLiveClient

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordJob: Job? = null
    private val audioBuffer = ByteArrayOutputStream()

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        cameraStatusText = findViewById(R.id.cameraStatusText)
        btnSettings = findViewById(R.id.btnSettings)
        userTurnText = findViewById(R.id.userTurnText)
        aiTurnText = findViewById(R.id.aiTurnText)
        holdToSpeakBtn = findViewById(R.id.holdToSpeakBtn)

        tts = TextToSpeech(this, this)
        geminiClient = GeminiLiveClient(this)

        geminiClient.setCallback(object : GeminiLiveClient.Callback {
            override fun onConnected() {
                runOnUiThread {
                    cameraStatusText.text = "Đã kết nối Gemini Live"
                }
            }

            override fun onDisconnected(reason: String) {
                runOnUiThread {
                    cameraStatusText.text = "Ngắt kết nối: $reason"
                }
            }

            override fun onUserTurn(text: String) {
                runOnUiThread {
                    userTurnText.text = text
                }
            }

            override fun onAiTurn(text: String, isComplete: Boolean, hasNativeAudio: Boolean) {
                runOnUiThread {
                    aiTurnText.text = text
                    if (isComplete && text.isNotBlank()) {
                        cameraStatusText.text = "Hoàn tất - Giữ nút để hỏi tiếp"
                        if (!hasNativeAudio) {
                            speakOut(text)
                        }
                    }
                }
            }

            override fun onThinkingStatus(text: String) {
                runOnUiThread {
                    cameraStatusText.text = text
                }
            }

            override fun onError(error: String) {
                runOnUiThread {
                    aiTurnText.text = error
                    cameraStatusText.text = "Lỗi xử lý"
                    Toast.makeText(this@MainActivity, error, Toast.LENGTH_LONG).show()
                }
            }
        })

        btnSettings.setOnClickListener {
            val dialog = SettingsDialog(this) {
                updateStatusWithCurrentConfig()
            }
            dialog.show()
        }

        // Hold-to-speak button listener
        holdToSpeakBtn.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    onHoldStart()
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    onHoldEnd()
                    true
                }
                else -> false
            }
        }

        requestNeededPermissions()
        updateStatusWithCurrentConfig()
    }

    private fun updateStatusWithCurrentConfig() {
        val key = AppPreferences.getApiKey(this)
        val model = AppPreferences.getModel(this)
        if (key.isBlank()) {
            cameraStatusText.text = "Chưa có API Key! Nhấn ⚙ CÀI ĐẶT để thêm"
            aiTurnText.text = "Vui lòng nhấn nút [⚙ CÀI ĐẶT] ở góc trên bên phải để cấu hình Gemini API Key."
        } else {
            cameraStatusText.text = "Mô hình: $model"
            aiTurnText.text = "Sẵn sàng. Giữ nút màu vàng để nói & chụp ảnh."
            // Pre-connect WebSocket session for instant zero-latency conversation
            geminiClient.ensureConnected {}
        }
    }

    private fun onHoldStart() {
        val apiKey = AppPreferences.getApiKey(this)
        if (apiKey.isBlank()) {
            Toast.makeText(this, "Vui lòng nhập API Key trong Cài Đặt!", Toast.LENGTH_SHORT).show()
            val dialog = SettingsDialog(this) { updateStatusWithCurrentConfig() }
            dialog.show()
            return
        }

        geminiClient.interrupt()
        tts?.stop()

        holdToSpeakBtn.text = "🔴 ĐANG LẮNG NGHE... (THẢ ĐỂ DỪNG)"
        holdToSpeakBtn.backgroundTintList = ContextCompat.getColorStateList(this, android.R.color.holo_red_dark)
        userTurnText.text = "Đang truyền giọng nói & hình ảnh trực tiếp đến Gemini..."
        aiTurnText.text = "Gemini Live đang lắng nghe..."
        cameraStatusText.text = "Đang thu âm & truyền thời gian thực..."

        // Ensure session is connected, capture frame and stream audio immediately
        geminiClient.ensureConnected {
            // Capture and stream camera frame at the start of turn
            val bitmap = viewFinder.bitmap
            bitmap?.let { bmp ->
                val stream = ByteArrayOutputStream()
                bmp.compress(Bitmap.CompressFormat.JPEG, 75, stream)
                geminiClient.sendCameraFrame(stream.toByteArray())
            }
        }

        startRecordingAudio()
    }

    private fun onHoldEnd() {
        if (!isRecording) return

        holdToSpeakBtn.text = "🎤 GIỮ ĐỂ NÓI (HOLD TO SPEAK)"
        holdToSpeakBtn.backgroundTintList = ContextCompat.getColorStateList(this, android.R.color.holo_orange_light)
        cameraStatusText.text = "Đã gửi câu hỏi. Đang nhận phản hồi..."

        stopRecordingAudio()
        geminiClient.finishUserTurn()
    }

    @SuppressLint("MissingPermission")
    private fun startRecordingAudio() {
        try {
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                minBuf * 2
            )

            audioRecord?.startRecording()
            isRecording = true

            recordJob = lifecycleScope.launch(Dispatchers.IO) {
                // Stream 3200 bytes (~100ms chunks) directly in real-time
                val buffer = ByteArray(3200)
                while (isRecording && isActive) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        val chunk = if (read == buffer.size) buffer else buffer.copyOf(read)
                        geminiClient.sendRealtimeAudioChunk(chunk)
                    }
                }
            }
        } catch (e: Exception) {
            isRecording = false
            Toast.makeText(this, "Lỗi ghi âm: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopRecordingAudio() {
        isRecording = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            recordJob?.cancel()
        } catch (e: Exception) {}
    }

    private fun createWavFile(pcmData: ByteArray, sampleRate: Int): ByteArray {
        val totalAudioLen = pcmData.size.toLong()
        val totalDataLen = totalAudioLen + 36
        val longSampleRate = sampleRate.toLong()
        val channels = 1
        val byteRate = 16 * sampleRate * channels / 8

        val header = ByteArray(44)
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = (totalDataLen shr 8 and 0xff).toByte()
        header[6] = (totalDataLen shr 16 and 0xff).toByte()
        header[7] = (totalDataLen shr 24 and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // PCM
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (longSampleRate and 0xff).toByte()
        header[25] = (longSampleRate shr 8 and 0xff).toByte()
        header[26] = (longSampleRate shr 16 and 0xff).toByte()
        header[27] = (longSampleRate shr 24 and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = (byteRate shr 8 and 0xff).toByte()
        header[30] = (byteRate shr 16 and 0xff).toByte()
        header[31] = (byteRate shr 24 and 0xff).toByte()
        header[32] = (channels * 16 / 8).toByte()
        header[33] = 0
        header[34] = 16
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = (totalAudioLen shr 8 and 0xff).toByte()
        header[42] = (totalAudioLen shr 16 and 0xff).toByte()
        header[43] = (totalAudioLen shr 24 and 0xff).toByte()

        val out = ByteArrayOutputStream()
        out.write(header)
        out.write(pcmData)
        return out.toByteArray()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview)
            } catch (exc: Exception) {
                cameraStatusText.text = "Camera lỗi: ${exc.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun speakOut(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "BYVVResponse")
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale("vi", "VN")
        }
    }

    private fun requestNeededPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), 101)
        } else {
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            }
        }
    }

    override fun onDestroy() {
        geminiClient.release()
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }
}
