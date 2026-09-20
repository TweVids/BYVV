package com.valkeryne.multimodal

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
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
import java.io.ByteArrayOutputStream
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
                    userTurnText.text = "Bạn: $text"
                }
            }

            override fun onAiTurn(text: String, isComplete: Boolean, hasNativeAudio: Boolean) {
                runOnUiThread {
                    if (text.isNotBlank()) {
                        aiTurnText.text = text
                    }
                    if (isComplete) {
                        cameraStatusText.text = "Hoàn tất - Giữ nút để hỏi tiếp"
                        // Fallback to TTS only if native 24kHz audio was not streamed
                        if (!hasNativeAudio && text.isNotBlank()) {
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
        userTurnText.text = "Đang lắng nghe..."
        aiTurnText.text = "Gemini Live đang lắng nghe..."
        cameraStatusText.text = "Đang thu âm & truyền thời gian thực..."

        // Ensure session is connected
        geminiClient.ensureConnected {
            // Capture and scale camera frame in background thread to avoid freezing UI
            val bmp = viewFinder.bitmap
            if (bmp != null) {
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        val maxDim = 1024
                        val scaled = if (bmp.width > maxDim || bmp.height > maxDim) {
                            val ratio = maxDim.toFloat() / maxOf(bmp.width, bmp.height)
                            Bitmap.createScaledBitmap(
                                bmp,
                                (bmp.width * ratio).toInt(),
                                (bmp.height * ratio).toInt(),
                                true
                            )
                        } else {
                            bmp
                        }
                        val stream = ByteArrayOutputStream()
                        scaled.compress(Bitmap.CompressFormat.JPEG, 70, stream)
                        geminiClient.sendCameraFrame(stream.toByteArray())
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Error capturing camera frame: ${e.message}")
                    }
                }
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
            val bufferSize = maxOf(minBuf * 2, 6400)

            // Try VOICE_RECOGNITION first, fallback to MIC
            var record = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )

            if (record.state != AudioRecord.STATE_INITIALIZED) {
                record.release()
                record = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    bufferSize
                )
            }

            if (record.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("MainActivity", "AudioRecord initialization failed!")
                Toast.makeText(this, "Không thể khởi tạo micro!", Toast.LENGTH_SHORT).show()
                return
            }

            audioRecord = record
            audioRecord?.startRecording()
            isRecording = true

            recordJob = lifecycleScope.launch(Dispatchers.IO) {
                val buffer = ByteArray(3200) // 100ms at 16kHz 16-bit mono
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
            Log.e("MainActivity", "AudioRecord start error: ${e.message}", e)
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
