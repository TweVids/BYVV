package com.valkeryne.multimodal

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var engine: ValkeryneMultimodalEngine
    private lateinit var viewFinder: PreviewView
    private lateinit var userTurnText: TextView
    private lateinit var aiTurnText: TextView
    private lateinit var actionBtn: Button
    private lateinit var cameraStatusText: TextView
    private var tts: TextToSpeech? = null
    private lateinit var modelDir: File

    private val downloadReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            intent?.let {
                val status = it.getStringExtra(DownloadService.EXTRA_STATUS) ?: ""
                val hasSuccess = it.hasExtra(DownloadService.EXTRA_SUCCESS)

                if (hasSuccess) {
                    val success = it.getBooleanExtra(DownloadService.EXTRA_SUCCESS, false)
                    if (success) {
                        cameraStatusText.text = "Models ready - Tap to describe"
                        aiTurnText.text = "Tải & kiểm tra mô hình thành công! Nhấn nút để bắt đầu."
                        actionBtn.isEnabled = true
                        actionBtn.text = "CHỤP & HỎI (TAP TO ASK)"
                    } else {
                        cameraStatusText.text = "Download paused or incomplete"
                        aiTurnText.text = "Nhấn nút để tiếp tục tải mô hình."
                        actionBtn.isEnabled = true
                        actionBtn.text = "TIẾP TỤC TẢI (RESUME)"
                    }
                } else if (status.isNotEmpty()) {
                    cameraStatusText.text = status
                    aiTurnText.text = status
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        userTurnText = findViewById(R.id.userTurnText)
        aiTurnText = findViewById(R.id.aiTurnText)
        actionBtn = findViewById(R.id.actionBtn)
        cameraStatusText = findViewById(R.id.cameraStatusText)

        tts = TextToSpeech(this, this)

        modelDir = getExternalFilesDir(null) ?: filesDir
        engine = ValkeryneMultimodalEngine(this, modelDir)

        requestNeededPermissions()

        val filter = IntentFilter(DownloadService.BROADCAST_DOWNLOAD_PROGRESS)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(downloadReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(downloadReceiver, filter)
        }

        checkModelStatus()

        actionBtn.setOnClickListener {
            lifecycleScope.launch {
                val isReady = withContext(Dispatchers.IO) {
                    ModelDownloader.areModelsDownloadedAndValid(modelDir)
                }
                if (!isReady) {
                    startBackgroundDownload()
                } else {
                    runAccessibleInference()
                }
            }
        }

        viewFinder.setOnClickListener {
            lifecycleScope.launch {
                val isReady = withContext(Dispatchers.IO) {
                    ModelDownloader.areModelsDownloadedAndValid(modelDir)
                }
                if (isReady) {
                    runAccessibleInference()
                }
            }
        }
    }

    override fun onDestroy() {
        try {
            unregisterReceiver(downloadReceiver)
        } catch (e: Exception) {}
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    private fun checkModelStatus() {
        if (ModelDownloader.isDownloading()) {
            cameraStatusText.text = "Downloading models in background..."
            actionBtn.isEnabled = false
            return
        }

        lifecycleScope.launch(Dispatchers.IO) {
            val valid = ModelDownloader.areModelsDownloadedAndValid(modelDir)
            withContext(Dispatchers.Main) {
                if (valid) {
                    cameraStatusText.text = "Models ready - Tap to describe"
                    aiTurnText.text = "Đang chờ bạn gửi câu hỏi..."
                    actionBtn.isEnabled = true
                    actionBtn.text = "CHỤP & HỎI (TAP TO ASK)"
                } else {
                    cameraStatusText.text = "Models missing. Downloading in queue..."
                    aiTurnText.text = "Đang tải tuần tự các mô hình ONNX trong nền..."
                    actionBtn.text = "ĐANG TẢI (DOWNLOADING...)"
                    startBackgroundDownload()
                }
            }
        }
    }

    private fun startBackgroundDownload() {
        actionBtn.isEnabled = false
        DownloadService.start(this)
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
                cameraStatusText.text = "Camera error: ${exc.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun runAccessibleInference() {
        actionBtn.isEnabled = false
        userTurnText.text = "Đang lắng nghe & chụp hình ảnh..."
        aiTurnText.text = "Đang phân tích khung cảnh..."

        lifecycleScope.launch {
            try {
                val dummyAudio = File(modelDir, "user_prompt.wav")
                val dummyPatches = FloatArray(256 * 1536)
                val dummyGrid = longArrayOf(1, 16, 16)
                val dummyTokens = LongArray(90) { 100L }

                val (response, _) = engine.runMultimodalInference(
                    dummyAudio,
                    dummyPatches,
                    dummyGrid,
                    dummyTokens
                ) { status ->
                    runOnUiThread {
                        cameraStatusText.text = status
                    }
                }

                userTurnText.text = "Hãy miêu tả chi tiết hình ảnh này bằng tiếng Việt."
                aiTurnText.text = response
                cameraStatusText.text = "Đã hoàn thành"

                speakOut(response)

            } catch (e: Exception) {
                aiTurnText.text = "Lỗi: ${e.message}"
                cameraStatusText.text = "Lỗi thực thi"
            } finally {
                actionBtn.isEnabled = true
            }
        }
    }

    private fun speakOut(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "ValkeryneResponse")
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
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

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
}
