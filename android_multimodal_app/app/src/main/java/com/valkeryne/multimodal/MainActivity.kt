package com.valkeryne.multimodal

import android.Manifest
import android.content.pm.PackageManager
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
import kotlinx.coroutines.launch
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        userTurnText = findViewById(R.id.userTurnText)
        aiTurnText = findViewById(R.id.aiTurnText)
        actionBtn = findViewById(R.id.actionBtn)
        cameraStatusText = findViewById(R.id.cameraStatusText)

        tts = TextToSpeech(this, this)

        val modelDir = getExternalFilesDir(null) ?: filesDir
        engine = ValkeryneMultimodalEngine(this, modelDir)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO),
                101
            )
        }

        actionBtn.setOnClickListener {
            runAccessibleInference(modelDir)
        }

        // Tap camera area also triggers capture for accessibility
        viewFinder.setOnClickListener {
            runAccessibleInference(modelDir)
        }
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
                cameraStatusText.text = "Camera Active - Tap to describe"
            } catch (exc: Exception) {
                cameraStatusText.text = "Camera error: ${exc.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun runAccessibleInference(modelDir: File) {
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

                // Read out loud for blind user
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

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    private fun allPermissionsGranted() = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101 && allPermissionsGranted()) {
            startCamera()
        }
    }
}
