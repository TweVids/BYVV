package com.valkeryne.multimodal

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File

class DownloadService : Service() {

    private val serviceJob = Job()
    private val serviceScope = CoroutineScope(Dispatchers.Main + serviceJob)
    private var wakeLock: PowerManager.WakeLock? = null

    companion object {
        const val CHANNEL_ID = "BYVV_DOWNLOAD_CHANNEL"
        const val NOTIFICATION_ID = 1001
        const val ACTION_START_DOWNLOAD = "ACTION_START_DOWNLOAD"
        const val BROADCAST_DOWNLOAD_PROGRESS = "com.valkeryne.multimodal.DOWNLOAD_PROGRESS"
        const val EXTRA_STATUS = "EXTRA_STATUS"
        const val EXTRA_PERCENT = "EXTRA_PERCENT"
        const val EXTRA_SUCCESS = "EXTRA_SUCCESS"

        fun start(context: Context) {
            val intent = Intent(context, DownloadService::class.java).apply {
                action = ACTION_START_DOWNLOAD
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()

        // Acquire WakeLock so phone sleep/lockscreen doesn't interrupt download
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "BYVV:DownloadWakeLock"
        ).apply {
            acquire(3 * 60 * 60 * 1000L) // 3 hours maximum
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = createNotification("Starting download...", 0)
        startForeground(NOTIFICATION_ID, notification)

        val modelDir = getExternalFilesDir(null) ?: filesDir

        serviceScope.launch {
            val success = ModelDownloader.downloadAllModels(modelDir) { fileName, percent, status ->
                updateNotification(status, percent)
                sendBroadcast(Intent(BROADCAST_DOWNLOAD_PROGRESS).apply {
                    putExtra(EXTRA_STATUS, status)
                    putExtra(EXTRA_PERCENT, percent)
                })
            }

            sendBroadcast(Intent(BROADCAST_DOWNLOAD_PROGRESS).apply {
                putExtra(EXTRA_SUCCESS, success)
                putExtra(EXTRA_STATUS, if (success) "Completed" else "Failed")
            })

            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }

        return START_NOT_STICKY
    }

    private fun createNotification(content: String, progress: Int): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("BYVV AI - Downloading Models")
            .setContentText(content)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, progress, progress == 0)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(content: String, progress: Int) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, createNotification(content, progress))
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "BYVV Model Downloader",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows progress when downloading ONNX models in background"
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        wakeLock?.let {
            if (it.isHeld) it.release()
        }
        serviceJob.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
