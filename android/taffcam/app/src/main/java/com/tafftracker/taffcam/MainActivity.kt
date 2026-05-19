package com.tafftracker.taffcam

import android.Manifest
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.lang.ref.WeakReference

class MainActivity : Activity() {
    private lateinit var statusView: TextView
    private lateinit var controller: TaffCamController
    private var resumed = false
    private var pendingCommand: String? = null
    private val commandReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            handleCommandIntent(intent)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activeActivity = WeakReference(this)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        controller = TaffCamController(this) { text -> runOnUiThread { appendStatus(text) } }
        setContentView(buildUi())
        registerReceiver(
            commandReceiver,
            IntentFilter().apply {
                addAction(ACTION_START)
                addAction(ACTION_STOP)
            }
        )

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            appendStatus("Camera permission granted.")
            handleCommandIntent(intent)
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            appendStatus("Camera permission granted.")
            handleCommandIntent(intent)
        } else {
            appendStatus("Camera permission denied.")
        }
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        runPendingCommand()
    }

    override fun onPause() {
        resumed = false
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            Log.i(LOG_TAG, "Window focused.")
            resumed = true
            runPendingCommand()
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleCommandIntent(intent)
    }

    override fun onStop() {
        controller.stop()
        super.onStop()
    }

    override fun onDestroy() {
        if (activeActivity?.get() === this) activeActivity = null
        try {
            unregisterReceiver(commandReceiver)
        } catch (_: IllegalArgumentException) {
        }
        controller.close()
        super.onDestroy()
    }

    private fun buildUi(): LinearLayout {
        statusView = TextView(this).apply {
            textSize = 13f
            setTextIsSelectable(true)
            text = "TaffCam idle.\n"
        }

        val startButton = Button(this).apply {
            text = "Start 640x480"
            setOnClickListener {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    controller.start()
                } else {
                    requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
                }
            }
        }

        val stopButton = Button(this).apply {
            text = "Stop"
            setOnClickListener { controller.stop() }
        }

        val capsButton = Button(this).apply {
            text = "Capabilities"
            setOnClickListener { showCapabilities() }
        }

        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            addView(startButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(stopButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(capsButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }

        val scroll = ScrollView(this).apply {
            addView(statusView)
        }

        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            addView(buttons, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        }
    }

    private fun showCapabilities() {
        val text = controller.describeCapabilities()
        appendStatus(text)
    }

    fun handleCommandIntent(intent: Intent?) {
        if (intent == null || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        Log.i(LOG_TAG, "Command intent action=${intent.action}")
        when {
            intent.action == ACTION_START || intent.getBooleanExtra(EXTRA_START, false) -> {
                pendingCommand = COMMAND_START
                runPendingCommand()
            }
            intent.action == ACTION_STOP || intent.getBooleanExtra(EXTRA_STOP, false) -> {
                pendingCommand = COMMAND_STOP
                runPendingCommand()
            }
        }
    }

    private fun runPendingCommand() {
        val command = pendingCommand ?: return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.i(LOG_TAG, "Deferring command=$command until camera permission is granted")
            return
        }
        pendingCommand = null
        val delayMs = if (resumed) 0L else 750L
        if (!resumed) Log.i(LOG_TAG, "Running command=$command after foreground delay")
        statusView.postDelayed({
            when (command) {
                COMMAND_START -> {
                    appendStatus("ADB start command.")
                    controller.applyLaunchIntent(intent)
                    controller.start()
                }
                COMMAND_STOP -> {
                    appendStatus("ADB stop command.")
                    controller.stop()
                }
            }
        }, delayMs)
    }

    private fun appendStatus(text: String) {
        statusView.append(text.trimEnd() + "\n")
        Log.i(LOG_TAG, text.trimEnd())
    }

    companion object {
        const val CAMERA_REQUEST = 1001
        const val ACTION_START = "com.tafftracker.taffcam.START"
        const val ACTION_STOP = "com.tafftracker.taffcam.STOP"
        const val EXTRA_START = "taff_start"
        const val EXTRA_STOP = "taff_stop"
        const val COMMAND_START = "start"
        const val COMMAND_STOP = "stop"
        const val LOG_TAG = "TaffCam"
        var activeActivity: WeakReference<MainActivity>? = null
    }
}
