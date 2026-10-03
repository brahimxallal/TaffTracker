package com.tafftracker.taffcam

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

class TaffCommandReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        val activity = MainActivity.activeActivity?.get()
        Log.i(MainActivity.LOG_TAG, "Receiver action=${intent?.action} active=${activity != null}")
        if (activity == null) return
        activity.runOnUiThread { activity.handleCommandIntent(intent) }
    }
}
