package com.tafftracker.taffcam

import org.json.JSONObject
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

class TaffControlServer(
    private val host: String,
    private val port: Int,
    private val handler: (JSONObject) -> JSONObject,
    private val status: (String) -> Unit
) {
    private val running = AtomicBoolean(false)
    private var worker: Thread? = null
    @Volatile
    private var activeSocket: Socket? = null

    fun start() {
        if (running.getAndSet(true)) return
        worker = Thread(::runClient, "TaffCamControl").apply { start() }
    }

    fun stop() {
        running.set(false)
        try {
            activeSocket?.close()
        } catch (_: Exception) {
        }
        activeSocket = null
        worker?.interrupt()
        worker = null
    }

    private fun runClient() {
        while (running.get()) {
            try {
                Socket().use { socket ->
                    activeSocket = socket
                    socket.tcpNoDelay = true
                    socket.soTimeout = READ_TIMEOUT_MS
                    socket.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MS)
                    status("Control connected to $host:$port.")
                    BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8)).use { reader ->
                        BufferedWriter(OutputStreamWriter(socket.getOutputStream(), Charsets.UTF_8)).use { writer ->
                            while (running.get() && socket.isConnected) {
                                val line = readBoundedLine(reader) ?: break
                                val response = handleLine(line)
                                writer.write(response.toString())
                                writer.write("\n")
                                writer.flush()
                            }
                        }
                    }
                    activeSocket = null
                }
            } catch (_: InterruptedException) {
                activeSocket = null
                Thread.currentThread().interrupt()
                return
            } catch (ex: Exception) {
                activeSocket = null
                if (running.get()) {
                    status("Control waiting for $host:$port (${ex.message}).")
                    try {
                        Thread.sleep(RETRY_SLEEP_MS)
                    } catch (_: InterruptedException) {
                        Thread.currentThread().interrupt()
                        return
                    }
                }
            }
        }
    }

    private fun readBoundedLine(reader: BufferedReader): String? {
        val builder = StringBuilder()
        while (running.get()) {
            val value = reader.read()
            if (value == -1) {
                return if (builder.isEmpty()) null else builder.toString()
            }
            if (value == '\n'.code) {
                return builder.toString().trimEnd('\r')
            }
            if (builder.length >= MAX_COMMAND_CHARS) {
                throw IllegalArgumentException("control command exceeds $MAX_COMMAND_CHARS chars")
            }
            builder.append(value.toChar())
        }
        return null
    }

    private fun handleLine(line: String): JSONObject {
        return try {
            val request = JSONObject(line)
            handler(request)
        } catch (ex: Exception) {
            JSONObject().put("ok", false).put("error", ex.message ?: "invalid command")
        }
    }

    private companion object {
        const val CONNECT_TIMEOUT_MS = 500
        const val READ_TIMEOUT_MS = 10000
        const val MAX_COMMAND_CHARS = 8192
        const val RETRY_SLEEP_MS = 500L
    }
}
