package com.tafftracker.taffcam

import java.io.BufferedOutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

data class YuvFramePacket(
    val sequence: Long,
    val timestampNs: Long,
    val width: Int,
    val height: Int,
    val exposureNs: Long,
    val iso: Int,
    val focusDiopters: Float,
    val payload: ByteArray
)

class YuvTcpStreamer(
    private val host: String,
    private val port: Int,
    private val status: (String) -> Unit
) {
    private val running = AtomicBoolean(false)
    private val latest = AtomicReference<YuvFramePacket?>(null)
    private var worker: Thread? = null

    fun start() {
        if (running.getAndSet(true)) return
        worker = Thread(::runLoop, "TaffCamYuvTcp").apply { start() }
    }

    fun stop() {
        running.set(false)
        latest.set(null)
        worker?.interrupt()
        worker = null
    }

    fun offer(packet: YuvFramePacket) {
        latest.set(packet)
    }

    private fun runLoop() {
        while (running.get()) {
            try {
                Socket().use { socket ->
                    socket.tcpNoDelay = true
                    socket.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MS)
                    status("YUV stream connected to $host:$port.")
                    BufferedOutputStream(socket.getOutputStream(), 1 shl 20).use { out ->
                        while (running.get() && socket.isConnected) {
                            val packet = latest.getAndSet(null)
                            if (packet == null) {
                                Thread.sleep(IDLE_SLEEP_MS)
                                continue
                            }
                            out.write(packet.header())
                            out.write(packet.payload)
                            out.flush()
                        }
                    }
                }
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            } catch (ex: Exception) {
                if (running.get()) {
                    status("YUV stream waiting for $host:$port (${ex.message}).")
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

    private fun YuvFramePacket.header(): ByteArray {
        /*
         * Little-endian frame header followed by I420 bytes:
         * magic[8]="TAFFYUV1", version u16=1, header_size u16=56,
         * sequence u64, timestamp_ns i64, width u16, height u16,
         * pixel_format u16=1, flags u16, exposure_ns i64, iso i32,
         * focus_diopters f32, payload_len u32.
         */
        return ByteBuffer.allocate(HEADER_SIZE)
            .order(ByteOrder.LITTLE_ENDIAN)
            .put(MAGIC)
            .putShort(VERSION.toShort())
            .putShort(HEADER_SIZE.toShort())
            .putLong(sequence)
            .putLong(timestampNs)
            .putShort(width.toShort())
            .putShort(height.toShort())
            .putShort(PIXEL_FORMAT_I420.toShort())
            .putShort(0)
            .putLong(exposureNs)
            .putInt(iso)
            .putFloat(focusDiopters)
            .putInt(payload.size)
            .array()
    }

    private companion object {
        val MAGIC: ByteArray = "TAFFYUV1".toByteArray(Charsets.US_ASCII)
        const val VERSION = 1
        const val HEADER_SIZE = 56
        const val PIXEL_FORMAT_I420 = 1
        const val CONNECT_TIMEOUT_MS = 500
        const val IDLE_SLEEP_MS = 2L
        const val RETRY_SLEEP_MS = 500L
    }
}
