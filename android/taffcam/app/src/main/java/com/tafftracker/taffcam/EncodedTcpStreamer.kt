package com.tafftracker.taffcam

import java.io.BufferedOutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

data class EncodedFramePacket(
    val sequence: Long,
    val timestampNs: Long,
    val width: Int,
    val height: Int,
    val codec: Int,
    val flags: Int,
    val exposureNs: Long,
    val iso: Int,
    val focusDiopters: Float,
    val payload: ByteArray
)

class EncodedTcpStreamer(
    private val host: String,
    private val port: Int,
    private val status: (String) -> Unit
) {
    private val running = AtomicBoolean(false)
    private val latest = AtomicReference<EncodedFramePacket?>(null)
    private val codecConfig = AtomicReference<EncodedFramePacket?>(null)
    private var worker: Thread? = null

    fun start() {
        if (running.getAndSet(true)) return
        worker = Thread(::runLoop, "TaffCamEncodedTcp").apply { start() }
    }

    fun stop() {
        running.set(false)
        latest.set(null)
        codecConfig.set(null)
        worker?.interrupt()
        worker = null
    }

    fun offer(packet: EncodedFramePacket) {
        if ((packet.flags and FLAG_CODEC_CONFIG) != 0) {
            codecConfig.set(packet)
            return
        }
        latest.set(packet)
    }

    private fun runLoop() {
        while (running.get()) {
            try {
                Socket().use { socket ->
                    socket.tcpNoDelay = true
                    socket.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MS)
                    status("MPEG stream connected to $host:$port.")
                    BufferedOutputStream(socket.getOutputStream(), 1 shl 20).use { out ->
                        var sentConfigSequence = -1L
                        while (running.get() && socket.isConnected) {
                            codecConfig.get()?.let { packet ->
                                if (packet.sequence != sentConfigSequence) {
                                    out.write(packet.wireHeader())
                                    out.write(packet.payload)
                                    out.flush()
                                    sentConfigSequence = packet.sequence
                                }
                            }
                            val packet = latest.getAndSet(null)
                            if (packet == null) {
                                Thread.sleep(IDLE_SLEEP_MS)
                                continue
                            }
                            out.write(packet.wireHeader())
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
                    status("MPEG stream waiting for $host:$port (${ex.message}).")
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

    private fun EncodedFramePacket.header(): ByteArray {
        return ByteBuffer.allocate(HEADER_SIZE)
            .order(ByteOrder.LITTLE_ENDIAN)
            .put(MAGIC)
            .putShort(VERSION.toShort())
            .putShort(HEADER_SIZE.toShort())
            .putLong(sequence)
            .putLong(timestampNs)
            .putShort(width.toShort())
            .putShort(height.toShort())
            .putShort(codec.toShort())
            .putShort(flags.toShort())
            .putLong(exposureNs)
            .putInt(iso)
            .putFloat(focusDiopters)
            .putInt(payload.size)
            .array()
    }

    private fun EncodedFramePacket.wireHeader(): ByteArray {
        val frameHeader = header()
        return ByteBuffer.allocate(LENGTH_PREFIX_SIZE + frameHeader.size)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(frameHeader.size + payload.size)
            .put(frameHeader)
            .array()
    }

    private companion object {
        val MAGIC: ByteArray = "TAFFENC1".toByteArray(Charsets.US_ASCII)
        const val VERSION = 1
        const val LENGTH_PREFIX_SIZE = 4
        const val HEADER_SIZE = 56
        const val FLAG_CODEC_CONFIG = 2
        const val CONNECT_TIMEOUT_MS = 500
        const val IDLE_SLEEP_MS = 1L
        const val RETRY_SLEEP_MS = 500L
    }
}
