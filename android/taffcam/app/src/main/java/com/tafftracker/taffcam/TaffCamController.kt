package com.tafftracker.taffcam

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.hardware.Camera as LegacyCamera
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class TaffCamController(
    private val context: Context,
    private val status: (String) -> Unit
) {
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val yuvStreamer = YuvTcpStreamer(STREAM_HOST, STREAM_PORT, status)
    private val encodedStreamer = EncodedTcpStreamer(STREAM_HOST, STREAM_PORT, status)
    private val controlServer = TaffControlServer(CONTROL_HOST, CONTROL_PORT, ::handleCommand, status)
    private val sequence = AtomicLong(0)
    private val running = AtomicBoolean(false)

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var encoder: MediaCodec? = null
    private var encoderSurface: Surface? = null
    private var cameraId: String? = null
    private var characteristics: CameraCharacteristics? = null
    private var previewSurface: Surface? = null
    private var requestBuilder: CaptureRequest.Builder? = null
    private var requestedCameraId: String? = null

    private var mode = CaptureMode(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS)
    private var captureMode: String = CAPTURE_MODE_AUTO
    private var streamFormat: String = STREAM_FORMAT_MPEG
    private var codecName: String = CODEC_H264
    private var bitrateBps: Int = DEFAULT_BITRATE_BPS
    private var keyframeIntervalS: Float = DEFAULT_KEYFRAME_INTERVAL_S
    private var highSpeedActive: Boolean = false
    private var vendorHfpsActive: Boolean = false
    private var vendorHfpsFallbackApplied: Boolean = false
    private var vendorHfpsLogged: Boolean = false
    private var vendorRecordStateLogged: Boolean = false
    @Volatile private var manualExposureEnabled: Boolean = false
    @Volatile private var exposureNs: Long = 0L
    @Volatile private var iso: Int = 0
    @Volatile private var manualFocusEnabled: Boolean = false
    @Volatile private var focusDiopters: Float = 0f
    @Volatile private var torchEnabled: Boolean = false
    @Volatile private var aeLocked: Boolean = false
    @Volatile private var awbLocked: Boolean = false
    @Volatile private var awbMode: Int = CaptureRequest.CONTROL_AWB_MODE_AUTO
    @Volatile private var zoom: Float = 1f
    @Volatile private var reportedExposureNs: Long = 0L
    @Volatile private var reportedIso: Int = 0
    @Volatile private var reportedFocusDiopters: Float = 0f
    private var aeFpsRange: Range<Int> = Range(DEFAULT_FPS, DEFAULT_FPS)

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            reportedExposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: exposureNs
            reportedIso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: iso
            reportedFocusDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: focusDiopters
        }
    }

    fun start() {
        if (running.getAndSet(true)) return
        if (context.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            status("Camera permission missing.")
            running.set(false)
            return
        }

        startCameraThread()
        controlServer.start()
        startActiveStreamer()
        openCamera()
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        closeCamera()
        yuvStreamer.stop()
        encodedStreamer.stop()
        controlServer.stop()
        stopCameraThread()
        status("TaffCam stopped.")
    }

    fun close() {
        stop()
    }

    fun describeCapabilities(): String {
        val root = JSONArray()
        for (id in cameraManager.cameraIdList) {
            val chars = cameraManager.getCameraCharacteristics(id)
            val item = JSONObject()
                .put("camera_id", id)
                .put("facing", facingName(chars.get(CameraCharacteristics.LENS_FACING)))
                .put("hardware_level", hardwareLevelName(chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)))
                .put("capabilities", intArrayJson(chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)))
                .put("ae_fps_ranges", rangeArrayJson(chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()))
                .put("vendor_request_keys", vendorKeyNamesJson(chars.availableCaptureRequestKeys))
                .put("vendor_result_keys", vendorKeyNamesJson(chars.availableCaptureResultKeys))
                .put("yuv", yuvCapabilityJson(chars))
                .put("mpeg", encodedCapabilityJson(chars))
                .put("high_speed_video", highSpeedCapabilityJson(chars))
                .put("legacy_camera1", legacyCapabilityJson(id))
                .put("exposure_ns_range", rangeJson(chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)))
                .put("iso_range", rangeJson(chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)))
                .put("min_focus_distance_diopters", chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f)
                .put("af_modes", intArrayJson(chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)))
                .put("awb_modes", intArrayJson(chars.get(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)))
                .put("flash_available", chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false)
            root.put(item)
        }
        val text = root.toString(2)
        Log.i(TAG, text)
        return text
    }

    fun applyLaunchIntent(intent: Intent?) {
        if (intent == null) return
        if (intent.hasExtra(EXTRA_WIDTH) || intent.hasExtra(EXTRA_HEIGHT) || intent.hasExtra(EXTRA_FPS) ||
            intent.hasExtra(EXTRA_STREAM_FORMAT) || intent.hasExtra(EXTRA_CODEC) ||
            intent.hasExtra(EXTRA_BITRATE_BPS) || intent.hasExtra(EXTRA_KEYFRAME_INTERVAL_S)
        ) {
            mode = CaptureMode(
                width = intent.getIntExtra(EXTRA_WIDTH, mode.width),
                height = intent.getIntExtra(EXTRA_HEIGHT, mode.height),
                fps = intent.getIntExtra(EXTRA_FPS, mode.fps)
            )
            streamFormat = normalizeStreamFormat(intent.getStringExtra(EXTRA_STREAM_FORMAT), streamFormat)
            codecName = normalizeCodec(intent.getStringExtra(EXTRA_CODEC), codecName)
            bitrateBps = intent.getIntExtra(EXTRA_BITRATE_BPS, bitrateBps).coerceAtLeast(MIN_BITRATE_BPS)
            keyframeIntervalS = intent.getFloatExtra(EXTRA_KEYFRAME_INTERVAL_S, keyframeIntervalS).coerceAtLeast(0f)
            captureMode = normalizeCaptureMode(intent.getStringExtra(EXTRA_CAPTURE_MODE), captureMode)
        }
        if (intent.hasExtra(EXTRA_EXPOSURE_NS) || intent.hasExtra(EXTRA_ISO)) {
            manualExposureEnabled = true
            exposureNs = intent.getLongExtra(EXTRA_EXPOSURE_NS, exposureNs)
            iso = intent.getIntExtra(EXTRA_ISO, iso)
        }
        if (intent.hasExtra(EXTRA_TORCH_ENABLED)) {
            torchEnabled = intent.getBooleanExtra(EXTRA_TORCH_ENABLED, torchEnabled)
        }
        if (intent.hasExtra(EXTRA_AWB_ENABLED)) {
            awbMode = if (intent.getBooleanExtra(EXTRA_AWB_ENABLED, true)) {
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            } else {
                CaptureRequest.CONTROL_AWB_MODE_OFF
            }
        }
        if (intent.hasExtra(EXTRA_AWB_LOCK)) {
            awbLocked = intent.getBooleanExtra(EXTRA_AWB_LOCK, awbLocked)
        }
        if (intent.hasExtra(EXTRA_FOCUS_DIOPTERS)) {
            manualFocusEnabled = true
            focusDiopters = intent.getFloatExtra(EXTRA_FOCUS_DIOPTERS, focusDiopters)
        }
    }

    private fun openCamera() {
        val handler = cameraHandler ?: return
        try {
            val selectedId = chooseCameraId()
            cameraId = selectedId
            characteristics = cameraManager.getCameraCharacteristics(selectedId)
            vendorHfpsFallbackApplied = false
            vendorHfpsLogged = false
            vendorRecordStateLogged = false
            @Suppress("MissingPermission")
            cameraManager.openCamera(selectedId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCaptureSession(camera)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    status("Camera disconnected.")
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    status("Camera error: $error")
                    camera.close()
                    cameraDevice = null
                    running.set(false)
                }
            }, handler)
        } catch (ex: Exception) {
            running.set(false)
            status("Open camera failed: ${ex.message}")
        }
    }

    private fun createCaptureSession(camera: CameraDevice) {
        val chars = characteristics ?: return
        val actual = chooseMode(chars, mode)
        mode = actual
        try {
            closeStreamSurfaces()
            previewSurface = if (isMpegStream()) {
                prepareEncoder(actual)
            } else {
                ImageReader.newInstance(actual.width, actual.height, ImageFormat.YUV_420_888, MAX_IMAGES).apply {
                    setOnImageAvailableListener({ reader ->
                        val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                        handleImage(image)
                    }, cameraHandler)
                    imageReader = this
                }.surface
            }
        } catch (ex: Exception) {
            status("Create stream surface failed: ${ex.message}")
            return
        }
        val surface = previewSurface ?: return

        try {
            requestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                applyCameraSettings(this, chars)
            }
            applyRepeatingRequest()
            val callback = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    applyRepeatingRequest()
                    val kind = when {
                        isMpegStream() -> codecName.uppercase()
                        highSpeedActive -> "high-speed"
                        else -> "YUV"
                    }
                    status(
                        "Streaming $kind ${actual.width}x${actual.height} requested=${actual.fps}fps " +
                            "ae=${aeFpsRange.lower}-${aeFpsRange.upper} to $STREAM_HOST:$STREAM_PORT."
                    )
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    status("Camera session configure failed (capture_mode=$captureMode high_speed=$highSpeedActive).")
                }
            }
            if (highSpeedActive) {
                camera.createConstrainedHighSpeedCaptureSession(listOf(surface), callback, cameraHandler)
            } else {
                camera.createCaptureSession(listOf(surface), callback, cameraHandler)
            }
        } catch (ex: CameraAccessException) {
            status("Create session failed: ${ex.message}")
        } catch (ex: RuntimeException) {
            status("Create session failed: ${ex.message}")
        }
    }

    private fun prepareEncoder(actual: CaptureMode): Surface? {
        val mime = codecMime(codecName)
        val format = buildEncoderFormat(mime, actual, preferCbr = true)
        val fallbackFormat = buildEncoderFormat(mime, actual, preferCbr = false)
        val codecList = MediaCodecList(MediaCodecList.REGULAR_CODECS)
        val encoderName = codecList.findEncoderForFormat(format)
            ?: codecList.findEncoderForFormat(fallbackFormat)
            ?: firstEncoderForMime(codecList, mime)
            ?: throw IllegalStateException("No encoder for $mime ${actual.width}x${actual.height}@${actual.fps}")
        var codec = MediaCodec.createByCodecName(encoderName)
        try {
            configureEncoder(codec, format, actual)
        } catch (ex: IllegalArgumentException) {
            try {
                codec.release()
            } catch (_: Exception) {
            }
            status("MPEG encoder rejected CBR settings; retrying default bitrate mode.")
            codec = MediaCodec.createByCodecName(encoderName)
            configureEncoder(codec, fallbackFormat, actual)
        }
        encoderSurface = codec.createInputSurface()
        encoder = codec
        codec.start()
        return encoderSurface
    }

    private fun configureEncoder(
        codec: MediaCodec,
        format: MediaFormat,
        actual: CaptureMode
    ) {
        codec.setCallback(object : MediaCodec.Callback() {
            override fun onInputBufferAvailable(codec: MediaCodec, index: Int) = Unit

            override fun onOutputBufferAvailable(
                codec: MediaCodec,
                index: Int,
                info: MediaCodec.BufferInfo
            ) {
                handleEncodedOutput(codec, index, info, actual)
            }

            override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) {
                status("MPEG encoder format: $format")
                sendCodecConfig(format, actual)
            }

            override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
                status("MPEG encoder error: ${e.message}")
            }
        }, cameraHandler)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    }

    private fun sendCodecConfig(format: MediaFormat, actual: CaptureMode) {
        val payload = codecConfigPayload(format) ?: return
        val packet = EncodedFramePacket(
            sequence = sequence.getAndIncrement(),
            timestampNs = 0L,
            width = actual.width,
            height = actual.height,
            codec = codecId(codecName),
            flags = ENCODED_FLAG_CODEC_CONFIG,
            exposureNs = if (reportedExposureNs > 0L) reportedExposureNs else exposureNs,
            iso = if (reportedIso > 0) reportedIso else iso,
            focusDiopters = if (reportedFocusDiopters > 0f) reportedFocusDiopters else focusDiopters,
            payload = payload
        )
        encodedStreamer.offer(packet)
    }

    private fun codecConfigPayload(format: MediaFormat): ByteArray? {
        val output = ByteArrayOutputStream()
        for (index in 0..3) {
            val key = "csd-$index"
            val buffer = try {
                format.getByteBuffer(key)
            } catch (_: Exception) {
                null
            } ?: continue
            val duplicate = buffer.duplicate()
            duplicate.position(0)
            val bytes = ByteArray(duplicate.remaining())
            duplicate.get(bytes)
            output.write(bytes)
        }
        return output.toByteArray().takeIf { it.isNotEmpty() }
    }

    private fun firstEncoderForMime(codecList: MediaCodecList, mime: String): String? {
        return codecList.codecInfos.firstOrNull { info ->
            info.isEncoder && info.supportedTypes.any { type -> type.equals(mime, ignoreCase = true) }
        }?.name
    }

    private fun buildEncoderFormat(
        mime: String,
        actual: CaptureMode,
        preferCbr: Boolean
    ): MediaFormat {
        return MediaFormat.createVideoFormat(mime, actual.width, actual.height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrateBps)
            setInteger(MediaFormat.KEY_FRAME_RATE, actual.fps)
            setFloat(MediaFormat.KEY_I_FRAME_INTERVAL, keyframeIntervalS)
            try {
                if (preferCbr) {
                    setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR)
                }
                setInteger("latency", 0)
                setInteger("priority", 0)
                setInteger("max-bframes", 0)
                setInteger("prepend-sps-pps-to-idr-frames", 1)
            } catch (_: Exception) {
            }
        }
    }

    private fun handleEncodedOutput(
        codec: MediaCodec,
        index: Int,
        info: MediaCodec.BufferInfo,
        actual: CaptureMode
    ) {
        try {
            if (info.size <= 0) return
            val buffer = codec.getOutputBuffer(index) ?: return
            buffer.position(info.offset)
            buffer.limit(info.offset + info.size)
            val payload = ByteArray(info.size)
            buffer.get(payload)
            val packet = EncodedFramePacket(
                sequence = sequence.getAndIncrement(),
                timestampNs = info.presentationTimeUs * 1000L,
                width = actual.width,
                height = actual.height,
                codec = codecId(codecName),
                flags = protocolFlags(info.flags),
                exposureNs = if (reportedExposureNs > 0L) reportedExposureNs else exposureNs,
                iso = if (reportedIso > 0) reportedIso else iso,
                focusDiopters = if (reportedFocusDiopters > 0f) reportedFocusDiopters else focusDiopters,
                payload = payload
            )
            encodedStreamer.offer(packet)
        } catch (ex: Exception) {
            Log.w(TAG, "Encoded output failed", ex)
        } finally {
            codec.releaseOutputBuffer(index, false)
        }
    }

    private fun handleImage(image: Image) {
        try {
            val width = image.width
            val height = image.height
            val payload = image.toI420()
            val packet = YuvFramePacket(
                sequence = sequence.getAndIncrement(),
                timestampNs = image.timestamp,
                width = width,
                height = height,
                exposureNs = if (reportedExposureNs > 0L) reportedExposureNs else exposureNs,
                iso = if (reportedIso > 0) reportedIso else iso,
                focusDiopters = if (reportedFocusDiopters > 0f) reportedFocusDiopters else focusDiopters,
                payload = payload
            )
            yuvStreamer.offer(packet)
        } catch (ex: Exception) {
            Log.w(TAG, "Image conversion failed", ex)
        } finally {
            image.close()
        }
    }

    private fun applyRepeatingRequest() {
        val builder = requestBuilder ?: return
        val session = captureSession ?: return
        val chars = characteristics
        try {
            applyCameraSettings(builder, chars)
            val request = builder.build()
            if (highSpeedActive && session is CameraConstrainedHighSpeedCaptureSession) {
                session.setRepeatingBurst(
                    session.createHighSpeedRequestList(request),
                    captureCallback,
                    cameraHandler
                )
            } else {
                session.setRepeatingRequest(request, captureCallback, cameraHandler)
            }
        } catch (ex: Exception) {
            if (vendorHfpsActive && !vendorHfpsFallbackApplied && chars != null) {
                vendorHfpsFallbackApplied = true
                vendorHfpsActive = false
                aeFpsRange = chooseFpsRange(chars, min(mode.fps, 30))
                status("MediaTek vendor hfpsMode rejected (${ex.message}); falling back to ${aeFpsRange.lower}-${aeFpsRange.upper}fps.")
                applyRepeatingRequest()
                return
            }
            status("Apply camera request failed: ${ex.message}")
        }
    }

    private fun handleCommand(command: JSONObject): JSONObject {
        val cmd = command.optString("cmd", command.optString("command", ""))
        return when (cmd) {
            "get_capabilities" -> JSONObject().put("ok", true).put("capabilities", JSONArray(describeCapabilities()))
            "set_mode" -> setMode(command)
            "set_focus" -> setFocus(command)
            "set_exposure" -> setExposure(command)
            "set_wb" -> setWhiteBalance(command)
            "set_torch" -> setTorch(command)
            "set_zoom" -> setZoom(command)
            "set_auto_locks" -> setAutoLocks(command)
            else -> JSONObject().put("ok", false).put("error", "unknown command '$cmd'")
        }
    }

    private fun setMode(command: JSONObject): JSONObject {
        val newMode = CaptureMode(
            width = command.optInt("width", DEFAULT_WIDTH),
            height = command.optInt("height", DEFAULT_HEIGHT),
            fps = command.optInt("fps", DEFAULT_FPS)
        )
        val requestedStreamFormat = command.optString("stream_format", command.optString("streamFormat", streamFormat))
            .lowercase()
        val requestedCaptureMode = command.optString("capture_mode", command.optString("captureMode", captureMode))
            .lowercase()
        val warnings = JSONArray()
        val newStreamFormat = when (requestedStreamFormat) {
            STREAM_FORMAT_YUV, STREAM_FORMAT_MPEG -> requestedStreamFormat
            else -> {
                warnings.put("unsupported stream_format '$requestedStreamFormat'; using '$STREAM_FORMAT_MPEG'")
                STREAM_FORMAT_MPEG
            }
        }
        val newCaptureMode = when (requestedCaptureMode) {
            CAPTURE_MODE_AUTO, CAPTURE_MODE_YUV, CAPTURE_MODE_NORMAL, CAPTURE_MODE_HIGH_SPEED -> requestedCaptureMode
            else -> {
                warnings.put("unsupported capture_mode '$requestedCaptureMode'; using '$CAPTURE_MODE_AUTO'")
                CAPTURE_MODE_AUTO
            }
        }
        val requestedCodec = command.optString("codec", codecName).lowercase()
        val newCodecName = when (requestedCodec) {
            CODEC_MPEG4, CODEC_H264 -> requestedCodec
            "mpeg" -> CODEC_MPEG4
            else -> {
                warnings.put("unsupported codec '$requestedCodec'; using '$CODEC_H264'")
                CODEC_H264
            }
        }
        val newBitrateBps = command.optInt("bitrate_bps", command.optInt("bitrateBps", bitrateBps))
            .coerceAtLeast(MIN_BITRATE_BPS)
        val newKeyframeIntervalS = command.optDouble(
            "keyframe_interval_s",
            command.optDouble("keyframeIntervalS", keyframeIntervalS.toDouble())
        ).toFloat().coerceAtLeast(0f)
        val newRequestedCameraId = command.optString("camera_id", "").takeIf { it.isNotBlank() }
            ?: requestedCameraId
        applyInlineCameraControls(command)
        val changed = newMode != mode ||
            newStreamFormat != streamFormat ||
            newCaptureMode != captureMode ||
            newCodecName != codecName ||
            newBitrateBps != bitrateBps ||
            newKeyframeIntervalS != keyframeIntervalS ||
            newRequestedCameraId != requestedCameraId
        mode = newMode
        streamFormat = newStreamFormat
        captureMode = newCaptureMode
        codecName = newCodecName
        bitrateBps = newBitrateBps
        keyframeIntervalS = newKeyframeIntervalS
        requestedCameraId = newRequestedCameraId
        if (running.get()) {
            if (changed) {
                closeCamera()
                restartActiveStreamer()
                openCamera()
            } else {
                characteristics?.let {
                    vendorHfpsActive = shouldUseVendorHfps(it, mode.fps)
                    aeFpsRange = chooseRequestedFpsRange(it, mode.fps)
                }
                applyRepeatingRequest()
            }
        }
        return JSONObject()
            .put("ok", true)
            .put("warnings", warnings)
            .put(
                "mode",
                JSONObject()
                    .put("width", mode.width)
                    .put("height", mode.height)
                    .put("fps", mode.fps)
                    .put("stream_format", streamFormat)
                    .put("capture_mode", captureMode)
                    .put("codec", codecName)
                    .put("bitrate_bps", bitrateBps)
                    .put("keyframe_interval_s", keyframeIntervalS)
            )
    }

    private fun setFocus(command: JSONObject): JSONObject {
        val auto = command.optBoolean("auto", false)
        manualFocusEnabled = !auto
        focusDiopters = command.optDouble("diopters", command.optDouble("focus_diopters", 0.0)).toFloat()
        applyRepeatingRequest()
        return JSONObject().put("ok", true).put("manual_focus", manualFocusEnabled).put("focus_diopters", focusDiopters)
    }

    private fun setExposure(command: JSONObject): JSONObject {
        val auto = command.optBoolean("auto", false)
        manualExposureEnabled = !auto
        if (!auto) {
            exposureNs = command.optLong("exposure_ns", command.optLong("exposureNs", exposureNs))
            iso = command.optInt("iso", iso)
        }
        applyRepeatingRequest()
        return JSONObject().put("ok", true).put("manual_exposure", manualExposureEnabled).put("exposure_ns", exposureNs).put("iso", iso)
    }

    private fun applyInlineCameraControls(command: JSONObject) {
        if (command.has("exposure_ns") || command.has("exposureNs") || command.has("iso")) {
            manualExposureEnabled = true
            exposureNs = command.optLong("exposure_ns", command.optLong("exposureNs", exposureNs))
            iso = command.optInt("iso", iso)
        }
        if (command.has("focus_diopters") || command.has("diopters")) {
            manualFocusEnabled = true
            focusDiopters = command.optDouble("focus_diopters", command.optDouble("diopters", focusDiopters.toDouble())).toFloat()
        }
        if (command.has("awb_enabled")) {
            awbMode = if (command.optBoolean("awb_enabled", true)) {
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            } else {
                CaptureRequest.CONTROL_AWB_MODE_OFF
            }
        }
        if (command.has("awb_lock")) {
            awbLocked = command.optBoolean("awb_lock", awbLocked)
        }
        if (command.has("torch_enabled")) {
            torchEnabled = command.optBoolean("torch_enabled", torchEnabled)
        }
        if (command.has("zoom_ratio")) {
            zoom = command.optDouble("zoom_ratio", zoom.toDouble()).toFloat().coerceAtLeast(1f)
        }
    }

    private fun setWhiteBalance(command: JSONObject): JSONObject {
        val auto = command.optBoolean("auto", true)
        val modeName = command.optString("mode", if (auto) "auto" else "off")
        awbMode = awbModeFromName(modeName)
        applyRepeatingRequest()
        return JSONObject().put("ok", true).put("awb_mode", awbMode)
    }

    private fun setTorch(command: JSONObject): JSONObject {
        torchEnabled = command.optBoolean("enabled", command.optBoolean("on", false))
        applyRepeatingRequest()
        return JSONObject().put("ok", true).put("torch", torchEnabled)
    }

    private fun setZoom(command: JSONObject): JSONObject {
        zoom = command.optDouble("zoom", command.optDouble("ratio", 1.0)).toFloat().coerceAtLeast(1f)
        applyRepeatingRequest()
        return JSONObject().put("ok", true).put("zoom", zoom)
    }

    private fun setAutoLocks(command: JSONObject): JSONObject {
        aeLocked = command.optBoolean("ae_lock", command.optBoolean("ae", aeLocked))
        awbLocked = command.optBoolean("awb_lock", command.optBoolean("awb", awbLocked))
        applyRepeatingRequest()
        return JSONObject().put("ok", true).put("ae_lock", aeLocked).put("awb_lock", awbLocked)
    }

    private fun applyCameraSettings(builder: CaptureRequest.Builder, chars: CameraCharacteristics?) {
        builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, aeFpsRange)
        applyVendorRecordState(builder, chars)
        applyVendorHfps(builder, chars)
        if (manualExposureEnabled) {
            builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            if (exposureNs > 0L) {
                val requestedFps = if (mode.fps > 0) mode.fps else aeFpsRange.upper
                val maxFrameExposureNs = if (requestedFps > 0) {
                    1_000_000_000L / requestedFps.toLong()
                } else {
                    Long.MAX_VALUE
                }
                val sensorRange = chars?.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                val sensorClampedExposureNs = sensorRange?.clamp(exposureNs) ?: exposureNs
                val effectiveExposureNs = min(sensorClampedExposureNs, maxFrameExposureNs)
                builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, effectiveExposureNs)
                if (requestedFps > 0) {
                    val frameDurationNs = max(effectiveExposureNs, maxFrameExposureNs)
                    builder.set(CaptureRequest.SENSOR_FRAME_DURATION, frameDurationNs)
                }
            }
            if (iso > 0) {
                val sensorIsoRange = chars?.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                val effectiveIso = sensorIsoRange?.clamp(iso) ?: iso
                builder.set(CaptureRequest.SENSOR_SENSITIVITY, effectiveIso)
            }
        } else {
            builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        }
        builder.set(CaptureRequest.CONTROL_AE_LOCK, aeLocked)
        builder.set(CaptureRequest.CONTROL_AWB_MODE, awbMode)
        builder.set(CaptureRequest.CONTROL_AWB_LOCK, awbLocked)
        if (manualFocusEnabled) {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            val maxFocus = chars?.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
            val effectiveFocus = if (maxFocus > 0f) focusDiopters.coerceIn(0f, maxFocus) else focusDiopters.coerceAtLeast(0f)
            builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, effectiveFocus)
        }
        builder.set(CaptureRequest.FLASH_MODE, if (torchEnabled) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF)
        applyZoom(builder, chars)
    }

    private fun applyVendorHfps(builder: CaptureRequest.Builder, chars: CameraCharacteristics?) {
        if (!vendorHfpsActive || chars == null) return
        try {
            @Suppress("UNCHECKED_CAST")
            val hfpsKey = (vendorHfpsRequestKey(chars) as? CaptureRequest.Key<IntArray>)
                ?: CaptureRequest.Key(VENDOR_HFPS_MODE, IntArray::class.java)
            builder.set(hfpsKey, intArrayOf(1))
            if (!vendorHfpsLogged) {
                status("Trying MediaTek vendor hfpsMode=1 with ae=${aeFpsRange.lower}-${aeFpsRange.upper}fps.")
                vendorHfpsLogged = true
            }
        } catch (ex: Exception) {
            try {
                val intKey: CaptureRequest.Key<Int> =
                    CaptureRequest.Key(VENDOR_HFPS_MODE, Int::class.javaObjectType)
                builder.set(intKey, 1)
                if (!vendorHfpsLogged) {
                    status("Trying MediaTek vendor hfpsMode=1(Integer) with ae=${aeFpsRange.lower}-${aeFpsRange.upper}fps.")
                    vendorHfpsLogged = true
                }
            } catch (fallback: Exception) {
                vendorHfpsActive = false
                if (!vendorHfpsLogged) {
                    status("MediaTek vendor hfpsMode unavailable: ${fallback.message ?: ex.message}")
                    vendorHfpsLogged = true
                }
            }
        }
    }

    private fun applyVendorRecordState(builder: CaptureRequest.Builder, chars: CameraCharacteristics?) {
        if (!isMpegStream() || chars == null || !hasMtkStreamingFeature(chars)) return
        try {
            @Suppress("UNCHECKED_CAST")
            val recordKey = (vendorRecordStateRequestKey(chars) as? CaptureRequest.Key<IntArray>)
                ?: CaptureRequest.Key(VENDOR_RECORD_STATE, IntArray::class.java)
            builder.set(recordKey, intArrayOf(1))
            if (!vendorRecordStateLogged) {
                status("Trying MediaTek recordState=1 for encoder surface.")
                vendorRecordStateLogged = true
            }
        } catch (ex: Exception) {
            try {
                val intKey: CaptureRequest.Key<Int> =
                    CaptureRequest.Key(VENDOR_RECORD_STATE, Int::class.javaObjectType)
                builder.set(intKey, 1)
                if (!vendorRecordStateLogged) {
                    status("Trying MediaTek recordState=1(Integer) for encoder surface.")
                    vendorRecordStateLogged = true
                }
            } catch (_: Exception) {
            }
        }
    }

    private fun applyZoom(builder: CaptureRequest.Builder, chars: CameraCharacteristics?) {
        if (chars == null) return
        val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return
        val maxZoom = chars.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f
        val ratio = zoom.coerceIn(1f, maxZoom)
        val cropWidth = (activeArray.width() / ratio).roundToInt()
        val cropHeight = (activeArray.height() / ratio).roundToInt()
        val left = activeArray.left + (activeArray.width() - cropWidth) / 2
        val top = activeArray.top + (activeArray.height() - cropHeight) / 2
        builder.set(CaptureRequest.SCALER_CROP_REGION, Rect(left, top, left + cropWidth, top + cropHeight))
    }

    private fun startCameraThread() {
        if (cameraThread != null) return
        cameraThread = HandlerThread("TaffCamCamera").also {
            it.start()
            cameraHandler = Handler(it.looper)
        }
    }

    private fun stopCameraThread() {
        cameraThread?.quitSafely()
        try {
            cameraThread?.join(1000)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        cameraThread = null
        cameraHandler = null
    }

    private fun closeCamera() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            closeStreamSurfaces()
            requestBuilder = null
            previewSurface = null
            highSpeedActive = false
            vendorHfpsActive = false
        } catch (ex: Exception) {
            Log.w(TAG, "Close camera failed", ex)
        }
    }

    private fun closeStreamSurfaces() {
        try {
            encoder?.stop()
        } catch (_: Exception) {
        }
        try {
            encoder?.release()
        } catch (_: Exception) {
        }
        encoder = null
        encoderSurface?.release()
        encoderSurface = null
        imageReader?.close()
        imageReader = null
    }

    private fun startActiveStreamer() {
        if (isMpegStream()) {
            yuvStreamer.stop()
            encodedStreamer.start()
        } else {
            encodedStreamer.stop()
            yuvStreamer.start()
        }
    }

    private fun restartActiveStreamer() {
        yuvStreamer.stop()
        encodedStreamer.stop()
        startActiveStreamer()
    }

    private fun chooseCameraId(): String {
        requestedCameraId?.let { requested ->
            if (requested in cameraManager.cameraIdList) return requested
        }
        var fallback: String? = null
        for (id in cameraManager.cameraIdList) {
            val chars = cameraManager.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            if (fallback == null) fallback = id
            if (facing == CameraCharacteristics.LENS_FACING_BACK) return id
        }
        return fallback ?: throw IllegalStateException("No camera found")
    }

    private fun chooseMode(chars: CameraCharacteristics, requested: CaptureMode): CaptureMode {
        if (isMpegStream()) {
            return chooseEncodedMode(chars, requested)
        }
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = map?.getOutputSizes(ImageFormat.YUV_420_888)?.toList().orEmpty()
        val targetFrameNs = if (requested.fps > 0) {
            1_000_000_000L / requested.fps.toLong()
        } else {
            Long.MAX_VALUE
        }
        val fpsCapableSizes = if (map != null && requested.fps > 0) {
            sizes.filter { size ->
                val minFrameNs = map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size)
                minFrameNs <= 0L || minFrameNs <= targetFrameNs
            }
        } else {
            sizes
        }
        val candidateSizes = fpsCapableSizes.ifEmpty { sizes }
        val size = candidateSizes.minByOrNull { abs(it.width - requested.width) + abs(it.height - requested.height) }
            ?: Size(requested.width, requested.height)
        val minFrameNs = map?.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size) ?: 0L
        val yuvSupportsRequested = requested.fps <= 0 || fpsCapableSizes.isNotEmpty()
        if (!yuvSupportsRequested && requested.fps > 0 && minFrameNs > targetFrameNs) {
            val maxFps = 1_000_000_000.0 / minFrameNs.toDouble()
            status("No YUV size reports ${requested.fps}fps; closest ${size.width}x${size.height} reports max ${"%.1f".format(maxFps)}fps.")
        }
        highSpeedActive = false
        vendorHfpsActive = false
        val shouldTryHighSpeed = captureMode == CAPTURE_MODE_HIGH_SPEED
        if (shouldTryHighSpeed) {
            val highSpeed = chooseHighSpeedMode(chars, requested)
            if (highSpeed != null) {
                highSpeedActive = true
                vendorHfpsActive = false
                aeFpsRange = highSpeed.second
                return highSpeed.first
            }
            if (captureMode == CAPTURE_MODE_HIGH_SPEED) {
                status("High-speed video mode unavailable for ${requested.width}x${requested.height}@${requested.fps}; falling back to YUV.")
            }
        }
        vendorHfpsActive = shouldUseVendorHfps(chars, requested.fps)
        aeFpsRange = chooseRequestedFpsRange(chars, requested.fps)
        return CaptureMode(size.width, size.height, requested.fps)
    }

    private fun chooseEncodedMode(chars: CameraCharacteristics, requested: CaptureMode): CaptureMode {
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = map?.getOutputSizes(MediaCodec::class.java)?.toList().orEmpty()
        val targetFrameNs = if (requested.fps > 0) 1_000_000_000L / requested.fps.toLong() else Long.MAX_VALUE
        val size = sizes.minByOrNull {
            abs(it.width - requested.width) + abs(it.height - requested.height)
        } ?: Size(requested.width, requested.height)
        val minFrameNs = map?.getOutputMinFrameDuration(MediaCodec::class.java, size) ?: 0L
        if (requested.fps > 0 && minFrameNs > targetFrameNs) {
            val maxFps = 1_000_000_000.0 / minFrameNs.toDouble()
            status("MPEG surface ${size.width}x${size.height} reports max ${"%.1f".format(maxFps)}fps; still requesting ${requested.fps}fps.")
        }
        val shouldTryHighSpeed = captureMode == CAPTURE_MODE_HIGH_SPEED ||
            (captureMode == CAPTURE_MODE_AUTO && requested.fps > 30)
        if (shouldTryHighSpeed) {
            val highSpeed = chooseHighSpeedMode(chars, requested, sizes.toSet())
            if (highSpeed != null) {
                highSpeedActive = true
                vendorHfpsActive = false
                aeFpsRange = highSpeed.second
                status(
                    "Using high-speed encoded surface ${highSpeed.first.width}x${highSpeed.first.height} " +
                        "at ${aeFpsRange.lower}-${aeFpsRange.upper}fps."
                )
                return highSpeed.first
            }
            if (captureMode == CAPTURE_MODE_HIGH_SPEED) {
                status("High-speed encoded mode unavailable for ${requested.width}x${requested.height}@${requested.fps}; falling back to normal encoded session.")
            }
        }
        highSpeedActive = false
        vendorHfpsActive = shouldUseVendorHfps(chars, requested.fps)
        aeFpsRange = chooseRequestedFpsRange(chars, requested.fps)
        if (vendorHfpsActive) {
            status(
                "Public Camera2 caps stop at ${publicMaxFps(chars)}fps; " +
                    "trying MediaTek hfps vendor path for ${requested.fps}fps."
            )
        }
        return CaptureMode(size.width, size.height, requested.fps)
    }

    private fun chooseHighSpeedMode(
        chars: CameraCharacteristics,
        requested: CaptureMode,
        allowedSizes: Set<Size>? = null
    ): Pair<CaptureMode, Range<Int>>? {
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val highSpeedSizes = map.highSpeedVideoSizes?.toList().orEmpty()
        if (highSpeedSizes.isEmpty()) return null
        val candidates = highSpeedSizes.flatMap { size ->
            map.getHighSpeedVideoFpsRangesFor(size).orEmpty().map { range -> size to range }
        }.filter { (size, _) -> allowedSizes == null || size in allowedSizes }
        if (candidates.isEmpty()) return null

        fun sizeDistance(size: Size): Int {
            return abs(size.width - requested.width) + abs(size.height - requested.height)
        }

        val exactOrFaster = candidates
            .filter { (_, range) -> requested.fps <= 0 || range.upper >= requested.fps }
            .minWithOrNull(
                compareBy<Pair<Size, Range<Int>>> { sizeDistance(it.first) }
                    .thenBy { abs(it.second.upper - requested.fps) }
                    .thenBy { it.second.upper - it.second.lower }
            )
        val fastestFallback = candidates.minWithOrNull(
            compareByDescending<Pair<Size, Range<Int>>> { it.second.upper }
                .thenBy { sizeDistance(it.first) }
        )
        val (size, range) = exactOrFaster ?: fastestFallback ?: return null
        return CaptureMode(size.width, size.height, range.upper) to range
    }

    private fun chooseFpsRange(chars: CameraCharacteristics, requestedFps: Int): Range<Int> {
        val fpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
        return fpsRanges
            .filter { requestedFps in it.lower..it.upper }
            .minWithOrNull(compareBy<Range<Int>> { it.upper - it.lower }.thenBy { abs(it.upper - requestedFps) })
            ?: fpsRanges.maxByOrNull { it.upper }
            ?: Range(requestedFps, requestedFps)
    }

    private fun chooseRequestedFpsRange(chars: CameraCharacteristics, requestedFps: Int): Range<Int> {
        return if (shouldUseVendorHfps(chars, requestedFps)) {
            Range(requestedFps, requestedFps)
        } else {
            chooseFpsRange(chars, requestedFps)
        }
    }

    private fun publicMaxFps(chars: CameraCharacteristics): Int {
        return chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            .orEmpty()
            .maxOfOrNull { it.upper }
            ?: 0
    }

    private fun shouldUseVendorHfps(chars: CameraCharacteristics, requestedFps: Int): Boolean {
        return requestedFps > publicMaxFps(chars) && hasMtkStreamingFeature(chars)
    }

    private fun vendorHfpsRequestKey(chars: CameraCharacteristics): CaptureRequest.Key<*>? {
        return chars.availableCaptureRequestKeys?.firstOrNull { it.name == VENDOR_HFPS_MODE }
    }

    private fun vendorRecordStateRequestKey(chars: CameraCharacteristics): CaptureRequest.Key<*>? {
        return chars.availableCaptureRequestKeys?.firstOrNull { it.name == VENDOR_RECORD_STATE }
    }

    private fun hasMtkStreamingFeature(chars: CameraCharacteristics): Boolean {
        return chars.availableCaptureRequestKeys
            .orEmpty()
            .any { it.name.startsWith("com.mediatek.streamingfeature.") }
    }

    private fun yuvCapabilityJson(chars: CameraCharacteristics): JSONArray {
        val result = JSONArray()
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return result
        val fpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
        for (size in map.getOutputSizes(ImageFormat.YUV_420_888).orEmpty()) {
            val minFrameDurationNs = map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size)
            val estimatedMaxFps = if (minFrameDurationNs > 0L) 1_000_000_000.0 / minFrameDurationNs else JSONObject.NULL
            result.put(
                JSONObject()
                    .put("width", size.width)
                    .put("height", size.height)
                    .put("min_frame_duration_ns", minFrameDurationNs)
                    .put("estimated_max_fps", estimatedMaxFps)
                    .put("supports_60_yuv", minFrameDurationNs <= 0L || minFrameDurationNs <= 16_666_667L)
                    .put("fps_ranges", rangeArrayJson(fpsRanges))
            )
        }
        return result
    }

    private fun encodedCapabilityJson(chars: CameraCharacteristics): JSONObject {
        val root = JSONObject()
        val sizesJson = JSONArray()
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        if (map != null) {
            val fpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
            for (size in map.getOutputSizes(MediaCodec::class.java).orEmpty()) {
                val minFrameDurationNs = map.getOutputMinFrameDuration(MediaCodec::class.java, size)
                val estimatedMaxFps = if (minFrameDurationNs > 0L) 1_000_000_000.0 / minFrameDurationNs else JSONObject.NULL
                sizesJson.put(
                    JSONObject()
                        .put("width", size.width)
                        .put("height", size.height)
                        .put("min_frame_duration_ns", minFrameDurationNs)
                        .put("estimated_max_fps", estimatedMaxFps)
                        .put("supports_60_surface", minFrameDurationNs <= 0L || minFrameDurationNs <= 16_666_667L)
                        .put("fps_ranges", rangeArrayJson(fpsRanges))
                )
            }
        }
        return root
            .put("sizes", sizesJson)
            .put("codecs", JSONArray().put(CODEC_H264).put(CODEC_MPEG4))
            .put("default_codec", CODEC_H264)
    }

    private fun highSpeedCapabilityJson(chars: CameraCharacteristics): JSONArray {
        val result = JSONArray()
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return result
        for (size in map.highSpeedVideoSizes.orEmpty()) {
            result.put(
                JSONObject()
                    .put("width", size.width)
                    .put("height", size.height)
                    .put("fps_ranges", rangeArrayJson(map.getHighSpeedVideoFpsRangesFor(size).orEmpty()))
            )
        }
        return result
    }

    private fun legacyCapabilityJson(cameraId: String): JSONObject {
        val root = JSONObject()
        val legacyId = cameraId.toIntOrNull()
            ?: return root.put("available", false).put("error", "non-numeric camera id")
        if (legacyId !in 0 until LegacyCamera.getNumberOfCameras()) {
            return root.put("available", false).put("error", "legacy camera id out of range")
        }

        var camera: LegacyCamera? = null
        return try {
            camera = LegacyCamera.open(legacyId)
            val params = camera.parameters
            Log.i(
                TAG,
                "LEGACY_CAPS camera=$cameraId fps=${legacyFpsRangesSummary(params.supportedPreviewFpsRange)} " +
                    "sizes=${legacySizesSummary(params.supportedPreviewSizes)}"
            )
            root
                .put("available", true)
                .put("preview_fps_ranges", legacyFpsRangesJson(params.supportedPreviewFpsRange))
                .put("preview_sizes", legacySizesJson(params.supportedPreviewSizes))
                .put("picture_sizes", legacySizesJson(params.supportedPictureSizes))
                .put("focus_modes", JSONArray(params.supportedFocusModes.orEmpty()))
                .put("flash_modes", JSONArray(params.supportedFlashModes.orEmpty()))
                .put("white_balance_modes", JSONArray(params.supportedWhiteBalance.orEmpty()))
        } catch (ex: Exception) {
            root.put("available", false).put("error", ex.message ?: ex.javaClass.simpleName)
        } finally {
            try {
                camera?.release()
            } catch (_: Exception) {
            }
        }
    }

    private fun legacyFpsRangesSummary(values: List<IntArray>?): String {
        return values.orEmpty().joinToString(prefix = "[", postfix = "]") { range ->
            if (range.size >= 2) {
                "${range[0] / 1000.0}-${range[1] / 1000.0}"
            } else {
                "invalid"
            }
        }
    }

    private fun legacySizesSummary(values: List<LegacyCamera.Size>?): String {
        return values.orEmpty()
            .take(12)
            .joinToString(prefix = "[", postfix = "]") { size -> "${size.width}x${size.height}" }
    }

    private fun legacyFpsRangesJson(values: List<IntArray>?): JSONArray {
        val result = JSONArray()
        values.orEmpty().forEach { range ->
            if (range.size >= 2) {
                result.put(
                    JSONObject()
                        .put("min", range[0] / 1000.0)
                        .put("max", range[1] / 1000.0)
                )
            }
        }
        return result
    }

    private fun legacySizesJson(values: List<LegacyCamera.Size>?): JSONArray {
        val result = JSONArray()
        values.orEmpty().forEach { size ->
            result.put(JSONObject().put("width", size.width).put("height", size.height))
        }
        return result
    }

    private fun Image.toI420(): ByteArray {
        val output = ByteArray(width * height * 3 / 2)
        val chromaWidth = width / 2
        val chromaHeight = height / 2
        copyPlane(planes[0], width, height, output, 0, width)
        copyPlane(planes[1], chromaWidth, chromaHeight, output, width * height, chromaWidth)
        copyPlane(planes[2], chromaWidth, chromaHeight, output, width * height + chromaWidth * chromaHeight, chromaWidth)
        return output
    }

    private fun copyPlane(
        plane: Image.Plane,
        width: Int,
        height: Int,
        out: ByteArray,
        offset: Int,
        outStride: Int
    ) {
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        for (row in 0 until height) {
            val rowStart = row * rowStride
            if (pixelStride == 1 && outStride == width) {
                buffer.position(rowStart)
                buffer.get(out, offset + row * outStride, width)
            } else {
                for (col in 0 until width) {
                    out[offset + row * outStride + col] = buffer.get(rowStart + col * pixelStride)
                }
            }
        }
    }

    private fun facingName(value: Int?): String = when (value) {
        CameraCharacteristics.LENS_FACING_BACK -> "back"
        CameraCharacteristics.LENS_FACING_FRONT -> "front"
        CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
        else -> "unknown"
    }

    private fun hardwareLevelName(value: Int?): String = when (value) {
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "legacy"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "limited"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "full"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "level_3"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "external"
        else -> "unknown"
    }

    private fun awbModeFromName(value: String): Int = when (value.lowercase()) {
        "off", "manual" -> CaptureRequest.CONTROL_AWB_MODE_OFF
        "incandescent" -> CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT
        "fluorescent" -> CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT
        "warm_fluorescent" -> CaptureRequest.CONTROL_AWB_MODE_WARM_FLUORESCENT
        "daylight" -> CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
        "cloudy" -> CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
        "twilight" -> CaptureRequest.CONTROL_AWB_MODE_TWILIGHT
        "shade" -> CaptureRequest.CONTROL_AWB_MODE_SHADE
        else -> CaptureRequest.CONTROL_AWB_MODE_AUTO
    }

    private fun normalizeStreamFormat(value: String?, fallback: String): String = when (value?.lowercase()) {
        STREAM_FORMAT_YUV, STREAM_FORMAT_MPEG -> value.lowercase()
        else -> fallback
    }

    private fun normalizeCaptureMode(value: String?, fallback: String): String = when (value?.lowercase()) {
        CAPTURE_MODE_AUTO, CAPTURE_MODE_YUV, CAPTURE_MODE_NORMAL, CAPTURE_MODE_HIGH_SPEED -> value.lowercase()
        else -> fallback
    }

    private fun normalizeCodec(value: String?, fallback: String): String = when (value?.lowercase()) {
        CODEC_MPEG4, CODEC_H264 -> value.lowercase()
        "mpeg" -> CODEC_MPEG4
        else -> fallback
    }

    private fun isMpegStream(): Boolean = streamFormat == STREAM_FORMAT_MPEG

    private fun codecMime(value: String): String = when (value.lowercase()) {
        CODEC_H264 -> MediaFormat.MIMETYPE_VIDEO_AVC
        else -> MediaFormat.MIMETYPE_VIDEO_MPEG4
    }

    private fun codecId(value: String): Int = when (value.lowercase()) {
        CODEC_H264 -> CODEC_ID_H264
        else -> CODEC_ID_MPEG4
    }

    private fun protocolFlags(mediaCodecFlags: Int): Int {
        var flags = 0
        if ((mediaCodecFlags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0) {
            flags = flags or ENCODED_FLAG_KEYFRAME
        }
        if ((mediaCodecFlags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
            flags = flags or ENCODED_FLAG_CODEC_CONFIG
        }
        return flags
    }

    private fun intArrayJson(values: IntArray?): JSONArray {
        val array = JSONArray()
        values?.forEach { array.put(it) }
        return array
    }

    private fun vendorKeyNamesJson(keys: List<*>?): JSONArray {
        val array = JSONArray()
        keys.orEmpty()
            .mapNotNull { key ->
                when (key) {
                    is CameraCharacteristics.Key<*> -> key.name
                    is CaptureRequest.Key<*> -> key.name
                    is CaptureResult.Key<*> -> key.name
                    else -> null
                }
            }
            .filter { it.startsWith("com.") }
            .sorted()
            .forEach { array.put(it) }
        return array
    }

    private fun rangeArrayJson(values: Array<out Range<Int>>): JSONArray {
        val array = JSONArray()
        values.forEach { array.put(JSONObject().put("min", it.lower).put("max", it.upper)) }
        return array
    }

    private fun <T : Comparable<T>> rangeJson(value: Range<T>?): JSONObject? {
        return value?.let { JSONObject().put("min", it.lower).put("max", it.upper) }
    }

    data class CaptureMode(val width: Int, val height: Int, val fps: Int)

    private companion object {
        const val TAG = "TaffCam"
        const val STREAM_HOST = "127.0.0.1"
        const val STREAM_PORT = 27183
        const val CONTROL_HOST = "127.0.0.1"
        const val CONTROL_PORT = 27184
        const val DEFAULT_WIDTH = 640
        const val DEFAULT_HEIGHT = 480
        const val DEFAULT_FPS = 60
        const val DEFAULT_BITRATE_BPS = 8_000_000
        const val MIN_BITRATE_BPS = 500_000
        const val DEFAULT_KEYFRAME_INTERVAL_S = 1.0f
        const val STREAM_FORMAT_YUV = "yuv"
        const val STREAM_FORMAT_MPEG = "mpeg"
        const val CODEC_MPEG4 = "mpeg4"
        const val CODEC_H264 = "h264"
        const val CODEC_ID_MPEG4 = 1
        const val CODEC_ID_H264 = 2
        const val EXTRA_WIDTH = "width"
        const val EXTRA_HEIGHT = "height"
        const val EXTRA_FPS = "fps"
        const val EXTRA_STREAM_FORMAT = "stream_format"
        const val EXTRA_CAPTURE_MODE = "capture_mode"
        const val EXTRA_CODEC = "codec"
        const val EXTRA_BITRATE_BPS = "bitrate_bps"
        const val EXTRA_KEYFRAME_INTERVAL_S = "keyframe_interval_s"
        const val EXTRA_EXPOSURE_NS = "exposure_ns"
        const val EXTRA_ISO = "iso"
        const val EXTRA_TORCH_ENABLED = "torch_enabled"
        const val EXTRA_AWB_ENABLED = "awb_enabled"
        const val EXTRA_AWB_LOCK = "awb_lock"
        const val EXTRA_FOCUS_DIOPTERS = "focus_diopters"
        const val ENCODED_FLAG_KEYFRAME = 1
        const val ENCODED_FLAG_CODEC_CONFIG = 2
        const val MAX_IMAGES = 3
        const val CAPTURE_MODE_AUTO = "auto"
        const val CAPTURE_MODE_YUV = "yuv"
        const val CAPTURE_MODE_NORMAL = "normal"
        const val CAPTURE_MODE_HIGH_SPEED = "high_speed"
        const val VENDOR_HFPS_MODE = "com.mediatek.streamingfeature.hfpsMode"
        const val VENDOR_RECORD_STATE = "com.mediatek.streamingfeature.recordState"
    }
}
