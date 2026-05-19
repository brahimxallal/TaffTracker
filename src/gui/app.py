from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from collections.abc import Sequence
from pathlib import Path

from src.config_loader import build_config_from_yaml, load_yaml_config
from src.gui.view_model import (
    LaunchSettings,
    TuningSettings,
    apply_launch_settings,
    apply_tuning_settings,
    build_dashboard_shell_spec,
    build_runtime_diagnostic_rows,
    launch_settings_from_config,
    runtime_control_specs,
    suggest_source_for_mode,
    summarize_config,
    tuning_settings_from_config,
    validate_launch_settings,
)


def main(argv: Sequence[str] | None = None) -> int:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Launch the TaffTracker desktop control plane.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args(argv)

    try:
        return _run_qt_app(args.config)
    except ImportError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            print(
                "PySide6 is not installed. Install GUI extras with: pip install -e .[gui]",
                file=sys.stderr,
            )
            return 1
        raise


def _run_qt_app(config_path: Path) -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QPalette, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStyle,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    from src.gui.runtime_session import RuntimeSession
    from src.gui.theme import APP_STYLESHEET

    config = build_config_from_yaml(load_yaml_config(config_path))
    config_holder = {"config": config}
    shell = build_dashboard_shell_spec()
    summary = summarize_config(config)
    session_holder: dict[str, RuntimeSession | None] = {"session": None}

    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0]])
    app.setApplicationName(shell.title)
    app.setFont(QFont("Segoe UI", 10))

    window = QMainWindow()
    window.setWindowTitle(shell.title)
    window.resize(1200, 780)

    root = QWidget()
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(12, 12, 12, 12)
    root_layout.setSpacing(10)

    top_bar = QFrame()
    top_bar.setObjectName("topBar")
    top_bar_layout = QGridLayout(top_bar)
    top_bar_layout.setContentsMargins(10, 8, 10, 8)
    top_bar_layout.setHorizontalSpacing(18)
    top_bar_layout.setVerticalSpacing(6)
    status_widgets: dict[str, QLabel] = {}
    status_rows = (
        (
            ("Mode", summary.mode, 86),
            ("Target", summary.target, 96),
            ("Source", summary.source, 220),
            ("Camera", summary.camera, 190),
            ("Comms", summary.comms, 160),
            ("Laser", summary.laser, 110),
        ),
        (
            ("FPS", "n/a", 86),
            ("Latency", "n/a", 120),
            ("Inference", "n/a", 110),
            ("Tracking", "n/a", 110),
            ("Post", "n/a", 110),
            ("Wait", "n/a", 110),
        ),
        (
            ("Lock", "n/a", 86),
            ("Link", "n/a", 170),
            ("Runtime", "stopped", 140),
        ),
    )
    for row, fields in enumerate(status_rows):
        for column, (label, value, min_width) in enumerate(fields):
            status = _status_label(label, value, min_width=min_width)
            status_widgets[label] = status
            top_bar_layout.addWidget(status, row, column)
    for column in range(6):
        top_bar_layout.setColumnStretch(column, 1)
    root_layout.addWidget(top_bar)

    body = QHBoxLayout()
    body.setSpacing(10)

    sidebar = QListWidget()
    sidebar.setFixedWidth(130)
    for panel in shell.panels:
        sidebar.addItem(panel.title)
    sidebar.setCurrentRow(0)
    body.addWidget(sidebar)

    video_panel = QFrame()
    video_panel.setObjectName("videoPanel")
    video_panel.setFrameShape(QFrame.Shape.StyledPanel)
    video_layout = QVBoxLayout(video_panel)
    video_layout.setContentsMargins(0, 0, 0, 0)
    video_placeholder = QLabel("Video observer standby")
    video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
    video_placeholder.setMinimumSize(480, 270)
    video_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    palette = video_placeholder.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#080b0f"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#d8dee9"))
    video_placeholder.setAutoFillBackground(True)
    video_placeholder.setPalette(palette)
    video_layout.addWidget(video_placeholder)
    body.addWidget(video_panel, 1)

    inspector = QFrame()
    inspector.setObjectName("inspectorPanel")
    inspector.setFixedWidth(420)
    inspector_outer_layout = QVBoxLayout(inspector)
    inspector_outer_layout.setContentsMargins(0, 0, 0, 0)
    inspector_scroll = QScrollArea()
    inspector_scroll.setObjectName("inspectorScroll")
    inspector_scroll.setWidgetResizable(True)
    inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
    inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inspector_contents = QWidget()
    inspector_contents.setObjectName("inspectorContents")
    inspector_layout = QVBoxLayout(inspector_contents)
    inspector_layout.setContentsMargins(10, 10, 10, 10)
    inspector_layout.setSpacing(8)

    inspector_title = QLabel("Inspector")
    inspector_title.setObjectName("panelTitle")
    inspector_layout.addWidget(inspector_title)

    pending_label = QLabel("Settings applied")
    pending_label.setObjectName("statusGood")
    inspector_layout.addWidget(pending_label)

    def make_double_spin(
        value: float,
        *,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int = 2,
        suffix: str = "",
        special_text: str = "",
    ):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        if special_text:
            spin.setSpecialValueText(special_text)
        spin.setValue(value)
        return spin

    launch_title = QLabel("Launch Setup")
    launch_title.setObjectName("panelTitle")
    inspector_layout.addWidget(launch_title)

    launch_settings = launch_settings_from_config(config)
    launch_controls = QGridLayout()
    launch_controls.setHorizontalSpacing(8)
    launch_controls.setVerticalSpacing(6)
    mode_combo = QComboBox()
    mode_combo.addItems(("video", "camera"))
    mode_combo.setCurrentText(launch_settings.mode)
    mode_combo.setToolTip("Choose the runtime input mode used by Start")
    target_combo = QComboBox()
    target_combo.addItems(("human", "dog"))
    target_combo.setCurrentText(launch_settings.target)
    target_combo.setToolTip("Choose the TensorRT model/pose schema used by Start")
    source_input = QLineEdit(launch_settings.source)
    source_input.setToolTip("Video path, stream URL, or camera index such as 0")
    source_browse_button = QPushButton("Browse")
    source_browse_button.setToolTip("Select a video file for video mode")
    camera_width_spin = QSpinBox()
    camera_width_spin.setRange(160, 4096)
    camera_width_spin.setSingleStep(16)
    camera_width_spin.setValue(launch_settings.camera_width)
    camera_width_spin.setToolTip("Frame width after capture letterbox")
    camera_height_spin = QSpinBox()
    camera_height_spin.setRange(160, 4096)
    camera_height_spin.setSingleStep(16)
    camera_height_spin.setValue(launch_settings.camera_height)
    camera_height_spin.setToolTip("Frame height after capture letterbox")
    camera_fps_spin = QSpinBox()
    camera_fps_spin.setRange(1, 240)
    camera_fps_spin.setValue(launch_settings.camera_fps)
    camera_fps_spin.setToolTip("Requested capture FPS and tracking dt baseline")
    source_backend_combo = QComboBox()
    source_backend_combo.addItems(("opencv", "phone_h264", "phone_mpeg", "phone_yuv", "droidcam"))
    source_backend_combo.setCurrentText(launch_settings.camera_source_backend)
    source_backend_combo.setToolTip("Capture source implementation for the next Start")
    backend_combo = QComboBox()
    backend_combo.addItems(("auto", "msmf", "dshow", "ffmpeg"))
    backend_combo.setCurrentText(launch_settings.camera_backend)
    backend_combo.setToolTip("OpenCV capture backend for the next Start")
    camera_fov_spin = QDoubleSpinBox()
    camera_fov_spin.setRange(0.0, 179.0)
    camera_fov_spin.setDecimals(1)
    camera_fov_spin.setSingleStep(0.1)
    camera_fov_spin.setSpecialValueText("auto")
    camera_fov_spin.setSuffix(" deg")
    camera_fov_spin.setValue(launch_settings.camera_fov or 0.0)
    camera_fov_spin.setToolTip("Horizontal camera FOV. Camera mode requires a value above 0.")
    precision_combo = QComboBox()
    precision_combo.addItems(("fp16", "int8"))
    precision_combo.setCurrentText(launch_settings.model_precision)
    precision_combo.setToolTip("TensorRT engine precision to prefer")
    image_size_spin = QSpinBox()
    image_size_spin.setRange(160, 1536)
    image_size_spin.setSingleStep(32)
    image_size_spin.setValue(launch_settings.model_image_size)
    image_size_spin.setToolTip("YOLO model image size")
    comms_checkbox = QCheckBox("Comms")
    comms_checkbox.setChecked(launch_settings.comms_enabled)
    comms_checkbox.setToolTip("Enable serial/UDP output to ESP32 for the next Start")
    apply_launch_button = QPushButton("Apply")
    apply_launch_button.setToolTip("Apply launch settings while the runtime is stopped")
    for compact_widget in (
        mode_combo,
        target_combo,
        camera_width_spin,
        camera_height_spin,
        camera_fps_spin,
        source_backend_combo,
        backend_combo,
        camera_fov_spin,
        precision_combo,
        image_size_spin,
    ):
        compact_widget.setMinimumWidth(74)
        compact_widget.setMaximumWidth(96)
    source_browse_button.setMaximumWidth(82)
    launch_controls.addWidget(QLabel("Mode"), 0, 0)
    launch_controls.addWidget(mode_combo, 0, 1)
    launch_controls.addWidget(QLabel("Target"), 0, 2)
    launch_controls.addWidget(target_combo, 0, 3)
    launch_controls.addWidget(QLabel("Source"), 1, 0)
    launch_controls.addWidget(source_input, 1, 1, 1, 2)
    launch_controls.addWidget(source_browse_button, 1, 3)
    launch_controls.addWidget(QLabel("Width"), 2, 0)
    launch_controls.addWidget(camera_width_spin, 2, 1)
    launch_controls.addWidget(QLabel("Height"), 2, 2)
    launch_controls.addWidget(camera_height_spin, 2, 3)
    launch_controls.addWidget(QLabel("FPS"), 3, 0)
    launch_controls.addWidget(camera_fps_spin, 3, 1)
    launch_controls.addWidget(QLabel("Source backend"), 3, 2)
    launch_controls.addWidget(source_backend_combo, 3, 3)
    launch_controls.addWidget(QLabel("Backend"), 4, 0)
    launch_controls.addWidget(backend_combo, 4, 1)
    launch_controls.addWidget(QLabel("FOV"), 4, 2)
    launch_controls.addWidget(camera_fov_spin, 4, 3)
    launch_controls.addWidget(QLabel("Precision"), 5, 0)
    launch_controls.addWidget(precision_combo, 5, 1)
    launch_controls.addWidget(QLabel("Image"), 5, 2)
    launch_controls.addWidget(image_size_spin, 5, 3)
    launch_controls.addWidget(comms_checkbox, 6, 0, 1, 2)
    launch_controls.addWidget(apply_launch_button, 6, 3)
    inspector_layout.addLayout(launch_controls)

    tuning_title = QLabel("Tracking And Gimbal Tuning")
    tuning_title.setObjectName("panelTitle")
    inspector_layout.addWidget(tuning_title)

    tuning_settings = tuning_settings_from_config(config)
    tuning_controls = QGridLayout()
    tuning_controls.setHorizontalSpacing(8)
    tuning_controls.setVerticalSpacing(6)
    confidence_spin = make_double_spin(
        tuning_settings.tracking_confidence_threshold,
        minimum=0.01,
        maximum=1.0,
        step=0.01,
    )
    confidence_spin.setToolTip("Minimum pose detection confidence")
    hold_spin = make_double_spin(
        tuning_settings.tracking_hold_time_s,
        minimum=0.0,
        maximum=5.0,
        step=0.05,
        suffix=" s",
    )
    hold_spin.setToolTip("How long to hold target state through brief losses")
    kp_spin = make_double_spin(tuning_settings.gimbal_kp, minimum=0.0, maximum=20.0, step=0.05)
    ki_spin = make_double_spin(tuning_settings.gimbal_ki, minimum=0.0, maximum=5.0, step=0.01)
    kd_spin = make_double_spin(tuning_settings.gimbal_kd, minimum=0.0, maximum=20.0, step=0.05)
    deadband_spin = make_double_spin(
        tuning_settings.gimbal_deadband_deg,
        minimum=0.0,
        maximum=30.0,
        step=0.1,
        suffix=" deg",
    )
    slew_spin = make_double_spin(
        tuning_settings.gimbal_slew_limit_dps,
        minimum=0.0,
        maximum=500.0,
        step=1.0,
        suffix=" dps",
    )
    kp_near_spin = make_double_spin(
        tuning_settings.gimbal_kp_near or 0.0,
        minimum=0.0,
        maximum=20.0,
        step=0.05,
        special_text="auto",
    )
    kp_far_spin = make_double_spin(
        tuning_settings.gimbal_kp_far or 0.0,
        minimum=0.0,
        maximum=20.0,
        step=0.05,
        special_text="auto",
    )
    lead_spin = make_double_spin(
        tuning_settings.gimbal_predictive_lead_s,
        minimum=0.0,
        maximum=2.0,
        step=0.01,
        suffix=" s",
    )
    pan_offset_spin = make_double_spin(
        tuning_settings.boresight_pan_offset_deg,
        minimum=-45.0,
        maximum=45.0,
        step=0.05,
        suffix=" deg",
    )
    tilt_offset_spin = make_double_spin(
        tuning_settings.boresight_tilt_offset_deg,
        minimum=-45.0,
        maximum=45.0,
        step=0.05,
        suffix=" deg",
    )
    relay_pulse_spin = QSpinBox()
    relay_pulse_spin.setRange(0, 5000)
    relay_pulse_spin.setSingleStep(25)
    relay_pulse_spin.setSuffix(" ms")
    relay_pulse_spin.setValue(tuning_settings.relay_pulse_ms)
    laser_startup_checkbox = QCheckBox("Laser startup")
    laser_startup_checkbox.setChecked(tuning_settings.laser_startup_enabled)
    apply_tuning_button = QPushButton("Apply Tuning")
    apply_tuning_button.setToolTip("Apply tuning for the next runtime start")
    for compact_widget in (
        confidence_spin,
        hold_spin,
        kp_spin,
        ki_spin,
        kd_spin,
        deadband_spin,
        slew_spin,
        kp_near_spin,
        kp_far_spin,
        lead_spin,
        pan_offset_spin,
        tilt_offset_spin,
        relay_pulse_spin,
    ):
        compact_widget.setMinimumWidth(74)
        compact_widget.setMaximumWidth(96)
    tuning_controls.addWidget(QLabel("Conf"), 0, 0)
    tuning_controls.addWidget(confidence_spin, 0, 1)
    tuning_controls.addWidget(QLabel("Hold"), 0, 2)
    tuning_controls.addWidget(hold_spin, 0, 3)
    tuning_controls.addWidget(QLabel("Kp"), 1, 0)
    tuning_controls.addWidget(kp_spin, 1, 1)
    tuning_controls.addWidget(QLabel("Ki"), 1, 2)
    tuning_controls.addWidget(ki_spin, 1, 3)
    tuning_controls.addWidget(QLabel("Kd"), 2, 0)
    tuning_controls.addWidget(kd_spin, 2, 1)
    tuning_controls.addWidget(QLabel("Deadband"), 2, 2)
    tuning_controls.addWidget(deadband_spin, 2, 3)
    tuning_controls.addWidget(QLabel("Slew"), 3, 0)
    tuning_controls.addWidget(slew_spin, 3, 1)
    tuning_controls.addWidget(QLabel("Lead"), 3, 2)
    tuning_controls.addWidget(lead_spin, 3, 3)
    tuning_controls.addWidget(QLabel("Near Kp"), 4, 0)
    tuning_controls.addWidget(kp_near_spin, 4, 1)
    tuning_controls.addWidget(QLabel("Far Kp"), 4, 2)
    tuning_controls.addWidget(kp_far_spin, 4, 3)
    tuning_controls.addWidget(QLabel("Pan off"), 5, 0)
    tuning_controls.addWidget(pan_offset_spin, 5, 1)
    tuning_controls.addWidget(QLabel("Tilt off"), 5, 2)
    tuning_controls.addWidget(tilt_offset_spin, 5, 3)
    tuning_controls.addWidget(QLabel("Relay"), 6, 0)
    tuning_controls.addWidget(relay_pulse_spin, 6, 1)
    tuning_controls.addWidget(laser_startup_checkbox, 6, 2)
    tuning_controls.addWidget(apply_tuning_button, 6, 3)
    inspector_layout.addLayout(tuning_controls)

    controls = QGridLayout()
    controls.setHorizontalSpacing(6)
    controls.setVerticalSpacing(6)
    start_button = QPushButton("Start")
    start_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    start_button.setToolTip("Start the tracker runtime")
    stop_button = QPushButton("Stop")
    stop_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
    stop_button.setToolTip("Stop the tracker runtime")
    emergency_button = QPushButton("E-Stop")
    emergency_button.setIcon(
        window.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
    )
    emergency_button.setToolTip("Emergency stop: laser off, relay off, request shutdown")
    emergency_button.setObjectName("emergencyButton")
    relock_button = QPushButton("Relock")
    relock_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
    relock_button.setToolTip("Release the current target lock")
    target_button = QPushButton("Cycle")
    target_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
    target_button.setToolTip("Cycle target selection")
    laser_button = QPushButton("Laser")
    laser_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
    laser_button.setToolTip("Enable or disable laser output")
    manual_button = QPushButton("Manual")
    manual_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
    manual_button.setToolTip("Toggle manual gimbal mode")
    for button in (
        start_button,
        stop_button,
        emergency_button,
        relock_button,
        target_button,
        laser_button,
        manual_button,
    ):
        button.setMinimumWidth(92)
        button.setMinimumHeight(32)
    controls.addWidget(start_button, 0, 0)
    controls.addWidget(stop_button, 0, 1)
    controls.addWidget(emergency_button, 1, 0, 1, 2)
    controls.addWidget(relock_button, 2, 0)
    controls.addWidget(target_button, 2, 1)
    controls.addWidget(laser_button, 3, 0)
    controls.addWidget(manual_button, 3, 1)
    inspector_layout.addLayout(controls)

    diagnostics_title = QLabel("Live Diagnostics")
    diagnostics_title.setObjectName("panelTitle")
    inspector_layout.addWidget(diagnostics_title)

    diagnostics_table = QTableWidget(0, 2)
    diagnostics_table.setMaximumHeight(220)
    diagnostics_table.setHorizontalHeaderLabels(("Metric", "Value"))
    diagnostics_table.verticalHeader().setVisible(False)
    diagnostics_table.horizontalHeader().setStretchLastSection(True)
    inspector_layout.addWidget(diagnostics_table)
    _populate_table(
        diagnostics_table,
        ((row.metric, row.value) for row in build_runtime_diagnostic_rows(None)),
    )

    processes_title = QLabel("Process Health")
    processes_title.setObjectName("panelTitle")
    inspector_layout.addWidget(processes_title)

    process_table = QTableWidget(0, 4)
    process_table.setMaximumHeight(130)
    process_table.setHorizontalHeaderLabels(("Process", "PID", "State", "Restarts"))
    process_table.verticalHeader().setVisible(False)
    process_table.horizontalHeader().setStretchLastSection(True)
    inspector_layout.addWidget(process_table)
    _populate_process_table(process_table, ())

    controls_title = QLabel("Control Plane")
    controls_title.setObjectName("panelTitle")
    inspector_layout.addWidget(controls_title)

    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(("Control", "Path", "Mode"))
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    for control in runtime_control_specs():
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(control.label))
        table.setItem(row, 1, QTableWidgetItem(control.path))
        mode = "restart" if control.requires_restart else control.scope
        if control.safety_critical:
            mode = f"{mode} safety"
        table.setItem(row, 2, QTableWidgetItem(mode))
    inspector_layout.addWidget(table, 1)
    inspector_scroll.setWidget(inspector_contents)
    inspector_outer_layout.addWidget(inspector_scroll)
    body.addWidget(inspector)
    root_layout.addLayout(body, 1)

    console = QTextEdit()
    console.setObjectName("consolePanel")
    console.setReadOnly(True)
    console.setMaximumHeight(130)
    console.append(f"Loaded {config_path}")
    console.append("Runtime bridge ready. Choose launch setup, then Start.")
    root_layout.addWidget(console)

    def log_action(action: str) -> None:
        console.append(action)

    def current_session() -> RuntimeSession | None:
        return session_holder["session"]

    def set_settings_status(text: str, object_name: str = "statusGood") -> None:
        pending_label.setText(text)
        pending_label.setObjectName(object_name)
        pending_label.style().unpolish(pending_label)
        pending_label.style().polish(pending_label)

    def mark_settings_pending() -> None:
        if current_session() is None:
            set_settings_status("Pending changes", "statusWarn")
        else:
            set_settings_status("Pending live tuning", "statusWarn")

    def set_runtime_buttons_enabled(running: bool) -> None:
        start_button.setEnabled(not running)
        stop_button.setEnabled(running)
        emergency_button.setEnabled(running)
        relock_button.setEnabled(running)
        target_button.setEnabled(running)
        laser_button.setEnabled(running)
        manual_button.setEnabled(running)

    def set_launch_controls_enabled(enabled: bool) -> None:
        mode_combo.setEnabled(enabled)
        target_combo.setEnabled(enabled)
        source_input.setEnabled(enabled)
        source_browse_button.setEnabled(enabled)
        camera_width_spin.setEnabled(enabled)
        camera_height_spin.setEnabled(enabled)
        camera_fps_spin.setEnabled(enabled)
        source_backend_combo.setEnabled(enabled)
        backend_combo.setEnabled(enabled)
        camera_fov_spin.setEnabled(enabled)
        precision_combo.setEnabled(enabled)
        image_size_spin.setEnabled(enabled)
        comms_checkbox.setEnabled(enabled)
        apply_launch_button.setEnabled(enabled)

    def set_tuning_controls_enabled(enabled: bool) -> None:
        confidence_spin.setEnabled(enabled)
        hold_spin.setEnabled(enabled)
        kp_spin.setEnabled(enabled)
        ki_spin.setEnabled(enabled)
        kd_spin.setEnabled(enabled)
        deadband_spin.setEnabled(enabled)
        slew_spin.setEnabled(enabled)
        kp_near_spin.setEnabled(enabled)
        kp_far_spin.setEnabled(enabled)
        lead_spin.setEnabled(enabled)
        pan_offset_spin.setEnabled(enabled)
        tilt_offset_spin.setEnabled(enabled)
        relay_pulse_spin.setEnabled(enabled)
        laser_startup_checkbox.setEnabled(enabled)
        apply_tuning_button.setEnabled(enabled)

    def read_launch_settings() -> LaunchSettings:
        mode = "camera" if mode_combo.currentText() == "camera" else "video"
        camera_fov = camera_fov_spin.value()
        return LaunchSettings(
            mode=mode,
            target="dog" if target_combo.currentText() == "dog" else "human",
            source=source_input.text(),
            comms_enabled=comms_checkbox.isChecked(),
            camera_width=camera_width_spin.value(),
            camera_height=camera_height_spin.value(),
            camera_fps=camera_fps_spin.value(),
            camera_source_backend=source_backend_combo.currentText(),
            camera_backend=backend_combo.currentText(),
            camera_fov=camera_fov if camera_fov > 0.0 else None,
            model_precision="int8" if precision_combo.currentText() == "int8" else "fp16",
            model_image_size=image_size_spin.value(),
        )

    def read_tuning_settings() -> TuningSettings:
        kp_near = kp_near_spin.value()
        kp_far = kp_far_spin.value()
        return TuningSettings(
            tracking_confidence_threshold=confidence_spin.value(),
            tracking_hold_time_s=hold_spin.value(),
            gimbal_kp=kp_spin.value(),
            gimbal_ki=ki_spin.value(),
            gimbal_kd=kd_spin.value(),
            gimbal_deadband_deg=deadband_spin.value(),
            gimbal_slew_limit_dps=slew_spin.value(),
            gimbal_kp_near=kp_near if kp_near > 0.0 else None,
            gimbal_kp_far=kp_far if kp_far > 0.0 else None,
            gimbal_predictive_lead_s=lead_spin.value(),
            laser_startup_enabled=laser_startup_checkbox.isChecked(),
            relay_pulse_ms=relay_pulse_spin.value(),
            boresight_pan_offset_deg=pan_offset_spin.value(),
            boresight_tilt_offset_deg=tilt_offset_spin.value(),
        )

    def refresh_config_status() -> None:
        latest_summary = summarize_config(config_holder["config"])
        _set_status_label(status_widgets["Mode"], "Mode", latest_summary.mode)
        _set_status_label(status_widgets["Target"], "Target", latest_summary.target)
        _set_status_label(status_widgets["Source"], "Source", latest_summary.source)
        _set_status_label(status_widgets["Camera"], "Camera", latest_summary.camera)
        _set_status_label(status_widgets["Comms"], "Comms", latest_summary.comms)
        _set_status_label(status_widgets["Laser"], "Laser", latest_summary.laser)

    def sync_launch_controls_from_config() -> None:
        current_settings = launch_settings_from_config(config_holder["config"])
        mode_combo.setCurrentText(current_settings.mode)
        target_combo.setCurrentText(current_settings.target)
        source_input.setText(current_settings.source)
        camera_width_spin.setValue(current_settings.camera_width)
        camera_height_spin.setValue(current_settings.camera_height)
        camera_fps_spin.setValue(current_settings.camera_fps)
        source_backend_combo.setCurrentText(current_settings.camera_source_backend)
        backend_combo.setCurrentText(current_settings.camera_backend)
        camera_fov_spin.setValue(current_settings.camera_fov or 0.0)
        precision_combo.setCurrentText(current_settings.model_precision)
        image_size_spin.setValue(current_settings.model_image_size)
        comms_checkbox.setChecked(current_settings.comms_enabled)

    def sync_tuning_controls_from_config() -> None:
        current_settings = tuning_settings_from_config(config_holder["config"])
        confidence_spin.setValue(current_settings.tracking_confidence_threshold)
        hold_spin.setValue(current_settings.tracking_hold_time_s)
        kp_spin.setValue(current_settings.gimbal_kp)
        ki_spin.setValue(current_settings.gimbal_ki)
        kd_spin.setValue(current_settings.gimbal_kd)
        deadband_spin.setValue(current_settings.gimbal_deadband_deg)
        slew_spin.setValue(current_settings.gimbal_slew_limit_dps)
        kp_near_spin.setValue(current_settings.gimbal_kp_near or 0.0)
        kp_far_spin.setValue(current_settings.gimbal_kp_far or 0.0)
        lead_spin.setValue(current_settings.gimbal_predictive_lead_s)
        laser_startup_checkbox.setChecked(current_settings.laser_startup_enabled)
        relay_pulse_spin.setValue(current_settings.relay_pulse_ms)
        pan_offset_spin.setValue(current_settings.boresight_pan_offset_deg)
        tilt_offset_spin.setValue(current_settings.boresight_tilt_offset_deg)

    def apply_launch_setup(*, silent: bool = False) -> bool:
        if current_session() is not None:
            sync_launch_controls_from_config()
            log_action("launch setup: stop runtime before changing startup parameters")
            return False
        launch_settings = read_launch_settings()
        validation_errors = validate_launch_settings(launch_settings)
        if validation_errors:
            set_settings_status("Invalid settings", "statusBad")
            log_action(f"launch setup invalid: {'; '.join(validation_errors)}")
            return False
        config_holder["config"] = apply_launch_settings(
            config_holder["config"],
            launch_settings,
        )
        sync_launch_controls_from_config()
        refresh_config_status()
        set_settings_status("Settings applied")
        summary = summarize_config(config_holder["config"])
        if not silent:
            log_action(
                f"launch setup applied: {summary.mode} source={summary.source} camera={summary.camera}"
            )
        return True

    def apply_tuning_setup(*, silent: bool = False) -> bool:
        settings = read_tuning_settings()
        session = current_session()
        config_holder["config"] = apply_tuning_settings(config_holder["config"], settings)
        if session is not None:
            version = session.apply_runtime_tuning(settings)
            refresh_config_status()
            set_settings_status(f"Runtime tuning queued v{version}", "statusWarn")
            if not silent:
                log_action(
                    f"runtime tuning queued v{version}: output/gimbal controls live; inference confidence applies next restart"
                )
            return True
        sync_tuning_controls_from_config()
        refresh_config_status()
        set_settings_status("Settings applied")
        if not silent:
            log_action("tuning setup applied")
        return True

    def apply_all_startup_settings(*, silent: bool = False) -> bool:
        if not apply_launch_setup(silent=True):
            return False
        if not apply_tuning_setup(silent=True):
            return False
        if not silent:
            log_action("all startup settings applied")
        return True

    def browse_source() -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            window,
            "Select video source",
            str(Path(source_input.text()).parent if source_input.text() else Path("videos")),
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.wmv);;All files (*)",
        )
        if path:
            try:
                source_input.setText(str(Path(path).resolve().relative_to(Path.cwd())))
            except ValueError:
                source_input.setText(path)
            mode_combo.setCurrentText("video")
            mark_settings_pending()

    def handle_mode_change(mode_text: str) -> None:
        mode = "camera" if mode_text == "camera" else "video"
        suggested_source = suggest_source_for_mode(mode, source_input.text())
        if suggested_source != source_input.text().strip():
            source_input.setText(suggested_source)

    def ensure_session() -> RuntimeSession:
        session = current_session()
        if session is None:
            session = RuntimeSession(config_holder["config"])
            session_holder["session"] = session
        return session

    def set_runtime_status(value: str) -> None:
        _set_status_label(status_widgets["Runtime"], "Runtime", value)

    def start_runtime() -> None:
        if current_session() is None and not apply_all_startup_settings(silent=True):
            return
        session = ensure_session()
        try:
            snapshot = session.start()
        except Exception as exc:
            log_action(f"start failed: {exc}")
            return
        set_runtime_status(_snapshot_status(snapshot))
        set_launch_controls_enabled(False)
        set_runtime_buttons_enabled(True)
        _populate_process_table(process_table, snapshot.process_states)
        summary = summarize_config(config_holder["config"])
        log_action(f"runtime started: {summary.mode} source={summary.source}")

    def stop_runtime() -> None:
        session = current_session()
        if session is None:
            log_action("stop: runtime bridge not connected")
            return
        snapshot = session.stop(join_timeout=1.0, terminate_timeout=0.5)
        set_runtime_status(_snapshot_status(snapshot))
        log_action("runtime stopped")
        session_holder["session"] = None
        set_launch_controls_enabled(True)
        set_runtime_buttons_enabled(False)
        _populate_process_table(process_table, ())
        video_placeholder.setPixmap(QPixmap())
        video_placeholder.setText("Video observer standby")

    def emergency_stop_runtime() -> None:
        session = current_session()
        if session is None:
            log_action("emergency_stop: runtime bridge not connected")
            return
        snapshot = session.emergency_stop()
        set_runtime_status(_snapshot_status(snapshot))
        _populate_process_table(process_table, snapshot.process_states)
        log_action("emergency stop asserted: laser off, relay off, shutdown requested")

    def request_relock() -> None:
        session = current_session()
        if session is None:
            log_action("relock: runtime bridge not connected")
            return
        session.request_relock()
        log_action("relock requested")

    def request_target_cycle() -> None:
        session = current_session()
        if session is None:
            log_action("cycle target: runtime bridge not connected")
            return
        session.request_cycle_target()
        log_action("target cycle requested")

    def toggle_laser() -> None:
        session = current_session()
        if session is None:
            log_action("laser toggle: runtime bridge not connected")
            return
        enabled = session.toggle_laser()
        log_action(f"laser {'enabled' if enabled else 'disabled'}")

    def toggle_manual() -> None:
        session = current_session()
        if session is None:
            log_action("manual toggle: runtime bridge not connected")
            return
        enabled = session.toggle_manual_mode()
        log_action(f"manual mode {'enabled' if enabled else 'disabled'}")

    def poll_runtime() -> None:
        session = current_session()
        if session is None:
            return
        snapshot = session.poll()
        set_runtime_status(_snapshot_status(snapshot))
        set_runtime_buttons_enabled(snapshot.running)
        if (
            snapshot.runtime_control_version > 0
            and pending_label.text().startswith("Runtime tuning")
        ):
            if snapshot.runtime_control_ack_version >= snapshot.runtime_control_version:
                set_settings_status(
                    f"Runtime tuning active v{snapshot.runtime_control_ack_version}",
                    "statusGood",
                )
            else:
                set_settings_status(
                    f"Runtime tuning queued v{snapshot.runtime_control_version}",
                    "statusWarn",
                )
        if snapshot.telemetry is not None:
            _set_status_label(status_widgets["FPS"], "FPS", f"{snapshot.telemetry.fps:.1f}")
            _set_status_label(
                status_widgets["Latency"],
                "Latency",
                f"{snapshot.telemetry.total_latency_ms:.1f} ms",
            )
            _set_status_label(
                status_widgets["Inference"],
                "Inference",
                f"{snapshot.telemetry.inference_ms:.1f} ms",
            )
            _set_status_label(
                status_widgets["Tracking"],
                "Tracking",
                f"{snapshot.telemetry.tracking_ms:.1f} ms",
            )
            _set_status_label(
                status_widgets["Post"],
                "Post",
                f"{snapshot.telemetry.postprocess_ms:.1f} ms",
            )
            _set_status_label(
                status_widgets["Wait"],
                "Wait",
                f"{snapshot.telemetry.wait_ms:.1f} ms",
            )
            lock_pct = (
                snapshot.telemetry.lock_frames / max(1, snapshot.telemetry.total_frames)
            ) * 100.0
            _set_status_label(status_widgets["Lock"], "Lock", f"{lock_pct:.0f}%")
            _set_status_label(status_widgets["Link"], "Link", snapshot.telemetry.transport_status)
            _populate_table(
                diagnostics_table,
                (
                    (row.metric, row.value)
                    for row in build_runtime_diagnostic_rows(snapshot.telemetry)
                ),
            )
        _populate_process_table(process_table, snapshot.process_states)
        for error in snapshot.error_summaries:
            if error not in console.toPlainText():
                log_action(f"runtime error: {error}")
        if not snapshot.running and not snapshot.stopping:
            session_holder["session"] = None
            set_launch_controls_enabled(True)
            set_runtime_buttons_enabled(False)

    def render_display_frame() -> None:
        session = current_session()
        if session is None:
            return
        frame = session.read_display_frame()
        if frame is None:
            return
        pixmap = _frame_to_pixmap(frame)
        video_placeholder.setText("")
        video_placeholder.setPixmap(
            pixmap.scaled(
                video_placeholder.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    start_button.clicked.connect(start_runtime)
    stop_button.clicked.connect(stop_runtime)
    emergency_button.clicked.connect(emergency_stop_runtime)
    relock_button.clicked.connect(request_relock)
    target_button.clicked.connect(request_target_cycle)
    laser_button.clicked.connect(toggle_laser)
    manual_button.clicked.connect(toggle_manual)
    apply_launch_button.clicked.connect(lambda: apply_launch_setup())
    apply_tuning_button.clicked.connect(lambda: apply_tuning_setup())
    source_browse_button.clicked.connect(browse_source)
    mode_combo.currentTextChanged.connect(handle_mode_change)
    for signal in (
        mode_combo.currentTextChanged,
        target_combo.currentTextChanged,
        source_input.textChanged,
        camera_width_spin.valueChanged,
        camera_height_spin.valueChanged,
        camera_fps_spin.valueChanged,
        source_backend_combo.currentTextChanged,
        backend_combo.currentTextChanged,
        camera_fov_spin.valueChanged,
        precision_combo.currentTextChanged,
        image_size_spin.valueChanged,
        comms_checkbox.stateChanged,
        confidence_spin.valueChanged,
        hold_spin.valueChanged,
        kp_spin.valueChanged,
        ki_spin.valueChanged,
        kd_spin.valueChanged,
        deadband_spin.valueChanged,
        slew_spin.valueChanged,
        kp_near_spin.valueChanged,
        kp_far_spin.valueChanged,
        lead_spin.valueChanged,
        pan_offset_spin.valueChanged,
        tilt_offset_spin.valueChanged,
        relay_pulse_spin.valueChanged,
        laser_startup_checkbox.stateChanged,
    ):
        signal.connect(lambda *_args: mark_settings_pending())
    set_runtime_buttons_enabled(False)
    set_tuning_controls_enabled(True)

    status_timer = QTimer(window)
    status_timer.setInterval(250)
    status_timer.timeout.connect(poll_runtime)
    status_timer.start()

    frame_timer = QTimer(window)
    frame_timer.setInterval(16)
    frame_timer.timeout.connect(render_display_frame)
    frame_timer.start()

    def cleanup_runtime() -> None:
        session = current_session()
        if session is not None:
            session.stop(join_timeout=1.0, terminate_timeout=0.5)

    app.aboutToQuit.connect(cleanup_runtime)

    window.setCentralWidget(root)
    window.setStyleSheet(APP_STYLESHEET)
    window.show()
    return int(app.exec())


def _status_label(label: str, value: str, *, min_width: int = 110):
    from PySide6.QtWidgets import QLabel

    widget = QLabel()
    widget.setMinimumWidth(min_width)
    _set_status_label(widget, label, value)
    return widget


def _set_status_label(widget, label: str, value: str) -> None:
    html_value = value.replace("\n", "<br>")
    widget.setText(f"<b>{label}</b><br>{html_value}")
    widget.setToolTip(f"{label}: {value}")


def _populate_table(table, rows) -> None:
    from PySide6.QtWidgets import QTableWidgetItem

    row_values = tuple(rows)
    table.setRowCount(len(row_values))
    for row, (left, right) in enumerate(row_values):
        table.setItem(row, 0, QTableWidgetItem(left))
        table.setItem(row, 1, QTableWidgetItem(right))


def _populate_process_table(table, process_states) -> None:
    from PySide6.QtWidgets import QTableWidgetItem

    states = tuple(process_states)
    table.setRowCount(len(states))
    for row, state in enumerate(states):
        table.setItem(row, 0, QTableWidgetItem(state.name))
        table.setItem(row, 1, QTableWidgetItem("n/a" if state.pid is None else str(state.pid)))
        table.setItem(row, 2, QTableWidgetItem("alive" if state.alive else "stopped"))
        table.setItem(row, 3, QTableWidgetItem(str(state.restarts)))


def _frame_to_pixmap(frame):
    import numpy as np
    from PySide6.QtGui import QImage, QPixmap

    if not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
    height, width, channels = frame.shape
    bytes_per_line = channels * width
    image = QImage(
        frame.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_BGR888,
    ).copy()
    return QPixmap.fromImage(image)


def _snapshot_status(snapshot) -> str:
    if snapshot.stopping:
        return "stopping"
    if not snapshot.running:
        return "stopped"
    alive = sum(1 for process in snapshot.process_states if process.alive)
    total = len(snapshot.process_states)
    return f"running {alive}/{total}"


if __name__ == "__main__":
    raise SystemExit(main())
