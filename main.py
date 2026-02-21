import csv
import importlib.util
import math
import os
import re
import sys
import time
from typing import Any, List, Optional, Tuple

TIME_PATTERN = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)(?:[.,](\d{1,2}))?$")
CPU_REQUIREMENTS_FILE = "requirements.cpu.txt"
GPU_CUDA_REQUIREMENTS_FILE = "requirements.gpu.cuda.txt"
GPU_DIRECTML_REQUIREMENTS_FILE = "requirements.gpu.directml.txt"
CUDA_PROVIDER_NAME = "CUDAExecutionProvider"
DIRECTML_PROVIDER_NAME = "DmlExecutionProvider"
PROCESSOR_MODE_AUTO = "Auto (Prefer DirectML, then CUDA)"
PROCESSOR_MODE_CPU = "CPU"
PROCESSOR_MODE_GPU_DIRECTML = "GPU (DirectML)"
PROCESSOR_MODE_GPU_CUDA = "GPU (CUDA)"
OCR_BACKEND_CPU = "cpu"
OCR_BACKEND_DIRECTML = "directml"
OCR_BACKEND_CUDA = "cuda"
BOOTSTRAP_REQUIRED_MODULES = (
    ("PySide6", "PySide6"),
    ("cv2", "opencv-python"),
    ("rapidocr", "rapidocr"),
    ("onnxruntime", "onnxruntime/onnxruntime-directml/onnxruntime-gpu"),
)


def _bootstrap_application_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _bootstrap_missing_modules() -> List[Tuple[str, str]]:
    missing: List[Tuple[str, str]] = []
    for module_name, package_name in BOOTSTRAP_REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, package_name))
    return missing


def _bootstrap_install_instructions_text(missing_module_names: Optional[List[str]] = None) -> str:
    base_dir = _bootstrap_application_base_dir()
    install_ps1 = os.path.join(base_dir, "install.ps1")
    install_bat = os.path.join(base_dir, "install.bat")
    cpu_req = os.path.join(base_dir, CPU_REQUIREMENTS_FILE)
    cuda_req = os.path.join(base_dir, GPU_CUDA_REQUIREMENTS_FILE)
    directml_req = os.path.join(base_dir, GPU_DIRECTML_REQUIREMENTS_FILE)

    lines: List[str] = []
    if missing_module_names:
        lines.append(
            "Missing required packages for GUI startup: "
            + ", ".join(missing_module_names)
            + "."
        )
        lines.append("")

    lines.extend(
        [
            "Install dependencies from this project folder with one of these commands:",
            f'powershell -NoProfile -ExecutionPolicy Bypass -File "{install_ps1}"',
            f'"{install_bat}"',
            "",
            "Manual commands (if needed):",
            f'{sys.executable} -m pip install -r "{directml_req}"',
            f'{sys.executable} -m pip install -r "{cuda_req}"',
            f'{sys.executable} -m pip install -r "{cpu_req}"',
        ]
    )
    return "\n".join(lines)


def _bootstrap_pause_before_exit() -> None:
    if sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass


def _bootstrap_show_install_failure_dialog(message: str) -> None:
    if sys.stdin and sys.stdin.isatty():
        return

    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return

    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        messagebox.showerror("Dependency Install Failed", message)
        root.destroy()
    except Exception:
        return


def _bootstrap_dependencies_if_needed() -> None:
    missing_modules = _bootstrap_missing_modules()
    if not missing_modules:
        return

    message = _bootstrap_install_instructions_text([package for _, package in missing_modules])
    print(f"\n{message}")
    _bootstrap_show_install_failure_dialog(message)
    _bootstrap_pause_before_exit()
    raise SystemExit(1)


_bootstrap_dependencies_if_needed()

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class DualHandleSlider(QWidget):
    valuesChanged = Signal(int, int)
    valuesChangeFinished = Signal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._low = 0
        self._high = 100
        self._active_handle: Optional[str] = None
        self._handle_radius = 8
        self._groove_height = 6
        self.setMinimumHeight(36)

    def setRange(self, minimum: int, maximum: int) -> None:
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        self.setValues(self._low, self._high)

    def setValues(self, low: int, high: int, emit_signal: bool = True) -> None:
        low = max(self._minimum, min(int(low), self._maximum))
        high = max(self._minimum, min(int(high), self._maximum))
        if low > high:
            low = high

        changed = (low != self._low) or (high != self._high)
        self._low = low
        self._high = high
        self.update()

        if changed and emit_signal:
            self.valuesChanged.emit(self._low, self._high)

    def values(self) -> Tuple[int, int]:
        return self._low, self._high

    def _groove_left(self) -> float:
        return float(self._handle_radius)

    def _groove_right(self) -> float:
        return float(self.width() - self._handle_radius)

    def _value_to_pos(self, value: int) -> float:
        if self._maximum == self._minimum:
            return self._groove_left()
        span = self._maximum - self._minimum
        ratio = (value - self._minimum) / span
        return self._groove_left() + ratio * (self._groove_right() - self._groove_left())

    def _pos_to_value(self, x: float) -> int:
        left = self._groove_left()
        right = self._groove_right()
        if right <= left:
            return self._minimum
        ratio = (x - left) / (right - left)
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        center_y = self.height() / 2.0
        left = self._groove_left()
        right = self._groove_right()

        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.gray)
        painter.drawRoundedRect(
            QRectF(left, center_y - self._groove_height / 2.0, right - left, self._groove_height),
            3,
            3,
        )

        low_x = self._value_to_pos(self._low)
        high_x = self._value_to_pos(self._high)
        painter.setBrush(Qt.darkCyan)
        painter.drawRoundedRect(
            QRectF(low_x, center_y - self._groove_height / 2.0, high_x - low_x, self._groove_height),
            3,
            3,
        )

        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(QPointF(low_x, center_y), self._handle_radius, self._handle_radius)
        painter.drawEllipse(QPointF(high_x, center_y), self._handle_radius, self._handle_radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        low_x = self._value_to_pos(self._low)
        high_x = self._value_to_pos(self._high)
        x = event.position().x()

        if abs(x - low_x) <= abs(x - high_x):
            self._active_handle = "low"
        else:
            self._active_handle = "high"
        self._move_active_handle(x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._active_handle:
            return
        self._move_active_handle(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._active_handle:
            self._active_handle = None
            self.valuesChangeFinished.emit(self._low, self._high)

    def _move_active_handle(self, x: float) -> None:
        value = self._pos_to_value(x)
        if self._active_handle == "low":
            self.setValues(value, self._high)
        elif self._active_handle == "high":
            self.setValues(self._low, value)

class VideoCanvas(QWidget):
    roiChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._roi_norm: Optional[QRectF] = None
        self._drag_start: Optional[QPointF] = None
        self._drawing_roi = False
        self.setMinimumSize(720, 405)

    def set_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def clear_frame(self) -> None:
        self._pixmap = None
        self.update()

    def has_roi(self) -> bool:
        return self._roi_norm is not None

    def clear_roi(self) -> None:
        self._roi_norm = None
        self.update()
        self.roiChanged.emit()

    def roi_in_frame(self, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        if not self._roi_norm:
            return None

        left = int(round(self._roi_norm.left() * width))
        right = int(round(self._roi_norm.right() * width))
        top = int(round(self._roi_norm.top() * height))
        bottom = int(round(self._roi_norm.bottom() * height))

        left = max(0, min(left, width - 1))
        right = max(1, min(right, width))
        top = max(0, min(top, height - 1))
        bottom = max(1, min(bottom, height))

        roi_width = max(1, right - left)
        roi_height = max(1, bottom - top)
        return left, top, roi_width, roi_height

    def _display_rect(self) -> QRectF:
        if not self._pixmap or self._pixmap.isNull():
            return QRectF()

        widget_w = float(self.width())
        widget_h = float(self.height())
        pixmap_w = float(self._pixmap.width())
        pixmap_h = float(self._pixmap.height())

        scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
        draw_w = pixmap_w * scale
        draw_h = pixmap_h * scale
        draw_x = (widget_w - draw_w) / 2.0
        draw_y = (widget_h - draw_h) / 2.0
        return QRectF(draw_x, draw_y, draw_w, draw_h)

    def _widget_to_norm(self, point: QPointF) -> QPointF:
        rect = self._display_rect()
        if rect.isEmpty():
            return QPointF(0.0, 0.0)

        nx = (point.x() - rect.left()) / rect.width()
        ny = (point.y() - rect.top()) / rect.height()
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return QPointF(nx, ny)

    def _norm_to_display(self, norm_rect: QRectF) -> QRectF:
        rect = self._display_rect()
        return QRectF(
            rect.left() + norm_rect.left() * rect.width(),
            rect.top() + norm_rect.top() * rect.height(),
            norm_rect.width() * rect.width(),
            norm_rect.height() * rect.height(),
        )

    def _normalized_rect(self, p1: QPointF, p2: QPointF) -> QRectF:
        left = min(p1.x(), p2.x())
        right = max(p1.x(), p2.x())
        top = min(p1.y(), p2.y())
        bottom = max(p1.y(), p2.y())
        return QRectF(left, top, right - left, bottom - top)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or not self._pixmap:
            return
        self._drawing_roi = True
        self._drag_start = self._widget_to_norm(event.position())
        self._roi_norm = QRectF(self._drag_start, self._drag_start)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._drawing_roi or self._drag_start is None:
            return
        current = self._widget_to_norm(event.position())
        self._roi_norm = self._normalized_rect(self._drag_start, current)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or not self._drawing_roi:
            return

        self._drawing_roi = False
        if self._roi_norm and (self._roi_norm.width() < 0.005 or self._roi_norm.height() < 0.005):
            self._roi_norm = None
        self.update()
        self.roiChanged.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if not self._pixmap or self._pixmap.isNull():
            painter.setPen(QPen(Qt.white, 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "Load an MP4 file to preview")
            return

        draw_rect = self._display_rect()
        painter.drawPixmap(draw_rect.toRect(), self._pixmap)

        if self._roi_norm:
            roi_rect = self._norm_to_display(self._roi_norm)
            painter.setPen(QPen(Qt.green, 2))
            painter.drawRect(roi_rect)
            if self._drawing_roi:
                painter.fillRect(roi_rect, QColor(0, 255, 0, 64))

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MP4 OCR Analyzer")

        self.video_path: Optional[str] = None
        self.video_capture = None
        self.video_fps = 30.0
        self.video_frame_count = 0
        self.current_frame_index = 0
        self.ocr_engine = None
        self.ocr_signature: Optional[Tuple[str, int]] = None
        self.ocr_backend_name = PROCESSOR_MODE_CPU
        self._onnxruntime_dlls_preloaded = False
        self._onnxruntime_dlls_preload_error = ""
        self._cuda_dll_search_paths_configured = False
        self._cuda_dll_dir_handles: List[Any] = []

        self._updating_slider = False
        self._resume_after_scrub = False
        self._prev_trim_low = 0
        self._prev_trim_high = 0

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self.resize(1360, 860)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        file_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse MP4")
        browse_button.clicked.connect(self._browse_video)

        file_layout.addWidget(QLabel("Video"))
        file_layout.addWidget(self.video_path_edit, 1)
        file_layout.addWidget(browse_button)
        main_layout.addLayout(file_layout)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        left_panel = QVBoxLayout()
        content_layout.addLayout(left_panel, 3)

        self.video_canvas = VideoCanvas()
        left_panel.addWidget(self.video_canvas, 1)

        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._on_scrub_started)
        self.position_slider.sliderReleased.connect(self._on_scrub_ended)
        self.position_slider.valueChanged.connect(self._on_position_slider_changed)

        self.current_time_label = QLabel("00:00:00.00")
        self.total_time_label = QLabel("/ 00:00:00.00")

        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.position_slider, 1)
        playback_layout.addWidget(self.current_time_label)
        playback_layout.addWidget(self.total_time_label)
        left_panel.addLayout(playback_layout)

        trim_layout = QVBoxLayout()
        trim_layout.addWidget(QLabel("Trim Range (drag both handles)"))
        self.trim_slider = DualHandleSlider()
        self.trim_slider.setRange(0, 0)
        self.trim_slider.setValues(0, 0)
        self.trim_slider.valuesChanged.connect(self._on_trim_values_changed)
        self.trim_slider.valuesChangeFinished.connect(self._on_trim_values_finished)
        trim_layout.addWidget(self.trim_slider)

        self.trim_label = QLabel("Trim: 00:00:00.00 to 00:00:00.00")
        trim_layout.addWidget(self.trim_label)
        left_panel.addLayout(trim_layout)

        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("Draw ROI directly on the preview to set OCR area."))
        clear_roi_button = QPushButton("Clear ROI")
        clear_roi_button.clicked.connect(self.video_canvas.clear_roi)
        roi_layout.addWidget(clear_roi_button)
        left_panel.addLayout(roi_layout)

        settings_panel = QVBoxLayout()
        content_layout.addLayout(settings_panel, 2)

        sampling_group = QGroupBox("Sampling")
        sampling_form = QFormLayout(sampling_group)
        self.sample_mode_combo = QComboBox()
        self.sample_mode_combo.addItems(["Frames", "Seconds"])
        self.sample_mode_combo.setCurrentIndex(1)
        self.sample_mode_combo.currentIndexChanged.connect(self._on_sample_mode_changed)

        self.sample_value_stack = QStackedWidget()
        self.sample_frames_spin = QSpinBox()
        self.sample_frames_spin.setRange(1, 1_000_000)
        self.sample_frames_spin.setValue(1)
        self.sample_frames_spin.valueChanged.connect(self._update_sampling_readout)

        self.sample_seconds_spin = QDoubleSpinBox()
        self.sample_seconds_spin.setDecimals(3)
        self.sample_seconds_spin.setRange(0.001, 10_000.0)
        self.sample_seconds_spin.setSingleStep(0.1)
        self.sample_seconds_spin.setValue(1.0)
        self.sample_seconds_spin.valueChanged.connect(self._update_sampling_readout)

        self.sample_value_stack.addWidget(self.sample_frames_spin)
        self.sample_value_stack.addWidget(self.sample_seconds_spin)
        self.sample_value_stack.setCurrentIndex(self.sample_mode_combo.currentIndex())

        sampling_form.addRow("Mode", self.sample_mode_combo)
        sampling_form.addRow("Every", self.sample_value_stack)
        self.sample_points_label = QLabel("Estimated data points: 0")
        self.sample_points_label.setWordWrap(True)
        sampling_form.addRow("", self.sample_points_label)

        ocr_group = QGroupBox("OCR")
        ocr_form = QFormLayout(ocr_group)

        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.0, 1.0)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setValue(0.50)

        self.include_negative_checkbox = QCheckBox("Include negative values")
        self.include_negative_checkbox.setChecked(True)
        self.include_decimals_checkbox = QCheckBox("Include decimals")
        self.include_decimals_checkbox.setChecked(True)
        self.only_numbers_checkbox = QCheckBox("Only include numbers")
        self.only_numbers_checkbox.setChecked(True)

        ocr_form.addRow("Confidence threshold", self.conf_threshold_spin)
        ocr_form.addRow("", self.include_negative_checkbox)
        ocr_form.addRow("", self.include_decimals_checkbox)
        ocr_form.addRow("", self.only_numbers_checkbox)

        performance_group = QGroupBox("Performance")
        performance_form = QFormLayout(performance_group)

        self.processor_combo = QComboBox()
        self.processor_combo.addItems(
            [
                PROCESSOR_MODE_AUTO,
                PROCESSOR_MODE_CPU,
                PROCESSOR_MODE_GPU_DIRECTML,
                PROCESSOR_MODE_GPU_CUDA,
            ]
        )
        self.processor_combo.currentTextChanged.connect(self._on_processor_changed)

        cpu_count = max(1, int(os.cpu_count() or 1))
        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(1, max(1, cpu_count * 2))
        self.cpu_threads_spin.setValue(cpu_count)
        self.cpu_threads_spin.setToolTip(
            "Number of CPU threads for OpenCV and ONNX Runtime when running in CPU mode."
        )
        self.cpu_threads_spin.valueChanged.connect(self._on_cpu_threads_changed)

        performance_form.addRow("Processor", self.processor_combo)
        performance_form.addRow("CPU threads", self.cpu_threads_spin)
        self.cpu_threads_label = performance_form.labelForField(self.cpu_threads_spin)
        self.processor_info_label = QLabel("")
        self.processor_info_label.setWordWrap(True)
        performance_form.addRow("", self.processor_info_label)

        time_group = QGroupBox("X-Axis")
        time_form = QFormLayout(time_group)

        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["video_time_sec", "time"])
        self.x_axis_combo.currentTextChanged.connect(self._on_x_axis_changed)

        self.start_time_edit = QLineEdit("00:00:00.00")
        self.use_end_time_checkbox = QCheckBox("Use end time (for sped-up or slowed video)")
        self.use_end_time_checkbox.setChecked(False)
        self.use_end_time_checkbox.toggled.connect(self._on_use_end_time_toggled)
        self.end_time_edit = QLineEdit("00:00:00.00")
        self.end_time_edit.setToolTip(
            "Used only when column is 'time'. If end is earlier than start, it is treated as next day."
        )

        time_form.addRow("Column", self.x_axis_combo)
        time_form.addRow("Start time (HH:MM:SS.xx)", self.start_time_edit)
        time_form.addRow("", self.use_end_time_checkbox)
        time_form.addRow("End time (HH:MM:SS.xx)", self.end_time_edit)

        export_group = QGroupBox("Export")
        export_form = QFormLayout(export_group)
        self.include_header_checkbox = QCheckBox("Include header row")
        self.include_header_checkbox.setChecked(True)
        export_form.addRow("", self.include_header_checkbox)

        self.run_button = QPushButton("Run OCR and Save CSV")
        self.run_button.clicked.connect(self._run_ocr_export)

        self.status_label = QLabel("Load an MP4 file to begin.")
        self.status_label.setWordWrap(True)
        self.run_stats_label = QLabel("Last run stats: none")
        self.run_stats_label.setWordWrap(True)

        settings_panel.addWidget(sampling_group)
        settings_panel.addWidget(ocr_group)
        settings_panel.addWidget(performance_group)
        settings_panel.addWidget(time_group)
        settings_panel.addWidget(export_group)
        settings_panel.addWidget(self.run_button)
        settings_panel.addWidget(self.status_label)
        settings_panel.addWidget(self.run_stats_label)
        settings_panel.addStretch(1)

        self._on_x_axis_changed(self.x_axis_combo.currentText())
        self._update_sampling_readout()
        self._update_performance_controls_visibility()
        self._update_processor_info()

    def closeEvent(self, event) -> None:
        self._release_capture()
        super().closeEvent(event)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _browse_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select MP4 File",
            "",
            "MP4 files (*.mp4);;All files (*.*)",
        )
        if not file_path:
            return
        self._load_video(file_path)

    def _load_video(self, file_path: str) -> None:
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            QMessageBox.warning(self, "Unable to Open File", "Could not open the selected MP4 file.")
            return

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if frame_count <= 0:
            capture.release()
            QMessageBox.warning(self, "Invalid Video", "Video has no readable frames.")
            return

        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        self._release_capture()
        self.video_capture = capture
        self.video_path = file_path
        self.video_frame_count = frame_count
        self.video_fps = fps
        self.current_frame_index = 0

        self.video_path_edit.setText(file_path)
        self.position_slider.setRange(0, frame_count - 1)
        self.trim_slider.setRange(0, frame_count - 1)
        self.trim_slider.setValues(0, frame_count - 1, emit_signal=False)
        self._prev_trim_low = 0
        self._prev_trim_high = frame_count - 1
        self._update_trim_label(0, frame_count - 1)

        self.play_timer.setInterval(max(1, int(round(1000.0 / self.video_fps))))
        self._show_frame(0)

        duration_seconds = frame_count / fps
        parsed_start_time = self._parse_clock_time(self.start_time_edit.text())
        if parsed_start_time < 0:
            parsed_start_time = 0.0
        self.end_time_edit.setText(self._format_clock(parsed_start_time + duration_seconds))
        self._set_status(
            f"Loaded video: {frame_count} frames at {fps:.2f} FPS ({self._format_clock(duration_seconds)} total)."
        )
        self._update_sampling_readout()

    def _release_capture(self) -> None:
        self._pause_playback()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def _on_sample_mode_changed(self, index: int) -> None:
        self.sample_value_stack.setCurrentIndex(index)
        self._update_sampling_readout()

    def _on_processor_changed(self, value: str) -> None:
        self.ocr_engine = None
        self.ocr_signature = None
        self.ocr_backend_name = PROCESSOR_MODE_CPU
        self._update_performance_controls_visibility()
        self._update_processor_info()

    def _on_cpu_threads_changed(self, value: int) -> None:
        self.ocr_engine = None
        self.ocr_signature = None

    def _on_x_axis_changed(self, value: str) -> None:
        use_clock_time = value == "time"
        self.start_time_edit.setEnabled(use_clock_time)
        self.use_end_time_checkbox.setEnabled(use_clock_time)
        self.end_time_edit.setEnabled(use_clock_time and self.use_end_time_checkbox.isChecked())

    def _on_use_end_time_toggled(self, checked: bool) -> None:
        self.end_time_edit.setEnabled(self.x_axis_combo.currentText() == "time" and checked)

    def _on_scrub_started(self) -> None:
        self._resume_after_scrub = self.play_timer.isActive()
        self._pause_playback()

    def _on_scrub_ended(self) -> None:
        if self._resume_after_scrub:
            self._start_playback()
        self._resume_after_scrub = False

    def _on_position_slider_changed(self, value: int) -> None:
        if self._updating_slider or self.video_capture is None:
            return
        self._show_frame(value)

    def _on_trim_values_changed(self, low: int, high: int) -> None:
        self._pause_playback()
        self._update_trim_label(low, high)
        self._update_sampling_readout()

        preview_index: Optional[int] = None
        if low != self._prev_trim_low:
            preview_index = low
        elif high != self._prev_trim_high:
            preview_index = high

        self._prev_trim_low = low
        self._prev_trim_high = high

        if preview_index is not None:
            self._show_frame(preview_index)

    def _on_trim_values_finished(self, low: int, high: int) -> None:
        if self.current_frame_index < low:
            self._show_frame(low)
        elif self.current_frame_index > high:
            self._show_frame(high)

    def _update_trim_label(self, low: int, high: int) -> None:
        start_text = self._format_clock(low / self.video_fps if self.video_fps > 0 else 0.0)
        end_text = self._format_clock(high / self.video_fps if self.video_fps > 0 else 0.0)
        self.trim_label.setText(f"Trim: {start_text} to {end_text}")

    def _toggle_playback(self) -> None:
        if self.play_timer.isActive():
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self.video_capture is None:
            return

        low, high = self.trim_slider.values()
        if self.current_frame_index < low or self.current_frame_index > high:
            self._show_frame(low)

        self.play_timer.start()
        self.play_button.setText("Pause")

    def _pause_playback(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.play_button.setText("Play")

    def _advance_frame(self) -> None:
        if self.video_capture is None:
            self._pause_playback()
            return

        low, high = self.trim_slider.values()
        next_index = self.current_frame_index + 1
        if next_index > high:
            self._pause_playback()
            self._show_frame(high)
            return

        if next_index < low:
            next_index = low
        self._show_frame(next_index)

    def _show_frame(self, frame_index: int) -> bool:
        if self.video_capture is None:
            return False

        frame_index = max(0, min(frame_index, self.video_frame_count - 1))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.video_capture.read()
        if not ok:
            return False

        self.current_frame_index = frame_index
        self.video_canvas.set_frame(frame)

        self._updating_slider = True
        self.position_slider.setValue(frame_index)
        self._updating_slider = False

        current_seconds = frame_index / self.video_fps
        total_seconds = self.video_frame_count / self.video_fps
        self.current_time_label.setText(self._format_clock(current_seconds))
        self.total_time_label.setText(f"/ {self._format_clock(total_seconds)}")
        return True

    def _run_ocr_export(self) -> None:
        if not self.video_path:
            QMessageBox.information(self, "No Video", "Select an MP4 file first.")
            return

        if not self.video_canvas.has_roi():
            QMessageBox.information(self, "No ROI", "Draw an OCR ROI box on the video before exporting.")
            return

        x_axis_column = self.x_axis_combo.currentText()
        start_seconds = 0.0
        end_seconds = 0.0
        use_end_time_mapping = False
        if x_axis_column == "time":
            start_seconds = self._parse_clock_time(self.start_time_edit.text())
            if start_seconds < 0:
                QMessageBox.warning(
                    self,
                    "Invalid Start Time",
                    "Use HH:MM:SS.xx format, for example 18:34:42.95",
                )
                return

            use_end_time_mapping = self.use_end_time_checkbox.isChecked()
            if use_end_time_mapping:
                end_seconds = self._parse_clock_time(self.end_time_edit.text())
                if end_seconds < 0:
                    QMessageBox.warning(
                        self,
                        "Invalid End Time",
                        "Use HH:MM:SS.xx format, for example 18:39:42.95",
                    )
                    return

                # Allow clock ranges that cross midnight.
                if end_seconds < start_seconds:
                    end_seconds += 24 * 60 * 60

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "output.csv",
            "CSV files (*.csv)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".csv"):
            save_path += ".csv"

        execution_provider = self._resolve_execution_provider()
        if execution_provider is None:
            return

        self._configure_runtime_threads(execution_provider)

        ocr_engine = self._get_ocr_engine(execution_provider)
        if ocr_engine is None:
            return

        sample_step = self._sampling_step_frames()
        trim_low, trim_high = self.trim_slider.values()
        frame_indices = list(range(trim_low, trim_high + 1, sample_step))
        if not frame_indices:
            QMessageBox.warning(self, "Invalid Sampling", "No frames were selected for processing.")
            return
        if frame_indices[-1] != trim_high:
            frame_indices.append(trim_high)

        processing_capture = cv2.VideoCapture(self.video_path)
        if not processing_capture.isOpened():
            QMessageBox.warning(self, "Error", "Could not open the selected video for processing.")
            return

        frame_width = int(processing_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(processing_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0 or frame_height <= 0:
            ok, first_frame = processing_capture.read()
            if not ok:
                processing_capture.release()
                QMessageBox.warning(self, "Error", "Unable to read frames from video.")
                return
            frame_height, frame_width = first_frame.shape[:2]
            processing_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        roi = self.video_canvas.roi_in_frame(frame_width, frame_height)
        if roi is None:
            processing_capture.release()
            QMessageBox.warning(self, "Invalid ROI", "Draw a valid OCR region first.")
            return

        x, y, w, h = roi
        if w < 2 or h < 2:
            processing_capture.release()
            QMessageBox.warning(self, "Invalid ROI", "The OCR region is too small.")
            return

        progress = QProgressDialog("Running OCR...", "Cancel", 0, len(frame_indices), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        rows = []
        conf_threshold = float(self.conf_threshold_spin.value())
        total_samples = len(frame_indices)
        passed_samples = 0
        threshold_failed_samples = 0
        processed_samples = 0
        process_start = time.perf_counter()

        self._set_status(f"Processing {total_samples} frame samples on {self.ocr_backend_name}...")

        canceled = False
        last_timing_update_second = -1
        elapsed_text = "00:00"
        eta_text = "Estimating..."
        for idx, frame_index in enumerate(frame_indices, start=1):
            done_count = idx - 1
            elapsed = time.perf_counter() - process_start
            elapsed_rounded_seconds = int(elapsed + 0.5)
            if elapsed_rounded_seconds != last_timing_update_second:
                last_timing_update_second = elapsed_rounded_seconds
                elapsed_text = self._format_duration_whole_seconds(elapsed_rounded_seconds)
                if done_count > 0:
                    avg_per_sample = elapsed / done_count
                    eta_seconds = max(0.0, avg_per_sample * (total_samples - done_count))
                    eta_text = self._format_duration_whole_seconds(eta_seconds)
                else:
                    eta_text = "Estimating..."

            progress.setValue(idx - 1)
            progress.setLabelText(
                f"Running OCR on sample {idx}/{total_samples}\n"
                f"Time elapsed: {elapsed_text}\n"
                f"Estimated time remaining: {eta_text}"
            )
            QApplication.processEvents()

            if progress.wasCanceled():
                canceled = True
                break

            processing_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = processing_capture.read()
            if not ok:
                value_text = ""
                threshold_failed_samples += 1
            else:
                crop = frame[y : y + h, x : x + w]
                value, confidence = self._ocr_value(ocr_engine, crop)
                if value is None or confidence < conf_threshold:
                    value_text = ""
                    threshold_failed_samples += 1
                else:
                    value_text = self._format_numeric_value(value)
                    passed_samples += 1

            if x_axis_column == "video_time_sec":
                x_value = f"{frame_index / self.video_fps:.2f}"
            else:
                if use_end_time_mapping and trim_high > trim_low:
                    position_ratio = (frame_index - trim_low) / (trim_high - trim_low)
                    mapped_seconds = start_seconds + ((end_seconds - start_seconds) * position_ratio)
                else:
                    mapped_seconds = start_seconds + (frame_index / self.video_fps)
                x_value = self._format_clock(mapped_seconds)

            rows.append([x_value, value_text])
            processed_samples += 1

        progress.setValue(processed_samples)
        progress.close()
        processing_capture.release()

        elapsed_total = time.perf_counter() - process_start

        if canceled and not rows:
            self._set_status("Export canceled.")
            self.run_stats_label.setText(
                f"Last run stats: canceled before completion after {self._format_duration(elapsed_total)}"
            )
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as output_file:
                writer = csv.writer(output_file)
                if self.include_header_checkbox.isChecked():
                    writer.writerow([x_axis_column, "value"])
                writer.writerows(rows)
        except OSError as exc:
            QMessageBox.warning(self, "Save Error", f"Could not save CSV:\n{exc}")
            return

        pass_rate = (passed_samples / processed_samples * 100.0) if processed_samples else 0.0
        canceled_note = " (canceled early)" if canceled else ""
        self._set_status(f"Saved {len(rows)} samples to {save_path}{canceled_note}")
        self.run_stats_label.setText(
            "Last run stats:\n"
            f"Process time: {self._format_duration(elapsed_total)}\n"
            f"OCR pass rate: {pass_rate:.1f}% ({passed_samples}/{processed_samples})\n"
            f"Frames below confidence threshold: {threshold_failed_samples}"
            f"{canceled_note}"
        )
        QMessageBox.information(self, "Export Complete", f"Saved CSV:\n{save_path}")

    def _sampling_step_frames(self) -> int:
        mode = self.sample_mode_combo.currentText()
        if mode == "Frames":
            return int(self.sample_frames_spin.value())
        seconds_step = float(self.sample_seconds_spin.value())
        return max(1, int(round(seconds_step * self.video_fps)))

    def _estimated_sample_count(self) -> int:
        if self.video_frame_count <= 0:
            return 0

        low, high = self.trim_slider.values()
        if high < low:
            return 0

        step = max(1, self._sampling_step_frames())
        delta = high - low
        count = (delta // step) + 1
        if delta % step != 0:
            count += 1
        return count

    def _update_sampling_readout(self) -> None:
        count = self._estimated_sample_count()
        if self.video_frame_count <= 0:
            self.sample_points_label.setText("Estimated data points: 0 (load a video)")
            return

        low, high = self.trim_slider.values()
        span_seconds = 0.0
        if self.video_fps > 0:
            span_seconds = max(0.0, (high - low) / self.video_fps)
        self.sample_points_label.setText(
            f"Estimated data points: {count} across trimmed span {self._format_clock(span_seconds)}"
        )

    def _update_performance_controls_visibility(self) -> None:
        mode = self.processor_combo.currentText()
        show_cpu_controls = mode in (PROCESSOR_MODE_AUTO, PROCESSOR_MODE_CPU)

        self.cpu_threads_spin.setVisible(show_cpu_controls)
        if self.cpu_threads_label is not None:
            self.cpu_threads_label.setVisible(show_cpu_controls)

    @staticmethod
    def _backend_name(execution_provider: str) -> str:
        if execution_provider == OCR_BACKEND_CUDA:
            return PROCESSOR_MODE_GPU_CUDA
        if execution_provider == OCR_BACKEND_DIRECTML:
            return PROCESSOR_MODE_GPU_DIRECTML
        return PROCESSOR_MODE_CPU

    def _effective_cpu_threads(self, execution_provider: str) -> int:
        if execution_provider != OCR_BACKEND_CPU:
            return max(1, int(os.cpu_count() or 1))
        return int(self.cpu_threads_spin.value())

    def _configure_runtime_threads(self, execution_provider: str) -> None:
        thread_count = self._effective_cpu_threads(execution_provider)
        try:
            cv2.setNumThreads(thread_count)
        except Exception:
            pass

    def _ensure_onnxruntime_cuda_dlls_loaded(self) -> Optional[str]:
        self._configure_windows_cuda_dll_search_paths()

        if self._onnxruntime_dlls_preloaded:
            return None
        if self._onnxruntime_dlls_preload_error:
            return self._onnxruntime_dlls_preload_error

        try:
            import onnxruntime as ort
        except Exception as exc:
            self._onnxruntime_dlls_preload_error = f"onnxruntime import failed: {exc}"
            return self._onnxruntime_dlls_preload_error

        preload = getattr(ort, "preload_dlls", None)
        if not callable(preload):
            self._onnxruntime_dlls_preloaded = True
            return None

        try:
            preload()
            self._onnxruntime_dlls_preloaded = True
            return None
        except Exception as exc:
            self._onnxruntime_dlls_preload_error = f"onnxruntime preload_dlls failed: {exc}"
            return self._onnxruntime_dlls_preload_error

    def _configure_windows_cuda_dll_search_paths(self) -> None:
        if os.name != "nt" or self._cuda_dll_search_paths_configured:
            return

        site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
        nvidia_root = os.path.join(site_packages, "nvidia")
        ort_capi_dir = os.path.join(site_packages, "onnxruntime", "capi")

        candidates: List[str] = []
        preferred_cuda_components = (
            "cuda_nvrtc",
            "cuda_runtime",
            "cublas",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "nvjitlink",
        )

        for component in preferred_cuda_components:
            bin_dir = os.path.join(nvidia_root, component, "bin")
            if os.path.isdir(bin_dir):
                candidates.append(bin_dir)

        if os.path.isdir(nvidia_root):
            try:
                for name in os.listdir(nvidia_root):
                    bin_dir = os.path.join(nvidia_root, name, "bin")
                    if os.path.isdir(bin_dir):
                        candidates.append(bin_dir)
            except OSError:
                pass

        if os.path.isdir(ort_capi_dir):
            candidates.append(ort_capi_dir)

        unique_dirs: List[str] = []
        seen = set()
        for path in candidates:
            norm = os.path.normcase(os.path.normpath(path))
            if norm in seen:
                continue
            seen.add(norm)
            unique_dirs.append(path)

        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        existing = {os.path.normcase(os.path.normpath(item)) for item in path_entries if item}
        new_paths: List[str] = []
        for directory in unique_dirs:
            norm = os.path.normcase(os.path.normpath(directory))
            if norm not in existing:
                new_paths.append(directory)
                existing.add(norm)

        if new_paths:
            os.environ["PATH"] = os.pathsep.join(new_paths + path_entries)

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            for directory in unique_dirs:
                try:
                    handle = add_dll_directory(directory)
                except OSError:
                    continue
                self._cuda_dll_dir_handles.append(handle)

        self._cuda_dll_search_paths_configured = True

    def _onnxruntime_diagnostics(self) -> Tuple[bool, bool, List[str], str]:
        preload_error = self._ensure_onnxruntime_cuda_dlls_loaded()
        try:
            import onnxruntime as ort
        except Exception as exc:
            return False, False, [], f"onnxruntime import failed: {exc}"

        ort_version = getattr(ort, "__version__", "unknown")
        try:
            providers = list(ort.get_available_providers())
        except Exception as exc:
            return False, False, [], f"onnxruntime={ort_version}, provider_check_error={exc}"

        cuda_available = CUDA_PROVIDER_NAME in providers
        directml_available = DIRECTML_PROVIDER_NAME in providers
        details = f"onnxruntime={ort_version}, providers={providers}"
        if preload_error:
            details = f"{details}, preload_error={preload_error}"
        return cuda_available, directml_available, providers, details

    def _update_processor_info(self) -> None:
        mode = self.processor_combo.currentText()
        cuda_available, directml_available, _, details = self._onnxruntime_diagnostics()

        if mode == PROCESSOR_MODE_CPU:
            text = f"CPU selected. {details}"
        elif mode == PROCESSOR_MODE_GPU_CUDA:
            if cuda_available:
                text = f"{PROCESSOR_MODE_GPU_CUDA} ready. {details}"
            else:
                text = f"{PROCESSOR_MODE_GPU_CUDA} not ready. {details}"
        elif mode == PROCESSOR_MODE_GPU_DIRECTML:
            if directml_available:
                text = f"{PROCESSOR_MODE_GPU_DIRECTML} ready. {details}"
            else:
                text = f"{PROCESSOR_MODE_GPU_DIRECTML} not ready. {details}"
        else:
            if directml_available:
                backend = PROCESSOR_MODE_GPU_DIRECTML
            elif cuda_available:
                backend = PROCESSOR_MODE_GPU_CUDA
            else:
                backend = PROCESSOR_MODE_CPU
            text = f"Auto mode will use {backend}. {details}"

        self.processor_info_label.setText(text)

    def _resolve_execution_provider(self) -> Optional[str]:
        mode = self.processor_combo.currentText()
        cuda_available, directml_available, _, details = self._onnxruntime_diagnostics()

        if mode == PROCESSOR_MODE_CPU:
            return OCR_BACKEND_CPU

        if mode == PROCESSOR_MODE_GPU_CUDA:
            if cuda_available:
                return OCR_BACKEND_CUDA

            install_ps1 = os.path.join(self._application_base_dir(), "install.ps1")
            cuda_req = os.path.join(self._application_base_dir(), GPU_CUDA_REQUIREMENTS_FILE)
            QMessageBox.warning(
                self,
                "GPU Not Available",
                f"{PROCESSOR_MODE_GPU_CUDA} mode was selected, but {CUDA_PROVIDER_NAME} is unavailable in ONNX Runtime.\n\n"
                f"{details}\n\n"
                "Install CUDA dependencies with:\n"
                f"powershell -NoProfile -ExecutionPolicy Bypass -File \"{install_ps1}\" -Profile cuda\n\n"
                "Manual fallback:\n"
                f"{sys.executable} -m pip install -r \"{cuda_req}\"",
            )
            return None

        if mode == PROCESSOR_MODE_GPU_DIRECTML:
            if directml_available:
                return OCR_BACKEND_DIRECTML

            install_ps1 = os.path.join(self._application_base_dir(), "install.ps1")
            directml_req = os.path.join(self._application_base_dir(), GPU_DIRECTML_REQUIREMENTS_FILE)
            QMessageBox.warning(
                self,
                "GPU Not Available",
                f"{PROCESSOR_MODE_GPU_DIRECTML} mode was selected, but {DIRECTML_PROVIDER_NAME} is unavailable in ONNX Runtime.\n\n"
                f"{details}\n\n"
                "Install DirectML dependencies with:\n"
                f"powershell -NoProfile -ExecutionPolicy Bypass -File \"{install_ps1}\" -Profile directml\n\n"
                "Manual fallback:\n"
                f"{sys.executable} -m pip install -r \"{directml_req}\"",
            )
            return None

        if directml_available:
            return OCR_BACKEND_DIRECTML

        if cuda_available:
            return OCR_BACKEND_CUDA

        return OCR_BACKEND_CPU

    def _build_rapidocr_params(
        self,
        engine_type,
        lang_det,
        lang_rec,
        execution_provider: str,
        thread_count: int,
    ) -> dict:
        use_cuda = execution_provider == OCR_BACKEND_CUDA
        use_directml = execution_provider == OCR_BACKEND_DIRECTML
        use_cpu = execution_provider == OCR_BACKEND_CPU
        return {
            "Global.log_level": "warning",
            "Global.text_score": 0.0,
            "Global.use_det": False,
            "Global.use_cls": False,
            "Global.use_rec": True,
            "Det.engine_type": engine_type.ONNXRUNTIME,
            "Cls.engine_type": engine_type.ONNXRUNTIME,
            "Rec.engine_type": engine_type.ONNXRUNTIME,
            "Det.lang_type": lang_det.EN,
            "Rec.lang_type": lang_rec.EN,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
            "EngineConfig.onnxruntime.use_dml": use_directml,
            "EngineConfig.onnxruntime.intra_op_num_threads": thread_count if use_cpu else -1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1 if use_cpu else -1,
        }

    def _auto_fallback_providers(self, initial_provider: str) -> List[str]:
        cuda_available, directml_available, _, _ = self._onnxruntime_diagnostics()
        candidates: List[str] = []
        if directml_available:
            candidates.append(OCR_BACKEND_DIRECTML)
        if cuda_available:
            candidates.append(OCR_BACKEND_CUDA)
        candidates.append(OCR_BACKEND_CPU)
        return [provider for provider in candidates if provider != initial_provider]

    def _get_ocr_engine(self, execution_provider: str):
        if execution_provider == OCR_BACKEND_CUDA:
            self._ensure_onnxruntime_cuda_dlls_loaded()

        thread_count = self._effective_cpu_threads(execution_provider)
        signature = (execution_provider, thread_count)
        if self.ocr_engine is not None and self.ocr_signature == signature:
            return self.ocr_engine

        self.ocr_engine = None
        self.ocr_signature = None

        try:
            from rapidocr import EngineType, LangDet, LangRec, RapidOCR
        except ImportError as exc:
            install_help = _bootstrap_install_instructions_text()
            QMessageBox.warning(
                self,
                "Missing Dependency",
                "RapidOCR dependencies are missing.\n\n"
                f"{install_help}\n\n"
                f"Import error: {exc}",
            )
            return None

        backend_name = self._backend_name(execution_provider)
        params = self._build_rapidocr_params(
            EngineType,
            LangDet,
            LangRec,
            execution_provider,
            thread_count,
        )

        self._set_status(f"Loading RapidOCR on {backend_name} (first run can take a bit)...")
        QApplication.processEvents()
        try:
            self.ocr_engine = RapidOCR(params=params)
            self.ocr_signature = signature
            self.ocr_backend_name = backend_name
        except Exception as exc:
            if self.processor_combo.currentText() == PROCESSOR_MODE_AUTO:
                for fallback_provider in self._auto_fallback_providers(execution_provider):
                    fallback_name = self._backend_name(fallback_provider)
                    fallback_thread_count = self._effective_cpu_threads(fallback_provider)
                    self._set_status(
                        f"{backend_name} initialization failed in Auto mode. Falling back to {fallback_name}..."
                    )
                    QApplication.processEvents()
                    try:
                        if fallback_provider == OCR_BACKEND_CUDA:
                            self._ensure_onnxruntime_cuda_dlls_loaded()
                        fallback_params = self._build_rapidocr_params(
                            EngineType,
                            LangDet,
                            LangRec,
                            fallback_provider,
                            fallback_thread_count,
                        )
                        self.ocr_engine = RapidOCR(params=fallback_params)
                        self.ocr_signature = (fallback_provider, fallback_thread_count)
                        self.ocr_backend_name = fallback_name
                        QMessageBox.information(
                            self,
                            "Backend Fallback",
                            f"{backend_name} initialization failed. OCR is running on {fallback_name} instead.",
                        )
                        return self.ocr_engine
                    except Exception:
                        continue
            QMessageBox.warning(self, "OCR Initialization Failed", str(exc))
            self.ocr_engine = None
            self.ocr_signature = None
        return self.ocr_engine

    @staticmethod
    def _extract_rapidocr_pairs(result: Any) -> List[Tuple[str, float]]:
        pairs: List[Tuple[str, float]] = []
        if result is None:
            return pairs

        txts = getattr(result, "txts", None)
        if txts is not None:
            scores = getattr(result, "scores", None)
            txt_list = list(txts)
            score_list = list(scores) if scores is not None else [0.0] * len(txt_list)
            if len(score_list) < len(txt_list):
                score_list.extend([0.0] * (len(txt_list) - len(score_list)))

            for idx, text in enumerate(txt_list):
                try:
                    confidence = float(score_list[idx])
                except Exception:
                    confidence = 0.0
                pairs.append((str(text), confidence))
            return pairs

        raw_rows: Any = None
        if isinstance(result, tuple) and result:
            raw_rows = result[0]
        elif isinstance(result, list):
            raw_rows = result

        if not isinstance(raw_rows, list):
            return pairs

        for row in raw_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            text = str(row[1])
            try:
                confidence = float(row[2])
            except Exception:
                confidence = 0.0
            pairs.append((text, confidence))

        return pairs

    def _run_rapidocr(self, ocr_engine, image) -> List[Tuple[str, float]]:
        try:
            result = ocr_engine(image)
        except Exception:
            return []
        return self._extract_rapidocr_pairs(result)

    @staticmethod
    def _application_base_dir() -> str:
        if getattr(sys, "frozen", False):
            return os.path.dirname(os.path.abspath(sys.executable))
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _candidate_digit_count(text: str) -> int:
        return sum(1 for ch in text if ch.isdigit())

    @staticmethod
    def _candidate_has_decimal(text: str) -> bool:
        return "." in text or "," in text

    def _select_numeric_candidate(self, pairs: List[Tuple[str, float]]) -> Tuple[Optional[float], float]:
        if not pairs:
            return None, 0.0

        texts = [str(text).strip() for text, _ in pairs if str(text).strip()]
        if not texts:
            return None, 0.0

        max_conf = max((float(conf) for _, conf in pairs), default=0.0)
        candidates: List[Tuple[str, float]] = [(text, float(conf)) for text, conf in pairs if str(text).strip()]

        joined_compact = "".join(texts)
        if joined_compact:
            candidates.append((joined_compact, max_conf))

        joined_spaced = " ".join(texts)
        if joined_spaced:
            candidates.append((joined_spaced, max_conf))

        best_value: Optional[float] = None
        best_confidence = 0.0
        best_key = (-1, -1, -1.0)

        for candidate_text, confidence in candidates:
            parsed = self._parse_numeric(candidate_text)
            if parsed is None:
                continue

            key = (
                self._candidate_digit_count(candidate_text),
                1 if self._candidate_has_decimal(candidate_text) else 0,
                float(confidence),
            )
            if key > best_key:
                best_key = key
                best_value = parsed
                best_confidence = float(confidence)

        return best_value, best_confidence

    def _ocr_value(self, ocr_engine, crop) -> Tuple[Optional[float], float]:
        processed = self._preprocess_crop(crop)

        pairs = self._run_rapidocr(ocr_engine, processed)
        value, confidence = self._select_numeric_candidate(pairs)
        if value is not None:
            return value, confidence

        # Fallback to raw crop only when preprocessed OCR could not yield a numeric value.
        raw_pairs = self._run_rapidocr(ocr_engine, crop)
        value, confidence = self._select_numeric_candidate(raw_pairs)
        if value is not None:
            return value, confidence

        max_conf = max(
            [float(conf) for _, conf in pairs] + [float(conf) for _, conf in raw_pairs],
            default=0.0,
        )
        return None, max_conf

    @staticmethod
    def _preprocess_crop(crop):
        if crop is None or crop.size == 0:
            return crop

        if len(crop.shape) == 2:
            gray = crop
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _parse_numeric(self, text: str) -> Optional[float]:
        cleaned = text.strip().replace(",", ".")
        if not cleaned:
            return None

        include_decimals = self.include_decimals_checkbox.isChecked()
        include_negative = self.include_negative_checkbox.isChecked()
        only_numbers = self.only_numbers_checkbox.isChecked()
        had_minus_sign = "-" in cleaned

        if only_numbers:
            allowed = set("0123456789")
            if include_decimals:
                allowed.add(".")
            if include_negative:
                allowed.add("-")
            cleaned = "".join(ch for ch in cleaned if ch in allowed)

        if not cleaned:
            return None

        if not include_negative and had_minus_sign:
            return None

        if include_decimals:
            base_pattern = r"\d+(?:\.\d+)?"
        else:
            base_pattern = r"\d+"

        if include_negative:
            pattern = rf"-?{base_pattern}"
        else:
            pattern = rf"(?<!-){base_pattern}"

        match = re.search(pattern, cleaned)
        if not match:
            return None

        token = match.group(0)
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def _format_numeric_value(value: float) -> str:
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        if text in {"", "-0"}:
            return "0"
        return text

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        total_seconds = max(0.0, float(total_seconds))
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds - (hours * 3600) - (minutes * 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        return f"{minutes:02d}:{seconds:05.2f}"

    @staticmethod
    def _format_duration_whole_seconds(total_seconds: float) -> str:
        rounded_seconds = int(max(0.0, float(total_seconds)) + 0.5)
        hours = rounded_seconds // 3600
        minutes = (rounded_seconds % 3600) // 60
        seconds = rounded_seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _parse_clock_time(text: str) -> float:
        match = TIME_PATTERN.match(text.strip())
        if not match:
            return -1.0

        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        centiseconds_str = match.group(4) or "0"
        centiseconds = int(centiseconds_str.ljust(2, "0"))

        return (hours * 3600) + (minutes * 60) + seconds + (centiseconds / 100.0)

    @staticmethod
    def _format_clock(total_seconds: float) -> str:
        centiseconds_day = 24 * 60 * 60 * 100
        total_centiseconds = int(round(total_seconds * 100.0)) % centiseconds_day

        hours = total_centiseconds // (3600 * 100)
        remainder = total_centiseconds % (3600 * 100)
        minutes = remainder // (60 * 100)
        remainder %= 60 * 100
        seconds = remainder // 100
        centiseconds = remainder % 100

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

