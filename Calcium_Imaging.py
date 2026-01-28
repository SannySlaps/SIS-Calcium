from posixpath import basename
import sys, os, time, json
from datetime import datetime
import numpy as np
import tifffile
import cv2
import serial
from queue import Queue, Empty
import threading
from threading import Lock

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QSpinBox,QSlider, QComboBox, QVBoxLayout, QGridLayout, QGroupBox, QProgressBar, QCheckBox,QFileDialog, QSizePolicy, QTextEdit, QFrame,)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from tifffile import imwrite, imread

from pymmcore_plus import CMMCorePlus

import logging
import os

log_path = os.path.expanduser("~\\AppData\\Local\\pymmcore-plus\\pymmcore-plus\\logs\\pymmcore-plus.log")
logger = logging.getLogger("pymmcore-plus")
logger.handlers = []  # remove default handlers
logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode="a")

# -------------------- Load Core Thread --------------------

class LoadCoreThread(QThread):
    core_loaded = pyqtSignal(bool, object)  # success flag, core object or None

    def __init__(self, cfg_path):
        super().__init__()
        self.cfg_path = cfg_path
        self.core = None

    def run(self):
        try:
            # Reset any existing instance
            core = CMMCorePlus.instance()
            self.core = core
            try:
                self.core.reset()
            except Exception:
                pass
            self.core.loadSystemConfiguration(self.cfg_path)
            cam = self.core.getCameraDevice()
            self.core_loaded.emit(True, core)
        except Exception as e:
            print(f"[ERROR] Failed to load configuration: {e}")
            self.core_loaded.emit(False, None)
    
    def stop(self):
        # safe stop/reset of any instance held by this thread
        try:
            core = CMMCorePlus.instance()
            if core is not None:
                try:
                    if getattr(core, "isSequenceRunning", lambda: False)():
                        core.stopSequenceAcquisition()
                except Exception:
                    pass
                try:
                    core.reset()
                except Exception:
                    pass
        except Exception:
            pass

# -------------------- Frame Writer Thread --------------------
class FrameWriterThread(QThread):
    log_event_signal = pyqtSignal(str, str)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            try:
                path, arr = self.queue.get(timeout=0.1)
                
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr, dtype=np.uint16)
                else:
                    arr = arr.astype(np.uint16)

                tifffile.imwrite(path, arr, photometric='minisblack')
                self.queue.task_done()
                # self.log_event_signal.emit(f"Saved {path} ({len(arr)} frames)", "green")

            except Empty:
                continue
            except Exception as e:
                self.log_event_signal.emit(f"Error saving {path}: {e}", "red")

    def stop(self):
        self.running = False
        self.wait()

# -------------------- Live Preview Thread --------------------
class LivePreviewThread(QThread):
    image_ready = pyqtSignal(np.ndarray)  # throttled preview
    new_frame = pyqtSignal(np.ndarray)    # all frames for burst
    log_event_signal = pyqtSignal(str, str)

    def __init__(self, core, lock=None, preview_fps=30):
        super().__init__()
        self.core = core
        self.lock = lock
        self.running = False
        self.preview_fps = preview_fps

    def run(self):
        self.running = True
        last_emit_time = time.time()

        while self.running:
            with self.lock:
                while self.core.getRemainingImageCount() > 0:
                    img = self.core.popNextImage()
                    frame = np.array(img, dtype=np.uint16)

                    # Emit to burst thread (all frames)
                    self.new_frame.emit(frame)

                    # Emit to GUI at throttled FPS
                    now = time.time()
                    if now - last_emit_time >= 1.0 / self.preview_fps:
                        self.image_ready.emit(frame)
                        last_emit_time = now

            time.sleep(0.001)  # slight throttle to avoid busy loop

    def stop(self):
        self.running = False

# -------------------- Burst Thread --------------------
class BurstThread(QThread):
    burst_done = pyqtSignal(int, object)      # burst_index, frames
    burst_started = pyqtSignal(int)
    log_event_signal = pyqtSignal(str, str)

    def __init__(self, burst_index, duration_s):
        super().__init__()
        self.burst_index = burst_index
        self.duration_s = duration_s
        self.frames = []
        self._stop_event = threading.Event()

    def collect_frame(self, frame):
        """Connect this to LivePreviewThread.new_frame"""
        self.frames.append(frame)

    def run(self):
        self.burst_started.emit(self.burst_index)
        start_time = time.time()
        while (time.time() - start_time) < self.duration_s and not self._stop_event.is_set():
            time.sleep(0.001)  # just wait; frames are collected via signal

        self.burst_done.emit(self.burst_index, self.frames)

    def stop(self):
        self._stop_event.set()

# -------------------- Live Preview Window --------------------
class LivePreviewWindow(QWidget):
    def __init__(self, core, lock=None):
        super().__init__()
        self.core = core
        self.camera_lock = lock
        self.setWindowTitle("Live Preview")
        self.label = QLabel("Live Preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setMinimumSize(800, 600)

    def update_frame(self, arr):
        # Convert to 8-bit grayscale
        arr8 = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)
        qimg = QImage(arr8.data(), arr.shape[1], arr.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.repaint()
        QApplication.processEvents()

# -------------------- Collapsible GroupBox --------------------
class CollapsibleGroupBox(QWidget):
    def __init__(self, title):
        super().__init__()

        # Toggle button
        self.toggle_btn = QPushButton(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setFlat(True)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                background-color: transparent;
                border: none;
                padding-left: 5px;
            }
        """)
        self.toggle_btn.clicked.connect(self.on_toggle)

        # Content frame with border
        self.content_frame = QFrame()
        self.content_frame.setFrameShape(QFrame.StyledPanel)
        self.content_frame.setStyleSheet("""
            QFrame {
                border: 2px solid white;
                border-radius: 5px;
                margin-top: 5px;
                background-color: #2e2e2e;
            }
        """)
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5,5,5,5)
        self.content_frame.setLayout(self.content_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toggle_btn)
        main_layout.addWidget(self.content_frame)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.content_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def set_layout(self, layout):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.content_layout.addLayout(layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def on_toggle(self):
        self.content_frame.setVisible(self.toggle_btn.isChecked())
        self.adjustSize()  # force the parent layout to recalc height
    # Force layout recalculation
        self.content_frame.updateGeometry()

# -------------------- Main GUI --------------------
class LiveImagingGUI(QWidget):
    def __init__(self, cfg_path):
            super().__init__()
            self.cfg_path = cfg_path
            self.setWindowTitle("Glutamate Chasers")

            self.settings_file = "stim_gui_settings.json"
            self.settings = {}
            self.apply_dark_mode()

            self.last_frame = None
            self.live_window = None
            self.core = None
            self.camera_lock = Lock()
            self.default_clear_mode = "Pre-Sequence"
            self.default_clear_cycles = 2

            self.log_queue = Queue()
            self.log_timer = QTimer(self)
            self.log_timer.timeout.connect(self.flush_log_queue)
            self.log_timer.start(50)

            self.burst_job_queue = None
            self.writer_thread = None

            self.arduino = None
            self.experiment_timer = None

            self.burst_index = 0
            self.burst_duration_s = 2.0
            self.pause_between_bursts_s = 8.0
            self.total_frames = 0
            self.frames_taken = 0
            self.burst_queue = None
            self.writer_thread = None

            self.build_ui()
            self.load_settings()
            self.show()

    def on_core_loaded(self, success, core):
        if not success:
            self.log_event("Core failed to load!", "red")
            self.live_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            return

        self.core = core
        cam = self.core.getCameraDevice()
        self.core.setCameraDevice(cam)
        self.core.setROI(0, 0, 600, 600)
        self.core.setExposure(self.exp_spin.value())
        try:
            self.core.setProperty(cam, "CircularBufferEnabled", "ON")
            self.core.setProperty(cam, "CircularBufferFrameCount", 2000)
            self.core.setProperty(cam,"ClearMode", "Pre-Sequence")
            self.core.setProperty(cam,"ClearCycles", 2)
            # self.core.setProperty(cam, "Binning", "2x2")
            self.core.startContinuousSequenceAcquisition(0)      # buffer size
        except Exception as e:
            self.log_event(f"Warning setting camera properties: [e]", "orange")
        try:
            if hasattr(self.core, "isSequenceRunning") and self.core.isSequenceRunning():
                pass
            else:
                self.core.startContinuousSequenceAcquisition(0)   # 0 = run until stopped
                self.log_event("Continuous sequence acquisition started", "green")
        except Exception as e:
            self.log_event(f"Could not start continuous sequence acquisition: {e}", "red")

        self.live_thread = LivePreviewThread(core=self.core, lock=self.camera_lock)
        self.live_thread.image_ready.connect(self.update_live_frame)
        # self.live_thread.burst_started_signal.connect(self.on_burst_started)
        self.live_thread.log_event_signal.connect(self.log_event)
        # self.live_thread.burst_done_signal.connect(self.on_burst_done)
        self.live_thread.start()


        self.log_event("Core loaded successfully", "green")
        self.log_event("Continuous sequence acquisition started", "green")

        self.live_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

# -------------------- Collapsible / Group Widgets --------------------
    def build_ui(self):
    # File Saving
        self.file_group = CollapsibleGroupBox("File Saving")
        file_layout = QGridLayout()
        self.save_path_edit = QLineEdit()
        file_layout.addWidget(self.save_path_edit, 0, 1, 1, 2)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_folder)
        file_layout.addWidget(browse_btn, 0, 3)
        self.expt_name_edit = QLineEdit("Stim_Exp")
        file_layout.addWidget(self.expt_name_edit, 1, 1)
        self.file_group.set_layout(file_layout)
        lbl = QLabel("Mouse ID")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        file_layout.addWidget(lbl, 1, 2)
        self.mouse_id_edit = QLineEdit("Mouse_001")
        file_layout.addWidget(self.mouse_id_edit, 1, 3)
        lbl = QLabel("Experiment Type")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        file_layout.addWidget(lbl, 2, 0)
        self.expt_type_edit = QLineEdit("GCamp")
        file_layout.addWidget(self.expt_type_edit, 2, 1)
        lbl = QLabel("Final Titer")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        file_layout.addWidget(lbl, 2, 2)
        self.final_titer_edit = QLineEdit("e11")
        file_layout.addWidget(self.final_titer_edit, 2, 3)
        lbl = QLabel("Save Folder")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        file_layout.addWidget(lbl, 0, 0)
        lbl = QLabel("Experiment Name")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        file_layout.addWidget(lbl, 1, 0)

# Acquisition Settings
        self.acq_group = CollapsibleGroupBox("Acquisition Settings")
        acq_layout = QGridLayout()

        lbl = QLabel("Total Duration (min)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 0, 0)
        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setValue(5)
        acq_layout.addWidget(self.total_time_spin, 0, 1)

        lbl = QLabel("Trigger Time (ms)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 0, 2)
        self.trigger_time_spin = QDoubleSpinBox()
        self.trigger_time_spin.setRange(0, 10000)  # 0 ms to 10,000 ms
        self.trigger_time_spin.setValue(1000)
        acq_layout.addWidget(self.trigger_time_spin, 0, 3)



        lbl = QLabel("Burst Duration (s)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")    
        acq_layout.addWidget(lbl, 1, 0)
        self.burst_duration_spin = QDoubleSpinBox()
        self.burst_duration_spin.setRange(0.1, 60.0)   # 0.1s to 60s
        self.burst_duration_spin.setDecimals(1)
        self.burst_duration_spin.setValue(2.0)         # default 2 seconds
        self.burst_duration_spin.setSuffix(" s")
        acq_layout.addWidget(self.burst_duration_spin, 1, 1)

        lbl = QLabel("Wait interval (s)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 1, 2)
        self.wait_interval_spin = QDoubleSpinBox()
        self.wait_interval_spin.setRange(0.0, 600.0)   # up to 10 minutes if needed
        self.wait_interval_spin.setDecimals(1)
        self.wait_interval_spin.setValue(8.0)          # default 8 seconds
        self.wait_interval_spin.setSuffix(" s")
        acq_layout.addWidget(self.wait_interval_spin, 1, 3)

        lbl = QLabel("Exposure")    
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 2, 0)
        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setRange(0,1000)
        self.exp_spin.setValue(10)
        acq_layout.addWidget(self.exp_spin, 2, 1)

        lbl = QLabel("Frames Per Second")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 2, 2)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["5", "10", "15", "30", "60"])
        self.fps_combo.setCurrentText("30")
        acq_layout.addWidget(self.fps_combo, 2, 3)

        self.acq_group.set_layout(acq_layout)
# Camera Controls
        self.camera_group = CollapsibleGroupBox("Camera Controls")
        cam_layout = QGridLayout()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        cam_layout.addWidget(self.brightness_slider, 0, 1, 1, 2)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        cam_layout.addWidget(self.contrast_slider, 1, 1, 1, 2)
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["50%", "100%", "200%", "300%", "400%"])
        self.zoom_combo.setCurrentText("100%")
        cam_layout.addWidget(self.zoom_combo, 2, 1, 1, 2)
        self.live_btn = QPushButton("Toggle Live Preview")
        self.live_btn.clicked.connect(self.toggle_live)
        cam_layout.addWidget(self.live_btn, 3, 0)
        self.start_btn = QPushButton("Start Experiment")
        self.start_btn.clicked.connect(self.start_experiment)
        cam_layout.addWidget(self.start_btn, 3, 1)
        self.stop_btn = QPushButton("Stop Experiment")
        self.stop_btn.clicked.connect(self.stop_experiment)
        cam_layout.addWidget(self.stop_btn, 3, 2)
        self.record_cb = QCheckBox("Record")
        self.record_cb.setChecked(False)
        cam_layout.addWidget(self.record_cb, 4, 2)
        self.camera_group.set_layout(cam_layout)
        lbl = QLabel("Brightness")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        cam_layout.addWidget(lbl, 0, 0)
        lbl = QLabel("Contrast")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        cam_layout.addWidget(lbl, 1, 0)
        lbl = QLabel("Zoom")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        cam_layout.addWidget(lbl, 2, 0)

        self.core_thread = LoadCoreThread(self.cfg_path)
        self.core_thread.core_loaded.connect(self.on_core_loaded)
        self.load_core = QPushButton("Load Camera Config")
        cam_layout.addWidget(self.load_core, 4, 0)
        self.load_core.clicked.connect(self.core_thread.start)
        self.stop_core = QPushButton("Refresh Camera Config")
        cam_layout.addWidget(self.stop_core, 4,1)
        self.stop_core.clicked.connect(self.core_reset)

    # Arduino Controls
        self.arduino_group = CollapsibleGroupBox("Arduino Controls")
        ar_layout = QGridLayout()
        self.serial_edit = QLineEdit("COM5"); ar_layout.addWidget(self.serial_edit,0,1)
        self.baud_combo = QComboBox(); self.baud_combo.addItems(["9600","115200","250000"]); ar_layout.addWidget(self.baud_combo,0,3)
        self.run_trigger_cb = QCheckBox("Send TTL"); self.run_trigger_cb.setCheckable(True); self.run_trigger_cb.setChecked(True)
        ar_layout.addWidget(self.run_trigger_cb,2,2)
        test_ttl_btn = QPushButton("Test TTL"); test_ttl_btn.clicked.connect(self.test_ttl); ar_layout.addWidget(test_ttl_btn,2,3)

        lbl = QLabel("Arduino Port")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl, 0, 0)
        lbl = QLabel("Baud Rate")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl, 0, 2)

        lbl = QLabel("Pulse Frequency (Hz)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl,1,0)
        self.ttl_frequency_spin = QSpinBox()
        self.ttl_frequency_spin.setRange(1, 1000) # Hz
        self.ttl_frequency_spin.setValue(40)
        ar_layout.addWidget(self.ttl_frequency_spin,1,1)

        lbl = QLabel("Train Duration (ms)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl,1,2)
        self.ttl_duration_spin = QSpinBox()
        self.ttl_duration_spin.setRange(1, 5000) # ms
        self.ttl_duration_spin.setValue(300)
        ar_layout.addWidget(self.ttl_duration_spin,1,3)

        lbl = QLabel("Pulse Mode")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl,2,0)
        self.ttl_mode_combo = QComboBox()
        self.ttl_mode_combo.addItems(["Single Pulse", "Train"])
        ar_layout.addWidget(self.ttl_mode_combo,2,1)
        self.arduino_group.set_layout(ar_layout)

        self.info_group = QGroupBox("")
        info_layout = QGridLayout()
        self.overlay_label = QLabel("READY"); self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("color: green; font-weight: bold; font-size:20px")
        info_layout.addWidget(self.overlay_label,1,0,1,2)
        self.progressbar = QProgressBar(); self.progressbar.setRange(0,100)
        info_layout.addWidget(self.progressbar,2,0,1,1)
        self.timer_label = QLabel("00:00 / 00:00"); self.timer_label.setAlignment(Qt.AlignCenter); info_layout.addWidget(self.timer_label,2,1)
        self.info_group.setLayout(info_layout)

# -------------------- Log --------------------
        self.log_group = CollapsibleGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_group.toggle_btn.setStyleSheet("""
        QPushButton {
            text-align: left;
            font-weight: bold;
            font-size: 12px;
            color: #f0f0f0;
            border: none;
            border-bottom: 2px solid #f0f0f0;  /* underline */
            padding-bottom: 4px;               /* spacing below the text */
            padding-left: 5px;
            background-color: transparent;
            }
        """)
        self.log_group.content_frame.setStyleSheet("""QFrame {border: none;background-color: #444;}""")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        #self.log_text.setStyleSheet("background-color:#444; color:#f0f0f0; font-family: monospace;")
        self.log_text.setMaximumHeight(150)
        self.log_text.setProperty("noBorder", True)
        self.log_text.setStyleSheet("""QTextEdit[noBorder='true'] {border: none;background-color: #444;color: #f0f0f0;font-family: monospace;}""")
        log_layout.addWidget(self.log_text)
        self.log_group.set_layout(log_layout)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.setSpacing(2)
        self.main_layout.addWidget(self.file_group)
        self.main_layout.addWidget(self.arduino_group)
        self.main_layout.addWidget(self.acq_group)
        self.main_layout.addWidget(self.camera_group)
        self.main_layout.addWidget(self.info_group)
        self.main_layout.addWidget(self.log_group)
        self.setLayout(self.main_layout)


    # Connect sliders & zoom to live update
        self.brightness_slider.valueChanged.connect(lambda _: self.update_live_display())
        self.contrast_slider.valueChanged.connect(lambda _: self.update_live_display())
        self.zoom_combo.currentTextChanged.connect(lambda _: self.update_live_display())

        print("[DEBUG] UI built successfully")

    # -------------------- Dark Mode --------------------
    def apply_dark_mode(self):
        self.setStyleSheet("""
            QWidget, QLabel { background-color:#2e2e2e; color:#f0f0f0; border:none }
            QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton, QComboBox { background-color:#444; color:#f0f0f0; }
            QProgressBar { background-color:#555; color:#f0f0f0; border:1px solid #888; }          
        """)
    # ---------------------Log events--------------------------
    def set_overlay(self, text, color="red"):
        self.overlay_label.setText(text)
        self.overlay_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

    def log_event(self, msg, color="white"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # include milliseconds
        self.log_text.append(f"<span style='color:{color}'>{timestamp} - {msg}</span>")

    def flush_log_queue(self):
        while not self.log_queue.empty():
            try:
                ts, msg, color = self.log_queue.get_nowait()  # expects exactly 3
                self.log_text.append(f"<span style='color:{color}'>{ts} - {msg}</span>")
            except Exception as e:
                print("Log flush error:", e)

    # -------------------- Folder & Arduino --------------------
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"Select Save Folder")
        if folder: self.save_path_edit.setText(folder)

    def test_ttl(self):
        if not self.arduino: self.open_arduino()
        if self.arduino: 
            try:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3] 
                self.log_queue.put((ts, "TTL sent successfully", "yellow"))
                self.arduino.write(b'H')
                self.arduino.flush()
                self.arduino.flushInput()  # Clear any previous input
                self.arduino.flushOutput()  # Clear any previous output
            except Exception as e:
                self.set_overlay("TTL ERROR", color="red")
                QTimer.singleShot(500, lambda: self.set_overlay("READY", color="green"))
                self.log_event(f"Error sending TTL pulse: {e}", color="red")

    def send_ttl_threaded(self, frequency_hz=40, duration_ms=300, mode="Train"):
        if not self.arduino:
            self.open_arduino()
        if not self.arduino:
            return

        def ttl_worker():
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            try:
                if mode == "Single Pulse":
                # Send one TTL pulse (1 ms)
                    self.arduino.write(b'H')
                    time.sleep(0.001)
                    self.arduino.write(b'L')
                    self.log_queue.put((ts, f"{mode} sent successfully", "yellow"))
                else:
                # Calculate interval between pulses
                    interval = 1.0 / frequency_hz
                    pulses = int(duration_ms / 1000 * frequency_hz)
                    for _ in range(pulses):
                        self.arduino.write(b'H')
                        time.sleep(0.001)  # pulse width 1 ms
                        self.arduino.write(b'L')
                        time.sleep(interval - 0.001)
                        self.log_queue.put((ts, f"{mode} sent {pulses} pulses at {frequency_hz} Hz for {duration_ms} ms successfully", "yellow"))

            except Exception as e:
                self.set_overlay("TTL ERROR", color="red")
                QTimer.singleShot(500, lambda: self.set_overlay("READY", color="green"))
                self.log_event(f"Error sending TTL pulse: {e}", color="red")

        threading.Thread(target=ttl_worker, daemon=True).start()

    def open_arduino(self):
        if self.run_trigger_cb.isChecked():
            try:
                self.arduino = serial.Serial(self.serial_edit.text(),int(self.baud_combo.currentText()),timeout=1)
            except Exception as e:
                self.log_event(f"Arduino error: {e}"); self.arduino=None
        else: 
            self.log_event("Arduino not enabled", color = "red")
            QTimer.singleShot(500, lambda: self.set_overlay("READY", color = "green"))

        # -------------------- Live Preview --------------------
    def toggle_live(self):
        # target_fps = int(self.fps_combo.currentText())
        if self.core is None:
            self.log_event("Cannot start live preview: core not ready", color="red")
            return

    # Stop live preview if running
        if self.live_window and self.live_window.isVisible():
            self.live_window.hide()
            self.set_overlay("LIVE OFF", "red")
            return

    # Show live preview window
        if self.live_window is None:
            self.live_window = LivePreviewWindow(core=self.core, lock=self.camera_lock)
        self.live_window.show()
        self.set_overlay("LIVE ON", "green")

    # Start thread if not running
        if self.live_thread is None or not self.live_thread.isRunning():
            fps = int(self.fps_combo.currentText())
            self.live_thread = LivePreviewThread(core=self.core, lock=self.camera_lock)
            self.live_thread.image_ready.connect(self.update_live_frame)
            self.live_thread.log_event_signal.connect(self.log_event)
            self.live_thread.burst_done_signal.connect(self.on_burst_done)
            self.live_thread.start()

    def start_live(self):
        if not self.live_window:
            self.live_window = LivePreviewWindow(core=self.core, lock=self.camera_lock)
        self.live_window.show()

        # (Re)start live thread at target fps
        if self.live_thread and self.live_thread.isRunning():
            self.live_thread.stop()
            self.live_thread.wait()

        self.live_thread = LivePreviewThread(self.core, lock=self.camera_lock)
        self.live_thread.image_ready.connect(self.update_live_frame)
        self.live_thread.start()
        self.overlay_label.setText("LIVE ON")

    def update_live_frame(self, arr):
        # print("[DEBUG] Received frame:", arr.shape, arr.min(), arr.max())
        self.last_frame = arr
        self.update_live_display()

    def update_live_display(self):
        if getattr(self, "last_frame", None) is None:
            return
        if self.live_window is None or not hasattr(self.live_window, "label"):
            return
        arr = self.last_frame
        brightness = self.brightness_slider.value() * 256
        contrast = self.contrast_slider.value()/100.0
        arr_adj = np.clip(arr*contrast+brightness,0,65535)
        arr8 = ((arr_adj-arr_adj.min())/(np.ptp(arr_adj)+1e-6)*255).astype(np.uint8)
        # print("[DEBUG] Frame min/max:", arr.min(), arr.max())
        qimg = QImage(arr8.copy(),arr.shape[1],arr.shape[0],QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        zoom = int(self.zoom_combo.currentText()[:-1])/100.0
        w,h = int(pixmap.width()*zoom), int(pixmap.height()*zoom)
        self.live_window.label.setPixmap(pixmap.scaled(w,h,Qt.KeepAspectRatio,Qt.SmoothTransformation))

    # -------------------- Experiment --------------------
    def start_experiment(self):
        if self.arduino is None and self.run_trigger_cb.isChecked():
            self.open_arduino()
            self.arduino.flushInput()  # Clear any previous input
            self.arduino.flushOutput()  # Clear any previous output
            if self.arduino is None:
                self.log_event("Cannot start experiment: Arduino not connected", "red")
                return
            
        if not self.core:
            self.log_event("Cannot start experiment: core not ready", "red")
            return
        
        if not self.record_cb.isChecked():
            self.record_cb.setChecked(True)
            self.log_event("Recording enabled for experiment", "yellow")
        
        base_folder = self.save_path_edit.text() or "."
        exp_folder = f"{self.expt_name_edit.text()}_{self.expt_type_edit.text()}_{self.final_titer_edit.text()}"
        file_name = f"{self.mouse_id_edit.text()}_{datetime.now():%H%M%S}_{datetime.now():%d%m%y}"
        self.title_folder = os.path.join(base_folder, exp_folder)
        self.session_folder = os.path.join(self.title_folder, file_name)
        os.makedirs(self.session_folder, exist_ok=True)

        self.experiment_duration_s = float(self.total_time_spin.value()) * 60.0  # convert to seconds
        self.burst_duration_s = float(self.burst_duration_spin.value())
        self.pause_between_bursts_s = float(self.wait_interval_spin.value())
        self.start_time = time.time()
        self.experiment_running = True
        self.burst_index = 0
        self.target_fps = int(self.fps_combo.currentText())
        self.total_bursts = int(self.experiment_duration_s / (self.burst_duration_s + self.pause_between_bursts_s))
        self.burst_job_queue = Queue(maxsize=2000)
        self.writer_thread = FrameWriterThread(self.burst_job_queue)
        self.writer_thread.log_event_signal.connect(self.log_event)
        self.writer_thread.start()

        cam = self.core.getCameraDevice()
        try:
            self.core.setProperty(cam,"ClearMode", "Never")
            self.core.setProperty(cam,"ClearCycles", 2)
        except Exception:
            pass

        if self.live_thread is None or not self.live_thread.isRunning():
            self.live_thread = LivePreviewThread(self.core, lock=self.camera_lock)
            self.live_thread.image_ready.connect(self.update_live_frame)
            self.live_thread.log_event_signal.connect(self.log_event)
            self.live_thread.burst_done_signal.connect(self.on_burst_done)
            self.live_thread.start()

        if self.live_window is None or not self.live_window.isVisible():
            self.live_window = LivePreviewWindow(core=self.core, lock=self.camera_lock)
            self.live_window.show()

        self.set_overlay("EXPERIMENT IN PROGRESS...", color="blue")
        QTimer.singleShot(0, self.start_burst_and_ttl)

    def on_burst_started(self, burst_idx):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_queue.put((ts, f"Burst {burst_idx} started", "green"))

    def start_burst_and_ttl(self):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ttl_freq = self.ttl_frequency_spin.value()
        ttl_duration = self.ttl_duration_spin.value()
        burst_number = self.burst_index + 1
        self.burst_index = burst_number
        burst_duration = float(self.burst_duration_spin.value()) # convert ms to s
        ttl_delay_ms = int(self.trigger_time_spin.value())

        if not self.experiment_running:
            return

        if burst_number == 1:
            self.log_queue.put((ts, f"Burst 1 scheduled to start immediately (TTL in {ttl_delay_ms} ms)", "orange"))

    # Schedule TTL independently

        # Start burst
        # self.burst_thread = BurstThread(burst_index=self.burst_index, duration_s=burst_duration)
        self.burst_thread = BurstThread(burst_index=burst_number, duration_s=burst_duration)

    # Connect live preview frames to burst collection
        self.live_thread.new_frame.connect(self.burst_thread.collect_frame)

    # Connect GUI logging
        self.burst_thread.burst_started.connect(self.on_burst_started)
        self.burst_thread.burst_done.connect(self.on_burst_done)
        self.burst_thread.log_event_signal.connect(self.log_event)
    # Start burst
        self.burst_thread.start()
        QTimer.singleShot(ttl_delay_ms,lambda: self.send_ttl_threaded(frequency_hz=ttl_freq,duration_ms=ttl_duration,mode=self.ttl_mode_combo.currentText()))

    def on_burst_done(self, burst_idx, frames_array):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_queue.put((ts, f"Burst {burst_idx} done, {len(frames_array)} frames captured", "green"))

    # Save burst to disk
        save_folder = self.session_folder
        os.makedirs(save_folder, exist_ok=True)
        out_path = os.path.join(save_folder, f"burst_{burst_idx:03d}.tif")
    
    # Queue the array to the writer
        self.burst_job_queue.put((out_path, frames_array))
        self.log_queue.put((ts, f"Burst {burst_idx} for Mouse {self.mouse_id_edit.text()} Saved to: {self.title_folder}", "green"))
        if self.experiment_running:
            burst_interval = float(self.wait_interval_spin.value()) * 1000
            burst_duration = float(self.burst_duration_spin.value()) * 1000  # ms
            ttl_delay_ms = int(self.trigger_time_spin.value())
            burst_number = self.burst_index + 1

        if self.burst_index < self.total_bursts:
            self.log_queue.put((ts, f"Burst {burst_number} scheduled in {(burst_interval) / 1000:.1f} s (TTL after {ttl_delay_ms} ms)" , "orange"))
            QTimer.singleShot(int(burst_interval), self.start_burst_and_ttl)
        else:
            self.finish_experiment()
            
    def finish_experiment(self):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.stop_experiment()
        self.log_queue.put((ts, "All bursts completed", "orange"))

    def stop_experiment(self):
        self.experiment_running = False
        self.experiment_stopped = True

        cam = self.core.getCameraDevice()
        self.core.setProperty(cam,"ClearMode", "Pre-Exposure")
        self.core.setProperty(cam,"ClearCycles", 2)

        if self.writer_thread and self.writer_thread.isRunning():
            self.writer_thread.stop()

        if hasattr(self, "current_burst_thread") and self.burst_thread.isRunning():
            self.burst_thread.stop()
            self.burst_thread.wait()

        self.set_overlay("EXPERIMENT STOPPED", color="red")
        QTimer.singleShot(2000, lambda: self.set_overlay("READY", color="green"))

    def core_reset(self):
        # stop acquisition safely
        try:
            if getattr(self, "core", None) and getattr(self.core, "isSequenceRunning", lambda: False)():
                self.core.stopSequenceAcquisition()
        except Exception:
            pass

        if getattr(self, "core", None):
            try:
                self.core.reset()
            except Exception as e:
                self.log_event(f"Error resetting core: {e}", "red")
            # purge the attribute and recreate a fresh instance
            del self.core
            self.core = None

        # restart core thread fresh
        try:
            if getattr(self, "core_thread", None) and self.core_thread.isRunning():
                self.core_thread.stop()
            self.core_thread = LoadCoreThread(self.cfg_path)
            self.core_thread.core_loaded.connect(self.on_core_loaded)
            self.core_thread.start()
        except Exception as e:
            self.log_event(f"Failed to restart core thread: {e}", "red")
        self.log_event("Core Refreshed", "red")


    # -------------------- Settings --------------------
    def load_settings(self):
        try:
            with open(self.settings_file, "r") as f:
                settings = json.load(f)
            self.settings = settings
            self.save_path_edit.setText(settings.get("save_path", ""))
            self.mouse_id_edit.setText(settings.get("mouse_id", "Mouse_001"))
            self.expt_type_edit.setText(settings.get("expt_type", "GCamp"))
            self.final_titer_edit.setText(settings.get("final_titer", "e11"))
            self.brightness_slider.setValue(settings.get("brightness", 0))
            self.contrast_slider.setValue(settings.get("contrast", 100))
            self.expt_name_edit.setText(settings.get("expt_name", "Stim_Exp"))
            self.total_time_spin.setValue(settings.get("total_time", 5))
            self.wait_interval_spin.setValue(settings.get("wait_interval", 10))
            self.exp_spin.setValue(settings.get("exp", 10))
            self.zoom_combo.setCurrentText(settings.get("zoom", "100%"))
            # self.batch_size_spin.setValue(settings.get("batch_size", 500))
            self.trigger_time_spin.setValue(settings.get("trigger_time", 2))
            self.serial_edit.setText(settings.get("arduino_port", "COM5"))
            self.baud_combo.setCurrentText(settings.get("baud_rate", "115200"))
        except FileNotFoundError:
            self.log_event("Settings file not found, using defaults.")        

    def save_settings(self):
        """Save current GUI settings to JSON file."""
        self.settings.update({
            "save_path": self.save_path_edit.text(),
            "expt_name": self.expt_name_edit.text(),
            "mouse_id": self.mouse_id_edit.text(),
            "expt_type": self.expt_type_edit.text(),
            "final_titer": self.final_titer_edit.text(),
            "total_time": self.total_time_spin.value(),
            "wait_interval": self.wait_interval_spin.value(),
            #"batch_size": self.batch_size_spin.value(),
            "trigger_time": self.trigger_time_spin.value(),
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_slider.value(),
            "zoom": self.zoom_combo.currentText(),
            "arduino_port": self.serial_edit.text(),
            "baud_rate": self.baud_combo.currentText(),
            "send_ttl": self.run_trigger_cb.isChecked(),
            "record": self.record_cb.isChecked(),
            "exp": self.exp_spin.value()
        })

        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.log_event(f"Error saving settings: {e}")

    def closeEvent(self, event):
    # Stop live preview thread
        if self.live_thread and self.live_thread.isRunning():
            self.live_thread.stop()
            self.live_thread.wait(2000)
            self.live_thread = None

    # Stop writer thread
        if self.writer_thread and self.writer_thread.isRunning():
            self.writer_thread.stop()
            self.writer_thread.wait(2000)
            self.writer_thread = None

    # Close Arduino
        if getattr(self, "arduino", None) and getattr(self.arduino, "is_open", False):
            self.arduino.close()

    # Close live window
        if self.live_window:
            self.live_window.close()

    # Reset camera core to release all devices
        if getattr(self, "core", None):
            try:
                self.core.stopSequenceAcquisition()
                self.core.reset()
            except Exception as e:
                self.log_event(f"Error resetting core: {e}", "red")
            del self.core
            self.core = None

    # Save settings
        self.save_settings()
        event.accept()
        super().closeEvent(event)

# -------------------- Main --------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Windows-safe

    cfg_path = "C:\\Program Files\\Micro-Manager-2.0\\Scientifica.cfg"  # Adjust  as needed

    app = QApplication(sys.argv)
    gui = LiveImagingGUI(cfg_path)
    gui.show()

    # core_thread = LoadCoreThread(cfg_path)
    # core_thread.core_loaded.connect(gui.on_core_loaded)
    # core_thread.start()

    def cleanup():
    # Safe stop live thread
        if getattr(gui, "live_thread", None) and gui.live_thread.isRunning():
            gui.live_thread.stop()
            gui.live_thread.wait()

    # Safe stop writer thread
        if getattr(gui, "writer_thread", None) and gui.writer_thread.isRunning():
            gui.writer_thread.stop()
            gui.writer_thread.wait()

app.aboutToQuit.connect(cleanup)
sys.exit(app.exec_())
