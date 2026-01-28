import json
import os
import sys
from datetime import datetime
import time
import numpy as np
import serial
import tifffile
import threading

from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton,
    QSlider, QComboBox, QSpinBox, QLineEdit,
    QTextEdit, QFileDialog, QDoubleSpinBox, QGridLayout, QCheckBox, QSizePolicy, QProgressBar, QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from pycromanager import Core

# -------------------- Live Preview Thread --------------------
class LivePreviewThread(QThread):
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, core, lock=None, interval_ms=100):
        super().__init__()
        self.core = core
        self.lock = lock
        self.interval_ms = interval_ms
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            try:
                if self.lock:
                    self.lock.acquire()
                try:
                    self.core.snap_image()
                    pixels = self.core.get_image()

                    if pixels is not None:
                        width = self.core.get_image_width()
                        height = self.core.get_image_height()
                        arr = np.array(pixels, dtype=np.uint16).reshape((height, width))
                        self.image_ready.emit(arr)
                finally:
                    if self.lock:
                        self.lock.release()
            except Exception as e:
                print(f"Live preview error: {e}")
            self.msleep(self.interval_ms)

    def stop(self):
        self.running = False
        self.wait()

# -------------------- Live Imaging Window --------------------
class LiveImageWindow(QWidget):
    def __init__(self, zoom_combo):
        super().__init__()
        self.setWindowTitle("Live Imaging")
        self.zoom_combo = zoom_combo
        self.zoom_factor = 1.0
        self.image_label = QLabel("Live image will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.setMinimumSize(800, 600)

    def set_zoom_factor(self, zoom_factor):
        self.zoom_factor = zoom_factor
        self.update_pixmap_size()

    def set_pixmap(self, pixmap):
        self.image_label.setPixmap(pixmap)
        self.update_pixmap_size()

    def update_pixmap_size(self):
        if self.image_label.pixmap():
            scaled = self.image_label.pixmap().scaled(
                int(self.width() * self.zoom_factor),
                int(self.height() * self.zoom_factor),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap_size()

# -------------------- Main GUI --------------------
class LiveImagingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.core = None
        self.settings = {}
        self.setWindowTitle("Timed Acquisition with Stim & Preview")
        self.settings_file = "stim_gui_settings.json"
        self.live_window = None
        self.tiff_stack = []
        self.batch_stack = []
        self.build_widgets()
        self.apply_dark_mode()
        self.load_settings()
        self.timer = QTimer()
        self.arduino = None
        self.camera_lock = None
        self.live_thread = None
        self.show()
        self.original_height = self.height()


    def build_widgets(self):

        #---------------------File Saving---------------------
        self.file_group = QGroupBox("File Saving")
        self.file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; 
                border: 2px solid #f0f0f0; 
                border-radius: 5px; 
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)
        file_group = QGridLayout()
        file_group.addWidget(QLabel("Save Folder"), 0, 0)
        self.save_path_edit = QLineEdit()
        file_group.addWidget(self.save_path_edit, 0, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        file_group.addWidget(browse_button, 0, 2)
        file_group.addWidget(QLabel("Experiment Name"), 1, 0)
        self.expt_name_edit = QLineEdit("Stim_Exp")
        file_group.addWidget(self.expt_name_edit, 1, 1)
        self.file_group.setLayout(file_group)

        # -------------------- Acquisition Settings --------------------
        self.acq_group = QGroupBox("Acquisition Settings")
        self.acq_group.setStyleSheet(self.file_group.styleSheet())
        acq_layout = QGridLayout()
        acq_layout.addWidget(QLabel("Total Duration (min)"), 0, 0)
        self.total_time_spin = QDoubleSpinBox(); self.total_time_spin.setValue(5)
        acq_layout.addWidget(self.total_time_spin, 0, 1)
        acq_layout.addWidget(QLabel("Acquisition Interval (ms)"), 1, 0)
        self.interval_spin = QDoubleSpinBox(); self.interval_spin.setValue(10); self.interval_spin.setMinimum(1); self.interval_spin.setMaximum(60000)
        acq_layout.addWidget(self.interval_spin, 1, 1)
        acq_layout.addWidget(QLabel("Batch Size"), 2, 0)
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setMinimum(1); self.batch_size_spin.setMaximum(1000); self.batch_size_spin.setValue(500)
        acq_layout.addWidget(self.batch_size_spin, 2, 1)
        acq_layout.addWidget(QLabel("Stim Time (min)"), 3, 0)
        self.trigger_time_spin = QDoubleSpinBox(); self.trigger_time_spin.setValue(2)
        acq_layout.addWidget(self.trigger_time_spin, 3, 1)
        self.acq_group.setLayout(acq_layout)

        # -------------------- Camera Controls --------------------
        self.camera_group = QGroupBox("Camera Controls")
        self.camera_group.setStyleSheet(self.file_group.styleSheet())
        cam_layout = QGridLayout()
        cam_layout.addWidget(QLabel("Zoom Level:"), 0, 0)
        self.zoom_combo = QComboBox(); self.zoom_combo.addItems(["50%", "100%", "200%", "300%", "400%"])
        self.zoom_combo.setCurrentText("100%"); self.zoom_combo.currentTextChanged.connect(self.update_zoom)
        cam_layout.addWidget(self.zoom_combo, 0, 1, 1, 2)
        cam_layout.addWidget(QLabel("Brightness"), 1, 0)
        self.brightness_slider = QSlider(Qt.Horizontal); self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100); self.brightness_slider.setValue(0)
        cam_layout.addWidget(self.brightness_slider, 1, 1, 1, 3)
        cam_layout.addWidget(QLabel("Contrast"), 2, 0)
        self.contrast_slider = QSlider(Qt.Horizontal); self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(300); self.contrast_slider.setValue(100)
        cam_layout.addWidget(self.contrast_slider, 2, 1, 1, 3)
        live_button = QPushButton("Live Preview"); live_button.clicked.connect(self.start_live)
        cam_layout.addWidget(live_button, 3, 0)
        self.start_button = QPushButton("Start Experiment"); self.start_button.clicked.connect(self.start_experiment)
        cam_layout.addWidget(self.start_button, 3, 1)
        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.clicked.connect(self.request_stop_experiment)
        cam_layout.addWidget(self.stop_button, 3, 2)
        self.overlay_label = QLabel("READY"); self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
        cam_layout.addWidget(self.overlay_label, 4, 0, 1, 2)
        self.progressbar = QProgressBar(); self.progressbar.setRange(0, 100)
        cam_layout.addWidget(self.progressbar, 5, 0, 1, 2)
        self.timer_label = QLabel("00:00 / 00:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        cam_layout.addWidget(self.timer_label, 5, 2, 1, 2)
        self.camera_group.setLayout(cam_layout)

        #--------------------- Arduino Controls ---------------------
        self.arduino_group = QGroupBox("Arduino Controls")
        self.arduino_group.setStyleSheet(self.file_group.styleSheet())
        arduino_layout = QGridLayout()
        arduino_layout.addWidget(QLabel("Arduino Port"), 2, 0)
        self.serial_port_edit = QLineEdit("COM5")
        arduino_layout.addWidget(self.serial_port_edit, 2, 1)
        arduino_layout.addWidget(QLabel("Baud Rate:"), 1, 0)
        self.baud_rate_combo = QComboBox()
        self.baud_rate_combo.addItems(["9600", "115200", "250000"])
        arduino_layout.addWidget(self.baud_rate_combo, 1, 1)
        test_ttl_button = QPushButton("Test TTL Trigger")
        test_ttl_button.clicked.connect(self.test_ttl_trigger)
        arduino_layout.addWidget(test_ttl_button, 3, 1)
        self.run_trigger_cb = QCheckBox("Send Arduino Trigger") 
        self.run_trigger_cb.setChecked(True)
        arduino_layout.addWidget(self.run_trigger_cb, 3, 0)
        self.arduino_group.setLayout(arduino_layout)


        # -------------------- Main Content Widget --------------------
        self.main_content_widget = QWidget()
        main_content_layout = QVBoxLayout()
        main_content_layout.addWidget(self.file_group)
        main_content_layout.addWidget(self.arduino_group)
        main_content_layout.addWidget(self.acq_group)   
        main_content_layout.addWidget(self.camera_group)
        self.main_content_widget.setLayout(main_content_layout)
        self.toggle_log_btn = QPushButton("Hide Log"); self.toggle_log_btn.setCheckable(True)
        self.toggle_log_btn.toggled.connect(self.toggle_log)
        main_content_layout.addWidget(self.toggle_log_btn)


        # -------------------- Log & TTL --------------------
        self.log_widget = QWidget()
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.log_widget.setLayout(log_layout)

        # -------------------- Splitter --------------------
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.main_content_widget)
        self.splitter.addWidget(self.log_widget)
        #self.splitter.setSizes([int(self.height() * 0.7), int(self.height() * 0.3)])  # initial proportion

        # -------------------- Outer Layout --------------------
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.splitter)
        self.setLayout(outer_layout)

        
    # -------------------- Utility Functions --------------------
    def apply_dark_mode(self):
        self.setStyleSheet("background-color: #2e2e2e; color: #f0f0f0;")

    def set_overlay(self, text, color="red"):
        self.overlay_label.setText(text)
        self.overlay_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

    def log(self, message):
        self.log_text.append(message)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.save_path_edit.setText(folder)

    def update_zoom(self, text):
        zoom_factor = int(text[:-1]) / 100
        if self.live_window:
            self.live_window.set_zoom_factor(zoom_factor)

    def apply_brightness_contrast(self, arr, dtype=np.uint16):
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        arr_adj = contrast * arr + brightness
        if dtype == np.uint8:
            return np.clip(arr_adj, 0, 255).astype(np.uint8)
        elif dtype == np.uint16:
            return np.clip(arr_adj, 0, 65535).astype(np.uint16)
        else:
            return arr_adj.astype(dtype)
        
    def toggle_log(self, checked):
        if checked:
            self.toggle_log_btn.setText("Show Log")
            self.log_widget.hide()
        else:
            self.toggle_log_btn.setText("Hide Log")
            self.log_widget.show()
    # Adjust main content to fill space proportionally
       # self.splitter.setSizes([int(self.height() * 0.9), int(self.height() * 0.1)])


    # -------------------- Live Preview --------------------
    def start_live(self):
        if self.live_window is None:
            self.live_window = LiveImageWindow(self.zoom_combo)
    # Restore previous geometry if available
            geom = self.settings.get("live_window_geometry")
            if geom:
                self.live_window.restoreGeometry(bytes.fromhex(geom))
            self.live_window.show()

        if self.live_thread is None:
            self.live_thread = LivePreviewThread(self.core, lock=self.camera_lock, interval_ms=100)
            self.live_thread.image_ready.connect(self.update_live_window)
            self.live_thread.start()

    def stop_live(self):
        if self.live_thread:
            self.live_thread.stop()
            self.live_thread = None

        if self.experiment_timer:
            self.experiment_timer.stop()
            
        if self.arduino and self.arduino.is_open:
            self.arduino.close()

        if self.batch_stack:
            self.save_batch_stack()
            self.merge_batches_to_final_tiff()
            self.set_overlay("EXPERIMENT STOPPED", color="red")
            self.log("Experiment stopped.")

        if self.live_window:
            self.live_window.close()
        return

    def request_stop_experiment(self):
        self.experiment_stopped = True

    def update_live_window(self, arr):
        # store latest frame
        self.last_frame = arr.copy()
        self._update_display_from_frame(arr)

    def refresh_display(self):
        """Reapply brightness/contrast to the last frame."""
        if self.last_frame is not None:
            self._update_display_from_frame(self.last_frame)
            self.last_frame = None  # Clear last frame to avoid reprocessing    

    def _update_display_from_frame(self, arr):
        """Helper to convert a frame to 8-bit with brightness/contrast and update live window."""
        arr8 = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255)
        arr8 = self.apply_brightness_contrast(arr8, dtype=np.uint8)
        qimg = QImage(arr8.data, arr.shape[1], arr.shape[0], QImage.Format_Grayscale8)
        self.live_window.set_pixmap(QPixmap.fromImage(qimg))

    # -------------------- Experiment Functions --------------------
    def start_experiment(self):
        if self.live_window is None:
            self.start_live()
        self.set_overlay("ACQUIRING IMAGES...", color="blue")
        self.log("Experiment starting...")
        self.total_duration = self.total_time_spin.value() * 60
        self.interval = self.interval_spin.value()
        self.trigger_time = self.trigger_time_spin.value() * 60
        self.batch_stack = []
        self.frames_taken = 0
        baud_rate = int(self.baud_rate_combo.currentText())

        total_seconds = self.total_duration
        interval_seconds = self.interval / 1000
        self.total_frames = int(total_seconds / interval_seconds)
        self.progressbar.setValue(0)

        if self.run_trigger_cb.isChecked():
            try:
                self.arduino = serial.Serial(self.serial_port_edit.text(), baud_rate, timeout=1)
                self.arduino.flush()
            except Exception as e:
                self.log(f"Could not open Arduino: {e}")
                self.arduino = None

        self.start_time = int(time.time() * 1000)  # store start time in ms
        self.last_image_time = 0
        self.ttl_pulse_sent = False
        self.experiment_stopped = False  # Flag for experiment stop

        self.experiment_timer = QTimer()
        self.experiment_timer.timeout.connect(self.run_experiment)
        self.experiment_timer.start(10)
        self.log("Experiment started.")

    def run_experiment(self):
        elapsed = time.time() - (self.start_time / 1000)
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        elapsed_ms_total = int(elapsed * 1000)
        total_min = int(self.total_duration // 60)
        total_sec = int(self.total_duration % 60)
        self.timer_label.setText(f"{elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}")

        # Snap image at interval
        if elapsed_ms_total - self.last_image_time >= self.interval:
            self.snap_and_store_image()
            self.last_image_time += self.interval

        # Single TTL pulse at trigger time
        if self.run_trigger_cb.isChecked() and not self.ttl_pulse_sent and elapsed >= self.trigger_time:
            self.send_ttl_pulse()
            self.ttl_pulse_sent = True

        # Stop experiment when total duration reached
        if elapsed >= self.total_duration:
            self.experiment_timer.stop()
            if self.arduino and self.arduino.is_open:
                self.arduino.close()

            if self.batch_stack:
                self.save_batch_stack()
            self.merge_batches_to_final_tiff()
            self.set_overlay("EXPERIMENT COMPLETE", color="green")
            self.log("Experiment completed.")

        elif self.experiment_stopped:
            self.experiment_timer.stop()
            if self.arduino and self.arduino.is_open:
                self.arduino.close()

            if self.batch_stack:
                self.save_batch_stack()
            self.merge_batches_to_final_tiff()
            self.set_overlay("EXPERIMENT STOPPED", color="red")
            self.log("Experiment stopped.")
    
    def snap_and_store_image(self):
        try:
            if self.camera_lock:
                self.camera_lock.acquire()
            try:
                self.core.snap_image()
                pixels = self.core.get_image()
                width = self.core.get_image_width()
                height = self.core.get_image_height()
                if pixels is not None:
                    arr = np.array(pixels, dtype=np.uint16).reshape((height, width))
                    self.batch_stack.append(arr)  # Only append once

            finally:
                if self.camera_lock:
                    self.camera_lock.release()

            if pixels is not None:
                percent = int((len(self.batch_stack) / self.total_frames) * 100)
                self.progressbar.setValue(percent)

                # Store for live re-render with sliders
               # self.last_frame = arr.copy()
                # self._update_display_from_frame(arr)

                if len(self.batch_stack) >= self.batch_size_spin.value():
                    self.save_batch_stack()
        except Exception as e:
            self.log(f"Error snapping image: {e}")

    def save_batch_stack(self):
        save_dir = self.save_path_edit.text()
        os.makedirs(save_dir, exist_ok=True)
        expt_name = self.expt_name_edit.text()
        date_str = datetime.now().strftime("%d%m%y")
        timestamp = int(time.time())
        folder = f"{date_str}_{expt_name}"
        batch_folder = f"{save_dir}/{folder}"
        os.makedirs(batch_folder, exist_ok=True)
        batch_file = f"{batch_folder}/{date_str}_{expt_name}__batch_{timestamp}.tiff"
        tifffile.imwrite(batch_file, np.array(self.batch_stack))
        self.log(f"Saved batch TIFF: {batch_file}"); self.log(f"Saved {len(self.batch_stack)} frames to batch.")
        self.batch_stack = []

    def merge_batches_to_final_tiff(self):
        import glob
        save_dir = self.save_path_edit.text()
        expt_name = self.expt_name_edit.text()
        timestamp = int(time.time())
        date_str = datetime.now().strftime("%d%m%y")
        folder = f"{date_str}_{expt_name}"
        batch_folder = f"{save_dir}/{folder}"
        batch_files = sorted(glob.glob(f"{batch_folder}/{date_str}_{expt_name}_batch_*.tiff"))

        if not batch_files:
            self.log("No batch files found to merge.")
            return

        all_frames = []
        for f in batch_files:
            stack = tifffile.imread(f)
            all_frames.append(stack)
        all_frames = np.concatenate(all_frames, axis=0)

        # Find a unique final filename
        base_final_file = f"{save_dir}/{folder}/{expt_name}_.tiff"
        final_file = base_final_file
        i = 1
        while os.path.exists(final_file):
            final_file = f"{save_dir}/{folder}/{expt_name}_E{i}.tiff"
            i += 1

        tifffile.imwrite(final_file, all_frames)
        self.log(f"Final TIFF saved: {final_file}")

        for f in batch_files:
            os.remove(f)

    # -------------------- TTL Trigger --------------------
    def test_ttl_trigger(self):
        if not self.run_trigger_cb.isChecked():
            self.log("TTL test skipped: Arduino trigger is disabled.")
            return
        if not self.arduino or not self.arduino.is_open:
            try:
                self.arduino = serial.Serial(self.serial_port_edit.text(), 115200, timeout=1)
                self.arduino.flush()
            except Exception as e:
                self.log(f"Could not open Arduino for TTL test: {e}")
                return
        self.set_overlay("TESTING TTL...", color="orange")
        self.log("[Stim] Sending test TTL pulse...")
        try:
            self.arduino.write(b'H')
            self.arduino.flush()
            self.log("[Stim] TTL pulse sent successfully.")
        except Exception as e:
            self.log(f"[Stim] Error sending TTL: {e}")
        QTimer.singleShot(500, lambda: self.set_overlay("READY", color="green"))

    def send_ttl_pulse(self):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(b'H')
                self.arduino.flush()
                self.log("[Stim] TTL pulse HIGH")
                self.set_overlay("TRIGGERING ARDUINO...", color="red")
                #QTimer.singleShot(50, self.reset_ttl)
                QTimer.singleShot(500, lambda: self.set_overlay("ACQUIRING IMAGES...", color="blue"))
            except Exception as e:
                self.log(f"[Stim] Error sending TTL: {e}")

    def reset_ttl(self):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(b'L')
                self.arduino.flush()
                self.log("[Stim] TTL pulse LOW")
            except Exception as e:
                self.log(f"[Stim] Error resetting TTL: {e}")

    # -------------------- Settings --------------------
    def load_settings(self):
        try:
            with open(self.settings_file, "r") as f:
                settings = json.load(f)
            self.settings = settings
            self.save_path_edit.setText(settings.get("save_path", ""))
            self.expt_name_edit.setText(settings.get("expt_name", "Stim_Exp"))
            self.total_time_spin.setValue(settings.get("total_time", 5))
            self.interval_spin.setValue(settings.get("interval", 10))
            self.batch_size_spin.setValue(settings.get("batch_size", 500))
            self.trigger_time_spin.setValue(settings.get("trigger_time", 2))
            self.serial_port_edit.setText(settings.get("serial_port", "COM5"))
            self.baud_rate_combo.setCurrentText(settings.get("baud_rate", "115200"))

        # Restore main window geometry
            geom = settings.get("main_window_geometry")
            if geom:
                self.restoreGeometry(bytes.fromhex(geom))

            self.settings.setdefault("live_window_geometry", None)
            self.log("Settings loaded.")
        except FileNotFoundError:
            self.log("Settings file not found, using defaults.")
        except Exception as e:
            self.log(f"Error loading settings: {e}")

    def save_settings(self):
        self.settings.update({
            "save_path": self.save_path_edit.text(),
            "expt_name": self.expt_name_edit.text(),
            "total_time": self.total_time_spin.value(),
            "interval": self.interval_spin.value(),
            "trigger_time": self.trigger_time_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "serial_port": self.serial_port_edit.text(),
            "baud_rate": self.baud_rate_combo.currentText(),
            "main_window_geometry": self.saveGeometry().toHex().data().decode(),
            "live_window_geometry": self.live_window.saveGeometry().toHex().data().decode() if self.live_window else self.settings.get("live_window_geometry")
        })
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
            self.log("Settings saved.")
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def closeEvent(self, event):
        self.settings["main_window_geometry"] = self.saveGeometry().toHex().data().decode()
        if self.live_window:
            self.settings["live_window_geometry"] = self.live_window.saveGeometry().toHex().data().decode()
        self.save_settings()
        if self.live_thread:
            self.live_thread.stop()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        if self.live_window:
            self.live_window.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = LiveImagingGUI()
    gui.show()
    sys.exit(app.exec_())
