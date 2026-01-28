import sys, os, time, json
from datetime import datetime
import numpy as np
import serial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QSpinBox,
    QSlider, QComboBox, QVBoxLayout, QGridLayout, QGroupBox, QProgressBar, QCheckBox,
    QFileDialog, QSizePolicy, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from pycromanager import Core, Acquisition
import tifffile
from queue import Queue
from tifffile import imwrite, imread


# -------------------- Frame Writer Thread --------------------

class FrameWriterThread(QThread):
    def __init__(self, frame_queue: Queue, folder: str, basename: str = "live", save_every: int = 1):
        super().__init__()
        self.queue = frame_queue
        self.folder = folder ##Send from main GUI
        os.makedirs(self.folder, exist_ok=True)
        self.basename = basename
        self.save_every = max(1, int(save_every))
        self._running = False
        self._idx = 0

    def run(self):
        self._running = True
        while self._running:
            try:
                arr = self.queue.get(timeout=0.1)
            except Exception:
                continue
            self._idx += 1
            if self._idx % self.save_every != 0:
                continue
            fname = os.path.join(self.folder, f"{self.basename}_{self._idx:06d}.tif")
            # self.log_event(f"Saved batch TIFF: {fname}"); self.log_event(f"Saved {len(arr)} frames to batch.")
            try:
                imwrite(fname, arr, photometric='minisblack')
            except Exception as e:
                # self.log_event(f"[Writer] save error: {e}")
                return

###### Maybe create another Class/Thread for batching and Merging####
    ####Figure out Logs between Classes####
    ###Batch Stack should be done from Directory####
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
            #self.log_event("No batch files found to merge.")
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
        #self.log_event(f"Final TIFF saved: {final_file}")

        for f in batch_files:
            os.remove(f)

    def stop(self):
        self._running = False
        self.wait()

    #def log_event(self, msg, color="white"):
        #timestamp = datetime.now().strftime("%H:%M:%S")
       #self.log_event.append(f"<span style='color:{color}'>{timestamp} - {msg}</span>")

# -------------------- Live Preview Thread --------------------
class LivePreviewThread(QThread):
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, core, lock=None, interval_ms=10, record_queue: Queue = None, record_stride: int = 1):
        super().__init__()
        self.core = Core()
        self.lock = lock
        self.interval_ms = interval_ms
        self.running = False
        self.record_queue = record_queue
        self.record_stride = max(1, int(record_stride))
        self._frame_count = 0

    def run(self):
        self.running = True
        while self.running:
            try:
                if self.lock: self.lock.acquire()
                self.core.snap_image()
                pixels = self.core.get_image()
                if pixels is not None:
                    width = self.core.get_image_width()
                    height = self.core.get_image_height()
                    arr = np.array(pixels, dtype=np.uint16).reshape((height, width))
                    self._frame_count += 1

                    # emit for display
                    self.image_ready.emit(arr)

                    # optionally enqueue for disk writing
                    if self.record_queue and (self._frame_count % self.record_stride == 0):
                        # copy to decouple from any downstream modifications
                        self.record_queue.put(arr.copy())
            except Exception as e:
                print(f"Live preview error: {e}")
            finally:
                if self.lock: self.lock.release()
            self.msleep(self.interval_ms)

    def stop(self):
        self.running = False
        self.wait()


# -------------------- Live Preview Window --------------------
class LivePreviewWindow(QWidget):
    def __init__(self, core, lock=None):
        super().__init__()
        self.core = core
        self.camera_lock = lock
        self.setWindowTitle("Live Preview")
        self.label = QLabel("Live preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setMinimumSize(800, 600)

# -------------------- Collapsible GroupBox --------------------
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame, QVBoxLayout

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
    def __init__(self):
        super().__init__()
        self.core = Core()
        self.setWindowTitle("Glutamate Chasers")
        self.camera_lock = None
        self.live_thread = None
        self.live_window = None
        self.last_frame = None
        self.arduino = None
        self.experiment_timer = None
        self.batch_stack = []
        self.experiment_stopped = False
        self.start_time = None
        self.total_frames = 0
        self.frames_taken = 0
        self.settings_file = "stim_gui_settings.json"
        self.settings = {}
        self.build_ui()
        self.apply_dark_mode()
        self.load_settings()
        self.record_queue = None
        self.writer_thread = None
        self.show()


# -------------------- Collapsible / Group Widgets --------------------
    def build_ui(self):
    # File Saving
        self.file_group = CollapsibleGroupBox("File Saving")
        file_layout = QGridLayout()
        self.save_path_edit = QLineEdit()
        file_layout.addWidget(self.save_path_edit, 0, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_folder)
        file_layout.addWidget(browse_btn, 0, 2)
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
        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setValue(5)
        acq_layout.addWidget(self.total_time_spin, 0, 1)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setValue(10)
        self.interval_spin.setMinimum(1)
        self.interval_spin.setMaximum(1000)
        acq_layout.addWidget(self.interval_spin, 1, 1)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(1000)
        self.batch_size_spin.setValue(500)
        acq_layout.addWidget(self.batch_size_spin, 1, 3)
        self.trigger_time_spin = QDoubleSpinBox()
        self.trigger_time_spin.setValue(2)
        acq_layout.addWidget(self.trigger_time_spin, 0, 3)
        self.acq_group.set_layout(acq_layout)
        lbl = QLabel("Total Duration (min)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 0, 0)
        lbl = QLabel("Interval (ms)")    
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 1, 0)
        lbl = QLabel("Batch Size")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 1, 2)
        lbl = QLabel("Trigger Time (min)")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        acq_layout.addWidget(lbl, 0, 2)

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
        cam_layout.addWidget(self.record_cb, 4, 1)
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

    # Arduino Controls
        self.arduino_group = CollapsibleGroupBox("Arduino Controls")
        ar_layout = QGridLayout()
        self.serial_edit = QLineEdit("COM5"); ar_layout.addWidget(self.serial_edit,0,1)
        self.baud_combo = QComboBox(); self.baud_combo.addItems(["9600","115200","250000"]); ar_layout.addWidget(self.baud_combo,0,3)
        self.run_trigger_cb = QCheckBox("Send TTL"); self.run_trigger_cb.setCheckable(True); self.run_trigger_cb.setChecked(True)
        ar_layout.addWidget(self.run_trigger_cb,1,0)
        test_ttl_btn = QPushButton("Test TTL"); test_ttl_btn.clicked.connect(self.test_ttl); ar_layout.addWidget(test_ttl_btn,1,3)
        self.arduino_group.set_layout(ar_layout)
        lbl = QLabel("Arduino Port")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl, 0, 0)
        lbl = QLabel("Baud Rate")
        lbl.setProperty("noBorder", True)
        lbl.setStyleSheet("QLabel[noBorder='true'] { border:none }")
        ar_layout.addWidget(lbl, 0, 2)

        self.info_group = QGroupBox("")
        info_layout = QGridLayout()
        self.overlay_label = QLabel("READY"); self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("color: green; font-weight: bold; font-size:16px")
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
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"<span style='color:{color}'>{timestamp} - {msg}</span>")

    # -------------------- Folder & Arduino --------------------
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"Select Save Folder")
        if folder: self.save_path_edit.setText(folder)

    def test_ttl(self):
        if not self.arduino: self.open_arduino()
        if self.arduino: 
            try: 
                self.arduino.write(b'H')
                self.arduino.flush()
                self.set_overlay("TTL Pulse Sent", color="yellow")
                self.log_event("TTL sent successfully", color="yellow")
                QTimer.singleShot(500, lambda: self.set_overlay("READY", color="green"))
                self.arduino.flushInput()  # Clear any previous input
                self.arduino.flushOutput()  # Clear any previous output
            except Exception as e:
                self.set_overlay("TTL ERROR", color="red")
                QTimer.singleShot(500, lambda: self.set_overlay("READY", color="green"))
                self.log_event(f"Error sending TTL pulse: {e}", color="red")

    def open_arduino(self):
        if self.run_trigger_cb.isChecked():
            try:
                self.arduino = serial.Serial(self.serial_edit.text(),int(self.baud_combo.currentText()),timeout=1)
                self.log_event(f"Arduino connected on {self.serial_edit.text()} at {self.baud_combo.currentText()} baud", color="green")
            except Exception as e:
                self.log_event(f"Arduino error: {e}"); self.arduino=None
        else: 
            self.log_event("Arduino not enabled", color = "red")
            QTimer.singleShot(500, lambda: self.set_overlay("READY", color = "green"))

# -------------------- Live Preview --------------------
    def toggle_live(self):
        # stopping?
        if self.live_thread and self.live_thread.isRunning():
            self.live_thread.stop()
            self.live_thread = None
            if self.live_window and self.live_window.isVisible():
                self.live_window.hide()
        # stop recording if it was tied to live
            self.stop_recording()
            self.set_overlay("LIVE OFF")
            QTimer.singleShot(500, lambda: self.overlay_label.setText("READY"))
            return

    # starting
        if not self.live_window:
            self.live_window = LivePreviewWindow(core=self.core, lock=self.camera_lock)
        self.live_window.show()

    # start recording if checkbox is on
        record_queue = None
        if getattr(self, "record_cb", None) and self.record_cb.isChecked():
            self.start_recording()
            record_queue = self.record_queue

        self.live_thread = LivePreviewThread(self.core, lock=self.camera_lock, interval_ms=10, record_queue=record_queue, record_stride=1)
        self.live_thread.image_ready.connect(self.update_live_frame)
        self.live_thread.start()
        self.overlay_label.setText("LIVE ON")

    def start_recording(self):
        if self.writer_thread:  # already running
            return
    # folder inside the experiment root, but separate from batch saves
        base_folder = self.save_path_edit.text() or "."
        exp_folder = f"{datetime.now():%d%m%y}_{self.expt_name_edit.text()}_{self.mouse_id_edit.text()}"
        os.makedirs(os.path.join(base_folder, exp_folder), exist_ok=True)
        live_folder = os.path.join(base_folder, exp_folder, "live_preview")
        if self.live_window is None:
            self.start_live()
        self.record_queue = Queue(maxsize=50)  # small buffer; adjust as needed
        self.writer_thread = FrameWriterThread(self.record_queue, folder=live_folder, basename="live", save_every=1)
        self.writer_thread.start()
        self.overlay_label.setText("RECORDING IN PROGRESS...")

    def stop_recording(self):
        if self.writer_thread:
            self.writer_thread.stop()
            self.writer_thread = None
        self.record_queue = None
    # don't override other status messages if an experiment is running

    def update_live_frame(self, arr):
        self.last_frame = arr
        self.update_live_display()

    def update_live_display(self):
        if self.last_frame is None or not self.live_window: return
        arr = self.last_frame
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()/100.0
        arr_adj = np.clip(arr*contrast+brightness,0,65535)
        arr8 = ((arr_adj-arr_adj.min())/(np.ptp(arr_adj)+1e-6)*255).astype(np.uint8)
        qimg = QImage(arr8.data,arr.shape[1],arr.shape[0],QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        zoom = int(self.zoom_combo.currentText()[:-1])/100.0
        w,h = int(pixmap.width()*zoom), int(pixmap.height()*zoom)
        self.live_window.label.setPixmap(pixmap.scaled(w,h,Qt.KeepAspectRatio,Qt.SmoothTransformation))

    # -------------------- Experiment --------------------
    def start_experiment(self):
        self.total_duration = self.total_time_spin.value()*60 # Convert to Seconds
        self.interval = self.interval_spin.value() # Convert to Milliseconds
        self.trigger_time = self.trigger_time_spin.value()*60 # Convert to Seconds
        self.batch_stack = []
        self.frames_taken = 0
        self.total_frames = int(self.total_duration/(self.interval/1000))
        self.progressbar.setValue(0)
        self.start_time = time.time()
        self.record_cb.setChecked(True)
        self.toggle_live()
        self.experiment_stopped = False
        self.log_event(f"Experiment started: {self.expt_name_edit.text()} at {self.save_path_edit.text()}")
        self.experiment_timer = QTimer()
        self.experiment_timer.timeout.connect(self.run_experiment)
        self.experiment_timer.start(5)


        self.start_time = int(time.time() * 1000)  # store start time in ms
        self.last_image_time = 0
        self.ttl_pulse_sent = False

        self.experiment_timer = QTimer()
        self.experiment_timer.timeout.connect(self.run_experiment)
        self.experiment_timer.start(10)

    def run_experiment(self):
        elapsed = time.time()- (self.start_time / 1000)
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        elapsed_ms_total = int(elapsed * 1000)
        total_min = int(self.total_duration // 60)
        total_sec = int(self.total_duration % 60)
        self.timer_label.setText(f"{(elapsed_min):02d}:{(elapsed_sec):02d} / {total_min:02d}:{total_sec:02d}")
        self.set_overlay(f"EXPERIMENT IN PROGRESS...", color="blue")

        if elapsed>=self.total_duration or self.experiment_stopped:
            self.stop_experiment()
            self.stop_recording()

        if self.run_trigger_cb.isChecked() and not self.ttl_pulse_sent and elapsed >= self.trigger_time:
            self.test_ttl()
            self.ttl_pulse_sent = True

    def stop_experiment(self):
        if self.experiment_timer: self.experiment_timer.stop()
        if self.arduino and self.arduino.is_open: self.arduino.close()
        if self.writer_thread: self.writer_thread.stop()
        self.set_overlay("EXPERIMENT COMPLETE", color="green")

    # -------------------- Settings --------------------
    def load_settings(self):
        try:
            with open(self.settings_file,"r") as f: self.settings=json.load(f)
            self.save_path_edit.setText(self.settings.get("save_path",""))
        except: pass

    def save_settings(self):
        self.settings["save_path"]=self.save_path_edit.text()
        with open(self.settings_file,"w") as f: json.dump(self.settings,f,indent=2)

    def closeEvent(self, event):
        if self.live_thread: self.live_thread.stop()
        if self.writer_thread: self.writer_thread.stop()
        if self.arduino and self.arduino.is_open: self.arduino.close()
        if self.live_window: self.live_window.close()
        self.save_settings()
        event.accept()


# -------------------- Main --------------------
if __name__=="__main__":
    app=QApplication(sys.argv)
    gui=LiveImagingGUI()
    sys.exit(app.exec_())
