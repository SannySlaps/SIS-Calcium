import threading
import queue
import time
import tifffile

class CameraWorker:
    def __init__(self, core, gui):
        self.core = core
        self.gui = gui
        self.frame_queue = queue.Queue(maxsize=10)   # live frames
        self.save_queue = queue.Queue(maxsize=100)   # burst save buffer
        self.burst_active = False
        self.running = True

        # Start acquisition
        self.core.startContinuousSequenceAcquisition(0)  

        # Launch threads
        threading.Thread(target=self._acquire_loop, daemon=True).start()
        threading.Thread(target=self._display_loop, daemon=True).start()
        threading.Thread(target=self._save_loop, daemon=True).start()

    def _acquire_loop(self):
        """Continuously pull frames from camera into frame_queue (and save_queue if burst is active)."""
        while self.running:
            if self.core.getRemainingImageCount() > 0:
                frame = self.core.getLastImage()
                # Push to live display
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                # Push to save if burst active
                if self.burst_active and not self.save_queue.full():
                    self.save_queue.put(frame, block=False)
            else:
                time.sleep(0.001)

    def _display_loop(self):
        """Update GUI preview from frame_queue."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                self.gui.update_preview(frame)
            except queue.Empty:
                pass

    def _save_loop(self):
        """Save frames to disk only during burst."""
        while self.running:
            try:
                frame = self.save_queue.get(timeout=1)
                if self.burst_active:
                    fname = f"{self.gui.save_dir}/frame_{int(time.time()*1000)}.tif"
                    tifffile.imwrite(fname, frame)
            except queue.Empty:
                pass

    def start_burst(self):
        self.burst_active = True
        self.gui.log("Burst started")

    def stop_burst(self):
        self.burst_active = False
        self.gui.log("Burst stopped")

    def shutdown(self):
        self.running = False
        self.core.stopSequenceAcquisition()
