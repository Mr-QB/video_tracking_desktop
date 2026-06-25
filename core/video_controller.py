import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread, QTimer

from core.mock_data import TOTAL_FRAMES, get_tracking_data_for_frame

class VideoController(QObject):
    """
    Controls video playback and synchronization between compressed and enhanced views.
    Handles reading frames and fetching tracking data.
    """
    frame_updated = Signal(int, np.ndarray, list, np.ndarray, list) # frame_idx, comp_img, comp_track, enh_img, enh_track
    playback_finished = Signal()

    def __init__(self, compressed_path=None, enhanced_path=None):
        super().__init__()
        self.compressed_path = compressed_path
        self.enhanced_path = enhanced_path
        
        self.comp_cap = None
        self.enh_cap = None
        
        # Try to open videos if paths provided
        if self.compressed_path:
            self.comp_cap = cv2.VideoCapture(self.compressed_path)
        if self.enhanced_path:
            self.enh_cap = cv2.VideoCapture(self.enhanced_path)
            
        self.current_frame = 0
        self.total_frames = TOTAL_FRAMES
        self.is_playing = False
        
        # Timer for playback loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playback_speed = 1.0
        self.base_interval_ms = int(1000 / 30) # Assume 30 FPS base
        
        self._update_timer_interval()

    def _update_timer_interval(self):
        interval = max(1, int(self.base_interval_ms / self.playback_speed))
        self.timer.setInterval(interval)

    def set_speed(self, speed):
        self.playback_speed = speed
        self._update_timer_interval()

    def play(self):
        if self.current_frame >= self.total_frames - 1:
            self.current_frame = 0
        self.is_playing = True
        self.timer.start()

    def pause(self):
        self.is_playing = False
        self.timer.stop()

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        self.pause()
        self.current_frame = 0
        self.update_frame_display()

    def set_frame(self, frame_idx):
        if 0 <= frame_idx < self.total_frames:
            self.current_frame = frame_idx
            self.update_frame_display()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame_display()
        else:
            self.pause()
            self.playback_finished.emit()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame_display()

    def update_frame_display(self):
        """Fetches the current frame (real or mock) and emits the update signal."""
        
        # 1. Fetch Images
        comp_img = self._get_frame_image(self.comp_cap, self.current_frame, is_enhanced=False)
        enh_img = self._get_frame_image(self.enh_cap, self.current_frame, is_enhanced=True)
        
        # 2. Fetch Tracking Data
        comp_tracks = get_tracking_data_for_frame(self.current_frame, is_enhanced=False)
        enh_tracks = get_tracking_data_for_frame(self.current_frame, is_enhanced=True)
        
        # 3. Emit Signal to UI
        self.frame_updated.emit(self.current_frame, comp_img, comp_tracks, enh_img, enh_tracks)

    def _get_frame_image(self, cap, frame_idx, is_enhanced):
        """Reads a frame from VideoCapture, or generates a placeholder if no video."""
        if cap and cap.isOpened():
            # OpenCV seeking is slow, but for simplicity in this demo we use set
            # For a production app, reading sequentially in a thread is better.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        # Generate Placeholder Mock Frame
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 240 # Light gray background
        
        # Add some mock road / environment markings
        cv2.line(img, (0, 500), (1280, 500), (200, 200, 200), 2)
        cv2.line(img, (640, 500), (640, 720), (200, 200, 200), 2)
        
        # Add text indicator
        text = "Enhanced Video" if is_enhanced else "Compressed Video"
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        
        # Simulate compression artifacts (blur/noise)
        if not is_enhanced:
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
        return img

    def cleanup(self):
        if self.comp_cap:
            self.comp_cap.release()
        if self.enh_cap:
            self.enh_cap.release()
