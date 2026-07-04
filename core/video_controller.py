import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
from pathlib import Path

from core.mock_data import TOTAL_FRAMES, get_tracking_data_for_frame

# Attempt to load YOLO from ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class VideoController(QObject):
    """
    Controls video playback and synchronization between compressed and enhanced views.
    Handles reading frames and fetching tracking data. Supports loading a single real
    image and pre-running YOLOv11 detection on startup to avoid UI thread lag.
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
        
        # Static YOLO and image variables for UI testing
        self.yolo_model = None
        self.image_files = []
        self.static_image = None
        self.static_tracks = []
        
        self._init_yolo_and_dataset()
        
        # Timer for playback loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playback_speed = 1.0
        self.base_interval_ms = int(1000 / 30) # Assume 30 FPS base
        
        self._update_timer_interval()

    def _init_yolo_and_dataset(self):
        if not ULTRALYTICS_AVAILABLE:
            print("[WARN] ultralytics package not available. Falling back to mock data.")
            return

        yolo_tracker_path = Path("D:/Dev/LAB_RESEARCH_PROJECT/NAFOSTED/yolo_tracker")
        val_images_path = yolo_tracker_path / "yolo_dataset_mot20" / "val" / "images"
        
        # 1. Load Image Files from Validation Set
        if val_images_path.exists():
            self.image_files = sorted(list(val_images_path.glob("*.jpg")))
            if self.image_files:
                self.total_frames = len(self.image_files)
                print(f"[INFO] Loaded {self.total_frames} validation images from {val_images_path}")
            else:
                print(f"[WARN] No images found in {val_images_path}. Falling back to mock data.")
        else:
            print(f"[WARN] Path {val_images_path} does not exist. Falling back to mock data.")
            
        # 2. Load Model Weights and Run Inference ONCE on startup
        if self.image_files:
            # Check weight options in order of preference
            weight_options = [
                yolo_tracker_path / "runs/detect/runs/detect/mot20_yolo26-3/weights/best.pt",
                yolo_tracker_path / "runs/detect/runs/detect/mot20_yolo26-2/weights/best.pt",
                yolo_tracker_path / "yolo26m.pt",
                yolo_tracker_path / "yolo26n.pt",
                "yolo11n.pt"  # online fallback if local files missing
            ]
            
            for wp in weight_options:
                wp_path = Path(wp)
                if wp_path.exists() or wp == "yolo11n.pt":
                    try:
                        print(f"[INFO] Loading YOLO model: {wp}...")
                        self.yolo_model = YOLO(str(wp))
                        print("[INFO] Model loaded successfully.")
                        
                        # Load first image and run static inference once
                        first_img_path = self.image_files[0]
                        print(f"[INFO] Running single static inference on {first_img_path}...")
                        img_bgr = cv2.imread(str(first_img_path))
                        if img_bgr is not None:
                            self.static_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            results = self.yolo_model(str(first_img_path), verbose=False)
                            
                            for i, box in enumerate(results[0].boxes):
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                class_id = int(box.cls[0].cpu().numpy())
                                
                                x = int(xyxy[0])
                                y = int(xyxy[1])
                                w = int(xyxy[2] - xyxy[0])
                                h = int(xyxy[3] - xyxy[1])
                                
                                self.static_tracks.append({
                                    'id': i + 1,
                                    'bbox': [x, y, w, h],
                                    'conf': conf,
                                    'class': class_id
                                })
                            print(f"[INFO] Inference complete. Found {len(self.static_tracks)} objects.")
                        else:
                            print("[ERROR] Failed to load the first image for static inference.")
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to load model weights or run inference on {wp}: {e}")

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
        # Check if real data and model are successfully loaded and static inference run
        if self.static_image is not None:
            # Enhanced stream: High resolution, clean detections
            enh_img = self.static_image
            enh_tracks = [t for t in self.static_tracks if t['conf'] > 0.25]
            
            # Compressed stream: Downsampled, pixelated, and blurred
            h_img, w_img, _ = self.static_image.shape
            small_img = cv2.resize(self.static_image, (w_img // 4, h_img // 4), interpolation=cv2.INTER_LINEAR)
            comp_img = cv2.resize(small_img, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            comp_img = cv2.GaussianBlur(comp_img, (5, 5), 0)
            
            # Degrade detections for Compressed stream: add jitter and drop lower-conf detections
            comp_tracks = []
            for t in self.static_tracks:
                if t['conf'] > 0.4:
                    np.random.seed(t['id'] + self.current_frame)
                    x_box, y_box, w_box, h_box = t['bbox']
                    x_p = int(x_box + np.random.uniform(-8, 8))
                    y_p = int(y_box + np.random.uniform(-8, 8))
                    w_p = int(w_box + np.random.uniform(-4, 4))
                    h_p = int(h_box + np.random.uniform(-4, 4))
                    
                    conf_p = max(0.0, t['conf'] - np.random.uniform(0.1, 0.25))
                    comp_tracks.append({
                        'id': t['id'],
                        'bbox': [x_p, y_p, w_p, h_p],
                        'conf': conf_p
                    })
            
            self.frame_updated.emit(self.current_frame, comp_img, comp_tracks, enh_img, enh_tracks)
            return
                
        # --- Fallback to mock data ---
        comp_img = self._get_frame_image(self.comp_cap, self.current_frame, is_enhanced=False)
        enh_img = self._get_frame_image(self.enh_cap, self.current_frame, is_enhanced=True)
        comp_tracks = get_tracking_data_for_frame(self.current_frame, is_enhanced=False)
        enh_tracks = get_tracking_data_for_frame(self.current_frame, is_enhanced=True)
        
        self.frame_updated.emit(self.current_frame, comp_img, comp_tracks, enh_img, enh_tracks)

    def _get_frame_image(self, cap, frame_idx, is_enhanced):
        """Reads a frame from VideoCapture, or generates a placeholder if no video."""
        if cap and cap.isOpened():
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
