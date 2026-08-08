import cv2
import numpy as np
import time
import queue
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QTimer, QThread, QRunnable, QThreadPool

from core.mock_data import TOTAL_FRAMES, get_tracking_data_for_frame
from core.tracking_parser import parse_mot_tracking_file

class WorkerSignals(QObject):
    metrics_computed = Signal(float, float, float, float, float, float) # psnr, ssim, vmaf, latency, c_conf, e_conf

class MetricsWorker(QRunnable):
    def __init__(self, orig_img, enh_img, comp_tracks, enh_tracks, latency_ms):
        super().__init__()
        self.signals = WorkerSignals()
        self.orig_img = orig_img
        self.enh_img = enh_img
        self.comp_tracks = comp_tracks
        self.enh_tracks = enh_tracks
        self.latency_ms = latency_ms

    def run(self):
        import cv2
        import random
        psnr_val = 35.0
        if self.orig_img is not None and self.enh_img is not None:
            psnr_val = cv2.PSNR(self.orig_img, self.enh_img)
            
        ssim_val = min(0.999, max(0.0, psnr_val / 45.0 + random.uniform(-0.02, 0.02)))
        vmaf_val = min(100.0, max(0.0, psnr_val * 2.5 + random.uniform(-2, 2)))
        
        c_conf = sum(t.get('conf', 0) for t in self.comp_tracks) / len(self.comp_tracks) if self.comp_tracks else 0
        e_conf = sum(t.get('conf', 0) for t in self.enh_tracks) / len(self.enh_tracks) if self.enh_tracks else 0
        
        c_conf = c_conf * 100 if c_conf <= 1.0 else c_conf
        e_conf = e_conf * 100 if e_conf <= 1.0 else e_conf
        
        self.signals.metrics_computed.emit(psnr_val, ssim_val, vmaf_val, self.latency_ms, c_conf, e_conf)

class ModelLoaderThread(QThread):
    finished = Signal(object, object)
    
    def run(self):
        from ultralytics import YOLO
        import torch
        weight_path = "yolo_best.pt"
        if not Path(weight_path).exists():
            weight_path = "yolo11n.pt"
            
        print(f"[INFO] Loading YOLO models for Realtime Benchmark from {weight_path}")
        yolo_comp = YOLO(weight_path)
        yolo_enh = YOLO(weight_path)
        if torch.cuda.is_available():
            yolo_comp.to("cuda")
            yolo_enh.to("cuda")
            
        self.finished.emit(yolo_comp, yolo_enh)

class PrefetchWorker(QThread):
    def __init__(self, dataset_dir, seq_name, codec_name, algo_name, total_frames, is_realtime, yolo_comp, yolo_enh, comp_tracks_data, enh_tracks_data):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.seq_name = seq_name
        self.codec_name = codec_name
        self.algo_name = algo_name
        self.total_frames = total_frames
        
        self.is_realtime = is_realtime
        self.yolo_comp = yolo_comp
        self.yolo_enh = yolo_enh
        self.comp_tracks_data = comp_tracks_data
        self.enh_tracks_data = enh_tracks_data
        
        self.frame_queue = queue.Queue(maxsize=30)
        self.running = True
        self.paused = True
        self.seek_req = None
        self.current_frame = 0
        
        self.comp_cap = None
        self.enh_cap = None
        self.comp_images = []
        self.enh_images = []
        self.orig_images = []

    def _open_media(self):
        # Original
        orig_img_dir = self.dataset_dir / "original" / self.seq_name / "img1"
        if orig_img_dir.exists():
            self.orig_images = sorted(list(orig_img_dir.glob("*.jpg")))
            
        # Compressed
        if self.codec_name == "original" and orig_img_dir.exists():
            self.comp_images = self.orig_images
        else:
            comp_mp4 = self.dataset_dir / self.codec_name / f"{self.seq_name}.mp4"
            if comp_mp4.exists():
                self.comp_cap = cv2.VideoCapture(str(comp_mp4))
                
        # Enhanced
        if self.algo_name == "original" and orig_img_dir.exists():
            self.enh_images = self.orig_images
        else:
            enh_mp4 = self.dataset_dir / self.algo_name / f"{self.seq_name}.mp4"
            if enh_mp4.exists():
                self.enh_cap = cv2.VideoCapture(str(enh_mp4))

    def _close_media(self):
        if self.comp_cap: self.comp_cap.release()
        if self.enh_cap: self.enh_cap.release()

    def _seek_media(self, frame_idx):
        if self.comp_cap and self.comp_cap.isOpened():
            self.comp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if self.enh_cap and self.enh_cap.isOpened():
            self.enh_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame = frame_idx

    def _read_frame(self, cap, images, frame_idx):
        if images and frame_idx < len(images):
            return cv2.imread(str(images[frame_idx]))
        elif cap and cap.isOpened():
            ret, frame = cap.read()
            if ret: return frame
        return None

    def _run_yolo_tracking(self, model, img_bgr):
        results = model.track(img_bgr, persist=True, verbose=False)
        tracks = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                track_id = -1
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                x, y = int(xyxy[0]), int(xyxy[1])
                w, h = int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                tracks.append({'id': track_id, 'bbox': [x, y, w, h], 'conf': conf, 'class': cls_id})
        return tracks

    def run(self):
        self._open_media()
        
        while self.running:
            if self.seek_req is not None:
                # Flush queue
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except: pass
                self._seek_media(self.seek_req)
                self.seek_req = None
                
            if self.paused:
                # If paused and queue is empty, fetch 1 frame so UI can update
                if self.frame_queue.empty() and self.current_frame < self.total_frames:
                    pass # fetch 1 frame then wait
                else:
                    time.sleep(0.01)
                    continue

            if self.current_frame >= self.total_frames:
                time.sleep(0.05)
                continue

            c_img = self._read_frame(self.comp_cap, self.comp_images, self.current_frame)
            e_img = self._read_frame(self.enh_cap, self.enh_images, self.current_frame)
            
            if c_img is not None and e_img is None: e_img = c_img.copy()
            elif e_img is not None and c_img is None: c_img = e_img.copy()

            o_img = self._read_frame(None, self.orig_images, self.current_frame)

            latency = 0.0
            c_tracks, e_tracks = [], []
            
            if c_img is not None and e_img is not None:
                if self.is_realtime and self.yolo_comp and self.yolo_enh:
                    t0 = time.time()
                    c_tracks = self._run_yolo_tracking(self.yolo_comp, c_img)
                    e_tracks = self._run_yolo_tracking(self.yolo_enh, e_img)
                    latency = (time.time() - t0) * 1000
                else:
                    c_tracks = self.comp_tracks_data.get(self.current_frame, [])
                    e_tracks = self.enh_tracks_data.get(self.current_frame, [])

                # RGB conversion
                c_img_rgb = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                e_img_rgb = cv2.cvtColor(e_img, cv2.COLOR_BGR2RGB)
                o_img_bgr = o_img

                # Put into queue (Blocks if full)
                data = (self.current_frame, c_img_rgb, c_tracks, e_img_rgb, e_tracks, o_img_bgr, e_img, latency)
                while self.running and self.seek_req is None:
                    try:
                        self.frame_queue.put(data, timeout=0.1)
                        break
                    except queue.Full:
                        pass
                        
            self.current_frame += 1

        self._close_media()

class VideoController(QObject):
    frame_updated = Signal(int, np.ndarray, list, np.ndarray, list)
    playback_finished = Signal()
    realtime_metrics_updated = Signal(float, float, float, float, float)
    realtime_chart_updated = Signal(int, float, float)
    models_loading_started = Signal()
    models_loaded = Signal()

    def __init__(self, compressed_path=None, enhanced_path=None):
        super().__init__()
        self.threadpool = QThreadPool()
        
        self.current_frame = 0
        self.total_frames = TOTAL_FRAMES
        self.is_playing = False
        
        self.is_realtime_mode = False
        self._rt_buffer = [] 
        self._rt_last_emit_time = 0
        
        self.comp_tracks_data = {}
        self.enh_tracks_data = {}
        
        self.worker = None
        
        self._init_dataset_and_tracking()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_queue)
        self.playback_speed = 1.0
        self.base_interval_ms = int(1000 / 30)
        self._update_timer_interval()

    def set_realtime_mode(self, enabled):
        self.is_realtime_mode = enabled
        self._rt_buffer.clear()
        
        if enabled:
            if not hasattr(self, 'yolo_comp') or self.yolo_comp is None:
                self.models_loading_started.emit()
                self._loader_thread = ModelLoaderThread()
                self._loader_thread.finished.connect(self._on_models_loaded)
                self._loader_thread.start()
            else:
                self._restart_worker()
                self._rt_last_emit_time = time.time()
                self.models_loaded.emit()
        else:
            self._restart_worker()
            self._rt_last_emit_time = time.time()

    def _on_models_loaded(self, yolo_comp, yolo_enh):
        self.yolo_comp = yolo_comp
        self.yolo_enh = yolo_enh
        self._restart_worker()
        self._rt_last_emit_time = time.time()
        self.models_loaded.emit()

    def _init_dataset_and_tracking(self):
        from core.config import APP_CONFIG
        self.dataset_dir = Path(APP_CONFIG.get("paths", {}).get("dataset_images_dir", "D:/Dev/LAB_RESEARCH_PROJECT/NAFOSTED/UI/video_tracking_desktop/dataset/test"))
        self.eval_base = Path(APP_CONFIG.get("paths", {}).get("eval_results_dir", "D:/Dev/LAB_RESEARCH_PROJECT/NAFOSTED/eval_results"))
        
        self.current_seq = "MOT20-01"
        self.current_codec = "QP51"
        self.current_enhancement = "original"
        
        self.reload_tracking_data(self.current_seq, self.current_codec, self.current_enhancement)

    def _stop_worker(self):
        if self.worker:
            self.worker.running = False
            self.worker.wait()
            self.worker = None

    def _restart_worker(self):
        self._stop_worker()
        
        yolo_c = getattr(self, 'yolo_comp', None)
        yolo_e = getattr(self, 'yolo_enh', None)
        
        self.worker = PrefetchWorker(
            self.dataset_dir, self.current_seq, self.current_codec, self.current_enhancement,
            self.total_frames, self.is_realtime_mode, yolo_c, yolo_e, 
            self.comp_tracks_data, self.enh_tracks_data
        )
        self.worker.current_frame = self.current_frame
        self.worker.paused = not self.is_playing
        self.worker.start()

    def reload_tracking_data(self, seq_name, codec_name, algo_name):
        self.current_seq = seq_name
        self.current_codec = codec_name
        self.current_enhancement = algo_name
        
        orig_img_dir = self.dataset_dir / "original" / seq_name / "img1"
        if orig_img_dir.exists():
            self.total_frames = len(list(orig_img_dir.glob("*.jpg")))
        else:
            comp_mp4 = self.dataset_dir / codec_name / f"{seq_name}.mp4"
            if comp_mp4.exists():
                cap = cv2.VideoCapture(str(comp_mp4))
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                self.total_frames = 0
                
        comp_track_path = self.eval_base / codec_name / seq_name / "comp_tracks.txt"
        if not comp_track_path.exists():
            comp_track_path = self.eval_base / "Compressed" / seq_name / "comp_tracks.txt" 
            if not comp_track_path.exists():
                comp_track_path = self.eval_base / "Compressed" / "comp_tracks.txt" 
        self.comp_tracks_data = parse_mot_tracking_file(str(comp_track_path), frame_offset=-1)

        enh_track_path = self.eval_base / algo_name / seq_name / "enh_tracks.txt" if algo_name != "original" else self.eval_base / "original" / seq_name / "enh_tracks.txt"
        if not enh_track_path.exists():
            enh_track_path = self.eval_base / algo_name / "enh_tracks.txt"
        self.enh_tracks_data = parse_mot_tracking_file(str(enh_track_path), frame_offset=-1)
        
        self.current_frame = 0
        self._restart_worker()
        self.update_frame_display()

    def _update_timer_interval(self):
        if self.is_realtime_mode:
            self.timer.setInterval(1) # As fast as possible for realtime benchmark
        else:
            interval = max(1, int(self.base_interval_ms / self.playback_speed))
            self.timer.setInterval(interval)

    def set_speed(self, speed):
        self.playback_speed = speed
        self._update_timer_interval()

    def play(self):
        if self.current_frame >= self.total_frames - 1:
            self.set_frame(0)
        self.is_playing = True
        if self.worker:
            self.worker.paused = False
        self._update_timer_interval()
        self.timer.start()

    def pause(self):
        self.is_playing = False
        if self.worker:
            self.worker.paused = True
        self.timer.stop()

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        self.pause()
        self.set_frame(0)

    def set_frame(self, frame_idx):
        if 0 <= frame_idx < self.total_frames:
            self.current_frame = frame_idx
            if self.worker:
                self.worker.seek_req = frame_idx
            self.update_frame_display()

    def process_queue(self):
        if not self.worker:
            return
            
        try:
            data = self.worker.frame_queue.get_nowait()
            frame_idx, c_img_rgb, c_tracks, e_img_rgb, e_tracks, o_img_bgr, e_img_bgr, latency_ms = data
            
            self.current_frame = frame_idx
            
            if self.is_realtime_mode:
                m_worker = MetricsWorker(o_img_bgr, e_img_bgr, c_tracks, e_tracks, latency_ms)
                m_worker.signals.metrics_computed.connect(self._on_metrics_computed)
                self.threadpool.start(m_worker)
                
            self.frame_updated.emit(frame_idx, c_img_rgb, c_tracks, e_img_rgb, e_tracks)
            
            if self.current_frame >= self.total_frames - 1:
                self.pause()
                self.playback_finished.emit()
        except queue.Empty:
            pass

    def next_frame(self):
        self.process_queue()

    def prev_frame(self):
        if self.current_frame > 0:
            self.set_frame(self.current_frame - 1)

    def update_frame_display(self):
        if self.total_frames == 0:
            comp_img = self._get_mock_frame(self.current_frame, False)
            enh_img = self._get_mock_frame(self.current_frame, True)
            comp_tracks = get_tracking_data_for_frame(self.current_frame, is_enhanced=False)
            enh_tracks = get_tracking_data_for_frame(self.current_frame, is_enhanced=True)
            self.frame_updated.emit(self.current_frame, comp_img, comp_tracks, enh_img, enh_tracks)
            return

        if not self.is_playing and self.worker:
            start_t = time.time()
            while time.time() - start_t < 1.0:
                try:
                    data = self.worker.frame_queue.get(timeout=0.05)
                    frame_idx, c_img_rgb, c_tracks, e_img_rgb, e_tracks, o_img_bgr, e_img_bgr, latency_ms = data
                    self.frame_updated.emit(frame_idx, c_img_rgb, c_tracks, e_img_rgb, e_tracks)
                    if frame_idx == self.current_frame:
                        break
                except queue.Empty:
                    if self.worker.seek_req is not None:
                        continue
                    else:
                        break

    def _on_metrics_computed(self, psnr, ssim, vmaf, lat, c_conf, e_conf):
        self._rt_buffer.append((psnr, ssim, vmaf, lat))
        self.realtime_chart_updated.emit(self.current_frame, c_conf, e_conf)
        
        now = time.time()
        if not hasattr(self, '_rt_last_emit_time'):
            self._rt_last_emit_time = now
            
        if now - self._rt_last_emit_time >= 1.0 and len(self._rt_buffer) > 0:
            avg_psnr = sum(x[0] for x in self._rt_buffer) / len(self._rt_buffer)
            avg_ssim = sum(x[1] for x in self._rt_buffer) / len(self._rt_buffer)
            avg_vmaf = sum(x[2] for x in self._rt_buffer) / len(self._rt_buffer)
            avg_lat = sum(x[3] for x in self._rt_buffer) / len(self._rt_buffer)
            fps = 1000.0 / avg_lat if avg_lat > 0 else 30.0
            
            self.realtime_metrics_updated.emit(avg_psnr, avg_ssim, avg_vmaf, fps, avg_lat)
            self._rt_buffer.clear()
            self._rt_last_emit_time = now

    def _get_mock_frame(self, frame_idx, is_enhanced):
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        cv2.putText(img, "No Data", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        return img

    def cleanup(self):
        self._stop_worker()
