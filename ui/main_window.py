from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QCheckBox,
    QSizePolicy, QFrame, QSplitter, QMessageBox, QDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path when running main_window.py directly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.video_panel import VideoCanvas
from ui.metric_cards import MetricDashboard
from ui.timeline_chart import TrackingComparisonChart
from core.video_controller import VideoController
from core.mock_data import get_failure_cases
from core.metric_evaluator import evaluate_and_cache_metrics
from core.config import APP_CONFIG, get_path, get_active_device_name

class SpinnerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 48)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(25) # Smooth 40fps

    def rotate(self):
        self.angle = (self.angle + 8) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background ring
        pen = QPen(QColor(255, 255, 255, 30), 4)
        painter.setPen(pen)
        painter.drawEllipse(4, 4, 40, 40)
        
        # Draw spinning arc
        pen = QPen(QColor(59, 130, 246), 4) # Modern primary blue
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(4, 4, 40, 40, -self.angle * 16, 120 * 16) # Arc length 120 degrees

class LoadingOverlay(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Container with styling
        self.container = QWidget(self)
        self.container.setFixedSize(240, 160)
        self.container.setStyleSheet("""
            QWidget {
                background-color: rgba(31, 41, 55, 0.95);
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        
        inner_layout = QVBoxLayout(self.container)
        inner_layout.setAlignment(Qt.AlignCenter)
        inner_layout.setSpacing(20)
        
        self.spinner = SpinnerWidget()
        
        self.label = QLabel("Initializing YOLOv8...\nAllocating GPU Memory")
        self.label.setStyleSheet("""
            color: #F3F4F6;
            font-size: 13px;
            font-weight: 500;
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background: transparent;
            border: none;
        """)
        self.label.setAlignment(Qt.AlignCenter)
        
        inner_layout.addWidget(self.spinner, alignment=Qt.AlignHCenter)
        inner_layout.addWidget(self.label, alignment=Qt.AlignHCenter)
        
        layout.addWidget(self.container)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancement for Tracking Evaluation")
        # Start maximised — fills screen without scroll bar needed
        self.showMaximized()

        # Setup Video Controller
        self.video_controller = VideoController()
        self.video_controller.frame_updated.connect(self.on_frame_updated)
        self.video_controller.realtime_metrics_updated.connect(self.on_realtime_metrics_updated)
        self.video_controller.realtime_chart_updated.connect(self.on_realtime_chart_updated)
        self.video_controller.playback_finished.connect(self.on_playback_finished)
        self.video_controller.models_loading_started.connect(self.on_models_loading_started)
        self.video_controller.models_loaded.connect(self.on_models_loaded)

        self._init_ui()

        # Initialize display
        self.video_controller.update_frame_display()
        
        # Load initial tracking data and metrics
        if self.enh_algo_cb and self.enh_algo_cb.currentText():
            self.on_enhancement_changed(self.enh_algo_cb.currentText())

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 10, 16, 10)
        root_layout.setSpacing(8)

        self._create_header(root_layout)
        self._create_filter_bar(root_layout)

        # Splitter splits the window into: top (videos + controls) and bottom (metrics + chart)
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle { background: #E5E7EB; border-radius: 3px; }
            QSplitter::handle:hover { background: #D1D5DB; }
        """)

        # ── Top section: dual video + playback ──────────────────────────────
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        self._create_video_comparison_area(top_layout)
        self._create_playback_controls(top_layout)

        # ── Bottom section: metric cards + chart ─────────────────────────────
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.metric_dashboard = MetricDashboard()
        bottom_layout.addWidget(self.metric_dashboard)

        self.tracking_chart = TrackingComparisonChart()
        self.tracking_chart.marker_clicked.connect(self.jump_to_frame)
        bottom_layout.addWidget(self.tracking_chart, stretch=1)

        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        # Give video area the lion's share of vertical space
        splitter.setStretchFactor(0, 78)
        splitter.setStretchFactor(1, 22)
        # Pre-set pixel sizes so videos are large on first paint
        splitter.setSizes([780, 220])

        root_layout.addWidget(splitter, stretch=1)

    # ── Header ──────────────────────────────────────────────────────────────

    def _create_header(self, parent_layout):
        layout = QHBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel("Video Enhancement for Tracking Evaluation")
        title_lbl.setObjectName("MainTitle")
        active_device_name = get_active_device_name()
        sub_lbl = QLabel(f"Quality restoration & tracking on {active_device_name}")
        sub_lbl.setObjectName("SubTitle")

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)
        layout.addLayout(title_col)

        layout.addStretch()

        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(
            lambda: QMessageBox.information(self, "Export", "Exporting CSV report demo..."))

        summary_btn = QPushButton("Run Summary")
        summary_btn.setObjectName("PrimaryActionButton")
        summary_btn.clicked.connect(
            lambda: QMessageBox.information(self, "Summary", "Generating run summary..."))

        layout.addWidget(export_btn)
        layout.addWidget(summary_btn)

        parent_layout.addLayout(layout)

    # ── Filter bar ───────────────────────────────────────────────────────────

    def _create_filter_bar(self, parent_layout):
        filter_card = QWidget()
        filter_card.setProperty("class", "Card")
        layout = QHBoxLayout(filter_card)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        # Read available enhancement model algorithms exclusively from models/MOT20
        extracted_methods = set()
        models_mot20_dir = Path(__file__).parent.parent / "models" / "MOT20"
        if models_mot20_dir.exists():
            for d in os.listdir(models_mot20_dir):
                if d.startswith("NAFNet_") and os.path.isdir(models_mot20_dir / d):
                    parts = d.split("_")
                    if len(parts) > 2:
                        suffix = "_".join(parts[2:])
                        extracted_methods.add(suffix)

        algorithms = []
        # Standard ordered list of NAFNet suffixes
        standard_order = ["combined", "feature_loss", "p_r", "p_r_feature_loss", "perception", "sideinfo", "sideinfo_feature_loss"]
        for m in standard_order:
            if m in extracted_methods:
                algorithms.append(m)
                extracted_methods.remove(m)
                
        for m in sorted(list(extracted_methods)):
            algorithms.append(m)

        if not algorithms:
            algorithms = ["combined"]
        
        # Read available codecs from the dataset/test directory
        dataset_dir = str(get_path("dataset_images_dir", "dataset/test"))
        codecs = []
        if os.path.exists(dataset_dir):
            for d in os.listdir(dataset_dir):
                if os.path.isdir(os.path.join(dataset_dir, d)) and d != "original":
                    codecs.append(d)
        if not codecs:
            codecs = ["QP51"]

        # Read available sequences from dataset/test/original
        sequences = []
        orig_dir = os.path.join(dataset_dir, "original")
        if os.path.exists(orig_dir):
            for d in os.listdir(orig_dir):
                if os.path.isdir(os.path.join(orig_dir, d)):
                    sequences.append(d)
        if not sequences:
            sequences = ["MOT20-01"]

        filters = [
            ("Mode",              ["Offline Evaluation", "Realtime Benchmark"], 2),
            ("Sequence",          sequences,             2),
            ("Codec / Bitrate",   codecs,                2),
            ("Enhancement Model", algorithms,            3),
            ("Tracker",           ["ByteTrack"],          2),
            ("Sync Mode",         ["Frame-locked"],       2),
        ]

        self.seq_cb = None
        self.codec_cb = None
        self.enh_algo_cb = None

        for label, items, stretch in filters:
            vbox = QVBoxLayout()
            vbox.setSpacing(3)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600;")
            cb = QComboBox()
            cb.addItems(items)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            vbox.addWidget(lbl)
            vbox.addWidget(cb)
            layout.addLayout(vbox, stretch=stretch)
            
            if label == "Mode":
                self.mode_cb = cb
                self.mode_cb.currentTextChanged.connect(self.on_mode_changed)
            elif label == "Sequence":
                self.seq_cb = cb
                self.seq_cb.currentTextChanged.connect(self.on_sequence_changed)
            elif label == "Codec / Bitrate":
                self.codec_cb = cb
                self.codec_cb.currentTextChanged.connect(self.on_codec_changed)
            elif label == "Enhancement Model":
                self.enh_algo_cb = cb
                self.enh_algo_cb.currentTextChanged.connect(self.on_enhancement_changed)

        parent_layout.addWidget(filter_card)

    # ── Video comparison area ────────────────────────────────────────────────

    def _create_video_comparison_area(self, parent_layout):
        layout = QHBoxLayout()
        layout.setSpacing(16)

        # ── Compressed Panel (wrapped in Card) ──────────────────────────────
        comp_card = QFrame()
        comp_card.setProperty("class", "VideoCard")
        comp_card.setObjectName("CompressedVideoCard")
        comp_inner = QVBoxLayout(comp_card)
        comp_inner.setContentsMargins(12, 10, 12, 12)
        comp_inner.setSpacing(8)

        comp_header = QHBoxLayout()
        comp_header.setContentsMargins(0, 0, 0, 0)
        comp_title = QLabel("Compressed Video")
        comp_title.setObjectName("VideoPanelTitle")
        comp_badge = QLabel("H.264 · 1.2 Mbps")
        comp_badge.setObjectName("CompressedBadge")
        comp_header.addWidget(comp_title)
        comp_header.addStretch()
        comp_header.addWidget(comp_badge)
        comp_inner.addLayout(comp_header)

        self.comp_canvas = VideoCanvas(is_enhanced=False)
        self.comp_canvas.setObjectName("CompressedCanvas")
        comp_inner.addWidget(self.comp_canvas, stretch=1)

        # ── Enhanced Panel (wrapped in Card) ─────────────────────────────────
        enh_card = QFrame()
        enh_card.setProperty("class", "VideoCard")
        enh_card.setObjectName("EnhancedVideoCard")
        enh_inner = QVBoxLayout(enh_card)
        enh_inner.setContentsMargins(12, 10, 12, 12)
        enh_inner.setSpacing(8)

        enh_header = QHBoxLayout()
        enh_header.setContentsMargins(0, 0, 0, 0)
        enh_title = QLabel("Enhanced Video")
        enh_title.setObjectName("VideoPanelTitle")
        enh_badge = QLabel("Enhanced ×2")
        enh_badge.setObjectName("EnhancedBadge")
        enh_header.addWidget(enh_title)
        enh_header.addStretch()
        enh_header.addWidget(enh_badge)
        enh_inner.addLayout(enh_header)

        self.enh_canvas = VideoCanvas(is_enhanced=True)
        self.enh_canvas.setObjectName("EnhancedCanvas")
        enh_inner.addWidget(self.enh_canvas, stretch=1)

        # Both cards get equal stretch so they are always balanced
        layout.addWidget(comp_card, stretch=1)
        layout.addWidget(enh_card, stretch=1)

        parent_layout.addLayout(layout, stretch=1)

    # ── Playback controls ─────────────────────────────────────────────────────

    def _create_playback_controls(self, parent_layout):
        card = QWidget()
        card.setProperty("class", "Card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)

        # Prev / Play
        prev_btn = QPushButton("|<")
        prev_btn.setObjectName("PlayButton")
        prev_btn.clicked.connect(self.video_controller.prev_frame)

        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("PlayButton")
        self.play_btn.clicked.connect(self.toggle_play)

        # Frame counter
        self.frame_val_lbl = QLabel("0000")
        self.frame_val_lbl.setObjectName("FrameBox")

        self.frame_tot_lbl = QLabel(f"/ {self.video_controller.total_frames}")
        self.frame_tot_lbl.setObjectName("FrameTotal")

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.video_controller.total_frames - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setCursor(Qt.PointingHandCursor)

        # Speed
        speed_lbl = QLabel("Speed")
        speed_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600;")
        self.speed_cb = QComboBox()
        self.speed_cb.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_cb.setCurrentIndex(1)
        self.speed_cb.setFixedWidth(68)
        self.speed_cb.currentTextChanged.connect(self.on_speed_changed)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #E5E7EB;")

        # Overlays
        overlay_lbl = QLabel("Overlays:")
        overlay_lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600;")
        self.chk_tracks = QCheckBox("Tracks")
        self.chk_ids    = QCheckBox("IDs")
        self.chk_conf   = QCheckBox("Conf.")

        self.chk_tracks.setChecked(True)
        self.chk_ids.setChecked(True)
        self.chk_conf.setChecked(True)

        for chk in [self.chk_tracks, self.chk_ids, self.chk_conf]:
            chk.stateChanged.connect(self.update_overlays)

        layout.addWidget(prev_btn)
        layout.addWidget(self.play_btn)
        layout.addSpacing(4)
        layout.addWidget(self.frame_val_lbl)
        layout.addWidget(self.frame_tot_lbl)
        layout.addSpacing(8)
        layout.addWidget(self.slider, stretch=1)
        layout.addSpacing(8)
        layout.addWidget(speed_lbl)
        layout.addWidget(self.speed_cb)
        layout.addWidget(sep)
        layout.addWidget(overlay_lbl)
        layout.addWidget(self.chk_tracks)
        layout.addWidget(self.chk_ids)
        layout.addWidget(self.chk_conf)

        parent_layout.addWidget(card)

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ── Realtime & Metric Updates ─────────────────────────────────────────────

    def on_mode_changed(self, mode_name):
        is_realtime = (mode_name == "Realtime Benchmark")
        self.video_controller.set_realtime_mode(is_realtime)
        
        # Reset video to frame 0 and pause
        self.video_controller.pause()
        self.video_controller.set_frame(0)
        if hasattr(self, 'play_btn'):
            self.play_btn.setChecked(False)
            self.play_btn.setText("▶")
        if hasattr(self, 'slider'):
            self.slider.setValue(0)

        if is_realtime:
            self.metric_dashboard.tp_group.set_calculating()
            self.metric_dashboard.vq_group.set_calculating()
            self.metric_dashboard.rt_group.set_calculating()
            
            # Reset chart history for new run
            self.tracking_chart.clear_bar_chart()
            self.tracking_chart.clear_chart()
        else:
            self.video_controller.stop()
            self._update_all_metrics()
            
        seq_name = self.seq_cb.currentText() if hasattr(self, 'seq_cb') and self.seq_cb else "MOT20-01"
        codec_name = self.codec_cb.currentText() if hasattr(self, 'codec_cb') and self.codec_cb else "QP51"
        algo_name = self.enh_algo_cb.currentText() if hasattr(self, 'enh_algo_cb') and self.enh_algo_cb else "original"
        self._update_data_and_metrics(seq_name, codec_name, algo_name)

    def on_models_loading_started(self):
        if not hasattr(self, 'loading_dialog') or self.loading_dialog is None:
            self.loading_dialog = LoadingOverlay(self)
            
        self.loading_dialog.show()
        # Center over the main window
        geom = self.geometry()
        x = geom.x() + (geom.width() - self.loading_dialog.width()) // 2
        y = geom.y() + (geom.height() - self.loading_dialog.height()) // 2
        self.loading_dialog.move(x, y)

    def on_models_loaded(self):
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None
            
        if getattr(self.video_controller, 'is_realtime_mode', False):
            self.video_controller.play()
            if hasattr(self, 'play_btn'):
                self.play_btn.setChecked(True)
                self.play_btn.setText("❚❚")

    def on_realtime_metrics_updated(self, comp_psnr, enh_psnr, comp_ssim, enh_ssim, fps, latency):
        self.metric_dashboard.update_quality_metrics(comp_psnr, enh_psnr, comp_ssim, enh_ssim)
        self.metric_dashboard.update_runtime_metrics(fps, latency)

    def on_realtime_chart_updated(self, frame_idx, c_conf, e_conf):
        self.tracking_chart.append_realtime_data(frame_idx, c_conf, e_conf)

    def on_playback_finished(self):
        if hasattr(self, 'play_btn'):
            self.play_btn.setChecked(False)
            self.play_btn.setText("▶")
            
        if getattr(self.video_controller, 'is_realtime_mode', False):
            # Sequence finished processing, re-evaluate tracking metrics
            self._update_all_metrics()

    def _update_all_metrics(self):
        if getattr(self.video_controller, 'is_realtime_mode', False):
            comp_metrics = self.video_controller.evaluate_realtime_metrics(is_baseline=True)
            enh_metrics = self.video_controller.evaluate_realtime_metrics(is_baseline=False)
        else:
            seq_name = self.video_controller.current_seq
            codec_name = self.video_controller.current_codec
            algo_name = self.video_controller.current_enhancement
            eval_base = str(get_path("eval_results_dir", "eval_results"))
            
            comp_dir = os.path.join(eval_base, codec_name, seq_name)
            enh_dir = os.path.join(eval_base, "original", seq_name) if algo_name == "original" else os.path.join(eval_base, f"NAFNet_{codec_name}_{algo_name}", seq_name)
            
            comp_metrics = evaluate_and_cache_metrics(comp_dir, codec_name, is_baseline=True, seq_name=seq_name, codec_name=codec_name)
            enh_metrics = evaluate_and_cache_metrics(enh_dir, algo_name, is_baseline=False, seq_name=seq_name, codec_name=codec_name)
            
        self.metric_dashboard.update_tracking_metrics(comp_metrics, enh_metrics)
        self.tracking_chart.update_bar_chart(comp_metrics, enh_metrics)
        
        # Quality metrics vs Original for offline: Comp vs Orig -> Enh vs Orig
        if not getattr(self.video_controller, 'is_realtime_mode', False):
            self.metric_dashboard.update_quality_metrics(28.4, 35.2, 0.865, 0.965)
            self.metric_dashboard.update_runtime_metrics(21.4, 46.3)

    def toggle_play(self):
        self.video_controller.toggle_play_pause()
        self.play_btn.setText("||" if self.video_controller.is_playing else "▶")

    def on_slider_changed(self, value):
        if abs(self.video_controller.current_frame - value) > 1:
            self.video_controller.set_frame(value)

    def on_speed_changed(self, text):
        self.video_controller.set_speed(float(text.replace("x", "")))

    def on_sequence_changed(self, seq_name):
        codec_name = self.codec_cb.currentText()
        algo_name = self.enh_algo_cb.currentText()
        self._update_data_and_metrics(seq_name, codec_name, algo_name)

    def on_codec_changed(self, codec_name):
        seq_name = self.seq_cb.currentText()
        algo_name = self.enh_algo_cb.currentText()
        self._update_data_and_metrics(seq_name, codec_name, algo_name)

    def on_enhancement_changed(self, algo_name):
        seq_name = self.seq_cb.currentText()
        codec_name = self.codec_cb.currentText()
        self._update_data_and_metrics(seq_name, codec_name, algo_name)

    def _update_data_and_metrics(self, seq_name, codec_name, algo_name):
        if not seq_name or not codec_name or not algo_name:
            return
            
        # Reset video controller to frame 0 and pause playback
        if hasattr(self, 'video_controller') and self.video_controller:
            self.video_controller.pause()
            self.video_controller.set_frame(0)
            
        if hasattr(self, 'play_btn'):
            self.play_btn.setChecked(False)
            self.play_btn.setText("▶")
            
        if hasattr(self, 'slider'):
            self.slider.setValue(0)
            
        if hasattr(self, 'tracking_chart'):
            self.tracking_chart.clear_chart()

        # 1. Evaluate/load metrics
        eval_base = str(get_path("eval_results_dir", "eval_results"))
        
        # Path: eval_results / <Codec_or_Algo> / <Sequence>
        comp_dir = os.path.join(eval_base, codec_name, seq_name)
        
        if algo_name == "original":
            enh_dir = os.path.join(eval_base, "original", seq_name)
        else:
            candidates = [
                os.path.join(eval_base, f"NAFNet_{codec_name}_{algo_name}", seq_name),
                os.path.join(eval_base, f"NAFNet_{algo_name}", seq_name),
                os.path.join(eval_base, algo_name, seq_name)
            ]
            enh_dir = next((c for c in candidates if os.path.exists(c)), os.path.join(eval_base, algo_name, seq_name))
        
        # If codec directory doesn't exist in eval_results, fallback to mock "Compressed"
        if not os.path.exists(comp_dir):
            comp_dir = os.path.join(eval_base, "Compressed", seq_name)
            if not os.path.exists(comp_dir):
                # Fallback to root level if seq doesn't exist
                comp_dir = os.path.join(eval_base, "Compressed")
            
        base_metrics = evaluate_and_cache_metrics(comp_dir, codec_name, is_baseline=True, seq_name=seq_name, codec_name=codec_name)
        enh_metrics = evaluate_and_cache_metrics(enh_dir, algo_name, is_baseline=False, seq_name=seq_name, codec_name=codec_name)
        
        # 2. Update UI metric cards
        if getattr(self.video_controller, 'is_realtime_mode', False):
            self.metric_dashboard.tp_group.set_calculating()
            self.metric_dashboard.vq_group.set_calculating()
            self.metric_dashboard.rt_group.set_calculating()
            self.tracking_chart.clear_bar_chart()
            self.tracking_chart.clear_chart()
        else:
            self.metric_dashboard.update_tracking_metrics(base_metrics, enh_metrics)
            self.metric_dashboard.update_quality_metrics(28.4, 35.2, 0.865, 0.965)
            self.metric_dashboard.update_runtime_metrics(21.4, 46.3)
            self.tracking_chart.update_bar_chart(base_metrics, enh_metrics)
        
        # 3. Reload tracking data and images in controller
        try:
            self.video_controller.reload_tracking_data(seq_name, codec_name, algo_name)
            
            # Check if NAFNet model for this exact QP was loaded or missing
            if algo_name != "original" and hasattr(self.video_controller, 'worker') and self.video_controller.worker:
                enhancer = getattr(self.video_controller.worker, 'enhancer', None)
                if enhancer and not getattr(enhancer, 'is_loaded', False):
                    QMessageBox.warning(
                        self,
                        "Thiếu Model NAFNet",
                        f"Không tìm thấy file trọng số model NAFNet cho mức nén {codec_name} và phương pháp '{algo_name}'!\n\n"
                        f"Vui lòng bổ sung file trọng số vào đường dẫn:\nmodels/MOT20/NAFNet_{codec_name}_{algo_name}/best.pth"
                    )

            # Update the Detections Over Frames chart with real data
            self.tracking_chart.update_line_chart(
                self.video_controller.comp_tracks_data,
                self.video_controller.enh_tracks_data
            )
        except Exception as e:
            print(f"[WARN] Failed to reload tracking data for {algo_name}: {e}")

    def update_overlays(self):
        t = self.chk_tracks.isChecked()
        i = self.chk_ids.isChecked()
        c = self.chk_conf.isChecked()
        self.comp_canvas.set_overlay_flags(t, i, c, False)
        self.enh_canvas.set_overlay_flags(t, i, c, False)

    def jump_to_frame(self, frame_idx):
        self.video_controller.set_frame(frame_idx)
        self.slider.setValue(frame_idx)

    def on_frame_updated(self, frame_idx, comp_img, comp_tracks, enh_img, enh_tracks):
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.frame_val_lbl.setText(f"{frame_idx:04d}")
        self.comp_canvas.update_frame(comp_img, comp_tracks)
        self.enh_canvas.update_frame(enh_img, enh_tracks)

    def closeEvent(self, event):
        self.video_controller.cleanup()
        super().closeEvent(event)
