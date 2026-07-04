from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QCheckBox,
    QSizePolicy, QFrame, QSplitter, QMessageBox
)
from PySide6.QtCore import Qt

from ui.video_panel import VideoCanvas
from ui.metric_cards import create_all_metric_groups
from ui.timeline_chart import TrackingComparisonChart
from core.video_controller import VideoController
from core.mock_data import get_failure_cases


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancement for Tracking Evaluation")
        # Start maximised — fills screen without scroll bar needed
        self.showMaximized()

        # Setup Video Controller
        self.video_controller = VideoController()
        self.video_controller.frame_updated.connect(self.on_frame_updated)

        self._init_ui()

        # Initialize display
        self.video_controller.update_frame_display()

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(20, 14, 20, 14)
        root_layout.setSpacing(10)

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

        bottom_layout.addWidget(create_all_metric_groups())

        self.tracking_chart = TrackingComparisonChart()
        self.tracking_chart.marker_clicked.connect(self.jump_to_frame)
        bottom_layout.addWidget(self.tracking_chart, stretch=1)

        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        # Ratio: ~58% for videos, ~42% for metrics+chart
        splitter.setStretchFactor(0, 58)
        splitter.setStretchFactor(1, 42)

        root_layout.addWidget(splitter, stretch=1)

    # ── Header ──────────────────────────────────────────────────────────────

    def _create_header(self, parent_layout):
        layout = QHBoxLayout()
        layout.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_lbl = QLabel("Video Enhancement for Tracking Evaluation")
        title_lbl.setObjectName("MainTitle")
        sub_lbl = QLabel("Visual analytics dashboard — quality restoration & tracking performance on Jetson AGX")
        sub_lbl.setObjectName("SubTitle")
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
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(14)

        filters = [
            ("Dataset",           ["MOT20"],             1),
            ("Sequence",          ["MOT20-02"],           2),
            ("Codec / Bitrate",   ["H.264 / 1.2 Mbps"],  2),
            ("Enhancement Model", ["Real-ESRGAN ×2"],     3),
            ("Tracker",           ["ByteTrack"],          2),
            ("Sync Mode",         ["Frame-locked"],       2),
        ]

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

        parent_layout.addWidget(filter_card)

    # ── Video comparison area ────────────────────────────────────────────────

    def _create_video_comparison_area(self, parent_layout):
        layout = QHBoxLayout()
        layout.setSpacing(14)

        # ── Compressed Panel ────────────────────────────────────────────────
        comp_layout = QVBoxLayout()
        comp_layout.setSpacing(6)

        comp_header = QHBoxLayout()
        comp_title = QLabel("Compressed Video")
        comp_title.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #111827;")
        comp_badge = QLabel("H.264 · 1.2 Mbps")
        comp_badge.setObjectName("CompressedBadge")
        comp_header.addWidget(comp_title)
        comp_header.addStretch()
        comp_header.addWidget(comp_badge)

        self.comp_canvas = VideoCanvas(is_enhanced=False)
        self.comp_canvas.setObjectName("CompressedCanvas")

        comp_layout.addLayout(comp_header)
        comp_layout.addWidget(self.comp_canvas, stretch=1)

        # ── Enhanced Panel ───────────────────────────────────────────────────
        enh_layout = QVBoxLayout()
        enh_layout.setSpacing(6)

        enh_header = QHBoxLayout()
        enh_title = QLabel("Enhanced Video")
        enh_title.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #111827;")
        enh_badge = QLabel("Enhanced ×2")
        enh_badge.setObjectName("EnhancedBadge")
        enh_header.addWidget(enh_title)
        enh_header.addStretch()
        enh_header.addWidget(enh_badge)

        self.enh_canvas = VideoCanvas(is_enhanced=True)
        self.enh_canvas.setObjectName("EnhancedCanvas")

        enh_layout.addLayout(enh_header)
        enh_layout.addWidget(self.enh_canvas, stretch=1)

        layout.addLayout(comp_layout, stretch=1)
        layout.addLayout(enh_layout, stretch=1)

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
    # ──────────────────────────────────────────────────────────────────────────

    def toggle_play(self):
        self.video_controller.toggle_play_pause()
        self.play_btn.setText("||" if self.video_controller.is_playing else "▶")

    def on_slider_changed(self, value):
        if abs(self.video_controller.current_frame - value) > 1:
            self.video_controller.set_frame(value)

    def on_speed_changed(self, text):
        self.video_controller.set_speed(float(text.replace("x", "")))

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
