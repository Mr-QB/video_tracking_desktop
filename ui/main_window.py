from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QSlider, QCheckBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt

from ui.video_panel import VideoCanvas
from ui.metric_cards import create_all_metric_groups
from ui.timeline_chart import TimelineChart
from ui.protocol_panel import ProtocolPanel
from core.video_controller import VideoController
from core.mock_data import get_failure_cases, TOTAL_FRAMES

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancement for Tracking Evaluation")
        self.resize(1500, 950)
        
        # Setup Video Controller
        self.video_controller = VideoController()
        self.video_controller.frame_updated.connect(self.on_frame_updated)
        
        self._init_ui()
        
        # Initialize display
        self.video_controller.update_frame_display()

    def _init_ui(self):
        # Main Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)
        
        main_widget = QWidget()
        scroll_area.setWidget(main_widget)
        
        self.main_layout = QVBoxLayout(main_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        self._create_header()
        self._create_filter_bar()
        self._create_video_comparison_area()
        self._create_playback_controls()
        
        # Add metric cards
        self.main_layout.addWidget(create_all_metric_groups())
        
        # Add timeline chart
        self.timeline_chart = TimelineChart()
        self.timeline_chart.marker_clicked.connect(self.jump_to_frame)
        self.main_layout.addWidget(self.timeline_chart)
        
        self._create_bottom_area()
        
        self.main_layout.addStretch()

    def _create_header(self):
        layout = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        title_lbl = QLabel("\u2582\u2585\u2587 Video Enhancement for Tracking Evaluation")
        title_lbl.setObjectName("MainTitle")
        sub_lbl = QLabel("Visual analytics dashboard for quality restoration and tracking performance")
        sub_lbl.setObjectName("SubTitle")
        
        title_layout.addWidget(title_lbl)
        title_layout.addWidget(sub_lbl)
        layout.addLayout(title_layout)
        
        layout.addStretch()
        
        export_btn = QPushButton("\u21A7 Export Report")
        export_btn.setObjectName("ActionButton")
        export_btn.clicked.connect(lambda: QMessageBox.information(self, "Export", "Exporting CSV report demo..."))
        
        summary_btn = QPushButton("\u2582\u2585\u2587 Run Summary")
        summary_btn.setObjectName("PrimaryActionButton")
        summary_btn.clicked.connect(lambda: QMessageBox.information(self, "Summary", "Generating run summary..."))
        
        layout.addWidget(export_btn)
        layout.addWidget(summary_btn)
        
        self.main_layout.addLayout(layout)

    def _create_filter_bar(self):
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        filters = [
            ("Dataset", ["MOT17"]),
            ("Sequence", ["MOT17-02"]),
            ("Codec / Bitrate", ["H.264 / 1.2 Mbps"]),
            ("Enhancement Model", ["Real-ESRGAN \u00D72"]),
            ("Tracker", ["ByteTrack"]),
            ("Sync Mode \u24D8", ["Frame-locked"])
        ]
        
        for label, items in filters:
            vbox = QVBoxLayout()
            vbox.setSpacing(4)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #556270; font-size: 11px;")
            cb = QComboBox()
            cb.addItems(items)
            
            # Make combo boxes expand
            cb.setSizePolicy(cb.sizePolicy().Policy.Expanding, cb.sizePolicy().Policy.Fixed)
            
            vbox.addWidget(lbl)
            vbox.addWidget(cb)
            layout.addLayout(vbox, stretch=1)
            
        self.main_layout.addLayout(layout)

    def _create_video_comparison_area(self):
        layout = QHBoxLayout()
        layout.setSpacing(20)
        
        # Compressed Panel
        comp_layout = QVBoxLayout()
        comp_header = QHBoxLayout()
        comp_title = QLabel("\uD83D\uDCF9 Compressed Video")
        comp_title.setProperty("class", "SectionTitle")
        comp_badge = QLabel("H.264 \u00B7 1.2 Mbps \u00B7 lower tracking quality")
        comp_badge.setObjectName("CompressedBadge")
        comp_header.addWidget(comp_title)
        comp_header.addStretch()
        comp_header.addWidget(comp_badge)
        
        self.comp_canvas = VideoCanvas(is_enhanced=False)
        self.comp_canvas.setObjectName("CompressedCanvas")
        self.comp_canvas.setMinimumSize(640, 360)
        
        comp_layout.addLayout(comp_header)
        comp_layout.addWidget(self.comp_canvas)
        
        # Enhanced Panel
        enh_layout = QVBoxLayout()
        enh_header = QHBoxLayout()
        enh_title = QLabel("\uD83D\uDCF9 Enhanced Video")
        enh_title.setProperty("class", "SectionTitle")
        enh_badge = QLabel("Enhanced \u00D72 \u00B7 improved tracking stability")
        enh_badge.setObjectName("EnhancedBadge")
        enh_header.addWidget(enh_title)
        enh_header.addStretch()
        enh_header.addWidget(enh_badge)
        
        self.enh_canvas = VideoCanvas(is_enhanced=True)
        self.enh_canvas.setObjectName("EnhancedCanvas")
        self.enh_canvas.setMinimumSize(640, 360)
        
        enh_layout.addLayout(enh_header)
        enh_layout.addWidget(self.enh_canvas)
        
        layout.addLayout(comp_layout)
        layout.addLayout(enh_layout)
        
        self.main_layout.addLayout(layout)

    def _create_playback_controls(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        
        # Play controls
        self.play_btn = QPushButton("\u25B6") # Play icon
        self.play_btn.setObjectName("PlayButton")
        self.play_btn.clicked.connect(self.toggle_play)
        
        prev_btn = QPushButton("|<")
        prev_btn.setObjectName("PlayButton")
        prev_btn.clicked.connect(self.video_controller.prev_frame)
        
        # Timeline
        lbl_frame_text = QLabel("Frame")
        
        self.frame_val_lbl = QLabel("0000")
        self.frame_val_lbl.setObjectName("FrameBox")
        
        self.frame_tot_lbl = QLabel(f" /{TOTAL_FRAMES}")
        self.frame_tot_lbl.setObjectName("FrameTotal")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, TOTAL_FRAMES - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setCursor(Qt.PointingHandCursor)
        
        # Speed
        speed_lbl = QLabel("Speed")
        self.speed_cb = QComboBox()
        self.speed_cb.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_cb.setCurrentIndex(1)
        self.speed_cb.currentTextChanged.connect(self.on_speed_changed)
        
        # Overlays
        overlay_lbl = QLabel("Overlays:")
        self.chk_tracks = QCheckBox("Tracks")
        self.chk_ids = QCheckBox("IDs")
        self.chk_conf = QCheckBox("Conf.")
        self.chk_det = QCheckBox("Detections")
        
        self.chk_tracks.setChecked(True)
        self.chk_ids.setChecked(True)
        self.chk_conf.setChecked(True)
        
        for chk in [self.chk_tracks, self.chk_ids, self.chk_conf, self.chk_det]:
            chk.stateChanged.connect(self.update_overlays)
            
        layout.addWidget(self.play_btn)
        layout.addWidget(prev_btn)
        layout.addSpacing(15)
        layout.addWidget(lbl_frame_text)
        layout.addWidget(self.frame_val_lbl)
        layout.addWidget(self.frame_tot_lbl)
        layout.addSpacing(15)
        layout.addWidget(self.slider, stretch=1)
        layout.addSpacing(15)
        layout.addWidget(speed_lbl)
        layout.addWidget(self.speed_cb)
        layout.addSpacing(25)
        layout.addWidget(overlay_lbl)
        layout.addWidget(self.chk_tracks)
        layout.addWidget(self.chk_ids)
        layout.addWidget(self.chk_conf)
        layout.addWidget(self.chk_det)
        
        self.main_layout.addLayout(layout)

    def _create_bottom_area(self):
        layout = QHBoxLayout()
        layout.setSpacing(20)
        
        # Failure Analysis Table
        table_layout = QVBoxLayout()
        table_title = QLabel("Key Cases / Failure Analysis")
        table_title.setProperty("class", "SectionTitle")
        
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Timestamp (Frame)", "Object ID", "Compressed Result", "Enhanced Result", "Status", "Notes"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        
        self._populate_table()
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        
        table_layout.addWidget(table_title)
        table_layout.addWidget(self.table)
        
        # Protocol Panel
        protocol_panel = ProtocolPanel()
        protocol_panel.setFixedWidth(400)
        
        layout.addLayout(table_layout)
        layout.addWidget(protocol_panel)
        
        self.main_layout.addLayout(layout)

    def _populate_table(self):
        cases = get_failure_cases()
        self.table.setRowCount(len(cases))
        
        for row, case in enumerate(cases):
            # Calculate mm:ss
            mins = case['frame'] // (30 * 60)
            secs = (case['frame'] // 30) % 60
            ts_str = f"{mins:02d}:{secs:02d} ({case['frame']})"
            
            item_ts = QTableWidgetItem(ts_str)
            item_ts.setData(Qt.UserRole, case['frame']) # Store actual frame for jumping
            
            self.table.setItem(row, 0, item_ts)
            self.table.setItem(row, 1, QTableWidgetItem(f"ID: {case['id']} ({case['type']})"))
            
            comp_item = QTableWidgetItem(case['comp_res'])
            if "Lost" in case['comp_res'] or "Wrong" in case['comp_res'] or "Switch" in case['comp_res']:
                comp_item.setForeground(Qt.red)
            self.table.setItem(row, 2, comp_item)
            
            enh_item = QTableWidgetItem(case['enh_res'])
            if "Maintained" in case['enh_res'] or "Stable" in case['enh_res'] or "Correct" in case['enh_res']:
                enh_item.setForeground(Qt.darkGreen)
            self.table.setItem(row, 3, enh_item)
            
            # Badge for Status
            status_item = QTableWidgetItem() # empty item
            self.table.setItem(row, 4, status_item)
            
            badge_widget = QWidget()
            badge_layout = QHBoxLayout(badge_widget)
            badge_layout.setContentsMargins(4, 2, 4, 2)
            badge_layout.setAlignment(Qt.AlignCenter)
            
            badge_lbl = QLabel(case['status'])
            if case['status'] == 'Improved':
                badge_lbl.setProperty("class", "BadgeImproved")
            else:
                badge_lbl.setProperty("class", "BadgeNeutral")
                
            badge_layout.addWidget(badge_lbl)
            self.table.setCellWidget(row, 4, badge_widget)
            
            self.table.setItem(row, 5, QTableWidgetItem(case['notes']))
            
            # Make read-only
            for col in range(6):
                item = self.table.item(row, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    # --- Callbacks ---

    def toggle_play(self):
        self.video_controller.toggle_play_pause()
        if self.video_controller.is_playing:
            self.play_btn.setText("||") # Pause icon
        else:
            self.play_btn.setText("\u25B6") # Play icon

    def on_slider_changed(self, value):
        # Only update if slider was moved by user, not by the timer
        if abs(self.video_controller.current_frame - value) > 1:
            self.video_controller.set_frame(value)

    def on_speed_changed(self, text):
        speed = float(text.replace("x", ""))
        self.video_controller.set_speed(speed)

    def update_overlays(self):
        t = self.chk_tracks.isChecked()
        i = self.chk_ids.isChecked()
        c = self.chk_conf.isChecked()
        d = self.chk_det.isChecked()
        
        self.comp_canvas.set_overlay_flags(t, i, c, d)
        self.enh_canvas.set_overlay_flags(t, i, c, d)

    def jump_to_frame(self, frame_idx):
        self.video_controller.set_frame(frame_idx)
        self.slider.setValue(frame_idx)

    def on_table_selection(self):
        items = self.table.selectedItems()
        if items:
            row = items[0].row()
            frame_item = self.table.item(row, 0)
            frame_idx = frame_item.data(Qt.UserRole)
            self.jump_to_frame(frame_idx)

    def on_frame_updated(self, frame_idx, comp_img, comp_tracks, enh_img, enh_tracks):
        # Update slider and label without triggering slider callback loop
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        
        self.frame_val_lbl.setText(f"{frame_idx:04d}")
        
        # Update canvases
        self.comp_canvas.update_frame(comp_img, comp_tracks)
        self.enh_canvas.update_frame(enh_img, enh_tracks)

    def closeEvent(self, event):
        self.video_controller.cleanup()
        super().closeEvent(event)
