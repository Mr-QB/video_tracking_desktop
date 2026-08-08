import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont
import numpy as np

from core.mock_data import generate_timeline_data, EVENTS

class TrackingComparisonChart(QWidget):
    """
    Side-by-side dual-panel bar+line chart showing tracking metrics
    comparison between Compressed and Enhanced video streams.
    Left: Bar chart comparing key tracking KPIs (HOTA, IDF1, MOTA, etc.)
    Right: Line chart showing IDF1 timeline over frames with event markers.
    """
    marker_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(6)

        # --- Header ---
        hdr = QHBoxLayout()
        title_lbl = QLabel("Tracking Performance Analysis")
        title_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #111827;")
        sub_lbl = QLabel("Compressed  vs  Enhanced — MOT20 Validation")
        sub_lbl.setStyleSheet("font-size: 10px; color: #6B7280;")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        hdr.addWidget(sub_lbl)
        outer.addLayout(hdr)

        # --- Dual chart row ---
        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        # ── Left: Bar chart ──────────────────────────────────────────────
        pg.setConfigOption('background', '#FFFFFF')
        pg.setConfigOption('foreground', '#374151')

        self.bar_widget = pg.PlotWidget()
        self.bar_widget.setMinimumHeight(120)
        self.bar_widget.setMaximumHeight(150)
        self.bar_widget.setSizePolicy(
            self.bar_widget.sizePolicy().horizontalPolicy(),
            self.bar_widget.sizePolicy().verticalPolicy()
        )
        self._build_bar_chart()
        charts_row.addWidget(self.bar_widget, stretch=1)

        # Thin divider
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFrameShadow(QFrame.Plain)
        div.setStyleSheet("color: #E5E7EB;")
        charts_row.addWidget(div)

        # ── Right: IDF1 timeline ─────────────────────────────────────────
        self.line_widget = pg.PlotWidget()
        self.line_widget.setMinimumHeight(120)
        self.line_widget.setMaximumHeight(150)
        self._build_line_chart()
        charts_row.addWidget(self.line_widget, stretch=2)

        outer.addLayout(charts_row)

        # --- Legend row ---
        legend_row = QHBoxLayout()
        legend_row.setSpacing(12)
        legend_row.addStretch()
        for color, label in [("#EF4444", "Compressed"), ("#0F766E", "Enhanced")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 11px;")
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 10px; color: #374151; font-weight: 500;")
            legend_row.addWidget(dot)
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        outer.addLayout(legend_row)

    def _build_bar_chart(self):
        """Grouped bar chart for KPI comparison."""
        pw = self.bar_widget
        pw.setLabel('left', '', units='')
        pw.getAxis('bottom').setTicks([
            [(0.5, 'HOTA'), (2.5, 'IDF1'), (4.5, 'MOTA'), (6.5, 'Recall'), (8.5, 'Prec.')]
        ])
        pw.getAxis('left').setStyle(tickTextOffset=4)
        pw.showGrid(x=False, y=True, alpha=0.2)
        pw.setYRange(0, 105)
        pw.setXRange(-0.4, 9.6)
        pw.getAxis('bottom').setPen(pg.mkPen('#D1D5DB'))
        pw.getAxis('left').setPen(pg.mkPen(None))
        pw.setMouseEnabled(x=False, y=False)
        pw.setTitle("Metric Comparison (% score)", color='#374151', size='10pt')

        # Bars will be added dynamically in update_bar_chart

    def clear_bar_chart(self):
        """Clears the bar chart."""
        self.bar_widget.clear()

    def update_bar_chart(self, comp_metrics, enh_metrics):
        """Update bar chart with new metrics dicts."""
        self.clear_bar_chart()
        
        # Extract metrics, fallback to 0
        def get_vals(m):
            if not m: return [0, 0, 0, 0, 0]
            # Mock MOTA, Recall, Prec if not in dict
            return [
                m.get('HOTA', 0),
                m.get('IDF1', 0),
                m.get('MOTA', m.get('HOTA', 0) - 5),
                m.get('Recall', 75.0),
                m.get('Precision', 80.0)
            ]
            
        comp_vals = get_vals(comp_metrics)
        enh_vals = get_vals(enh_metrics)
        x_positions = [0, 2, 4, 6, 8]
        bar_w = 0.7
        
        pw = self.bar_widget
        for i, (xp, cv, ev) in enumerate(zip(x_positions, comp_vals, enh_vals)):
            if cv > 0:
                bg_comp = pg.BarGraphItem(x=[xp], height=[cv], width=bar_w,
                                          brush='#EF4444', pen=pg.mkPen(None))
                pw.addItem(bg_comp)
                txt_comp = pg.TextItem(f"{cv:.0f}", color='#374151', anchor=(0.5, 0))
                txt_comp.setFont(QFont('Inter', 7, QFont.Bold))
                txt_comp.setPos(xp, cv + 1)
                pw.addItem(txt_comp)
                
            if ev > 0:
                bg_enh = pg.BarGraphItem(x=[xp + 1], height=[ev], width=bar_w,
                                         brush='#0F766E', pen=pg.mkPen(None))
                pw.addItem(bg_enh)
                txt_enh = pg.TextItem(f"{ev:.0f}", color='#374151', anchor=(0.5, 0))
                txt_enh.setFont(QFont('Inter', 7, QFont.Bold))
                txt_enh.setPos(xp + 1, ev + 1)
                pw.addItem(txt_enh)

    def _build_line_chart(self):
        """Average Confidence Score over time."""
        pw = self.line_widget
        pw.setLabel('left', 'Avg Confidence (%)')
        pw.setLabel('bottom', 'Frame')
        pw.showGrid(x=False, y=True, alpha=0.2)
        pw.setYRange(0, 100) 
        pw.getAxis('bottom').setPen(pg.mkPen('#D1D5DB'))
        pw.getAxis('left').setPen(pg.mkPen(None))
        pw.setMouseEnabled(x=True, y=True)
        pw.setTitle("Average Tracker Confidence Over Frames", color='#374151', size='10pt')
        
        # We start with empty data, it will be populated via update_line_chart()
        self.comp_curve = pw.plot([], [], pen=pg.mkPen(color='#EF4444', width=1.5), name='Compressed')
        self.enh_curve = pw.plot([], [], pen=pg.mkPen(color='#0F766E', width=1.5), name='Enhanced')
        
        self.fill_item = pg.FillBetweenItem(
            self.comp_curve,
            self.enh_curve,
            brush=(15, 118, 110, 30)
        )
        pw.addItem(self.fill_item)

    def update_line_chart(self, comp_tracks_data, enh_tracks_data):
        """
        Updates the line chart with real tracking data.
        comp_tracks_data: dict {frame_idx: [track1, track2, ...]}
        enh_tracks_data: dict {frame_idx: [track1, track2, ...]}
        """
        max_frame = 0
        if comp_tracks_data:
            max_frame = max(comp_tracks_data.keys())
        if enh_tracks_data:
            max_frame = max(max_frame, max(enh_tracks_data.keys()))
            
        if max_frame == 0:
            self.clear_chart()
            return
            
        frames = np.arange(1, max_frame + 1)
        comp_conf = np.zeros(max_frame)
        enh_conf = np.zeros(max_frame)
        
        for f in range(1, max_frame + 1):
            c_tracks = comp_tracks_data.get(f, [])
            if c_tracks:
                # conf is in the dictionary 'conf'
                c_conf = sum([t.get('conf', 0) for t in c_tracks]) / len(c_tracks)
                comp_conf[f - 1] = c_conf * 100 if c_conf <= 1.0 else c_conf
            
            e_tracks = enh_tracks_data.get(f, [])
            if e_tracks:
                e_conf = sum([t.get('conf', 0) for t in e_tracks]) / len(e_tracks)
                enh_conf[f - 1] = e_conf * 100 if e_conf <= 1.0 else e_conf
            
        self.comp_curve.setData(frames, comp_conf)
        self.enh_curve.setData(frames, enh_conf)
        
        # Auto-scale Y-Range to focus on differences
        min_val = min(np.min(comp_conf), np.min(enh_conf))
        max_val = max(np.max(comp_conf), np.max(enh_conf))
        padding = max(2.0, (max_val - min_val) * 0.2)
        self.line_widget.setYRange(max(0, min_val - padding), min(100, max_val + padding))

    def append_realtime_data(self, frame_idx, comp_conf, enh_conf):
        """Append real-time data to the line chart dynamically."""
        if not hasattr(self, '_rt_frames'):
            self._rt_frames = []
            self._rt_comp = []
            self._rt_enh = []
            
        self._rt_frames.append(frame_idx)
        self._rt_comp.append(comp_conf)
        self._rt_enh.append(enh_conf)
        
        # Keep last 150 frames for scrolling window
        max_window = 150
        if len(self._rt_frames) > max_window:
            self._rt_frames.pop(0)
            self._rt_comp.pop(0)
            self._rt_enh.pop(0)
            
        self.comp_curve.setData(self._rt_frames, self._rt_comp)
        self.enh_curve.setData(self._rt_frames, self._rt_enh)
        self.line_widget.setXRange(max(0, frame_idx - max_window), max(100, frame_idx + 10))
        
        # Auto-scale Y-Range dynamically for realtime data
        if self._rt_comp and self._rt_enh:
            min_val = min(min(self._rt_comp), min(self._rt_enh))
            max_val = max(max(self._rt_comp), max(self._rt_enh))
            padding = max(2.0, (max_val - min_val) * 0.2)
            self.line_widget.setYRange(max(0, min_val - padding), min(100, max_val + padding))

    def clear_chart(self):
        """Clears all data from the line chart."""
        self.comp_curve.setData([], [])
        self.enh_curve.setData([], [])
        if hasattr(self, '_rt_frames'):
            self._rt_frames.clear()
            self._rt_comp.clear()
            self._rt_enh.clear()
