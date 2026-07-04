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
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        # --- Header ---
        hdr = QHBoxLayout()
        title_lbl = QLabel("Tracking Performance Analysis")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #111827;")
        sub_lbl = QLabel("Compressed  vs  Enhanced — MOT20 Validation")
        sub_lbl.setStyleSheet("font-size: 11px; color: #6B7280;")
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
        self.bar_widget.setMinimumHeight(160)
        self.bar_widget.setMaximumHeight(200)
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
        self.line_widget.setMinimumHeight(160)
        self.line_widget.setMaximumHeight(200)
        self._build_line_chart()
        charts_row.addWidget(self.line_widget, stretch=2)

        outer.addLayout(charts_row)

        # --- Legend row ---
        legend_row = QHBoxLayout()
        legend_row.setSpacing(20)
        legend_row.addStretch()
        for color, label in [("#EF4444", "Compressed"), ("#0F766E", "Enhanced")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 11px; color: #374151; font-weight: 500;")
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

        # Data: [HOTA, IDF1, MOTA, Recall, Precision]
        comp_vals = [61.2, 66.3, 54.8, 72.1, 81.4]
        enh_vals  = [78.9, 84.7, 73.6, 88.5, 91.2]
        x_positions = [0, 2, 4, 6, 8]

        bar_w = 0.7
        for i, (xp, cv, ev) in enumerate(zip(x_positions, comp_vals, enh_vals)):
            # Compressed bar
            bg_comp = pg.BarGraphItem(x=[xp], height=[cv], width=bar_w,
                                      brush='#EF4444', pen=pg.mkPen(None))
            pw.addItem(bg_comp)
            # Enhanced bar
            bg_enh = pg.BarGraphItem(x=[xp + 1], height=[ev], width=bar_w,
                                     brush='#0F766E', pen=pg.mkPen(None))
            pw.addItem(bg_enh)

            # Value labels
            txt_comp = pg.TextItem(f"{cv:.0f}", color='#374151', anchor=(0.5, 0))
            txt_comp.setFont(QFont('Inter', 7, QFont.Bold))
            txt_comp.setPos(xp, cv + 1)
            pw.addItem(txt_comp)

            txt_enh = pg.TextItem(f"{ev:.0f}", color='#374151', anchor=(0.5, 0))
            txt_enh.setFont(QFont('Inter', 7, QFont.Bold))
            txt_enh.setPos(xp + 1, ev + 1)
            pw.addItem(txt_enh)

    def _build_line_chart(self):
        """IDF1 over time with event markers."""
        pw = self.line_widget
        pw.setLabel('left', 'IDF1 (%)')
        pw.setLabel('bottom', 'Frame')
        pw.showGrid(x=False, y=True, alpha=0.2)
        pw.setYRange(0, 105)
        pw.getAxis('bottom').setPen(pg.mkPen('#D1D5DB'))
        pw.getAxis('left').setPen(pg.mkPen(None))
        pw.setMouseEnabled(x=False, y=False)
        pw.setTitle("IDF1 Over Frames — Tracking Stability", color='#374151', size='10pt')

        frames, comp_idf1, enh_idf1 = generate_timeline_data()

        pw.plot(frames, comp_idf1, pen=pg.mkPen(color='#EF4444', width=1.5), name='Compressed')
        pw.plot(frames, enh_idf1, pen=pg.mkPen(color='#0F766E', width=1.5), name='Enhanced')

        # Fill area under enhanced line for emphasis
        fill = pg.FillBetweenItem(
            pw.plot(frames, comp_idf1, pen=pg.mkPen(None)),
            pw.plot(frames, enh_idf1, pen=pg.mkPen(None)),
            brush=(15, 118, 110, 30)
        )
        pw.addItem(fill)

        # Event markers (vertical dashed lines + text)
        for frame_idx, event in list(EVENTS.items())[:4]:  # show max 4 events
            line = pg.InfiniteLine(pos=frame_idx, angle=90,
                                   pen=pg.mkPen('#9CA3AF', width=1, style=Qt.DashLine))
            pw.addItem(line)
            txt = pg.TextItem(event['name'], color='#6B7280', anchor=(0, 1))
            txt.setFont(QFont('Inter', 7))
            txt.setPos(frame_idx + 10, 100)
            pw.addItem(txt)
