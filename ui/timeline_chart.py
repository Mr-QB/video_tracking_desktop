import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal
import numpy as np

from core.mock_data import generate_timeline_data, EVENTS

class TimelineChart(QWidget):
    marker_clicked = Signal(int) # Emits frame_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_lbl = QLabel("Metric Timeline \u2013 IDF1 Over Time \u24D8") # info icon
        title_lbl.setProperty("class", "SectionTitle")
        layout.addWidget(title_lbl)
        
        # Setup PyQtGraph PlotWidget
        pg.setConfigOption('background', '#FFFFFF')
        pg.setConfigOption('foreground', '#556270')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(250)
        layout.addWidget(self.plot_widget)
        
        # Plot styling
        self.plot_widget.setLabel('left', 'IDF1')
        self.plot_widget.setLabel('bottom', 'Frame')
        self.plot_widget.setXRange(0, 1500)
        self.plot_widget.setYRange(0, 110)
        self.plot_widget.showGrid(x=False, y=True, alpha=0.3)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(None))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen('#DDE2E8'))
        
        # Add legend
        self.legend = self.plot_widget.addLegend(offset=(30, 30))
        
        # Load data
        frames, comp_idf1, enh_idf1 = generate_timeline_data()
        
        # Plot lines
        pen_comp = pg.mkPen(color='#D95D39', width=2)
        self.plot_widget.plot(frames, comp_idf1, pen=pen_comp, name='Compressed')
        
        pen_enh = pg.mkPen(color='#1F5A94', width=2) # Changed to blue from #087E8B
        self.plot_widget.plot(frames, enh_idf1, pen=pen_enh, name='Enhanced')
        
        # Interaction
        self.plot_widget.setMouseEnabled(x=False, y=False) # Disable panning for simpler interaction


