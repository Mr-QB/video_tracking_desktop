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
        
        # Add Event Markers
        self._add_event_markers()
        
        # Interaction
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)
        self.plot_widget.setMouseEnabled(x=False, y=False) # Disable panning for simpler interaction

    def _add_event_markers(self):
        self.markers = []
        for frame, event_info in EVENTS.items():
            # Vertical line
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#DDE2E8', style=pg.QtCore.Qt.DashLine))
            v_line.setPos(frame)
            self.plot_widget.addItem(v_line)
            
            # Determine color based on event name/type for visual flair like image
            color = '#556270'
            if "Blur" in event_info['name']: color = '#F4B400' # Yellow
            elif "Occlusion" in event_info['name']: color = '#9B51E0' # Purple
            elif "Lost" in event_info['name']: color = '#D95D39' # Red
            elif "Switch" in event_info['name']: color = '#556270' # Gray
            elif "Recovery" in event_info['name']: color = '#0F9D58' # Green
            
            # Plot dot
            dot = pg.ScatterPlotItem([frame], [100], size=8, brush=pg.mkBrush(color), pen=pg.mkPen(None))
            self.plot_widget.addItem(dot)
            
            # Text label
            label = pg.TextItem(f"{event_info['name']}\n({event_info['type']})", anchor=(0.5, 1), color='#1D232B')
            label.setPos(frame, 95)
            # Make label clickable-ish by storing it
            self.plot_widget.addItem(label)
            self.markers.append({'frame': frame, 'line': v_line, 'label': label})

    def _on_mouse_click(self, evt):
        if evt.button() == pg.QtCore.Qt.LeftButton:
            # Get mouse position in plot coordinates
            pos = self.plot_widget.plotItem.vb.mapSceneToView(evt.scenePos())
            x_val = pos.x()
            
            # Find closest marker if within a reasonable distance (e.g. 50 frames)
            closest_frame = None
            min_dist = float('inf')
            
            for marker in self.markers:
                dist = abs(marker['frame'] - x_val)
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    closest_frame = marker['frame']
                    
            if closest_frame is not None:
                self.marker_clicked.emit(closest_frame)
