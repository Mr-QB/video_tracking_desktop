from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import Qt

class ProtocolPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_lbl = QLabel("Experiment Protocol")
        title_lbl.setProperty("class", "SectionTitle")
        layout.addWidget(title_lbl)
        
        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(20)
        
        protocols = [
            ("\u2699\uFE0E Detector", "YOLOX-X (same weights)"),
            ("\u2316\uFE0E Tracker", "ByteTrack (same config)"),
            ("\u25B6\uFE0E Sequence", "MOT17-02 (same frames)"),
            ("\u25A3\uFE0E Resolution", "1920 \u00D7 1080 (same)"),
            ("\u2295\uFE0E Detection Threshold", "0.5 (same)"),
            ("\u2296\uFE0E NMS Threshold", "0.7 (same)"),
            ("\u2728\uFE0E Enhancement Model", "Real-ESRGAN \u00D72 (general)"),
            ("\u25A6\uFE0E Codec / Bitrate", "H.264 / 1.2 Mbps (CRF 28)"),
            ("\u2699\uFE0E All other settings", "Identical")
        ]
        
        for i, (key, val) in enumerate(protocols):
            key_lbl = QLabel(key)
            key_lbl.setStyleSheet("color: #556270; font-weight: 500;")
            
            val_lbl = QLabel(val)
            val_lbl.setStyleSheet("color: #1D232B;")
            
            grid.addWidget(key_lbl, i, 0)
            grid.addWidget(val_lbl, i, 1)
            
        layout.addLayout(grid)
        layout.addStretch()
