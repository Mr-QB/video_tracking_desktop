from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt

class MetricCard(QWidget):
    def __init__(self, title, val_before, val_after, delta_text, is_positive_delta=True, is_neutral=False, parent=None):
        super().__init__(parent)
        # Removed .Card class from individual metric
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        
        # Title
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color: #1D232B; font-size: 13px; font-weight: bold;")
        title_lbl.setAlignment(Qt.AlignCenter)
        
        # Values
        val_layout = QHBoxLayout()
        val_layout.setAlignment(Qt.AlignCenter)
        val_layout.setSpacing(4)
        
        if val_before is not None:
            val_lbl = QLabel(f"{val_before} \u2192 {val_after}") # right arrow
        else:
            val_lbl = QLabel(f"{val_after}")
            
        val_lbl.setProperty("class", "MetricValue")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_layout.addWidget(val_lbl)
        
        # Delta
        delta_lbl = QLabel(delta_text)
        delta_lbl.setAlignment(Qt.AlignCenter)
        delta_lbl.setStyleSheet("font-size: 12px;")
        if is_neutral:
            delta_lbl.setProperty("class", "MetricNeutral")
        elif is_positive_delta:
            delta_lbl.setProperty("class", "MetricDeltaPositive")
        else:
            delta_lbl.setProperty("class", "MetricDeltaNegative")
            
        layout.addWidget(title_lbl)
        layout.addLayout(val_layout)
        if delta_text:
            layout.addWidget(delta_lbl)
            
        self.setMinimumWidth(120)

class MetricGroup(QWidget):
    def __init__(self, title_main, title_sub, metrics, color, parent=None):
        """
        metrics: list of dicts: {'title': str, 'before': str, 'after': str, 'delta': str, 'pos': bool, 'neutral': bool}
        """
        super().__init__(parent)
        self.setProperty("class", "Card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 15)
        layout.setSpacing(10)
        
        # Title Layout
        title_layout = QHBoxLayout()
        main_lbl = QLabel(title_main)
        main_lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
        
        sub_lbl = QLabel(title_sub)
        sub_lbl.setStyleSheet("color: #556270; font-size: 14px;")
        
        title_layout.addWidget(main_lbl)
        title_layout.addWidget(sub_lbl)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # Metrics line
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(5)
        for i, m in enumerate(metrics):
            card = MetricCard(
                m['title'], 
                m.get('before'), 
                m.get('after'), 
                m.get('delta'), 
                m.get('pos', True),
                m.get('neutral', False)
            )
            cards_layout.addWidget(card)
            
            # Add vertical line separator between metrics
            if i < len(metrics) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setStyleSheet("color: #E8ECEF;")
                cards_layout.addWidget(line)
                
        layout.addLayout(cards_layout)

def create_all_metric_groups():
    """Factory function to create the row of metric groups."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(20)
    
    # 1. Video Quality
    vq_metrics = [
        {'title': "VMAF \u2191", 'before': "58.3", 'after': "82.6", 'delta': "+24.3 (41.7%)", 'pos': True},
        {'title': "SSIM \u2191", 'before': "0.912", 'after': "0.971", 'delta': "+0.059 (6.5%)", 'pos': True},
        {'title': "PSNR \u2191", 'before': "26.1", 'after': "32.8", 'delta': "+6.7 dB (25.7%)", 'pos': True}
    ]
    layout.addWidget(MetricGroup("Video Quality", " (before \u2192 after)", vq_metrics, "#D95D39"))
    
    # 2. Tracking Performance
    tp_metrics = [
        {'title': "HOTA \u2191", 'before': "61.2", 'after': "78.9", 'delta': "+17.7 (28.9%)", 'pos': True},
        {'title': "IDF1 \u2191", 'before': "66.3", 'after': "84.7", 'delta': "+18.4 (27.8%)", 'pos': True},
        {'title': "ID Switches \u2193", 'before': "186", 'after': "62", 'delta': "\u2212124 (\u221266.7%)", 'pos': True},
        {'title': "False Negatives \u2193", 'before': "1,243", 'after': "512", 'delta': "\u2212731 (\u221258.8%)", 'pos': True}
    ]
    layout.addWidget(MetricGroup("Tracking Performance", " (before \u2192 after)", tp_metrics, "#087E8B"))
    
    # 3. Runtime
    rt_metrics = [
        {'title': "FPS (avg)", 'before': None, 'after': "21.4", 'delta': "frames / sec", 'neutral': True},
        {'title': "Latency (avg)", 'before': None, 'after': "46.3", 'delta': "ms / frame", 'neutral': True},
        {'title': "GPU Usage", 'before': None, 'after': "68%", 'delta': "RTX 3090", 'neutral': True}
    ]
    layout.addWidget(MetricGroup("Runtime / Deployment", " (enhanced)", rt_metrics, "#1F5A94"))
    
    layout.addStretch()
    
    return container
