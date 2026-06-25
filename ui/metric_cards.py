from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt

class MetricCard(QWidget):
    """
    Individual Metric Tile
    """
    def __init__(self, label, value, delta_text, is_positive=True, is_neutral=False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        
        # Label
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: 600;")
        lbl.setAlignment(Qt.AlignCenter)
        
        # Value
        val_lbl = QLabel(value)
        val_lbl.setProperty("class", "MetricValue")
        val_lbl.setAlignment(Qt.AlignCenter)
        
        # Delta
        delta_lbl = None
        if delta_text:
            delta_lbl = QLabel(delta_text)
            if is_neutral:
                delta_lbl.setProperty("class", "MetricNeutral")
            elif is_positive:
                delta_lbl.setProperty("class", "MetricDeltaPositive")
            else:
                delta_lbl.setProperty("class", "MetricDeltaNegative")
            delta_lbl.setAlignment(Qt.AlignCenter)
            
        layout.addWidget(lbl)
        layout.addWidget(val_lbl)
        if delta_lbl:
            layout.addWidget(delta_lbl)

class MetricGroup(QWidget):
    """
    Card containing a group of related metrics
    """
    def __init__(self, title, metrics, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 14px; font-weight: 650; color: #111827;")
        
        main_layout.addWidget(title_lbl)
        
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)
        
        for i, metric in enumerate(metrics):
            if i > 0:
                # Add thin separator
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFrameShadow(QFrame.Sunken)
                line.setStyleSheet("color: #E5E7EB;")
                metrics_layout.addWidget(line)
                
            card = MetricCard(
                label=metric['label'],
                value=metric['val'],
                delta_text=metric.get('delta', ''),
                is_positive=metric.get('is_positive', True),
                is_neutral=metric.get('is_neutral', False)
            )
            metrics_layout.addWidget(card, stretch=1)
            
        main_layout.addLayout(metrics_layout)

def create_all_metric_groups():
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(20)
    
    # 1. Video Quality
    vq_metrics = [
        {'label': 'VMAF', 'val': '8.3 \u2192 82.6', 'delta': '+74.3 (+895%)', 'is_positive': True},
        {'label': 'SSIM', 'val': '0.912 \u2192 0.971', 'delta': '+0.059 (+6.5%)', 'is_positive': True},
        {'label': 'PSNR', 'val': '26.1 \u2192 32.8', 'delta': '+6.7 dB (+25.7%)', 'is_positive': True}
    ]
    vq_group = MetricGroup("Video Quality (Before \u2192 After)", vq_metrics)
    
    # 2. Tracking Performance
    tp_metrics = [
        {'label': 'HOTA', 'val': '61.2 \u2192 78.9', 'delta': '+17.7 (+28.9%)', 'is_positive': True},
        {'label': 'IDF1', 'val': '66.3 \u2192 84.7', 'delta': '+18.4 (+27.8%)', 'is_positive': True},
        {'label': 'ID Switches', 'val': '186 \u2192 62', 'delta': '-124 (-66.7%)', 'is_positive': True}, # Less is better, so positive
        {'label': 'False Negatives', 'val': '1,243 \u2192 512', 'delta': '-731 (-58.8%)', 'is_positive': True}
    ]
    tp_group = MetricGroup("Tracking Performance", tp_metrics)
    
    # 3. Runtime / Deployment
    rt_metrics = [
        {'label': 'FPS (avg)', 'val': '21.4', 'delta': 'frames / sec', 'is_neutral': True},
        {'label': 'Latency (avg)', 'val': '46.3', 'delta': 'ms / frame', 'is_neutral': True},
        {'label': 'GPU Usage', 'val': '68%', 'delta': 'RTX 3090', 'is_neutral': True}
    ]
    rt_group = MetricGroup("Runtime / Deployment (Enhanced)", rt_metrics)
    
    layout.addWidget(vq_group, stretch=3)
    layout.addWidget(tp_group, stretch=4)
    layout.addWidget(rt_group, stretch=3)
    
    return container
