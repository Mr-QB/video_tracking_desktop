from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt
from core.config import get_active_device_name

class MetricCard(QWidget):
    """
    Individual Metric Tile
    """
    def __init__(self, label, value, delta_text, is_positive=True, is_neutral=False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(2)
        
        # Label
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600;")
        lbl.setAlignment(Qt.AlignCenter)
        
        # Value
        self.val_lbl = QLabel(value)
        self.val_lbl.setProperty("class", "MetricValue")
        self.val_lbl.setAlignment(Qt.AlignCenter)
        
        # Delta
        self.delta_lbl = QLabel(delta_text)
        self.delta_lbl.setAlignment(Qt.AlignCenter)
        self._apply_delta_style(is_positive, is_neutral)
            
        layout.addWidget(lbl)
        layout.addWidget(self.val_lbl)
        layout.addWidget(self.delta_lbl)

    def _apply_delta_style(self, is_positive, is_neutral):
        if is_neutral:
            self.delta_lbl.setProperty("class", "MetricNeutral")
        elif is_positive:
            self.delta_lbl.setProperty("class", "MetricDeltaPositive")
        else:
            self.delta_lbl.setProperty("class", "MetricDeltaNegative")
            
        # Force style update
        self.delta_lbl.style().unpolish(self.delta_lbl)
        self.delta_lbl.style().polish(self.delta_lbl)

    def update_data(self, value, delta_text, is_positive=True, is_neutral=False):
        self.val_lbl.setText(str(value))
        self.delta_lbl.setText(str(delta_text))
        self._apply_delta_style(is_positive, is_neutral)

    def set_calculating(self):
        self.val_lbl.setText("...")
        self.delta_lbl.setText("Calculating")
        self._apply_delta_style(is_positive=True, is_neutral=True)


class MetricGroup(QWidget):
    """
    Card containing a group of related metrics
    """
    def __init__(self, title, metrics, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        self.cards = {}
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(6)
        
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #111827;")
        
        main_layout.addWidget(title_lbl)
        
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(4)
        
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
                value=metric.get('val', '-'),
                delta_text=metric.get('delta', '-'),
                is_positive=metric.get('is_positive', True),
                is_neutral=metric.get('is_neutral', False)
            )
            self.cards[metric['id']] = card
            metrics_layout.addWidget(card, stretch=1)
            
        main_layout.addLayout(metrics_layout)

    def update_card(self, metric_id, value, delta_text, is_positive=True, is_neutral=False):
        if metric_id in self.cards:
            self.cards[metric_id].update_data(value, delta_text, is_positive, is_neutral)

    def set_calculating(self):
        for card in self.cards.values():
            card.set_calculating()


class MetricDashboard(QWidget):
    """
    Dashboard section combining VQ, TP, and RT metric groups
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # 1. Visual Quality (vs Original)
        vq_metrics = [
            {'id': 'psnr', 'label': 'PSNR (vs Orig)', 'val': '-', 'delta': '-', 'is_positive': True},
            {'id': 'ssim', 'label': 'SSIM (vs Orig)', 'val': '-', 'delta': '-', 'is_positive': True}
        ]
        self.vq_group = MetricGroup("Visual Quality (Comp \u2192 Enh)", vq_metrics)
        
        # 2. Tracking Performance
        tp_metrics = [
            {'id': 'hota', 'label': 'HOTA', 'val': '-', 'delta': '-', 'is_positive': True},
            {'id': 'idf1', 'label': 'IDF1', 'val': '-', 'delta': '-', 'is_positive': True},
            {'id': 'id_switches', 'label': 'ID Switches', 'val': '-', 'delta': '-', 'is_positive': True},
            {'id': 'false_negatives', 'label': 'False Negatives', 'val': '-', 'delta': '-', 'is_positive': True}
        ]
        self.tp_group = MetricGroup("Tracking Performance", tp_metrics)
        
        # 3. Runtime / Deployment
        device_name = get_active_device_name()
        rt_metrics = [
            {'id': 'fps', 'label': 'FPS (avg)', 'val': '21.4', 'delta': 'frames / sec', 'is_neutral': True},
            {'id': 'latency', 'label': 'Latency (avg)', 'val': '46.3', 'delta': 'ms / frame', 'is_neutral': True},
            {'id': 'gpu', 'label': 'Device Usage', 'val': '68%', 'delta': device_name, 'is_neutral': True}
        ]
        self.rt_group = MetricGroup("Runtime / Deployment (Enhanced)", rt_metrics)
        
        layout.addWidget(self.vq_group, stretch=3)
        layout.addWidget(self.tp_group, stretch=4)
        layout.addWidget(self.rt_group, stretch=3)

    def update_tracking_metrics(self, base_metrics, enh_metrics):
        if not base_metrics or not enh_metrics:
            return
            
        def _fmt(val):
            return f"{val:.1f}" if isinstance(val, float) else str(val)

        # HOTA
        hb = base_metrics.get('HOTA', 0)
        he = enh_metrics.get('HOTA', 0)
        dh = he - hb
        self.tp_group.update_card('hota', f"{_fmt(hb)} \u2192 {_fmt(he)}", f"{dh:+.1f} ({(dh/max(1e-5, hb))*100:+.1f}%)", is_positive=(dh >= 0))
        
        # IDF1
        ib = base_metrics.get('IDF1', 0)
        ie = enh_metrics.get('IDF1', 0)
        di = ie - ib
        self.tp_group.update_card('idf1', f"{_fmt(ib)} \u2192 {_fmt(ie)}", f"{di:+.1f} ({(di/max(1e-5, ib))*100:+.1f}%)", is_positive=(di >= 0))
        
        # ID Switches (Lower is better)
        sb = base_metrics.get('ID_Switches', 0)
        se = enh_metrics.get('ID_Switches', 0)
        ds = se - sb
        self.tp_group.update_card('id_switches', f"{sb} \u2192 {se}", f"{ds:+} ({(ds/max(1, sb))*100:+.1f}%)", is_positive=(ds <= 0))
        
        # False Negatives (Lower is better)
        fb = base_metrics.get('False_Negatives', 0)
        fe = enh_metrics.get('False_Negatives', 0)
        df = fe - fb
        self.tp_group.update_card('false_negatives', f"{fb} \u2192 {fe}", f"{df:+} ({(df/max(1, fb))*100:+.1f}%)", is_positive=(df <= 0))

    def update_quality_metrics(self, comp_psnr, enh_psnr, comp_ssim, enh_ssim):
        def _fmt_psnr(val):
            return f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
            
        def _fmt_ssim(val):
            return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)

        # PSNR (vs Orig): Comp -> Enh
        if isinstance(comp_psnr, (int, float)) and isinstance(enh_psnr, (int, float)):
            delta_p = enh_psnr - comp_psnr
            pct_p = (delta_p / max(1e-5, comp_psnr)) * 100
            self.vq_group.update_card('psnr', f"{_fmt_psnr(comp_psnr)} \u2192 {_fmt_psnr(enh_psnr)} dB", f"{delta_p:+.1f} dB ({pct_p:+.1f}%)", is_positive=(delta_p >= 0))
        else:
            self.vq_group.update_card('psnr', f"{comp_psnr} \u2192 {enh_psnr}", "dB", is_positive=True, is_neutral=True)

        # SSIM (vs Orig): Comp -> Enh
        if isinstance(comp_ssim, (int, float)) and isinstance(enh_ssim, (int, float)):
            delta_s = enh_ssim - comp_ssim
            pct_s = (delta_s / max(1e-5, comp_ssim)) * 100
            self.vq_group.update_card('ssim', f"{_fmt_ssim(comp_ssim)} \u2192 {_fmt_ssim(enh_ssim)}", f"{delta_s:+.3f} ({pct_s:+.1f}%)", is_positive=(delta_s >= 0))
        else:
            self.vq_group.update_card('ssim', f"{comp_ssim} \u2192 {enh_ssim}", "index", is_positive=True, is_neutral=True)

    def update_runtime_metrics(self, fps, latency, device_usage="68%", device_name=None):
        def _fmt(val):
            return f"{val:.1f}" if isinstance(val, float) else str(val)
            
        if device_name is None:
            device_name = get_active_device_name()
            
        self.rt_group.update_card('fps', _fmt(fps), "frames / sec", is_positive=True, is_neutral=True)
        self.rt_group.update_card('latency', _fmt(latency), "ms / frame", is_positive=True, is_neutral=True)
        self.rt_group.update_card('gpu', device_usage, device_name, is_positive=True, is_neutral=True)
