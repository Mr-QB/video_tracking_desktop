import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PySide6.QtCore import Qt, QRect, QSize

class VideoCanvas(QWidget):
    """
    Custom widget to display a video frame and overlay tracking bounding boxes.
    """
    def __init__(self, is_enhanced=False, parent=None):
        super().__init__(parent)
        self.is_enhanced = is_enhanced
        
        # Overlay configurations
        self.show_tracks = True
        self.show_ids = True
        self.show_confidence = True
        self.show_detections = False
        
        self.current_image = None  # Will be set when first frame arrives
        self.current_tracks = []
        
        # Layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        # CRITICAL: Set Ignored so QLabel does NOT drive the layout size based on pixmap.
        # Without this, every setPixmap() call changes the sizeHint of the label,
        # causing a layout recalculation → resizeEvent → update_display() → infinite loop.
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(1, 1)
        self._layout.addWidget(self.image_label)
        
        # Dark background so collapsed state looks clean
        self.setStyleSheet("background-color: #1a1a2e;")
        
        # Colors
        self.box_color = QColor("#0F766E") if is_enhanced else QColor("#EF4444")
        self.text_bg_color = QColor("#111827")
        self.text_bg_color.setAlpha(220)
        
        # Lock in a 16:9 aspect ratio for the canvas widget itself.
        # QSizePolicy.Expanding on both axes + heightForWidth drives the layout.
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        
        # Minimum height so the canvas never collapses to 0
        self.setMinimumHeight(180)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return int(width * 9 / 16)

    def sizeHint(self):
        return QSize(640, 360)  # Sensible default so layout has a starting size

    def set_overlay_flags(self, tracks, ids, conf, detections):
        self.show_tracks = tracks
        self.show_ids = ids
        self.show_confidence = conf
        self.show_detections = detections
        self.update_display()

    def update_frame(self, img_np, tracks):
        """Receives RGB numpy array and tracks list, updates display."""
        self.current_image = img_np
        self.current_tracks = tracks
        self.update_display()

    def update_display(self):
        if self.current_image is None:
            return
            
        # Convert numpy array to QImage
        h, w, ch = self.current_image.shape
        bytes_per_line = ch * w
        q_img = QImage(self.current_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create a pixmap to draw on
        pixmap = QPixmap.fromImage(q_img)
        
        # Draw overlays if requested
        if self.show_tracks or self.show_ids or self.show_confidence:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            pen = QPen(self.box_color)
            pen.setWidth(2)
            painter.setPen(pen)
            
            font = QFont("Inter", 10, QFont.Bold)
            painter.setFont(font)
            
            for track in self.current_tracks:
                x, y, bw, bh = track['bbox']
                t_id = track['id']
                conf = track['conf']
                
                # Draw Bounding Box
                if self.show_tracks:
                    painter.setPen(pen)
                    painter.drawRect(x, y, bw, bh)
                
                # Prepare Label Text
                label_parts = []
                if self.show_ids:
                    label_parts.append(f"ID: {t_id}")
                if self.show_confidence:
                    label_parts.append(f"{conf:.2f}")
                    
                if label_parts:
                    label_text = " | ".join(label_parts)
                    
                    # Draw label background
                    fm = painter.fontMetrics()
                    text_width = fm.horizontalAdvance(label_text)
                    text_height = fm.height()
                    
                    bg_rect = QRect(x, max(0, y - text_height - 4), text_width + 8, text_height + 4)
                    painter.fillRect(bg_rect, self.text_bg_color)
                    
                    # Draw label text
                    painter.setPen(QColor("#FFFFFF"))
                    painter.drawText(x + 4, max(text_height, y - 4), label_text)
                    
            painter.end()

        # Scale pixmap to fit the widget while maintaining aspect ratio.
        # Since the label's sizePolicy is Ignored, this scale uses the widget's size
        # (not the label's size) so there is NO feedback loop.
        target_size = self.size()
        if target_size.width() > 0 and target_size.height() > 0:
            scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # Widget not laid out yet: scale to sizeHint so there's something to show
            scaled_pixmap = pixmap.scaled(self.sizeHint(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Redraw at the new size. Safe because the label's sizePolicy is Ignored,
        # so setPixmap() does NOT trigger another layout/resize cycle.
        self.update_display()
