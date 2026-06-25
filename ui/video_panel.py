import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PySide6.QtCore import Qt, QRect

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
        
        self.current_image = None
        self.current_tracks = []
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        
        # Transparent background so it doesn't show black blocks if aspect ratio isn't perfect
        self.setStyleSheet("background-color: transparent;")
        
        # Colors
        self.box_color = QColor("#0F766E") if is_enhanced else QColor("#EF4444")
        self.text_bg_color = QColor("#111827")
        self.text_bg_color.setAlpha(220)
        
        # Aspect ratio enforcement
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return int(width * 9 / 16)

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
            
            font = QFont("Inter", 11, QFont.Bold)
            painter.setFont(font)
            
            for track in self.current_tracks:
                x, y, bw, bh = track['bbox']
                t_id = track['id']
                conf = track['conf']
                
                # Draw Bounding Box
                if self.show_tracks:
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
                    
                    bg_rect = QRect(x, y - text_height - 4, text_width + 8, text_height + 4)
                    painter.fillRect(bg_rect, self.text_bg_color)
                    
                    # Draw label text
                    painter.setPen(QColor("#FFFFFF"))
                    painter.drawText(x + 4, y - 4, label_text)
                    painter.setPen(pen) # Restore pen
                    
            painter.end()

        # Scale pixmap to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()
