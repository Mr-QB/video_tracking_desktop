import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def load_stylesheet(app):
    style_path = os.path.join(os.path.dirname(__file__), 'styles', 'app_style.qss')
    if os.path.exists(style_path):
        with open(style_path, 'r') as f:
            app.setStyleSheet(f.read())
    else:
        print("Warning: Stylesheet not found.")

def main():
    app = QApplication(sys.argv)
    
    # Load global styles
    load_stylesheet(app)
    
    # Initialize and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
