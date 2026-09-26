"""Shared desktop image views and action styling."""
import numpy as np
from .qt import QtCore, QtGui, QtWidgets

def style_action(button, color):
    button.setStyleSheet(f'''
        QPushButton {{ background: {color}; color: white; border: 2px solid transparent;
                       border-radius: 5px; padding: 6px 10px; font-weight: 600; }}
        QPushButton:hover {{ border-color: #b9d9ed; }}
        QPushButton:pressed {{ border-color: #172331; padding-top: 7px; padding-bottom: 5px; }}
        QPushButton:focus {{ border-color: #f5d45c; }}
        QPushButton:disabled {{ background: #d8dde2; color: #626b75; }}
    ''')


class ImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 170)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: #151c25")
        self.original = None

    def set_image(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        self.original = QtGui.QPixmap.fromImage(QtGui.QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy())
        self._fit()

    def _fit(self):
        if self.original is not None:
            self.setPixmap(self.original.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()
