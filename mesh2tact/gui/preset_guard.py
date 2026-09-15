"""Intercept user edits before protected preset controls can change."""
from .qt import QtCore, QtWidgets


class PresetEditGuard(QtCore.QObject):
    def __init__(self, parent, controls, protected, prompt):
        super().__init__(parent)
        self.controls, self.protected, self.prompt = controls, protected, prompt
        self.busy = False
        for control in controls:
            control.installEventFilter(self)
            for child in control.findChildren(QtWidgets.QWidget):
                child.installEventFilter(self)

    def eventFilter(self, watched, event):
        if self.busy or event.type() not in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.Wheel, QtCore.QEvent.KeyPress):
            return False
        if not isinstance(watched, QtWidgets.QWidget):
            return False
        if event.type() == QtCore.QEvent.KeyPress and event.key() in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab, QtCore.Qt.Key_Escape):
            return False
        if not any(watched is control or control.isAncestorOf(watched) for control in self.controls):
            return False
        if not self.protected():
            return False
        self.busy = True
        try:
            self.prompt()
        finally:
            self.busy = False
        return True


def edit_controls(panel, excluded=()):
    return [w for w in panel.findChildren(QtWidgets.QWidget)
            if isinstance(w, (QtWidgets.QAbstractSpinBox, QtWidgets.QAbstractSlider,
                              QtWidgets.QAbstractButton, QtWidgets.QComboBox))
            and not any(w is parent or parent.isAncestorOf(w) for parent in excluded)]
