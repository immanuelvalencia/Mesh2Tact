"""Shared per-checkpoint image-type annotations and a constrained table editor."""
import hashlib
from pathlib import Path

from .qt import QtCore, QtWidgets
from ..prediction import IMAGE_TYPES


class ModelTypeStore(QtCore.QObject):
    changed = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QtCore.QSettings("Mesh2Tact", "ModelImageTypes")

    @staticmethod
    def path_key(path):
        return str(Path(path).resolve()).casefold()

    def key(self, path):
        return "models/" + hashlib.sha256(self.path_key(path).encode("utf-8")).hexdigest()

    def get(self, path, fallback):
        value = self.settings.value(self.key(path), fallback)
        return value if value in IMAGE_TYPES else fallback

    def set(self, path, value):
        if value not in IMAGE_TYPES:
            raise ValueError("Choose a supported image type")
        self.settings.setValue(self.key(path), value)
        self.changed.emit(self.path_key(path), value)


def model_type_store():
    app = QtWidgets.QApplication.instance()
    if not hasattr(app, "_model_type_store"):
        app._model_type_store = ModelTypeStore(app)
    return app._model_type_store


class ImageTypeDelegate(QtWidgets.QStyledItemDelegate):
    """Allow editing only the image-type column, keeping paths and scores read-only."""

    def __init__(self, column, parent=None):
        super().__init__(parent)
        self.column = column

    def createEditor(self, parent, option, index):
        if index.column() != self.column:
            return None
        editor = QtWidgets.QComboBox(parent)
        editor.addItems(IMAGE_TYPES)
        editor.activated.connect(lambda *_: self.commitData.emit(editor))
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentText(index.data())

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), QtCore.Qt.EditRole)
