"""Selectable diagnostic images, arranged responsively."""
from .qt import QtCore, QtWidgets
import json
from pathlib import Path

PREFERENCES_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'preview_preferences.json'


class PreviewGrid(QtWidgets.QWidget):
    applied = QtCore.Signal()
    labels = {'rgb': 'Tactile image', 'depth': 'Geometric depth',
              'clean': 'Clean tactile image', 'raw': 'Raw depth',
              'contact': 'Contact mask', 'normals': 'Surface normals'}

    def __init__(self, view_factory, preferences_path=None):
        super().__init__()
        self.preferences_path = Path(preferences_path or PREFERENCES_PATH)
        layout = QtWidgets.QVBoxLayout(self)
        choices = QtWidgets.QGroupBox('Preview panels')
        choice_layout = QtWidgets.QGridLayout(choices)
        self.checkboxes = {}
        self._selected = ['rgb', 'depth']
        for index, (key, label) in enumerate(self.labels.items()):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(key in self._selected)
            self.checkboxes[key] = checkbox
            choice_layout.addWidget(checkbox, index//2, index%2)
        self.apply_button = QtWidgets.QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply_selection)
        choice_layout.addWidget(self.apply_button, 3, 0, 1, 2)
        layout.addWidget(choices)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        self.content = QtWidgets.QWidget()
        scroll.setWidget(self.content)
        self.grid = QtWidgets.QGridLayout(self.content)
        self.cards, self.views = {}, {}
        for key, label in self.labels.items():
            card = QtWidgets.QGroupBox(self.content)
            card_layout = QtWidgets.QVBoxLayout(card)
            title = QtWidgets.QLabel(label)
            title.setWordWrap(True)
            title.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
            card_layout.addWidget(title)
            view = view_factory()
            view.setMinimumSize(140, 100)
            view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            card_layout.addWidget(view, 1)
            self.cards[key], self.views[key] = card, view
        self.empty = QtWidgets.QLabel('Select preview panels above, then click Apply.')
        layout.addWidget(self.empty)
        self.persistence_status = QtWidgets.QLabel()
        self.persistence_status.setWordWrap(True)
        self.persistence_status.hide()
        layout.addWidget(self.persistence_status)
        for checkbox in self.checkboxes.values():
            checkbox.toggled.connect(self.mark_pending)
        if self.preferences_path.exists():
            try:
                raw = json.loads(self.preferences_path.read_text(encoding='utf-8'))
                if not isinstance(raw, dict) or raw.get('version') != 1:
                    raise ValueError('Unsupported preview preferences')
                self.set_selected(raw['panels'], save=False)
            except (OSError, ValueError, KeyError, TypeError):
                self.persistence_status.setText('Could not load preview choices; using tactile image and depth. Click Apply to save your choices.')
                self.persistence_status.show()
        self.reflow()

    def selected(self):
        return list(self._selected)

    def apply_selection(self):
        keys = [key for key, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        self.set_selected(keys)

    def mark_pending(self):
        self.persistence_status.setText('Click Apply to save preview choices.')
        self.persistence_status.show()

    def set_selected(self, keys, save=True):
        if not isinstance(keys, list) or any(key not in self.labels for key in keys):
            raise ValueError('Unknown preview panel')
        keys = [key for key in self.labels if key in keys]
        if save and not self.save_preferences(keys):
            return False
        self._selected = keys
        for key, checkbox in self.checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(key in keys)
            checkbox.blockSignals(False)
        if not save:
            self.persistence_status.hide()
        self.reflow()
        self.applied.emit()
        return True

    def save_preferences(self, keys=None):
        keys = self.selected() if keys is None else keys
        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
            output = QtCore.QSaveFile(str(self.preferences_path))
            if not output.open(QtCore.QIODevice.WriteOnly):
                raise OSError(output.errorString())
            encoded = (json.dumps({'version': 1, 'panels': keys}, indent=2)+'\n').encode('utf-8')
            if output.write(encoded) != len(encoded):
                output.cancelWriting()
                raise OSError('Incomplete preferences write')
            if not output.commit():
                raise OSError(output.errorString())
            self.persistence_status.setText('Preview choices saved.')
            self.persistence_status.show()
            return True
        except OSError as exc:
            self.persistence_status.setText(f'Preview choices could not be saved: {exc}')
            self.persistence_status.show()
            return False

    def reflow(self, *_):
        while self.grid.count():
            self.grid.takeAt(0)
        selected = self.selected()
        columns = 1 if len(selected) <= 2 else min(3, max(1, self.width()//210), (len(selected)+1)//2)
        for column in range(3):
            self.grid.setColumnStretch(column, int(column < columns))
        rows = (len(selected)+columns-1)//columns
        for row in range(6):
            self.grid.setRowStretch(row, int(row < rows))
        for key, card in self.cards.items():
            card.setVisible(key in selected)
        for index, key in enumerate(selected):
            self.grid.addWidget(self.cards[key], index//columns, index % columns)
        self.empty.setVisible(not selected)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()
