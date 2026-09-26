"""Named section presets with a protected default and atomic JSON persistence."""
from copy import deepcopy
import json
from pathlib import Path
from .qt import QtWidgets


class AppearancePresets(QtWidgets.QWidget):
    def __init__(self, path, capture, apply, default, validate):
        super().__init__()
        self.path = Path(path)
        self.capture, self.apply, self.default, self.validate = capture, apply, default, validate
        self.presets = {}
        self.read_error = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.selector = QtWidgets.QComboBox()
        layout.addWidget(self.selector)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for name, value in raw.items():
                    if name.casefold() in ("custom", "default"):
                        continue
                    validate(value)
                    self.presets[name] = value
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.read_error = True
            self.status.setText(f"Could not read presets: {exc}. Existing file preserved.")
        self.selector.addItems(["Custom", "Default", *self.presets])
        self.selector.currentTextChanged.connect(self.select)
        row = QtWidgets.QHBoxLayout()
        self.create_button = QtWidgets.QPushButton("Create preset…")
        self.update_button = QtWidgets.QPushButton("Update preset")
        self.delete_button = QtWidgets.QPushButton("Delete preset")
        for button, callback in [(self.create_button, self.create), (self.update_button, self.update), (self.delete_button, self.delete)]:
            row.addWidget(button)
            button.clicked.connect(callback)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.buttons()

    def buttons(self):
        editable = self.selector.currentText() in ('Default', *self.presets) and not self.read_error
        self.update_button.setEnabled(editable)
        self.delete_button.setEnabled(editable)
        self.create_button.setEnabled(not self.read_error)

    def select(self, name):
        if name == "Default" or name in self.presets:
            self.apply(deepcopy(self.default if name == "Default" else self.presets[name]))
        self.buttons()

    def persist(self, entries):
        if self.read_error or any(name.casefold() in ('default', 'custom') for name in entries):
            self.status.setText('Built-in presets are read-only. Create a new named preset.')
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(entries, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not save presets: {exc}")
            return False
        self.presets = entries
        self.status.setText("Saved. Edit controls, then Update preset to keep changes.")
        return True

    def create(self, _checked=False, protected=False):
        name, ok = QtWidgets.QInputDialog.getText(self, "Protected preset" if protected else "Create preset",
            "Default cannot be edited, overwritten, or deleted. Create a new preset. Name:" if protected else "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        if name.casefold() in {n.casefold() for n in ["Default", "Custom", *self.presets]}:
            self.status.setText("Choose a unique name. Default is protected.")
            return
        value = self.capture()
        self.validate(value)
        if self.persist({**self.presets, name: value}):
            self.selector.addItem(name)
            self.selector.setCurrentText(name)

    def update(self):
        name = self.selector.currentText()
        if name == 'Default':
            self.create(protected=True)
            return
        if name in self.presets:
            value = self.capture()
            self.validate(value)
            self.persist({**self.presets, name: value})

    def delete(self):
        name = self.selector.currentText()
        if name == 'Default':
            self.create(protected=True)
            return
        if name in self.presets and self.persist({k: v for k, v in self.presets.items() if k != name}):
            self.selector.blockSignals(True)
            self.selector.removeItem(self.selector.currentIndex())
            self.selector.setCurrentText("Custom")
            self.selector.blockSignals(False)
            self.buttons()
