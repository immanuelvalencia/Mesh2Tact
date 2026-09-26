"""Directory-backed sensor-only configurations. Object poses are independent."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re

from .qt import QtCore, QtWidgets
from ..config import SensorConfig
from ..lighting import validate_lighting
from ..render.gel import GelLighting
from ..render.effects import ImageEffects
from ..preview import migrate_preview, preview_size, link_capture_controls

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "sensors"
PROTECTED_NAMES = {'gelsightv1', 'real gelsight reference'}


class SensorConfigPanel(QtWidgets.QGroupBox):
    def __init__(self, window, directory=None):
        super().__init__("Saved sensor configurations")
        self.window = window
        self.directory = Path(directory or CONFIG_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = None
        self.preferences = QtCore.QSettings("Mesh2Tact", "SensorConfigs")
        self.selection_key = 'selected/' + self.directory.resolve().as_posix()
        layout = QtWidgets.QVBoxLayout(self)
        self.selector = QtWidgets.QComboBox()
        self.selector.currentIndexChanged.connect(self.select)
        layout.addWidget(self.selector)
        self.protected = QtWidgets.QCheckBox("Protected")
        self.protected.setToolTip("Prevent edits and deletion. Uncheck to unlock the selected configuration.")
        self.protected.clicked.connect(self.set_protected)
        layout.addWidget(self.protected)
        grid = QtWidgets.QGridLayout()
        for index, (title, callback) in enumerate([("Add current…", self.add_dialog),
              ("Update selected", self.update_selected), ("Delete selected", self.delete_selected), ("Refresh list", lambda: self.scan(force=True)),
              ("Load config file…", self.load_file_dialog)]):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(callback)
            grid.addWidget(button, index//2, index%2)
        self.rename_button = QtWidgets.QPushButton("Rename selected…")
        self.rename_button.clicked.connect(self.rename_dialog)
        grid.addWidget(self.rename_button, 2, 1)
        layout.addLayout(grid)
        self.message = QtWidgets.QLabel("Select a config to apply it. Edit below, then Update selected.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.scan_timer = QtCore.QTimer(self)
        self.scan_timer.timeout.connect(self.scan)
        self.scan_timer.start(2000)
        self.scan()

    def load_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load sensor config',
            str(self.directory.parent/'calibration_sessions'/'runs'), 'Sensor configuration (*.json)')
        if path:
            try:
                self.load(Path(path))
                self.selector.blockSignals(True)
                self.selector.setCurrentIndex(0)
                self.selector.blockSignals(False)
                self.sync_protection()
                self.remember_selection()
                self.message.setText('Loaded '+str(path))
            except Exception as exc:
                self.message.setText(str(exc))

    def scan(self, force=False):
        paths = sorted(self.directory.glob("*.json"), key=lambda p: p.name.lower())
        try:
            fingerprint = [(p.name, p.stat().st_mtime_ns) for p in paths]
        except OSError:
            return
        if not force and fingerprint == self.fingerprint:
            return
        self.fingerprint = fingerprint
        selected = self.selector.currentData()
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItem("Current / unsaved settings", None)
        for path in paths:
            self.selector.addItem(path.stem, str(path))
            if self.is_protected(path):
                self.selector.setItemData(self.selector.count()-1, 'Protected: uncheck Protected to edit or delete.', QtCore.Qt.ToolTipRole)
        index = self.selector.findData(selected)
        self.selector.setCurrentIndex(max(index, 0))
        self.selector.blockSignals(False)
        self.sync_protection()

    def data(self):
        self.window.apply_controls()
        window = self.window
        sensor = asdict(window.sim.cfg)
        sensor['camera']['width'], sensor['camera']['height'] = window.capture_resolution
        return dict(format="mesh2tact-sensor-config", version=1, sensor=sensor,
                    max_penetration_mm=window.max_penetration.value(),
                    gel_lighting=asdict(window.sim.gel_lighting), effects=asdict(window.sim.effects),
                    controls={key: getattr(window, key).value() for key in
                              ("softness", "width", "height", "depth_range", "output_w", "output_h", "preview_scale")})

    def write(self, path):
        path = Path(path)
        if self.is_protected(path):
            raise ValueError('Uncheck Protected before updating this sensor config.')
        if path.resolve().parent != self.directory.resolve():
            raise ValueError("Config must be inside the sensor config directory")
        data = self.data()
        data['protected'] = False
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temp.replace(path)
        if path.stem.casefold() == 'default':
            from ..geometric import invalidate_default_sensor_profile
            invalidate_default_sensor_profile()
        self.scan(force=True)
        self.selector.blockSignals(True)
        self.selector.setCurrentIndex(self.selector.findData(str(path)))
        self.selector.blockSignals(False)
        self.sync_protection()
        self.message.setText(f"Saved {path.name}")
        self.remember_selection()

    def add(self, name):
        name = re.sub(r'[^\w .-]', "_", name).strip(" .")
        if not name:
            raise ValueError("Enter a config name")
        path = self.directory / (name + ".json")
        if path.exists():
            raise ValueError("That name already exists; use Update selected or another name")
        self.write(path)
        return path

    def add_dialog(self):
        name, accepted = QtWidgets.QInputDialog.getText(self, "Add sensor config", "Config name")
        if accepted:
            try:
                self.add(name)
            except Exception as exc:
                self.message.setText(str(exc))

    def rename_selected(self, name):
        selected = self.selector.currentData()
        if not selected:
            raise ValueError("Select a saved configuration to rename.")
        path = Path(selected)
        if path.stem.casefold() == 'default':
            raise ValueError("The default configuration cannot be renamed.")
        if self.is_protected(path):
            raise ValueError("Uncheck Protected before renaming this configuration.")
        name = name.strip()
        if not name or name.endswith(('.', ' ')) or re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
            raise ValueError("Enter a valid file name without slashes or special filename characters.")
        if name.casefold() == 'default':
            raise ValueError("The name default is reserved.")
        target = self.directory / (name + '.json')
        if path.resolve().parent != self.directory.resolve() or target.resolve().parent != self.directory.resolve():
            raise ValueError("Config must be inside the sensor directory.")
        if target == path:
            return
        if target.exists():
            raise ValueError("That name already exists. Choose another name.")
        path.rename(target)
        self.scan(force=True)
        self.selector.blockSignals(True)
        self.selector.setCurrentIndex(self.selector.findData(str(target)))
        self.selector.blockSignals(False)
        self.sync_protection()
        self.message.setText(f"Renamed {path.stem} to {target.stem}.")
        self.remember_selection()

    def rename_dialog(self):
        selected = self.selector.currentData()
        if not selected:
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Rename sensor configuration", "New name:", text=Path(selected).stem)
        if accepted:
            try:
                self.rename_selected(name)
            except Exception as exc:
                self.message.setText(str(exc))

    def is_protected(self, path=None):
        path = path or self.selector.currentData()
        if not path:
            return False
        path = Path(path)
        default = path.stem.casefold() in PROTECTED_NAMES
        try:
            return bool(json.loads(path.read_text(encoding="utf-8")).get('protected', default))
        except (OSError, ValueError, AttributeError):
            return default

    def sync_protection(self):
        self.protected.setEnabled(bool(self.selector.currentData()))
        self.protected.setChecked(self.is_protected())
        selected = self.selector.currentData()
        self.rename_button.setEnabled(bool(selected) and Path(selected).stem.casefold() != 'default'
                                      and not self.is_protected())

    def set_protected(self, checked):
        selected = self.selector.currentData()
        if not selected:
            self.sync_protection()
            return
        try:
            path = Path(selected)
            if path.resolve().parent != self.directory.resolve():
                raise ValueError("Config is outside the sensor directory")
            data = json.loads(path.read_text(encoding="utf-8"))
            data['protected'] = bool(checked)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            temp.replace(path)
            self.scan(force=True)
            self.message.setText(f"{path.name}: {'protected' if checked else 'unlocked for editing and deletion'}.")
        except Exception as exc:
            self.sync_protection()
            self.message.setText(f"Could not change protection: {exc}")

    def create_copy(self):
        name, accepted = QtWidgets.QInputDialog.getText(self, 'Protected sensor configuration',
            'This config is protected. Uncheck Protected to unlock it, or create an editable copy.\nCopy name:')
        if accepted:
            try:
                self.add(name)
            except Exception as exc:
                self.message.setText(str(exc))

    def update_selected(self):
        path = self.selector.currentData()
        if self.is_protected(path):
            self.create_copy()
            return
        if not path:
            self.message.setText("Select a config to update, or Add current.")
            return
        try:
            self.write(path)
        except Exception as exc:
            self.message.setText(f"Update failed: {exc}")

    def delete_selected(self):
        path = self.selector.currentData()
        if self.is_protected(path):
            self.message.setText("Uncheck Protected before deleting this configuration.")
            return
        if not path:
            return
        try:
            path = Path(path)
            if path.resolve().parent != self.directory.resolve():
                raise ValueError("Config is outside the sensor directory")
            path.unlink()
            self.scan(force=True)
            self.remember_selection()
            self.message.setText(f"Deleted {path.name}. Current sensor settings retained.")
        except Exception as exc:
            self.message.setText(f"Delete failed: {exc}")

    def select(self, *_):
        self.sync_protection()
        path = self.selector.currentData()
        if path:
            try:
                self.load(path)
                self.remember_selection()
            except Exception as exc:
                self.message.setText(f"Cannot apply config: {exc}")
        else:
            self.remember_selection()

    def remember_selection(self):
        path = self.selector.currentData()
        if path:
            self.preferences.setValue(self.selection_key, Path(path).name)
        else:
            self.preferences.remove(self.selection_key)

    def restore_selection(self):
        name = self.preferences.value(self.selection_key, '')
        if not name or Path(name).name != name:
            return False
        path = self.directory / name
        if not path.is_file():
            self.preferences.remove(self.selection_key)
            return False
        try:
            self.load(path)
        except Exception as exc:
            self.message.setText(f"Could not restore {name}: {exc}")
            return False
        self.selector.blockSignals(True)
        self.selector.setCurrentIndex(self.selector.findData(str(path)))
        self.selector.blockSignals(False)
        self.sync_protection()
        return True

    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") not in ("mesh2tact-sensor-config", "vtsim-sensor-config") or data.get("version") != 1:
            raise ValueError("Not a sensor configuration (version 1)")
        w = self.window
        maximum = data.get('max_penetration_mm', data['sensor']['camera']['max_depth']*1000)
        w.validate_widget(w.max_penetration, maximum, 'maximum indentation depth')
        data["controls"] = link_capture_controls(migrate_preview(data["controls"]))
        expected = {"softness", "width", "height", "depth_range", "output_w", "output_h", "preview_scale"}
        if set(data["controls"]) != expected:
            raise ValueError("Config has missing or unsupported sensor controls")
        assignments = [(getattr(w, key), value, key) for key, value in data["controls"].items()]
        effects = ImageEffects(**data["effects"])
        assignments += [(widget, getattr(effects, key), key) for key, widget in w.effect_widgets.items()]
        assignments += [(w.effect_seed, effects.seed, "effect seed"), (w.effects_enabled, effects.enabled, "effects enabled")]
        for widget, value, key in assignments:
            w.validate_widget(widget, value, key)
        cfg = SensorConfig.from_dict(data["sensor"])
        validate_lighting(cfg.optics)
        staged = deepcopy(w.sim)
        staged.max_penetration = maximum/1000
        staged.cfg, staged.effects = cfg, effects
        staged.gel_lighting = GelLighting.from_dict(data["gel_lighting"])
        controls = data["controls"]
        staged.cfg.gel.size_x, staged.cfg.gel.size_y = controls["width"]/1000, controls["height"]/1000
        staged.softness = controls["softness"]/1000
        staged.cfg.camera.max_depth = controls["depth_range"]/1000
        staged.cfg.camera.width, staged.cfg.camera.height = preview_size(controls)
        staged._key = None
        staged.render()
        w.timer.stop()
        for widget, value, key in assignments:
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        w.sim = staged
        w.max_penetration.blockSignals(True)
        w.max_penetration.setValue(maximum)
        w.max_penetration.blockSignals(False)
        w.lighting.set_optics(cfg.optics)
        w.gel_panel.set_settings(staged.gel_lighting)
        w.sync_appearance_panels()
        w._scene_key = None
        w.refresh()
        self.message.setText(f"Applied {Path(path).name}. " +
            ('Protected config: uncheck Protected to unlock, or create a copy.' if self.is_protected(path)
             else 'Edit below, then Update selected.'))
        training = [m for m in data.get('calibration', {}).get('metrics', []) if not m.get('validation')]
        if training and sum(m['after_mae'] for m in training) > sum(m['before_mae'] for m in training)+1e-7:
            self.message.setText(f'Applied {Path(path).name}, but its saved calibration increased image error. '
                'This file needs refitting with the corrected RGB sampler. Open Calibration and align the original references.')
