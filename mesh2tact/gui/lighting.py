"""Live manual LED controls and reusable lighting presets."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .qt import QtCore, QtGui, QtWidgets
from ..config import LEDConfig, OpticsConfig
from ..lighting import load_lighting, save_lighting
from ..lighting import validate_lighting
from .appearance_presets import AppearancePresets

PRESET_DIR = Path(__file__).resolve().parents[2] / "configs" / "lighting"
LAST_PRESET = PRESET_DIR / "last_saved.json"


class LightingPanel(QtWidgets.QGroupBox):
    changed = QtCore.Signal(object)

    def __init__(self, show_presets=True):
        super().__init__("Tactile lighting")
        self.optics = OpticsConfig()
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)
        self.response_notice = QtWidgets.QLabel()
        self.response_notice.setWordWrap(True)
        layout.addWidget(self.response_notice)
        hint = QtWidgets.QLabel("Each colored light is an array spanning the entire selected edge. Light reaches the reflective membrane through the clear gel; deformation changes its reflected color. Select Top, Bottom, Left or Right for each array.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.side = QtWidgets.QCheckBox("Use full-edge LED arrays")
        self.side.toggled.connect(self._edit)
        layout.addWidget(self.side)
        self.side_distance = self._control(layout, "Distance from center (1 = sensor edge)", 1, 3, .05, " ×")
        self.side_falloff = self._control(layout, "Distance falloff (0 = none, 2 = inverse square)", 0, 2, .05, "")
        self.selector = QtWidgets.QComboBox()
        self.selector.currentIndexChanged.connect(self._select)
        layout.addWidget(self.selector)
        row = QtWidgets.QHBoxLayout()
        for title, callback in [("Add light", self._add), ("Remove", self._remove)]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.color_button = QtWidgets.QPushButton()
        self.color_button.clicked.connect(lambda: self._pick_color(False))
        layout.addWidget(self.color_button)
        layout.addWidget(QtWidgets.QLabel("Light side (as viewed in the sensor image)"))
        self.light_side = QtWidgets.QComboBox()
        for label, angle in [("Top", 270), ("Bottom", 90), ("Left", 180), ("Right", 0)]:
            self.light_side.addItem(label, angle)
        self.light_side.currentIndexChanged.connect(self._choose_side)
        layout.addWidget(self.light_side)
        self.azimuth = self._control(layout, "Around gel (azimuth)", -360, 360, 1, "°")
        # Keep the legacy numeric control for directional mode and old presets.
        self.azimuth.setToolTip("In array mode, angles select the nearest complete edge.")
        self.elevation = self._control(layout, "Above plane (elevation)", 0, 90, 1, "°")
        self.intensity = self._control(layout, "Light intensity (0 = off)", 0, 5, .01, "")
        self.reference = QtWidgets.QCheckBox("Blue-green reference tint")
        self.reference.setToolTip("Turn off for lighting without the reference color correction.")
        self.reference.toggled.connect(self._edit)
        layout.addWidget(self.reference)
        self.ambient_button = QtWidgets.QPushButton()
        self.ambient_button.clicked.connect(lambda: self._pick_color(True))
        layout.addWidget(self.ambient_button)
        self.exposure = self._control(layout, "Overall exposure", 0, 5, .01, "")
        self.diffuse = self._control(layout, "Diffuse reflection", 0, 3, .01, "")
        self.specular = self._control(layout, "Specular reflection", 0, 3, .01, "")
        self.shininess = self._control(layout, "Highlight sharpness", 1, 256, 1, "")
        self.noise = self._control(layout, "Image noise", 0, .1, .001, "")
        row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save preset…")
        save.setToolTip("Save a JSON preset and remember it for the next launch.")
        save.clicked.connect(self._save_dialog)
        load = QtWidgets.QPushButton("Load preset…")
        load.clicked.connect(self._load_dialog)
        row.addWidget(save)
        row.addWidget(load)
        if not show_presets:
            save.hide()
            load.hide()
        # Keep persistence accessible without scrolling through reflection controls.
        layout.insertLayout(1, row)
        self.reset_button = QtWidgets.QPushButton("Reset lighting to defaults")
        self.reset_button.clicked.connect(self.reset_defaults)
        layout.insertWidget(2, self.reset_button)
        self.message = QtWidgets.QLabel("Save remembers this lighting for the next launch.")
        self.message.setWordWrap(True)
        layout.insertWidget(2, self.message)
        self.set_optics(self.optics)
        self.presets = AppearancePresets(PRESET_DIR.parent / "directional_presets.json",
            lambda: asdict(self.optics), self.apply_named_preset,
            asdict(OpticsConfig()), self.decode_preset)
        layout.insertWidget(0, self.presets)
        reference_button = QtWidgets.QPushButton("Apply real GelSight reference lighting")
        reference_button.setToolTip("Applies the lighting fitted to the supplied real frames. For the background and effects too, select Real GelSight reference in Saved sensor configs.")
        reference_button.clicked.connect(self.apply_real_reference)
        layout.insertWidget(1, reference_button)
        from .preset_guard import PresetEditGuard, edit_controls
        self.edit_guard = PresetEditGuard(self, edit_controls(self, (self.presets, self.selector)),
            lambda: self.presets.selector.currentText() == 'Default',
            lambda: self.presets.create(protected=True))
        if not show_presets:
            self.message.hide()

    @staticmethod
    def decode_preset(raw):
        raw = deepcopy(raw)
        raw["leds"] = [LEDConfig(**led) for led in raw["leds"]]
        return validate_lighting(OpticsConfig(**raw))

    def apply_real_reference(self):
        import json
        path = PRESET_DIR.parent / "sensors" / "Real GelSight reference.json"
        try:
            self.apply_named_preset(json.loads(path.read_text())["sensor"]["optics"])
            self.presets.selector.setCurrentText("Custom")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Reference lighting", str(exc))

    def apply_named_preset(self, raw):
        self.set_optics(self.decode_preset(raw))
        self.changed.emit(deepcopy(self.optics))

    def reset_defaults(self):
        self.set_optics(OpticsConfig())
        self.presets.selector.setCurrentText('Default')
        self.message.setText("Default lighting restored. Save to keep it as your preset.")
        self.changed.emit(deepcopy(self.optics))

    def _control(self, layout, label, lo, hi, step, suffix):
        layout.addWidget(QtWidgets.QLabel(label))
        row = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(round(lo / step), round(hi / step))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(0 if step == 1 else (3 if step == .001 else 2))
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(True)
        spin.setMinimumWidth(85)
        slider.valueChanged.connect(lambda value: spin.setValue(value * step))
        def synchronize(value):
            slider.blockSignals(True)
            slider.setValue(round(value / step))
            slider.blockSignals(False)
            self._edit()
        spin.valueChanged.connect(synchronize)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def set_optics(self, optics):
        self._loading = True
        self.optics = deepcopy(optics)
        self.response_notice.setText('Calibrated spatial colour response is active. LED direction and colour controls do not affect this mode; exposure still applies.'
                                     if optics.spatial_response is not None else '')
        self.response_notice.setVisible(optics.spatial_response is not None)
        index = max(self.selector.currentIndex(), 0)
        self.selector.clear()
        self.selector.addItems([f"Light {i+1}" for i in range(len(optics.leds))])
        self.selector.setCurrentIndex(min(index, len(optics.leds)-1))
        self.reference.setChecked(optics.reference_background)
        self.side.setChecked(optics.side_lighting)
        self.side_distance.setValue(optics.side_distance)
        self.side_falloff.setValue(optics.side_falloff)
        for widget, value in [(self.exposure, optics.exposure), (self.diffuse, optics.diffuse_gain),
                              (self.specular, optics.specular_gain), (self.shininess, optics.shininess),
                              (self.noise, optics.noise_sigma)]:
            widget.setValue(value)
        self._loading = False
        self._select()

    def _select(self, *_):
        if self._loading or self.selector.currentIndex() < 0:
            return
        self._loading = True
        led = self.optics.leds[self.selector.currentIndex()]
        self.azimuth.setValue(led.azimuth)
        self._sync_side(led.azimuth)
        self.elevation.setValue(led.elevation)
        self.intensity.setValue(led.intensity)
        self._swatch(self.color_button, led.color, "Light color")
        self._swatch(self.ambient_button, self.optics.ambient, "Ambient color")
        self._loading = False

    def _choose_side(self, *_):
        if not self._loading and self.light_side.currentData() is not None:
            self.azimuth.setValue(self.light_side.currentData())

    def _sync_side(self, azimuth):
        angle = azimuth % 360
        snapped = (int((angle+45)//90)*90) % 360
        index = next(i for i in range(4) if self.light_side.itemData(i) == snapped)
        self.light_side.blockSignals(True)
        self.light_side.setCurrentIndex(index)
        self.light_side.blockSignals(False)

    @staticmethod
    def _swatch(button, color, label):
        qcolor = QtGui.QColor.fromRgbF(*color)
        button.setText(f"{label}: {qcolor.name()}")
        ink = "black" if qcolor.lightnessF() > .55 else "white"
        button.setStyleSheet(f"background-color: {qcolor.name()}; color: {ink};")

    def _pick_color(self, ambient):
        led = self.optics.leds[self.selector.currentIndex()]
        initial = self.optics.ambient if ambient else led.color
        dialog = QtWidgets.QColorDialog(QtGui.QColor.fromRgbF(*initial), self)
        dialog.setWindowTitle("Choose RGB color — live preview")
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)
        def preview(color):
            if not color.isValid():
                return
            rgb = (color.redF(), color.greenF(), color.blueF())
            if ambient:
                self.optics.ambient = rgb
            else:
                led.color = rgb
            self._select()
            self._edit()
        dialog.currentColorChanged.connect(preview)
        try:
            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                preview(QtGui.QColor.fromRgbF(*initial))
        finally:
            dialog.deleteLater()

    def _edit(self, *_):
        if self._loading or self.selector.currentIndex() < 0:
            return
        led = self.optics.leds[self.selector.currentIndex()]
        led.azimuth, led.elevation, led.intensity = self.azimuth.value(), self.elevation.value(), self.intensity.value()
        self._sync_side(led.azimuth)
        self.optics.reference_background = self.reference.isChecked()
        self.optics.side_lighting = self.side.isChecked()
        self.optics.side_distance = self.side_distance.value()
        self.optics.side_falloff = self.side_falloff.value()
        for name, widget in [("exposure", self.exposure), ("diffuse_gain", self.diffuse),
                             ("specular_gain", self.specular), ("shininess", self.shininess),
                             ("noise_sigma", self.noise)]:
            setattr(self.optics, name, widget.value())
        self.optics.calibration_path = None
        self.message.setText("Unsaved manual lighting. Save to reuse next time.")
        self.changed.emit(deepcopy(self.optics))

    def _add(self):
        if len(self.optics.leds) >= 16:
            return
        self.optics.leds.append(LEDConfig(intensity=.5))
        self.set_optics(self.optics)
        self.selector.setCurrentIndex(len(self.optics.leds)-1)
        self._edit()

    def _remove(self):
        if len(self.optics.leds) <= 1:
            return
        self.optics.leds.pop(self.selector.currentIndex())
        self.set_optics(self.optics)
        self._edit()

    def save_to(self, path):
        save_lighting(path, self.optics)
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != LAST_PRESET.resolve():
            save_lighting(LAST_PRESET, self.optics)
        self.message.setText(f"Saved {Path(path).name}. Restored on next launch.")

    def load_from(self, path):
        optics = load_lighting(path)  # validate fully before replacing live settings
        self.set_optics(optics)
        self.changed.emit(deepcopy(optics))
        self.message.setText(f"Loaded {Path(path).name}. Save to use at startup.")

    def _save_dialog(self):
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save lighting preset", str(PRESET_DIR / "my_lighting.json"), "Lighting preset (*.json)")
        if path:
            try:
                self.save_to(path if Path(path).suffix else path + ".json")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not save lighting", str(exc))

    def _load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load lighting preset", str(PRESET_DIR), "Lighting preset (*.json)")
        if path:
            try:
                self.load_from(path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not load lighting", str(exc))
