from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
from .qt import QtCore, QtGui, QtWidgets
from .lighting import LightingPanel
from ..render.gel import GelLighting, GelGlow

BACKGROUND_PRESETS = {"GelSight": (.19, .28, .29)}


class GelLightingPanel(QtWidgets.QGroupBox):
    changed = QtCore.Signal(object)

    def __init__(self, preset_path=None):
        super().__init__("Gel background and color gradients")
        self.settings = GelLighting()
        self.loading = False
        self.measured_background_active = False
        self.preset_path = Path(preset_path) if preset_path else Path(__file__).resolve().parents[2] / "configs" / "gel_backgrounds.json"
        self.presets = {}
        if self.preset_path.exists():
            try:
                self.presets = {name: GelLighting.from_dict(raw) for name, raw in
                                json.loads(self.preset_path.read_text()).items() if name.casefold() not in ("gelsight", "custom")}
            except (ValueError, TypeError, KeyError, OSError):
                QtWidgets.QMessageBox.warning(self, "Background presets", "Could not read background presets. The existing file will not be changed until you save a preset.")
        form = QtWidgets.QFormLayout(self)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.source_notice = QtWidgets.QLabel()
        self.source_notice.setWordWrap(True)
        form.addRow(self.source_notice)
        self.enabled = QtWidgets.QCheckBox("Use adjustable gel lighting")
        self.enabled.toggled.connect(self.edit)
        form.addRow(self.enabled)
        self.preserve_contact = QtWidgets.QCheckBox('Preserve contact colours under gradients')
        self.preserve_contact.setToolTip('Fit the empty-pad gradient without attenuating the contact shading.')
        self.preserve_contact.toggled.connect(self.edit)
        form.addRow(self.preserve_contact)
        self.background_preset = QtWidgets.QComboBox()
        self.background_preset.addItems(["Custom", "GelSight", *self.presets])
        self.background_preset.currentTextChanged.connect(self.apply_background_preset)
        form.addRow("Background preset", self.background_preset)
        preset_actions = QtWidgets.QHBoxLayout()
        for label, callback in [("Create preset…", self.create_preset), ("Update preset", self.update_preset), ("Delete preset", self.delete_preset)]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            preset_actions.addWidget(button)
        self.update_button = preset_actions.itemAt(1).widget()
        self.delete_button = preset_actions.itemAt(2).widget()
        form.addRow(preset_actions)
        self.background = QtWidgets.QPushButton()
        self.background.clicked.connect(lambda: self.pick(True))
        form.addRow(self.background)
        brightness_row = QtWidgets.QHBoxLayout()
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(0, 300)
        self.brightness = QtWidgets.QSpinBox()
        self.brightness.setRange(0, 300)
        self.brightness.setSuffix(" %")
        self.brightness_slider.valueChanged.connect(self.brightness.setValue)
        self.brightness.valueChanged.connect(self.edit_brightness)
        brightness_row.addWidget(self.brightness_slider, 1)
        brightness_row.addWidget(self.brightness)
        form.addRow("Master brightness", brightness_row)
        self.color_controls = {}
        for key, title, lo, hi, step in [("contrast", "Contrast", 0, 3, .05),
                ("hue", "Hue shift (°)", -180, 180, 1),
                ("saturation", "Saturation", 0, 3, .05), ("gamma", "Gamma", .1, 3, .05)]:
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(lo, hi)
            control.setSingleStep(step)
            control.valueChanged.connect(self.edit)
            self.color_controls[key] = control
            form.addRow(title, control)
        self.selector = QtWidgets.QComboBox()
        self.selector.currentIndexChanged.connect(self.select)
        form.addRow("Gradient layer", self.selector)
        self.color = QtWidgets.QPushButton()
        self.color.clicked.connect(lambda: self.pick(False))
        form.addRow(self.color)
        self.controls = {}
        self.gradient_sliders = {}
        for key, title, lo, hi, step in [
            ("strength", "Color strength", 0, 1, .01),
            ("x", "Center X (0 = left)", 0, 1, .01),
            ("y", "Center Y (0 = top)", 0, 1, .01),
            ("width", "Horizontal spread", .01, 1, .01),
            ("height", "Vertical spread", .01, 1, .01),
            ("angle", "Ellipse angle (°)", -180, 180, 1),
            ("vignette", "Gel corner darkening", 0, 1, .01)]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(3)
            spin.setSingleStep(step)
            spin.valueChanged.connect(self.edit)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(round(lo*1000), round(hi*1000))
            slider.setSingleStep(max(1, round(step*1000)))
            slider.valueChanged.connect(lambda value, target=spin: target.setValue(value/1000))
            def sync(value, target=slider):
                target.blockSignals(True)
                target.setValue(round(value*1000))
                target.blockSignals(False)
            spin.valueChanged.connect(sync)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(slider, 1)
            row.addWidget(spin)
            form.addRow(title, row)
            self.gradient_sliders[key] = slider
            self.controls[key] = spin
        row = QtWidgets.QHBoxLayout()
        self.gradient_actions = []
        for label, callback in [("Add gradient", self.add), ("Remove gradient", self.remove)]:
            button = QtWidgets.QPushButton(label)
            self.gradient_actions.append(button)
            button.clicked.connect(callback)
            row.addWidget(button)
        form.addRow(row)
        reset = QtWidgets.QPushButton("Restore to default")
        reset.clicked.connect(self.reset)
        self.reset_button = reset
        form.addRow(reset)
        hint = QtWidgets.QLabel("Soft blue and pink center glows inspired by your reference. "
            "Positions and spreads are fractions of the image. Gradient layers blend in list order. "
            "Master brightness scales the background, every gradient and contact lighting together.")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.set_settings(self.settings)
        from .preset_guard import PresetEditGuard, edit_controls
        self.edit_guard = PresetEditGuard(self, edit_controls(self,
            (self.background_preset, self.selector, *[preset_actions.itemAt(i).widget() for i in range(3)])),
            lambda: self.background_preset.currentText() == 'GelSight',
            lambda: self.create_preset(protected=True))

    def set_settings(self, settings):
        self.settings = deepcopy(settings)
        self.loading = True
        index = max(0, self.selector.currentIndex())
        self.selector.clear()
        self.selector.addItems([f"Gradient {i+1}" for i in range(len(settings.glows))])
        self.selector.setCurrentIndex(min(index, len(settings.glows)-1))
        self.enabled.setChecked(settings.enabled)
        self.preserve_contact.setChecked(settings.preserve_contact)
        self.brightness.setValue(round(settings.brightness*100))
        for key, control in self.color_controls.items():
            control.setValue(getattr(settings, key))
        self.loading = False
        self.select()

    def set_measured_background_active(self, active):
        self.measured_background_active = bool(active)
        self.sync_source_controls()

    def sync_source_controls(self):
        active = not self.measured_background_active
        self.source_notice.setText('Measured image background is active. Its preview and image controls are in Background and measured lighting. Manual colour and gradient values below are retained but inactive. Master colour adjustments still apply.' if not active else 'Manual background controls are active. Measured image controls are in Background and measured lighting.')
        for widget in (self.enabled, self.preserve_contact, self.background, self.background_preset,
                       self.selector, *self.gradient_actions):
            widget.setEnabled(active)
        has_glow = bool(self.settings.glows)
        self.color.setEnabled(active and has_glow)
        for key, control in self.controls.items():
            enabled = active and (key == 'vignette' or has_glow)
            control.setEnabled(enabled)
            self.gradient_sliders[key].setEnabled(enabled)

    def select(self, *_):
        if self.loading:
            return
        self.loading = True
        glow = self.settings.glows[self.selector.currentIndex()] if self.settings.glows else None
        for name, control in self.controls.items():
            control.setEnabled(name == "vignette" or glow is not None)
            self.gradient_sliders[name].setEnabled(name == "vignette" or glow is not None)
            if name == "vignette" or glow is not None:
                control.setValue(self.settings.vignette if name == "vignette" else getattr(glow, name))
        LightingPanel._swatch(self.background, self.settings.background, "Background color")
        self.background_preset.blockSignals(True)
        if self.background_preset.currentText() == "GelSight" and self.settings != GelLighting():
            self.background_preset.setCurrentText("Custom")
        self.background_preset.blockSignals(False)
        self.color.setEnabled(glow is not None)
        if glow is not None:
            LightingPanel._swatch(self.color, glow.color, "Gradient color")
        else:
            self.color.setText("No gradients")
        editable = self.background_preset.currentText() in ('GelSight', *self.presets)
        self.update_button.setEnabled(editable)
        self.delete_button.setEnabled(editable)
        self.loading = False
        self.sync_source_controls()

    def apply_background_preset(self, name):
        if not self.loading:
            if name == "GelSight" or name in self.presets:
                self.set_settings(GelLighting() if name == "GelSight" else self.presets[name])
            self.select()
            self.changed.emit(deepcopy(self.settings))

    def save_library(self):
        if any(name.casefold() in ('gelsight', 'custom') for name in self.presets):
            raise ValueError('Built-in backgrounds are read-only. Create a new named preset.')
        self.preset_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.preset_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({name: asdict(value) for name, value in self.presets.items()}, indent=2))
        temporary.replace(self.preset_path)

    def create_preset(self, _checked=False, protected=False):
        name, ok = QtWidgets.QInputDialog.getText(self, "Protected background preset" if protected else "Create background preset",
            "GelSight cannot be edited, overwritten, or deleted. Create a new preset. Name:" if protected else "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        if name.casefold() in {n.casefold() for n in ["GelSight", "Custom", *self.presets]}:
            QtWidgets.QMessageBox.warning(self, "Preset name", "Choose a unique name. GelSight is protected.")
            return
        self.presets[name] = deepcopy(self.settings)
        self.save_library()
        self.background_preset.addItem(name)
        self.background_preset.setCurrentText(name)

    def update_preset(self):
        name = self.background_preset.currentText()
        if name == 'GelSight':
            self.create_preset(protected=True)
            return
        if name in self.presets:
            self.presets[name] = deepcopy(self.settings)
            self.save_library()

    def delete_preset(self):
        name = self.background_preset.currentText()
        if name == 'GelSight':
            self.create_preset(protected=True)
            return
        if name in self.presets:
            del self.presets[name]
            self.save_library()
            self.background_preset.blockSignals(True)
            self.background_preset.removeItem(self.background_preset.currentIndex())
            self.background_preset.setCurrentText("Custom")
            self.background_preset.blockSignals(False)
            self.select()

    def edit_brightness(self, value):
        self.brightness_slider.blockSignals(True)
        self.brightness_slider.setValue(value)
        self.brightness_slider.blockSignals(False)
        if not self.loading:
            self.settings.brightness = value/100
            self.select()
            self.changed.emit(deepcopy(self.settings))

    def edit(self, *_):
        if self.loading:
            return
        self.settings.enabled = self.enabled.isChecked()
        self.settings.preserve_contact = self.preserve_contact.isChecked()
        for key, control in self.color_controls.items():
            setattr(self.settings, key, control.value())
        glow = self.settings.glows[self.selector.currentIndex()] if self.settings.glows else None
        for name, control in self.controls.items():
            if name == "vignette" or glow is not None:
                setattr(self.settings if name == "vignette" else glow, name, control.value())
        self.select()
        self.changed.emit(deepcopy(self.settings))

    def pick(self, background):
        glow = self.settings.glows[self.selector.currentIndex()] if self.settings.glows else None
        if not background and glow is None:
            return
        original = self.settings.background if background else glow.color
        dialog = QtWidgets.QColorDialog(QtGui.QColor.fromRgbF(*original), self)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)
        def preview(color):
            rgb = (color.redF(), color.greenF(), color.blueF())
            if background:
                self.settings.background = rgb
            else:
                glow.color = rgb
            self.select()
            self.changed.emit(deepcopy(self.settings))
        dialog.currentColorChanged.connect(preview)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            preview(QtGui.QColor.fromRgbF(*original))
        dialog.deleteLater()

    def add(self):
        if len(self.settings.glows) < 8:
            self.settings.glows.append(GelGlow())
            self.set_settings(self.settings)
            self.selector.setCurrentIndex(len(self.settings.glows)-1)
            self.changed.emit(deepcopy(self.settings))

    def remove(self):
        if self.settings.glows:
            self.settings.glows.pop(self.selector.currentIndex())
            self.set_settings(self.settings)
            self.changed.emit(deepcopy(self.settings))

    def reset(self):
        self.set_settings(GelLighting())
        self.background_preset.setCurrentText("GelSight")
        self.changed.emit(deepcopy(self.settings))
