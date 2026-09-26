"""Complete, versioned workbench presets with validation before live changes."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import os
import tempfile

import numpy as np

from .qt import QtWidgets
from ..config import SensorConfig
from ..geometric import GeometricSim
from ..geometry.mesh import primitive
from ..lighting import validate_lighting
from ..render.effects import ImageEffects
from ..render.gel import GelLighting
from ..gather import GatherSettings
from ..preview import migrate_preview, preview_size, link_capture_controls


class PresetMixin:
    def preset_controls(self):
        names = ("rx", "ry", "rz", "x", "y", "cut", "softness", "width", "height", "depth_range",
                 "output_w", "output_h", "preview_scale", "move_step", "scale")
        return {name: getattr(self, name) for name in names}

    def preset_data(self):
        self.apply_controls()
        sensor = asdict(self.sim.cfg)
        sensor["camera"]["width"], sensor["camera"]["height"] = self.capture_resolution
        return dict(format="mesh2tact-geometric-preset", version=1,
                    object=dict(source=self.sim.source, units=self.sim.units, scale=self.sim.scale,
                                base_rotation=list(self.sim.base_rotation),
                                quality=self.sim.quality_level, smoothing=self.sim.smoothing_iterations),
                    controls={name: widget.value() for name, widget in self.preset_controls().items()},
                    sensor=sensor, gel_lighting=asdict(self.sim.gel_lighting), effects=asdict(self.sim.effects),
                    full_preview=self.preview_scale.value() == 100, gizmo=True,
                    import_units=self.units.currentText(), gather=asdict(self.gather_panel.settings()),
                    gather_folder=str(self.data_root),
                    batch_gather=self.batch_gather_panel.data(),
                    gather_range_mode=self.gather_panel.range_mode.currentData(),
                    max_penetration_mm=self.max_penetration.value(),
                    preview_panels=self.preview_grid.selected())

    def save_preset(self, path):
        path = Path(path)
        if (path.resolve().parent == self.sensor_configs.directory.resolve()
                and self.sensor_configs.is_protected(path)):
            raise ValueError('Built-in sensor configs are read-only. Save under a new name.')
        data = self.preset_data()
        encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
        # A complete JSON replaces the target only after a successful write.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.statusBar().showMessage(f"Saved settings: {path}")

    @staticmethod
    def validate_widget(widget, value, name):
        if isinstance(widget, QtWidgets.QCheckBox):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not widget.minimum() <= value <= widget.maximum():
                raise ValueError(f"{name} is outside the supported range")
            if isinstance(widget, QtWidgets.QSpinBox) and int(value) != value:
                raise ValueError(f"{name} must be an integer")

    def load_preset(self, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") not in ("mesh2tact-geometric-preset", "vtsim-geometric-preset") or data.get("version") != 1:
            raise ValueError("Choose a complete Mesh2Tact settings preset (version 1)")
        controls = self.preset_controls()
        maximum = data.get('max_penetration_mm', data['sensor']['camera']['max_depth']*1000)
        self.validate_widget(self.max_penetration, maximum, 'maximum indentation depth')
        previews = data.get('preview_panels', self.preview_grid.selected())
        if not isinstance(previews, list) or any(key not in self.preview_grid.labels for key in previews):
            raise ValueError('Unknown preview panel')
        data["controls"] = link_capture_controls(migrate_preview(data["controls"], data.get("full_preview", False)))
        if set(data["controls"]) != set(controls):
            raise ValueError("Preset has missing or unsupported controls")
        assignments = [(widget, data["controls"][name], name) for name, widget in controls.items()]
        effects = ImageEffects(**data["effects"])
        assignments += [(widget, getattr(effects, name), name) for name, widget in self.effect_widgets.items()]
        assignments += [(self.effect_seed, effects.seed, "effect seed"),
                        (self.effects_enabled, effects.enabled, "effects enabled")]
        obj = data["object"]
        assignments += [(self.mesh_smoothing, obj["smoothing"], "shape smoothing")]
        if obj["quality"] not in (-3, -2, -1, 0, 1, 2, 3) or isinstance(obj["quality"], bool):
            raise ValueError("Unsupported object quality")
        if data["import_units"] not in ("mm", "cm", "m", "in"):
            raise ValueError("Unsupported import units")
        if not isinstance(data["gather_folder"], str):
            raise ValueError("Gather folder must be a path string")
        gathering = GatherSettings(**data["gather"])
        gathering.validate()
        range_mode = data.get('gather_range_mode', 'exact')
        if range_mode not in ('exact', 'symmetric'):
            raise ValueError('Unsupported gathering range mode')
        if range_mode == 'symmetric':
            for lows, highs in ((gathering.rotation_min, gathering.rotation_max),
                               (gathering.xy_min_mm, gathering.xy_max_mm)):
                if any(low != -high for low, high in zip(lows, highs)):
                    raise ValueError('Symmetric gathering ranges must have matching negative and positive bounds')
        gathering.cut_min_mm = min(maximum, max(0, gathering.cut_min_mm))
        gathering.cut_max_mm = min(maximum, max(0, gathering.cut_max_mm))
        panel = self.gather_panel
        assignments += [(panel.output_checks[name], value, f'{name} output')
                        for name, value in asdict(gathering.outputs).items()]
        assignments += [(panel.require_contact, gathering.require_contact, 'require contact'),
                        (panel.min_contact_pixels, gathering.min_contact_pixels, 'minimum contact pixels'),
                        (panel.attempts_per_sample, gathering.attempts_per_sample, 'attempts per sample')]
        assignments += [(widget, value, 'rotation axis') for widget, value in zip(panel.rotation_axes, gathering.rotation_axes)]
        assignments += [(panel.count, gathering.count, "sample count"), (panel.seed, gathering.seed if gathering.seed is not None else 0, "sampling seed"),
                        (panel.scale_steps, gathering.scale_steps or 5, 'scale steps'),
                        (panel.fresh_seed, gathering.seed is None, "fresh sampling seed"),
                        (panel.balanced, gathering.balanced, "balanced sampling"),
                        (panel.scale, gathering.random_scale, "vary scale"),
                        (panel.rotation, gathering.random_rotation, "random rotation"),
                        (panel.cut, gathering.random_cut, "random cut"), (panel.xy, gathering.random_xy, "random XY"),
                        (panel.effects, gathering.random_effect_seed, "random effects")]
        for rows, lows, highs in [(panel.rotation_ranges, gathering.rotation_min, gathering.rotation_max),
                                  ([panel.scale_range], [gathering.scale_min], [gathering.scale_max]),
                                  (panel.xy_ranges, gathering.xy_min_mm, gathering.xy_max_mm),
                                  ([panel.cut_range], [gathering.cut_min_mm], [gathering.cut_max_mm])]:
            for row, low, high in zip(rows, lows, highs):
                assignments.extend([(row[0], low, "range minimum"), (row[1], high, "range maximum")])
        for widget, value, name in assignments:
            if widget in panel.cut_range:
                if not 0 <= value <= maximum:
                    raise ValueError('Indentation range exceeds the configured limit')
                continue
            self.validate_widget(widget, value, name)
        cfg = SensorConfig.from_dict(data["sensor"])
        validate_lighting(cfg.optics)
        if cfg.optics.calibration_path:
            calibration = Path(cfg.optics.calibration_path)
            if not calibration.is_absolute():
                calibration = path.parent / calibration
            cfg.optics.calibration_path = str(calibration)
        staged = GeometricSim(cfg)
        staged.max_penetration = maximum/1000
        staged.effects = effects
        staged.gel_lighting = GelLighting.from_dict(data["gel_lighting"])
        source = obj["source"]
        if not isinstance(source, str):
            raise ValueError("Preset object source must be a string")
        if source.startswith("primitive:"):
            kind = source.split(":", 1)[1]
            staged.set_mesh(primitive(kind, .004), source)
            staged.set_scale(obj['scale'])
        else:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            if not source_path.is_file():
                raise ValueError(f"Preset mesh is missing: {source_path}")
            staged.load(source_path, units=obj["units"], scale=obj["scale"])
        staged.set_quality(obj["quality"], obj["smoothing"])
        staged.set_base_rotation(obj.get('base_rotation', (0., 0., 0.)))
        numbers = data["controls"]
        staged.rotation = tuple(numbers[name] for name in ("rx", "ry", "rz"))
        staged.offset = tuple(numbers[name]/1000 for name in ("x", "y"))
        staged.cut_depth, staged.softness = numbers["cut"]/1000, numbers["softness"]/1000
        staged.cfg.gel.size_x, staged.cfg.gel.size_y = numbers["width"]/1000, numbers["height"]/1000
        staged.cfg.camera.max_depth = numbers["depth_range"]/1000
        staged.cfg.camera.width, staged.cfg.camera.height = preview_size(numbers)
        staged.render()  # Resolve geometry and calibration before changing any live state.
        self.timer.stop()
        for widget in panel.cut_range:
            widget.setRange(0, maximum)
        for widget, value, _ in assignments:
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        panel.restore_range_mode(range_mode)
        panel.geometry_backend.setCurrentIndex(panel.geometry_backend.findData(gathering.geometry_backend))
        panel.save_layout.setCurrentIndex(panel.save_layout.findData(gathering.save_layout))
        panel.object_label.setText(gathering.object_label)
        panel.update_destination()
        panel.update_rotation_axes()
        panel.update_option_visibility()
        panel.update_scale_distribution()
        panel.seed.setEnabled(not panel.fresh_seed.isChecked())
        for spin in panel.scale_range:
            spin.setEnabled(panel.scale.isChecked())
        self.sim = staged
        self.base_flip.blockSignals(True)
        angles = tuple(staged.base_rotation)
        index = next((i for i in range(self.base_flip.count())
                      if tuple(self.base_flip.itemData(i)) == angles), 0)
        self.base_flip.setCurrentIndex(index)
        self.base_flip.blockSignals(False)
        self.max_penetration.blockSignals(True)
        self.max_penetration.setValue(maximum)
        self.max_penetration.blockSignals(False)
        self.preview_grid.set_selected(previews, save=False)
        self._has_object = True
        self.units.setCurrentText(data["import_units"])
        self.quality.blockSignals(True)
        self.quality.setCurrentIndex(self.quality.findData(obj["quality"]))
        self.quality.blockSignals(False)
        self.lighting.set_optics(staged.cfg.optics)
        self.gel_panel.set_settings(staged.gel_lighting)
        self.sync_appearance_panels()
        # Legacy presets cannot redirect the automatic data directory.
        self.model_label.setText(staged.source)
        self.resolution_preset.setCurrentIndex(0)
        self.sync_slider(self.cut.value())
        self._scene_key = None
        self._reset_camera = True
        self.refresh()
        if 'batch_gather' in data:
            self.batch_gather_panel.restore(data['batch_gather'])
        self.statusBar().showMessage(f"Loaded settings: {path}")

    def save_preset_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save all settings", "geometric_settings.json", "Settings (*.json)")
        if path:
            try:
                self.save_preset(path if Path(path).suffix else path + ".json")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not save settings", str(exc))

    def load_preset_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load all settings", "", "Settings (*.json)")
        if path:
            try:
                self.load_preset(path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not load settings", str(exc))
