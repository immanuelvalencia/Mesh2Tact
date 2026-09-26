"""Collection controls and background worker for the geometric workbench."""
from threading import Event
import json
from pathlib import Path

from .qt import QtCore, QtWidgets
from ..gather import GatherSettings, gather
from ..outputs import SaveOptions, model_name, next_capture_index


OUTPUT_PREFERENCES_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'gather_output_preferences.json'


class SignedRange(QtWidgets.QWidget):
    """Two bounds, or one nonnegative magnitude centered on zero."""
    def __init__(self, limit, start, end, unit):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.bounds = []
        for label, value in (("Min ", start), ("Max ", end)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-limit, limit)
            spin.setPrefix(label)
            spin.setSuffix(f" {unit}")
            spin.setValue(value)
            layout.addWidget(spin)
            self.bounds.append(spin)
        self.magnitude = QtWidgets.QDoubleSpinBox()
        self.magnitude.setDecimals(3)
        self.magnitude.setRange(0, limit)
        self.magnitude.setPrefix("± ")
        self.magnitude.setSuffix(f" {unit}")
        self.magnitude.setToolTip("One magnitude around zero: 5 means −5 to +5. Zero fixes this axis at zero.")
        self.summary = QtWidgets.QLabel()
        layout.addWidget(self.magnitude)
        layout.addWidget(self.summary)
        self.unit = unit
        self.magnitude.valueChanged.connect(self.apply_magnitude)
        self.set_symmetric(False)

    def __getitem__(self, index):
        return self.bounds[index]

    def __iter__(self):
        return iter(self.bounds)

    def set_symmetric(self, enabled):
        if enabled:
            self.magnitude.blockSignals(True)
            self.magnitude.setValue(max(abs(spin.value()) for spin in self.bounds))
            self.magnitude.blockSignals(False)
            self.apply_magnitude(self.magnitude.value())
        for spin in self.bounds:
            spin.setVisible(not enabled)
        self.magnitude.setVisible(enabled)
        self.summary.setVisible(enabled)

    def apply_magnitude(self, value):
        self.bounds[0].setValue(-value)
        self.bounds[1].setValue(value)
        self.summary.setText(f"{ -value:g} to {value:g} {self.unit}")


class GatherWorker(QtCore.QThread):
    sample_ready = QtCore.Signal(object)
    result_ready = QtCore.Signal(object)

    def __init__(self, sim, directory, settings, parent=None):
        super().__init__(parent)
        self.sim, self.directory, self.settings = sim, directory, settings
        self.stop_event = Event()

    def run(self):
        try:
            result = gather(self.sim, self.directory, self.settings,
                            self.stop_event.is_set, self.sample_ready.emit)
        except Exception as exc:
            result = dict(status="failed", completed=0, error=str(exc), directory=self.directory)
        self.result_ready.emit(result)


class GatherPanel(QtWidgets.QWidget):
    def __init__(self, window, output_preferences_path=None):
        super().__init__()
        self.window = window
        self.output_preferences_path = Path(output_preferences_path or OUTPUT_PREFERENCES_PATH)
        layout = QtWidgets.QVBoxLayout(self)
        self.inputs = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.inputs)
        self.form = form
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.save_layout = QtWidgets.QComboBox()
        self.save_layout.addItem("Date and time folders", "date_time")
        self.save_layout.addItem("Object label with index", "object_label")
        self.save_layout.currentIndexChanged.connect(self.update_destination)
        form.addRow("Save layout", self.save_layout)
        self.object_label = QtWidgets.QLineEdit()
        self.object_label.setPlaceholderText("Example: sphere_red_20mm")
        self.object_label.textChanged.connect(self.update_destination)
        form.addRow("Object label", self.object_label)
        self.destination = QtWidgets.QLabel()
        self.destination.setWordWrap(True)
        form.addRow("Next destination", self.destination)
        self.output_checks = {}
        primary_outputs = {'tactile', 'default_tactile', 'clean', 'contact'}
        self.more_outputs = QtWidgets.QWidget()
        extra_form = QtWidgets.QFormLayout(self.more_outputs)
        extra_form.setContentsMargins(12, 0, 0, 0)
        for name, label in [("tactile", "Tactile RGB"), ("default_tactile", "Tactile RGB (Default profile)"),
                            ("clean", "Clean RGB"),
                            ("depth_image", "Depth color image"), ("depth_array", "Depth array (.npy)"),
                            ("raw_depth", "Raw depth array (.npy)"), ("contact", "Contact mask (mask folder)"),
                            ("settings", "Per-sample settings"), ("mesh", "Processed 3D mesh")]:
            target_form = form if name in primary_outputs else extra_form
            self.output_checks[name] = self.check(target_form, label, True)
        self.more_outputs_toggle = QtWidgets.QToolButton()
        self.more_outputs_toggle.setText('More settings')
        self.more_outputs_toggle.setCheckable(True)
        self.more_outputs_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.more_outputs_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.more_outputs_toggle.toggled.connect(self.toggle_more_outputs)
        form.addRow(self.more_outputs_toggle)
        form.addRow(self.more_outputs)
        self.more_outputs.hide()
        self.output_defaults_status = QtWidgets.QLabel()
        self.output_defaults_status.setWordWrap(True)
        form.addRow(self.output_defaults_status)
        self.save_output_defaults = QtWidgets.QPushButton("Save checked outputs as defaults")
        self.save_output_defaults.clicked.connect(self.save_output_preferences)
        form.addRow(self.save_output_defaults)
        self.load_output_preferences()
        hint = QtWidgets.QLabel("Only the checked output types are saved.")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.count = window.integer_control(form, "Number of samples", 1, 100000, 100, schedule=False)
        self.geometry_backend = QtWidgets.QComboBox()
        self.geometry_backend.addItem('CPU', 'cpu')
        self.geometry_backend.addItem('GPU (CUDA)', 'cuda')
        self.geometry_backend.addItem('Auto (CUDA when available)', 'auto')
        self.geometry_backend.setToolTip('Accelerates geometry rasterization only. Appearance effects and file saving use CPU. CUDA requires the torch_gpu environment.')
        form.addRow('Geometry processing', self.geometry_backend)
        self.fresh_seed = self.check(form, "Fresh sampling seed for every run", True)
        self.seed = window.integer_control(form, "Fixed sampling seed", 0, 2_147_483_647, 0, schedule=False)
        self.seed.setEnabled(False)
        self.fresh_seed.toggled.connect(lambda enabled: self.seed.setEnabled(not enabled))
        self.fresh_seed.setToolTip("Uses system randomness each run. Turn off to repeat a run with a fixed seed. The actual seed is reported after gathering and saved when per-sample settings are selected.")
        self.balanced = self.check(form, "Balanced coverage of each enabled range", True)
        balance_hint = QtWidgets.QLabel("Each range gets one saved sample per equal interval. Contact retries stay in the assigned intervals. Only completed runs guarantee coverage; combinations and object classes are not balanced automatically.")
        balance_hint.setWordWrap(True)
        form.addRow(balance_hint)
        self.scale = self.check(form, "Vary object scale", False)
        self.scale_range = self.range_row(form, "Scale min / max (1 = original size)", .001, 1000, .8, 1.2)
        self.scale_steps = QtWidgets.QSpinBox()
        self.scale_steps.setRange(1, 10)
        self.scale_steps.setValue(5)
        self.scale_steps.setToolTip('Number of evenly spaced scale values. 1 uses the midpoint; 2–10 include both endpoints. Each value receives equal saved samples; sample count must be divisible by the number of values.')
        form.addRow('Scale steps (1–10)', self.scale_steps)
        self.scale_distribution = QtWidgets.QLabel()
        self.scale_distribution.setWordWrap(True)
        form.addRow(self.scale_distribution)
        for spin in self.scale_range:
            spin.setDecimals(6)
            spin.setSingleStep(.001)
            spin.setEnabled(False)
            self.scale.toggled.connect(spin.setEnabled)
            spin.setToolTip("Absolute scale, like Import scale. Arrow steps are 0.001; type up to six decimal places for finer bounds, e.g. 0.08 to 0.2. Samples vary continuously within the range. Off keeps the current object size.")
        self.require_contact = self.check(form, "Contact checker: skip empty / insufficient masks", True)
        self.min_contact_pixels = window.integer_control(form, "Minimum contact pixels", 1, 100000000, 1, schedule=False)
        self.attempts_per_sample = window.integer_control(form, "Attempt budget per requested sample", 1, 1000, 20, schedule=False)
        contact_hint = QtWidgets.QLabel("Checks the binary contact mask before saving, even if mask output is off. Retries until the requested number is saved or the attempt budget is reached. Capture current checks once.")
        contact_hint.setWordWrap(True)
        form.addRow(contact_hint)
        self.range_mode = QtWidgets.QComboBox()
        self.range_mode.addItem("Exact min / max", "exact")
        self.range_mode.addItem("Symmetric ± (around zero)", "symmetric")
        self.range_mode.setToolTip("Applies to XYZ rotations and XY position. Switching to ± uses the larger absolute bound.")
        form.addRow("Position / rotation ranges", self.range_mode)
        self.rotation = self.check(form, "Random rotations", True)
        self.rotation_ranges = []
        self.rotation_axes = []
        for axis in "XYZ":
            self.rotation_axes.append(self.check(form, f"Random {axis} rotation (off = keep current angle)", True))
            row = SignedRange(180, -180, 180, "°")
            form.addRow(f"{axis} rotation", row)
            self.rotation_ranges.append(row)
        self.cut = self.check(form, "Random indentation depth (object Z)", True)
        self.cut_range = self.range_row(form, "Indentation min / max (mm)", 0, 1000, .1, 2.)
        self.xy = self.check(form, "Random X/Y position", False)
        self.xy_ranges = []
        for axis in "XY":
            row = SignedRange(1000, -2, 2, "mm")
            form.addRow(f"{axis} position", row)
            self.xy_ranges.append(row)
        self.range_mode.currentIndexChanged.connect(self.update_range_mode)
        self.position_overlay = self.check(form, "Show 3×3 XY range overlay", True)
        overlay_hint = QtWidgets.QLabel("In Data gathering, cyan objects mark X/Y min, midpoint and max on the 3D gel, even with random XY off. Uses current rotation and Z. Overlapping positions are shown once.")
        overlay_hint.setWordWrap(True)
        form.addRow(overlay_hint)
        self.view_xy_grid = QtWidgets.QPushButton("View XY grid from above")
        self.view_xy_grid.clicked.connect(self.focus_position_overlay)
        form.addRow(self.view_xy_grid)
        self.overlay_timer = QtCore.QTimer(self)
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.setInterval(60)
        self.overlay_timer.timeout.connect(self.refresh_position_overlay)
        self.rotation_overlay = self.check(form, "Show rotation range overlay", False)
        rotation_hint = QtWidgets.QLabel("Red X, green Y, blue Z: min / midpoint / max angles at the current position and object Z. Each axis is previewed separately; other angles stay at the current pose. Symmetric objects may look unchanged when rotated.")
        rotation_hint.setWordWrap(True)
        form.addRow(rotation_hint)
        self.rotation_overlay.toggled.connect(self.queue_position_overlay)
        self.rotation.toggled.connect(self.update_rotation_axes)
        for checkbox, row in zip(self.rotation_axes, self.rotation_ranges):
            checkbox.toggled.connect(self.update_rotation_axes)
            for spin in row:
                spin.valueChanged.connect(self.queue_position_overlay)
        self.update_rotation_axes()
        self.position_overlay.toggled.connect(self.queue_position_overlay)
        self.xy.toggled.connect(self.queue_position_overlay)
        for row in self.xy_ranges:
            for spin in row:
                spin.valueChanged.connect(self.queue_position_overlay)
        self.effects = self.check(form, "Vary noise / texture seed", False)
        self.effects.setToolTip("Uses a distinct pattern seed for every saved sample, excluding the starting seed. Does not change pose sampling or noise strength. Enable noise or texture in Image effects to see a difference. Fixed sampling seeds reproduce the sequence; fresh runs may share individual seeds.")
        inputs_scroll = QtWidgets.QScrollArea()
        inputs_scroll.setWidgetResizable(True)
        inputs_scroll.setWidget(self.inputs)
        layout.addWidget(inputs_scroll, 1)
        row = QtWidgets.QHBoxLayout()
        self.capture = QtWidgets.QPushButton("Capture current")
        self.capture.clicked.connect(window.capture_to_folder)
        self.start = QtWidgets.QPushButton("Start auto gather")
        self.start.clicked.connect(window.start_gather)
        self.stop = QtWidgets.QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(window.stop_gather)
        for button, color, hover in ((self.capture, '#1769aa', '#12558b'),
                                     (self.start, '#18764a', '#105e39'),
                                     (self.stop, '#b83232', '#962525')):
            button.setMinimumHeight(38)
            button.setStyleSheet(f'''
                QPushButton {{ background: {color}; color: white; border: 2px solid transparent;
                               border-radius: 6px; padding: 7px 12px; font-weight: 600; }}
                QPushButton:hover {{ background: {hover}; }}
                QPushButton:pressed {{ background: {hover}; border-color: #162331; }}
                QPushButton:focus {{ border-color: #efc94d; }}
                QPushButton:disabled {{ background: #d8dde2; color: #626b75; border-color: #c4cbd2; }}
            ''')
        layout.addWidget(self.capture)
        row.addWidget(self.start)
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready. Uses the current object and capture resolution.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        hint = QtWidgets.QLabel("Unchecked random options keep the current value. Rotations use independent uniform "
            "XYZ angle ranges (not uniform 3D orientations). The contact checker skips masks below its pixel threshold. "
            "Stop finishes the current sample and keeps completed captures. The current preview pose is preserved.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.update_destination()
        for checkbox in (self.scale, self.rotation, self.cut, self.xy,
                         self.require_contact, self.fresh_seed, *self.rotation_axes):
            checkbox.toggled.connect(self.update_option_visibility)
        self.update_option_visibility()
        for spin in (*self.scale_range, self.scale_steps, self.count):
            spin.valueChanged.connect(self.update_scale_distribution)
        self.scale.toggled.connect(self.update_scale_distribution)
        self.update_scale_distribution()

    def check(self, form, label, checked):
        box = QtWidgets.QCheckBox(label)
        box.setChecked(checked)
        form.addRow(box)
        return box

    def toggle_more_outputs(self, expanded):
        self.more_outputs.setVisible(expanded)
        self.more_outputs_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)

    def update_option_visibility(self, *_):
        """Keep option switches visible; reveal their fields only when enabled."""
        visibility = {
            self.scale_range[0]: self.scale.isChecked(),
            self.scale_steps: self.scale.isChecked(),
            self.scale_distribution: self.scale.isChecked(),
            self.cut_range[0]: self.cut.isChecked(),
            self.min_contact_pixels: self.require_contact.isChecked(),
            self.attempts_per_sample: self.require_contact.isChecked(),
            self.seed: not self.fresh_seed.isChecked(),
            self.range_mode: self.rotation.isChecked() or self.xy.isChecked(),
        }
        for checkbox, row in zip(self.rotation_axes, self.rotation_ranges):
            visibility[checkbox] = self.rotation.isChecked()
            visibility[row] = self.rotation.isChecked() and checkbox.isChecked()
        for row in self.xy_ranges:
            visibility[row] = self.xy.isChecked()
        for field, visible in visibility.items():
            for index in range(self.form.rowCount()):
                item = self.form.itemAt(index, QtWidgets.QFormLayout.FieldRole)
                if item is None:
                    continue
                layout = item.layout()
                matches = item.widget() is field or (layout is not None and
                    any(layout.itemAt(i).widget() is field for i in range(layout.count())))
                if not matches:
                    continue
                label = self.form.itemAt(index, QtWidgets.QFormLayout.LabelRole)
                if label and label.widget():
                    label.widget().setVisible(visible)
                if layout:
                    for i in range(layout.count()):
                        if layout.itemAt(i).widget():
                            layout.itemAt(i).widget().setVisible(visible)
                elif item.widget():
                    item.widget().setVisible(visible)
                break

    def update_scale_distribution(self, *_):
        try:
            settings = GatherSettings(count=self.count.value(), random_scale=True,
                scale_min=self.scale_range[0].value(), scale_max=self.scale_range[1].value(),
                scale_steps=self.scale_steps.value())
            settings.validate()
            if settings.scale_steps is not None:
                levels = settings.scale_levels()
                self.scale_distribution.setText(f'{len(levels)} scale values; {settings.count//len(levels)} saved samples per value, per model. Random order; rejected contacts retain their assigned scale.')
            else:
                self.scale_distribution.setText('Continuous scales. Enable Balanced coverage for equal-interval coverage.')
        except (ValueError, TypeError, ArithmeticError) as exc:
            self.scale_distribution.setText(f'Check settings: {exc}')

    def update_range_mode(self):
        symmetric = self.range_mode.currentData() == "symmetric"
        for row in self.rotation_ranges + self.xy_ranges:
            row.set_symmetric(symmetric)

    def queue_position_overlay(self, *_):
        self.overlay_timer.start()

    def update_rotation_axes(self, *_):
        for checkbox, row in zip(self.rotation_axes, self.rotation_ranges):
            checkbox.setEnabled(self.rotation.isChecked())
            row.setEnabled(self.rotation.isChecked() and checkbox.isChecked())
        self.queue_position_overlay()

    def rotation_preview_poses(self, current):
        poses = []
        if self.rotation.isChecked():
            for axis, (checkbox, row) in enumerate(zip(self.rotation_axes, self.rotation_ranges)):
                low, high = row[0].value(), row[1].value()
                if checkbox.isChecked() and low <= high:
                    for angle in sorted(set((low, (low+high)/2, high))):
                        angles = list(current)
                        angles[axis] = angle
                        poses.append((axis, tuple(angles)))
        return poses

    def refresh_position_overlay(self):
        if hasattr(self.window, 'update_gather_overlay'):
            self.window.update_gather_overlay()

    def focus_position_overlay(self):
        self.position_overlay.setChecked(True)
        if hasattr(self.window, 'focus_gather_overlay'):
            self.window.focus_gather_overlay()

    def xy_preview_positions(self):
        """Return unique positions in top-to-bottom, left-to-right order (mm)."""
        xs, ys = [(row[0].value(), row[1].value()) for row in self.xy_ranges]
        if xs[0] > xs[1] or ys[0] > ys[1]:
            return []
        xs = sorted(set((xs[0], sum(xs)/2, xs[1])))
        ys = sorted(set((ys[0], sum(ys)/2, ys[1])), reverse=True)
        return [(x, y) for y in ys for x in xs]

    def restore_range_mode(self, mode):
        self.range_mode.blockSignals(True)
        self.range_mode.setCurrentIndex(self.range_mode.findData(mode))
        self.range_mode.blockSignals(False)
        self.update_range_mode()

    def range_row(self, form, label, low, high, start, end):
        row = QtWidgets.QHBoxLayout()
        values = []
        for value in (start, end):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(3)
            spin.setValue(value)
            row.addWidget(spin)
            values.append(spin)
        form.addRow(label, row)
        return values

    def update_destination(self):
        indexed = self.save_layout.currentData() == "object_label"
        self.object_label.setEnabled(indexed)
        if not indexed:
            self.destination.setText(str(self.window.data_root / "tactile" / model_name(self.window.sim.source) / "<date-time>"))
            return
        label = self.object_label.text().strip()
        if not label:
            self.destination.setText("Enter an object label. Existing data will not be replaced.")
            return
        parent = self.window.data_root / "tactile" / model_name(label)
        try:
            next_index = next_capture_index([
                self.window.data_root / branch / model_name(label)
                for branch in ('tactile', 'clean', 'default', 'mask')])
            self.destination.setText(str(parent / f"sample_{next_index:06d}_tactile.png") +
                                     '\nCapture current: ' + str(parent / 'stills' / '<date-time>_tactile.png'))
        except OSError as exc:
            self.destination.setText(f"Cannot inspect destination: {exc}")

    def outputs(self):
        return SaveOptions(**{name: check.isChecked() for name, check in self.output_checks.items()})

    def load_output_preferences(self):
        if not self.output_preferences_path.exists():
            return
        try:
            data = json.loads(self.output_preferences_path.read_text(encoding='utf-8'))
            values = data['outputs']
            expected = set(self.output_checks)
            if data.get('version') != 1 or not set(values).issubset(expected) or not all(isinstance(value, bool) for value in values.values()):
                raise ValueError('Unsupported output preferences')
            for name, checkbox in self.output_checks.items():
                checkbox.setChecked(values.get(name, False if name == 'default_tactile' else checkbox.isChecked()))
            self.output_defaults_status.setText('Saved output defaults loaded.')
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.output_defaults_status.setText('Could not load saved output defaults; using the built-in selection.')

    def save_output_preferences(self):
        try:
            self.output_preferences_path.parent.mkdir(parents=True, exist_ok=True)
            output = QtCore.QSaveFile(str(self.output_preferences_path))
            if not output.open(QtCore.QIODevice.WriteOnly):
                raise OSError(output.errorString())
            data = {'version': 1, 'outputs': {name: checkbox.isChecked() for name, checkbox in self.output_checks.items()}}
            encoded = (json.dumps(data, indent=2) + '\n').encode('utf-8')
            if output.write(encoded) != len(encoded):
                output.cancelWriting()
                raise OSError('Incomplete preferences write')
            if not output.commit():
                raise OSError(output.errorString())
            self.output_defaults_status.setText('Output defaults saved.')
            return True
        except OSError as exc:
            self.output_defaults_status.setText(f'Output defaults could not be saved: {exc}')
            return False

    def settings(self):
        return GatherSettings(count=self.count.value(), seed=None if self.fresh_seed.isChecked() else self.seed.value(),
            geometry_backend=self.geometry_backend.currentData(),
            balanced=self.balanced.isChecked(), random_scale=self.scale.isChecked(),
            scale_min=self.scale_range[0].value(), scale_max=self.scale_range[1].value(),
            scale_steps=self.scale_steps.value(),
            random_rotation=self.rotation.isChecked(),
            rotation_axes=tuple(box.isChecked() for box in self.rotation_axes),
            require_contact=self.require_contact.isChecked(), min_contact_pixels=self.min_contact_pixels.value(),
            attempts_per_sample=self.attempts_per_sample.value(),
            rotation_min=tuple(r[0].value() for r in self.rotation_ranges),
            rotation_max=tuple(r[1].value() for r in self.rotation_ranges),
            random_cut=self.cut.isChecked(), cut_min_mm=self.cut_range[0].value(), cut_max_mm=self.cut_range[1].value(),
            random_xy=self.xy.isChecked(), xy_min_mm=tuple(r[0].value() for r in self.xy_ranges),
            xy_max_mm=tuple(r[1].value() for r in self.xy_ranges), random_effect_seed=self.effects.isChecked(),
            save_layout=self.save_layout.currentData(), object_label=self.object_label.text().strip(), outputs=self.outputs())

    def set_running(self, running):
        self.inputs.setEnabled(not running)
        self.capture.setEnabled(not running)
        self.start.setEnabled(not running)
        self.stop.setEnabled(running)

    def restore_settings(self, values, range_mode='exact'):
        """Restore shared gathering controls from a validated settings mapping."""
        from dataclasses import asdict
        settings = GatherSettings(**values)
        settings.validate()
        if range_mode not in ('exact', 'symmetric'):
            raise ValueError('Unsupported gathering range mode')
        self.restore_range_mode('exact')
        self.scale_steps.setValue(settings.scale_steps or 5)
        self.geometry_backend.setCurrentIndex(self.geometry_backend.findData(settings.geometry_backend))
        for name, value in [('count', settings.count), ('seed', settings.seed or 0),
                            ('min_contact_pixels', settings.min_contact_pixels),
                            ('attempts_per_sample', settings.attempts_per_sample)]:
            getattr(self, name).setValue(value)
        for name, value in [('fresh_seed', settings.seed is None), ('balanced', settings.balanced),
                            ('scale', settings.random_scale), ('rotation', settings.random_rotation),
                            ('cut', settings.random_cut), ('xy', settings.random_xy),
                            ('effects', settings.random_effect_seed), ('require_contact', settings.require_contact)]:
            getattr(self, name).setChecked(value)
        for checkbox, value in zip(self.rotation_axes, settings.rotation_axes):
            checkbox.setChecked(value)
        for rows, lows, highs in [(self.rotation_ranges, settings.rotation_min, settings.rotation_max),
                                  (self.xy_ranges, settings.xy_min_mm, settings.xy_max_mm),
                                  ([self.scale_range], [settings.scale_min], [settings.scale_max]),
                                  ([self.cut_range], [settings.cut_min_mm], [settings.cut_max_mm])]:
            for row, low, high in zip(rows, lows, highs):
                row[0].setValue(low)
                row[1].setValue(high)
        for name, value in asdict(settings.outputs).items():
            self.output_checks[name].setChecked(value)
        self.save_layout.setCurrentIndex(self.save_layout.findData(settings.save_layout))
        self.object_label.setText(settings.object_label)
        self.restore_range_mode(range_mode)
        self.update_option_visibility()
        self.update_scale_distribution()
