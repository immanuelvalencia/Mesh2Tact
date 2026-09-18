"""Collection controls and background worker for the geometric workbench."""
from threading import Event
import json
from pathlib import Path

from .qt import QtCore, QtWidgets
from ..gather import GatherSettings, gather
from ..outputs import SaveOptions, model_name


OUTPUT_PREFERENCES_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'gather_output_preferences.json'


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
        for name, label in [("tactile", "Tactile RGB"), ("default_tactile", "Tactile RGB (Default profile)"),
                            ("clean", "Clean RGB"),
                            ("depth_image", "Depth color image"), ("depth_array", "Depth array (.npy)"),
                            ("raw_depth", "Raw depth array (.npy)"), ("contact", "Contact mask"),
                            ("settings", "Per-sample settings"), ("mesh", "Processed 3D mesh")]:
            self.output_checks[name] = self.check(form, label, True)
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
        self.seed = window.integer_control(form, "Sampling seed", 0, 1_000_000, 0, schedule=False)
        self.rotation = self.check(form, "Random rotations", True)
        self.rotation_ranges = [self.range_row(form, f"{axis} rotation min / max (°)", -180, 180, -180, 180) for axis in "XYZ"]
        self.cut = self.check(form, "Random indentation depth (object Z)", True)
        self.cut_range = self.range_row(form, "Indentation min / max (mm)", 0, 1000, .1, 2.)
        self.xy = self.check(form, "Random X/Y position", False)
        self.xy_ranges = [self.range_row(form, f"{axis} min / max (mm)", -1000, 1000, -2, 2) for axis in "XY"]
        self.effects = self.check(form, "Vary noise / texture seed", False)
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
            "XYZ angles. All samples are saved, including empty contacts. "
            "Stop finishes the current sample and keeps completed captures. The current preview pose is preserved.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.update_destination()

    def check(self, form, label, checked):
        box = QtWidgets.QCheckBox(label)
        box.setChecked(checked)
        form.addRow(box)
        return box

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
            import re
            pattern = re.compile(r'^(?:sample|run|manifest|processed_mesh|\.sample)_(\d{6})')
            indices = [int(match.group(1)) for path in parent.iterdir()
                       if (match := pattern.match(path.name))] if parent.is_dir() else []
            next_index = max(indices, default=0) + 1
            self.destination.setText(str(parent / f"sample_{next_index:06d}_tactile.png"))
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
        return GatherSettings(count=self.count.value(), seed=self.seed.value(),
            random_rotation=self.rotation.isChecked(),
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
