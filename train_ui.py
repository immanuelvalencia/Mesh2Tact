"""Desktop launcher for the sequential torchvision training pipeline."""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from mesh2tact.architectures import ALL_MODELS, SUITES
from train import IMAGE_TYPES, PROGRESS_PREFIX, resolve_datasets


def dataset_modes(root: Path) -> list[tuple[str, str]]:
    root = root.expanduser().resolve()
    if all((root / split).is_dir() for split in ("train", "val", "test")):
        return [("This dataset", "single")]
    if all((root / name).is_dir() for name in IMAGE_TYPES):
        return [("All three image types (separate models)", "all"),
                *((name.title(), name) for name in IMAGE_TYPES)]
    return []


def inspect_dataset(root: Path, mode: str) -> dict:
    if mode == "single":
        selected = resolve_datasets(root, None, False)
    elif mode == "all":
        selected = resolve_datasets(root, None, True)
    elif mode in IMAGE_TYPES:
        selected = resolve_datasets(root, mode, False)
    else:
        raise ValueError("Select a prepared dataset type.")
    labels = None
    counts = {}
    for name, folder in selected:
        label_file = folder / "labels.txt"
        if not label_file.is_file():
            raise ValueError(f"Missing labels.txt: {label_file}")
        current = label_file.read_text(encoding="utf-8-sig").splitlines()
        if len(current) < 2 or len(current) != len(set(current)) or any(not label for label in current):
            raise ValueError(f"Invalid labels.txt: {label_file}")
        if labels is not None and current != labels:
            raise ValueError("Selected datasets do not have identical labels.")
        labels = current
        branch_counts = {}
        for split in ("train", "val", "test"):
            split_folder = folder / split
            if not split_folder.is_dir():
                raise ValueError(f"Missing split folder: {split_folder}")
            found = sorted(path.name for path in split_folder.iterdir() if path.is_dir())
            if found != labels:
                raise ValueError(f"Class order in {split_folder} does not match labels.txt.")
            branch_counts[split] = {}
            for label in labels:
                files = [path for path in (split_folder / label).rglob("*")
                         if path.is_file() and path.suffix.lower() in
                         {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}]
                if not files:
                    raise ValueError(f"No images in {split_folder / label}")
                branch_counts[split][label] = len(files)
        counts[name or folder.name] = branch_counts
    if mode == "all":
        first = counts[IMAGE_TYPES[0]]
        if any(counts[name] != first for name in IMAGE_TYPES[1:]):
            raise ValueError("Image-type branches have different class counts per split.")
    return {"labels": labels, "counts": counts, "datasets": selected}


class TrainingWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh2Tact model training")
        self.resize(1040, 850)
        self.confirmed = False
        self.dataset_info = None
        self.process = None
        self.stream_buffer = ""
        self.session_path = None
        self.stop_directory = None
        self.stop_path = None
        self.stop_requested = False
        self.force_stop_timer = QtCore.QTimer(self)
        self.force_stop_timer.setSingleShot(True)
        self.force_stop_timer.timeout.connect(self.force_stop_if_running)
        main = QtWidgets.QVBoxLayout(self)

        folders = QtWidgets.QFormLayout()
        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setPlaceholderText("Prepared ml_dataset folder or a single image-type branch")
        self.dataset_edit.textChanged.connect(self.invalidate_confirmation)
        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.addWidget(self.dataset_edit)
        browse_dataset = QtWidgets.QPushButton("Browse…")
        browse_dataset.clicked.connect(self.browse_dataset)
        self.browse_dataset_button = browse_dataset
        dataset_row.addWidget(browse_dataset)
        folders.addRow("Dataset", dataset_row)
        self.dataset_mode = QtWidgets.QComboBox()
        self.dataset_mode.currentIndexChanged.connect(self.invalidate_confirmation)
        folders.addRow("Train on", self.dataset_mode)
        output_row = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit(str((Path.cwd() / "train").resolve()))
        output_row.addWidget(self.output_edit)
        browse_output = QtWidgets.QPushButton("Browse…")
        browse_output.clicked.connect(self.browse_output)
        self.browse_output_button = browse_output
        output_row.addWidget(browse_output)
        folders.addRow("Export folder", output_row)
        main.addLayout(folders)

        label_row = QtWidgets.QHBoxLayout()
        self.confirm_button = QtWidgets.QPushButton("Detect and confirm labels")
        self.confirm_button.clicked.connect(self.confirm_labels)
        label_row.addWidget(self.confirm_button)
        self.labels_view = QtWidgets.QLabel("No dataset confirmed")
        self.labels_view.setWordWrap(True)
        label_row.addWidget(self.labels_view, 1)
        main.addLayout(label_row)

        model_box = QtWidgets.QGroupBox(f"Architectures ({len(ALL_MODELS)} available)")
        self.model_box = model_box
        model_layout = QtWidgets.QVBoxLayout(model_box)
        self.select_all = QtWidgets.QCheckBox("Select all architectures")
        self.select_all.toggled.connect(self.toggle_all)
        model_layout.addWidget(self.select_all)
        self.model_tree = QtWidgets.QTreeWidget()
        self.model_tree.setHeaderHidden(True)
        self.model_tree.setMinimumHeight(270)
        for family, names in SUITES.items():
            parent = QtWidgets.QTreeWidgetItem(self.model_tree, [f"{family.title()} ({len(names)})"])
            parent.setFlags(parent.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsAutoTristate)
            parent.setCheckState(0, QtCore.Qt.Unchecked)
            for name in names:
                child = QtWidgets.QTreeWidgetItem(parent, [name])
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(0, QtCore.Qt.Checked if name == "resnet18" else QtCore.Qt.Unchecked)
        self.model_tree.expandAll()
        model_layout.addWidget(self.model_tree)
        self.model_count = QtWidgets.QLabel()
        model_layout.addWidget(self.model_count)
        self.model_tree.itemChanged.connect(self.update_model_count)
        self.update_model_count()
        main.addWidget(model_box)

        parameters = QtWidgets.QGroupBox("Training parameters")
        self.parameters_box = parameters
        grid = QtWidgets.QGridLayout(parameters)
        self.epochs = self._spin(1, 10000, 20)
        self.batch_size = self._spin(1, 4096, 16)
        self.patience = self._spin(1, 10000, 5)
        self.seed = self._spin(0, 2147483647, 42)
        self.workers = self._spin(0, 128, 0)
        self.grid_per_class = self._spin(1, 20, 4)
        self.learning_rate = QtWidgets.QDoubleSpinBox()
        self.learning_rate.setRange(0.0000001, 1.0)
        self.learning_rate.setDecimals(7)
        self.learning_rate.setSingleStep(0.00001)
        self.learning_rate.setValue(0.0001)
        self.weights = QtWidgets.QComboBox()
        self.weights.addItem("Pretrained default", "default")
        self.weights.addItem("From scratch", "none")
        self.weights.addItem("All available weight variants", "all")
        self.device = QtWidgets.QComboBox()
        for value in ("auto", "cuda", "cpu"):
            self.device.addItem(value.upper(), value)
        settings = (("Epochs", self.epochs), ("Batch size", self.batch_size),
                    ("Learning rate", self.learning_rate), ("Patience", self.patience),
                    ("Seed", self.seed), ("Data workers", self.workers),
                    ("Test photos per label", self.grid_per_class),
                    ("Weights", self.weights), ("Device", self.device))
        for index, (title, widget) in enumerate(settings):
            column = (index % 3) * 2
            row = index // 3
            grid.addWidget(QtWidgets.QLabel(title), row, column)
            grid.addWidget(widget, row, column + 1)
        main.addWidget(parameters)

        actions = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Train selected models sequentially")
        self.start_button.clicked.connect(self.start_training)
        actions.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("Stop training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        actions.addWidget(self.stop_button)
        main.addLayout(actions)
        self.status = QtWidgets.QLabel("Choose a prepared dataset, confirm labels, then train.")
        self.status.setWordWrap(True)
        main.addWidget(self.status)
        self.current_progress = QtWidgets.QProgressBar()
        self.current_progress.setRange(0, 1000)
        self.current_progress.setFormat("Current model: 0.0%")
        main.addWidget(self.current_progress)
        self.overall_progress = QtWidgets.QProgressBar()
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setFormat("Overall queue: 0.0%")
        main.addWidget(self.overall_progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        main.addWidget(self.log, 1)

    @staticmethod
    def _spin(minimum, maximum, value):
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def invalidate_confirmation(self, *_):
        self.confirmed = False
        self.dataset_info = None
        self.labels_view.setText("No dataset confirmed")

    def browse_dataset(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a prepared dataset folder")
        if not folder:
            return
        root = Path(folder)
        self.dataset_edit.setText(str(root))
        self.dataset_mode.clear()
        for label, mode in dataset_modes(root):
            self.dataset_mode.addItem(label, mode)
        if self.dataset_mode.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Not prepared", "Choose ml_dataset or one of its split branches.")
        if self.output_edit.text() == str((Path.cwd() / "train").resolve()):
            parent = root.parent.parent if root.name in IMAGE_TYPES and (root.parent / "paired_manifest.csv").is_file() else root.parent
            self.output_edit.setText(str((parent / "train").resolve()))

    def browse_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose model export folder")
        if folder:
            self.output_edit.setText(folder)

    def selected_models(self):
        selected = []
        for index in range(self.model_tree.topLevelItemCount()):
            parent = self.model_tree.topLevelItem(index)
            for child_index in range(parent.childCount()):
                child = parent.child(child_index)
                if child.checkState(0) == QtCore.Qt.Checked:
                    selected.append(child.text(0))
        return selected

    def toggle_all(self, checked):
        self.model_tree.blockSignals(True)
        for index in range(self.model_tree.topLevelItemCount()):
            self.model_tree.topLevelItem(index).setCheckState(
                0, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self.model_tree.blockSignals(False)
        self.update_model_count()

    def update_model_count(self, *_):
        count = len(self.selected_models())
        self.model_count.setText(f"{count} of {len(ALL_MODELS)} architectures selected; runs execute one after another.")
        self.select_all.blockSignals(True)
        self.select_all.setChecked(count == len(ALL_MODELS))
        self.select_all.blockSignals(False)

    def confirm_labels(self):
        try:
            folder = Path(self.dataset_edit.text().strip()).expanduser()
            if not self.dataset_edit.text().strip() or not folder.is_dir():
                raise ValueError("Choose an existing prepared dataset folder.")
            if self.dataset_mode.count() == 0:
                self.dataset_mode.clear()
                for label, mode in dataset_modes(folder):
                    self.dataset_mode.addItem(label, mode)
            info = inspect_dataset(folder, self.dataset_mode.currentData())
        except (OSError, ValueError) as exc:
            self.invalidate_confirmation()
            QtWidgets.QMessageBox.warning(self, "Dataset check failed", str(exc))
            return
        lines = [f"Labels ({len(info['labels'])}): {', '.join(info['labels'])}"]
        for branch, splits in info["counts"].items():
            lines.append(f"{branch}: " + ", ".join(
                f"{split} {sum(values.values())}" for split, values in splits.items()))
        message = "\n".join(lines) + "\n\nUse these labels for every selected model?"
        if QtWidgets.QMessageBox.question(self, "Confirm detected labels", message,
                                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                          QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            self.invalidate_confirmation()
            return
        self.dataset_info = info
        self.confirmed = True
        self.labels_view.setText("Confirmed: " + ", ".join(info["labels"]))
        self.status.setText("Dataset ready. Select architectures and training parameters.")

    def command_arguments(self):
        if not self.confirmed:
            raise ValueError("Detect and confirm the labels before training.")
        models = self.selected_models()
        if not models:
            raise ValueError("Select at least one architecture.")
        output = self.output_edit.text().strip()
        if not output:
            raise ValueError("Choose an export folder.")
        folder = Path(self.dataset_edit.text().strip()).expanduser().resolve()
        export = Path(output).expanduser().resolve()
        dataset_base = (folder.parent if folder.name in IMAGE_TYPES
                        and (folder.parent / "paired_manifest.csv").is_file() else folder)
        if export == dataset_base or export.is_relative_to(dataset_base):
            raise ValueError("The export folder must be outside the training dataset.")
        args = [str(Path(__file__).with_name("train.py")), "--dataset-dir", str(folder),
                "--output-dir", str(export)]
        mode = self.dataset_mode.currentData()
        if mode == "all":
            args.append("--all-image-types")
        elif mode in IMAGE_TYPES:
            args.extend(("--image-type", mode))
        args.extend(("--models", *models, "--epochs", str(self.epochs.value()),
                     "--batch-size", str(self.batch_size.value()),
                     "--lr", str(self.learning_rate.value()),
                     "--patience", str(self.patience.value()),
                     "--seed", str(self.seed.value()),
                     "--workers", str(self.workers.value()),
                     "--test-grid-per-class", str(self.grid_per_class.value()),
                     "--weights", self.weights.currentData(),
                     "--device", self.device.currentData(), "--ui-progress"))
        return args

    def start_training(self):
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            return
        try:
            args = self.command_arguments()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot start", str(exc))
            return
        variants = len(self.dataset_info["datasets"])
        planned = len(self.selected_models()) * variants
        if planned > 5 or self.weights.currentData() == "all":
            message = (f"Start at least {planned} sequential model run(s)?\n"
                       "Large architectures or pretrained weights may need substantial GPU memory and downloads.")
            if QtWidgets.QMessageBox.question(self, "Confirm training queue", message,
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                              QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
        self.stop_directory = tempfile.TemporaryDirectory(prefix="mesh2tact_training_")
        self.stop_path = Path(self.stop_directory.name) / "stop.request"
        self.stop_requested = False
        args.extend(("--stop-file", str(self.stop_path)))
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished_training)
        self.process.errorOccurred.connect(self.process_error)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_inputs_enabled(False)
        self.log.clear()
        self.log.appendPlainText("Selected models, in training order: " + ", ".join(self.selected_models()))
        self.log.appendPlainText("Only one model is trained at a time. Stop ends the queue after the current batch.")
        self.stream_buffer = ""
        self.session_path = None
        self.current_progress.setValue(0)
        self.current_progress.setFormat("Current model: 0.0%")
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("Overall queue: 0.0%")
        self.status.setText(f"Training {len(self.selected_models())} architecture(s) sequentially…")
        self.process.start(sys.executable, ["-u", *args])

    def set_inputs_enabled(self, enabled):
        for widget in (self.dataset_edit, self.dataset_mode, self.output_edit,
                       self.browse_dataset_button, self.browse_output_button,
                       self.confirm_button, self.model_box, self.parameters_box):
            widget.setEnabled(enabled)

    def cleanup_stop_request(self):
        self.force_stop_timer.stop()
        if self.stop_directory is not None:
            try:
                self.stop_directory.cleanup()
            except OSError as exc:
                self.log.appendPlainText(f"Could not remove temporary stop request: {exc}")
            self.stop_directory = None
            self.stop_path = None

    def read_output(self):
        if self.process is not None:
            content = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.consume_output(content)

    def consume_output(self, content):
        self.stream_buffer += content.replace("\r", "\n")
        while "\n" in self.stream_buffer:
            line, self.stream_buffer = self.stream_buffer.split("\n", 1)
            if line.startswith(PROGRESS_PREFIX):
                try:
                    self.handle_progress_event(json.loads(line[len(PROGRESS_PREFIX):]))
                except (ValueError, TypeError):
                    self.log.appendPlainText(line)
            elif line:
                self.log.appendPlainText(line)
        self.log.ensureCursorVisible()

    def handle_progress_event(self, event):
        overall = max(0.0, min(100.0, float(event.get("overall_percent", 0))))
        self.overall_progress.setValue(round(overall * 10))
        self.overall_progress.setFormat(f"Overall queue: {overall:.1f}%")
        if "model_percent" in event:
            current = max(0.0, min(100.0, float(event["model_percent"])))
            self.current_progress.setValue(round(current * 10))
            self.current_progress.setFormat(f"Current model: {current:.1f}%")
        kind = event.get("event")
        if kind == "plan":
            self.status.setText(f"Queued {event['runs']} model run(s); training sequentially.")
            queue = event.get("queue", [])
            if queue:
                preview = ", ".join(f"{item['architecture']} / {item['image_type']}"
                                    for item in queue[:12])
                if len(queue) > 12:
                    preview += f", … and {len(queue) - 12} more"
                self.log.appendPlainText("Run queue: " + preview)
            return
        if kind == "session":
            self.session_path = event["path"]
            self.status.setText(f"Session folder: {self.session_path}")
            return
        run = f"{event['run']}/{event['runs']}"
        model = f"{event['image_type']} / {event['architecture']} / {event['weights']}"
        if kind == "run_start":
            self.current_progress.setValue(0)
            self.current_progress.setFormat("Current model: 0.0%")
            self.status.setText(f"Run {run}: {model} — initializing")
        elif kind == "batch":
            phase = event["phase"]
            epoch = (f"Epoch {event['epoch']}/{event['epochs']} · "
                     if phase != "test" else "")
            loss = f" · loss {event['loss']:.4f}" if "loss" in event else ""
            self.status.setText(
                f"Run {run}: {model} · {epoch}{phase} {event['phase_percent']:.1f}% "
                f"({event['batch']}/{event['batches']} batches){loss} · "
                f"accuracy {event['accuracy'] * 100:.1f}%")
        elif kind in ("test_start", "export"):
            self.status.setText(f"Run {run}: {model} — "
                                + ("testing the best checkpoint" if kind == "test_start"
                                   else "exporting metrics and artifacts"))
        elif kind == "run_complete":
            self.status.setText(f"Run {run}: {model} complete · test accuracy "
                                f"{event['test_accuracy'] * 100:.1f}%")
        elif kind == "stopped":
            self.status.setText(f"Stopped after {event['completed']} completed model run(s).")

    def finished_training(self, exit_code, _status):
        self.read_output()
        if self.stream_buffer:
            self.log.appendPlainText(self.stream_buffer)
            self.stream_buffer = ""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_inputs_enabled(True)
        self.cleanup_stop_request()
        if exit_code == 0 and not self.stop_requested:
            self.current_progress.setValue(1000)
            self.current_progress.setFormat("Current model: 100.0%")
            self.overall_progress.setValue(1000)
            self.overall_progress.setFormat("Overall queue: 100.0%")
        if self.stop_requested:
            self.status.setText("Training stopped. Completed model exports were kept. "
                                + (f"Session: {self.session_path}" if self.session_path else ""))
        else:
            self.status.setText(f"Training complete. Session: {self.session_path}" if exit_code == 0
                                else f"Training failed (exit code {exit_code}). Review the log.")

    def process_error(self, error):
        self.status.setText(f"Could not run training: {self.process.errorString()} ({error}).")
        if self.process.state() == QtCore.QProcess.NotRunning:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.set_inputs_enabled(True)
            self.cleanup_stop_request()

    def stop_training(self):
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            if self.stop_requested:
                return
            self.stop_requested = True
            self.stop_button.setEnabled(False)
            self.status.setText("Stop requested. Finishing the current batch; force-stop in 8 seconds if needed…")
            self.log.appendPlainText("STOP REQUESTED: no further models will start.")
            try:
                self.stop_path.touch()
            except OSError as exc:
                self.log.appendPlainText(f"Could not send stop request ({exc}); force-stopping now.")
                self.process.kill()
                return
            self.force_stop_timer.start(8000)

    def force_stop_if_running(self):
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            self.log.appendPlainText("Stop timeout reached; force-stopping the training process.")
            self.process.kill()

    def closeEvent(self, event):
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, "Training active", "Stop training before closing this window.")
            event.ignore()
        else:
            event.accept()


def launch():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    return app.exec_()
