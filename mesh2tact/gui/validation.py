"""Validation controls and a viewer that replaces the 3D scene."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from threading import Event

from .qt import QtCore, QtGui, QtWidgets
from .predict import PredictPanel, PhotoPreview, style_button
from .model_types import ImageTypeDelegate, model_type_store
from ..prediction import infer_image_type
from ..validation import VALIDATION_ROOT, discover_split, sample_split, run_validation, update_run_image_type


class ValidationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    model_ready = QtCore.Signal(str)
    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.options = options
        self.stop = Event()

    def run(self):
        try:
            directory = run_validation(**self.options, progress=self.progress.emit,
                                       cancelled=self.stop.is_set, model_ready=self.model_ready.emit)
            self.completed.emit(str(directory))
        except Exception as exc:
            self.failed.emit(str(exc))


class ClassPhotoBrowser(QtWidgets.QWidget):
    """Class-organized, paginated thumbnails of the exact sampled photos."""
    PAGE_SIZE = 100

    def __init__(self):
        super().__init__()
        self.samples = {}
        self.paths = []
        self.page = 0
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel("Choose a processed folder to view photos by class.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        split = QtWidgets.QSplitter()
        self.classes = QtWidgets.QTreeWidget()
        self.classes.setHeaderLabels(("Class", "Selected", "In split"))
        self.classes.setRootIsDecorated(False)
        self.classes.setMinimumWidth(200)
        self.classes.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.classes.currentItemChanged.connect(self.choose_class)
        split.addWidget(self.classes)
        gallery = QtWidgets.QWidget()
        gallery_layout = QtWidgets.QVBoxLayout(gallery)
        self.photos = QtWidgets.QListWidget()
        self.photos.setViewMode(QtWidgets.QListView.IconMode)
        self.photos.setResizeMode(QtWidgets.QListView.Adjust)
        self.photos.setMovement(QtWidgets.QListView.Static)
        self.photos.setIconSize(QtCore.QSize(120, 90))
        self.photos.setGridSize(QtCore.QSize(155, 125))
        self.photos.setWordWrap(True)
        self.photos.currentItemChanged.connect(self.preview_photo)
        gallery_layout.addWidget(self.photos, 1)
        navigation = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QPushButton("Previous")
        self.next = QtWidgets.QPushButton("Next")
        self.page_label = QtWidgets.QLabel()
        self.previous.clicked.connect(lambda: self.change_page(-1))
        self.next.clicked.connect(lambda: self.change_page(1))
        for widget in (self.previous, self.page_label, self.next):
            navigation.addWidget(widget)
        gallery_layout.addLayout(navigation)
        self.preview = PhotoPreview()
        gallery_layout.addWidget(self.preview, 1)
        self.caption = QtWidgets.QLabel()
        self.caption.setWordWrap(True)
        gallery_layout.addWidget(self.caption)
        split.addWidget(gallery)
        split.setSizes([240, 600])
        layout.addWidget(split, 1)

    def set_samples(self, samples, totals, description):
        self.samples = {label: [] for label in totals}
        for label, path in samples:
            self.samples.setdefault(label, []).append(Path(path))
        self.classes.clear()
        self.paths = []
        self.page = 0
        self.summary.setText(f"{description}\n{len(samples)} selected photos. Choose a class to browse its sample.")
        for label, paths in self.samples.items():
            item = QtWidgets.QTreeWidgetItem((label, str(len(paths)), str(totals.get(label, len(paths)))))
            self.classes.addTopLevelItem(item)
        self.show_page()
        if self.classes.topLevelItemCount():
            self.classes.setCurrentItem(self.classes.topLevelItem(0))

    def choose_class(self, item, *_):
        self.paths = self.samples.get(item.text(0), []) if item else []
        self.page = 0
        self.show_page()

    def change_page(self, offset):
        self.page += offset
        self.show_page()

    def show_page(self):
        self.photos.clear()
        self.preview.original = QtGui.QPixmap()
        self.preview.setText("Select a photo to preview")
        self.caption.clear()
        start = self.page * self.PAGE_SIZE
        for path in self.paths[start:start + self.PAGE_SIZE]:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, path)
            item.setToolTip(str(path))
            reader = QtGui.QImageReader(str(path))
            reader.setAutoTransform(True)
            if reader.size().isValid():
                reader.setScaledSize(reader.size().scaled(120, 90, QtCore.Qt.KeepAspectRatio))
            item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(reader.read())))
            self.photos.addItem(item)
        self.page_label.setText(f"{start + 1 if self.paths else 0}–{min(start + self.PAGE_SIZE, len(self.paths))} / {len(self.paths)}")
        self.previous.setEnabled(self.page > 0)
        self.next.setEnabled(start + self.PAGE_SIZE < len(self.paths))
        if self.photos.count():
            self.photos.setCurrentRow(0)

    def preview_photo(self, item, *_):
        if item:
            path = item.data(QtCore.Qt.UserRole)
            self.preview.show_photo(path)
            self.caption.setText(str(path))


class ValidationResults(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.directory = None
        self.model_directories = []
        self.type_store = model_type_store()
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)
        self.description = QtWidgets.QLabel("Validation results will appear here and be saved automatically.")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        controls = QtWidgets.QHBoxLayout()
        hint = QtWidgets.QLabel("Drag the divider below the model table to resize it.")
        hint.setWordWrap(True)
        controls.addWidget(hint, 1)
        self.expand_models = QtWidgets.QPushButton("Expand model table")
        self.expand_models.setCheckable(True)
        self.expand_models.setToolTip("Use the full results area for models; click again to restore metrics and graphs.")
        self.expand_models.toggled.connect(self.toggle_model_expansion)
        controls.addWidget(self.expand_models)
        layout.addLayout(controls)
        self.results_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setHandleWidth(9)
        self.results_splitter.setStyleSheet("QSplitter::handle:vertical { background: #cbd3db; }")
        self._details_sizes = None
        self.summary = QtWidgets.QTableWidget(0, 8)
        self.summary.setHorizontalHeaderLabels(("Model", "Status", "Evaluated", "Errors", "Accuracy", "Macro F1", "Coverage", "Image type"))
        self.summary.horizontalHeader().moveSection(7, 1)
        self.summary.setItemDelegate(ImageTypeDelegate(7, self.summary))
        self.summary.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.summary.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.summary.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.summary.itemChanged.connect(self.edit_image_type)
        self.summary.currentCellChanged.connect(self.show_model)
        self.columns_button = PredictPanel.configure_columns(
            self.summary, self.summary.horizontalHeader(), "results", (300, 100, 90, 75, 100, 100, 100, 120),
            controls, settings_group="ValidationResults", version=2)
        self.results_splitter.addWidget(self.summary)
        self.details = QtWidgets.QTabWidget()
        self.metrics = QtWidgets.QTextBrowser()
        self.details.addTab(self.metrics, "Metrics")
        self.per_class = QtWidgets.QTableWidget()
        self.per_class.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.details.addTab(self.per_class, "Per class")
        self.counts = PhotoPreview()
        self.normalized = PhotoPreview()
        self.details.addTab(self.counts, "Confusion counts")
        self.details.addTab(self.normalized, "Confusion normalized")
        self.predictions = QtWidgets.QTableWidget()
        self.predictions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        prediction_page = QtWidgets.QWidget()
        prediction_layout = QtWidgets.QVBoxLayout(prediction_page)
        prediction_layout.addWidget(self.predictions, 1)
        pagination = QtWidgets.QHBoxLayout()
        self.prediction_previous = QtWidgets.QPushButton("Previous")
        self.prediction_next = QtWidgets.QPushButton("Next")
        self.prediction_page_label = QtWidgets.QLabel()
        self.prediction_previous.clicked.connect(lambda: self.show_predictions(self.prediction_page - 1))
        self.prediction_next.clicked.connect(lambda: self.show_predictions(self.prediction_page + 1))
        for control in (self.prediction_previous, self.prediction_page_label, self.prediction_next):
            pagination.addWidget(control)
        prediction_layout.addLayout(pagination)
        self.details.addTab(prediction_page, "Predictions / errors")
        self.prediction_rows = []
        self.prediction_page = 0
        self.results_splitter.addWidget(self.details)
        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 3)
        self.results_splitter.setSizes([240, 540])
        layout.addWidget(self.results_splitter, 1)

    def toggle_model_expansion(self, expanded):
        if expanded:
            self._details_sizes = self.results_splitter.sizes()
        self.details.setVisible(not expanded)
        self.expand_models.setText("Show metrics and graphs" if expanded else "Expand model table")
        if not expanded and self._details_sizes:
            self.results_splitter.setSizes(self._details_sizes)

    @staticmethod
    def value(value):
        if value is None:
            return "—"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    def load_run(self, directory):
        self.directory = Path(directory)
        run = json.loads((self.directory / "run.json").read_text(encoding="utf-8"))
        self.description.setText(f"{self.directory.name} • {run['status']} • {run['split']} • "
                                 f"{run['selected']} images • {run['percentage']}% requested • seed {run['seed']}\n"
                                 f"{run['dataset']}\nMetrics exclude failed images; compare coverage alongside scores.")
        self._loading = True
        self.summary.setRowCount(0)
        self.model_directories = []
        self.metrics.clear()
        self.per_class.setRowCount(0)
        self.prediction_rows = []
        self.show_predictions(0)
        for view in (self.counts, self.normalized):
            view.original = QtGui.QPixmap()
            view.setText("No model results")
        for model in run["models"]:
            folder = (self.directory / model["directory"]).resolve()
            if not folder.is_relative_to(self.directory.resolve()):
                raise ValueError("Saved model directory is outside the run")
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            row = self.summary.rowCount()
            self.model_directories.append(folder)
            self.summary.insertRow(row)
            image_type = metrics.get("image_type") or self.type_store.get(metrics["checkpoint"], infer_image_type(metrics["checkpoint"]))
            values = (Path(metrics["checkpoint"]).name, metrics["status"], metrics["evaluated"], metrics["errors"],
                      metrics["accuracy"], metrics["macro_f1"], metrics["coverage"], image_type)
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(self.value(value))
                item.setToolTip(metrics["checkpoint"])
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if column == 7:
                    item.setData(QtCore.Qt.UserRole, (metrics["checkpoint"], image_type))
                    if run["status"] != "running":
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                    item.setToolTip("Double-click to correct the image type. Changes update this run's metrics and summary exports. "
                                    "Editing is available after the run finishes.")
                self.summary.setItem(row, column, item)
        self._loading = False
        if self.model_directories:
            self.summary.setCurrentCell(0, 0)
        return run

    def edit_image_type(self, item):
        if self._loading or item.column() != 7:
            return
        checkpoint, previous = item.data(QtCore.Qt.UserRole)
        value = item.text()
        if value == previous:
            return
        try:
            update_run_image_type(self.directory, self.model_directories[item.row()], value)
        except (OSError, ValueError, KeyError) as exc:
            self.summary.blockSignals(True)
            item.setText(previous)
            self.summary.blockSignals(False)
            self.description.setText(f"Could not save image type: {exc}")
            return
        self.summary.blockSignals(True)
        item.setData(QtCore.Qt.UserRole, (checkpoint, value))
        self.summary.blockSignals(False)
        self.type_store.set(checkpoint, value)
        self.show_model(item.row())

    def show_model(self, row, *_):
        if not 0 <= row < len(self.model_directories):
            return
        folder = self.model_directories[row]
        try:
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            self.metrics.setPlainText("\n".join(f"{key.replace('_', ' ').title()}: {self.value(value)}"
                                               for key, value in metrics.items()
                                               if key not in ("per_class", "confusion_matrix")))
            fields = list(metrics["per_class"][0])
            self.per_class.setColumnCount(len(fields))
            self.per_class.setHorizontalHeaderLabels([field.replace("_", " ").title() for field in fields])
            self.per_class.setRowCount(len(metrics["per_class"]))
            for index, values in enumerate(metrics["per_class"]):
                for column, key in enumerate(fields):
                    self.per_class.setItem(index, column, QtWidgets.QTableWidgetItem(self.value(values[key])))
            self.per_class.resizeColumnsToContents()
            self.counts.show_photo(folder / "confusion_counts.png")
            self.normalized.show_photo(folder / "confusion_normalized.png")
            with (folder / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.prediction_rows = list(csv.DictReader(stream))
            self.show_predictions(0)
        except (OSError, ValueError, KeyError) as exc:
            self.metrics.setPlainText(f"Could not open model results: {exc}")

    def show_predictions(self, page):
        self.prediction_page = max(0, page)
        start = self.prediction_page * 200
        rows = self.prediction_rows[start:start + 200]
        fields = ("path", "actual", "predicted", "confidence", "correct", "error")
        self.predictions.setColumnCount(len(fields))
        self.predictions.setHorizontalHeaderLabels([field.title() for field in fields])
        self.predictions.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, key in enumerate(fields):
                self.predictions.setItem(row, column, QtWidgets.QTableWidgetItem(values.get(key, "")))
        self.prediction_page_label.setText(f"{start + 1 if rows else 0}–{start + len(rows)} / {len(self.prediction_rows)}")
        self.prediction_previous.setEnabled(start > 0)
        self.prediction_next.setEnabled(start + len(rows) < len(self.prediction_rows))


class ValidationPanel(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.dataset_root = None
        self.classes = {}
        self.samples = []
        self.actual_split = "test"
        self.worker = None
        self.active_run = None
        self.settings = QtCore.QSettings("Mesh2Tact", "ValidationPanel")
        self.output_root = Path(self.settings.value("output_root", str(VALIDATION_ROOT)))
        self.workspace = QtWidgets.QTabWidget()
        self.photos = ClassPhotoBrowser()
        self.results = ValidationResults()
        self.workspace.addTab(self.photos, "Photos by class")
        self.workspace.addTab(self.results, "Validation results")
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(contents)
        scroll.setWidget(contents)
        outer.addWidget(scroll)
        intro = QtWidgets.QLabel("Evaluate saved models on a processed dataset. Choose test or valid, "
                                "sample the split, and compare models on exactly the same photos.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        dataset_row = QtWidgets.QHBoxLayout()
        self.dataset = QtWidgets.QLineEdit()
        self.dataset.setReadOnly(True)
        self.dataset.setPlaceholderText("Processed folder containing train / test / val or valid")
        self.browse_dataset = QtWidgets.QPushButton("Browse processed…")
        self.browse_dataset.clicked.connect(self.choose_dataset)
        dataset_row.addWidget(self.dataset, 1)
        dataset_row.addWidget(self.browse_dataset)
        layout.addLayout(dataset_row)
        options = QtWidgets.QHBoxLayout()
        self.split = QtWidgets.QComboBox()
        self.split.addItem("Test", "test")
        self.split.addItem("Valid (val / valid)", "valid")
        self.percentage = QtWidgets.QDoubleSpinBox()
        self.percentage.setRange(.1, 100)
        self.percentage.setValue(100)
        self.percentage.setSuffix(" %")
        self.percentage.setDecimals(1)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)
        for label, control in (("Split", self.split), ("Use", self.percentage), ("Seed", self.seed)):
            options.addWidget(QtWidgets.QLabel(label))
            options.addWidget(control)
        layout.addLayout(options)
        note = QtWidgets.QLabel("Samples are random within each class, rounded up to at least one photo per nonempty class. "
                               "The seed makes selection repeatable; the effective percentage may be higher for small classes.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.sample_summary = QtWidgets.QLabel("No dataset selected")
        self.sample_summary.setWordWrap(True)
        layout.addWidget(self.sample_summary)
        self.sample_timer = QtCore.QTimer(self)
        self.sample_timer.setSingleShot(True)
        self.sample_timer.setInterval(250)
        self.sample_timer.timeout.connect(self.refresh_samples)
        self.split.currentIndexChanged.connect(self.schedule_samples)
        self.percentage.valueChanged.connect(self.schedule_samples)
        self.seed.valueChanged.connect(self.schedule_samples)
        self.selector = PredictPanel(window, selector_only=True)
        layout.addWidget(self.selector, 1)
        output_row = QtWidgets.QHBoxLayout()
        self.output = QtWidgets.QLineEdit(str(self.output_root))
        self.output.setReadOnly(True)
        self.browse_output = QtWidgets.QPushButton("Validation folder…")
        self.browse_output.clicked.connect(self.choose_output)
        output_row.addWidget(self.output, 1)
        output_row.addWidget(self.browse_output)
        layout.addLayout(output_row)
        actions = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QPushButton("Run validation")
        self.run.clicked.connect(self.start_validation)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_validation)
        actions.addWidget(self.run)
        actions.addWidget(self.cancel)
        layout.addLayout(actions)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Each run automatically exports metrics, predictions and confusion plots under validation/runs.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        saved = QtWidgets.QHBoxLayout()
        self.runs = QtWidgets.QComboBox()
        self.runs.setMinimumContentsLength(20)
        self.runs.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.refresh = QtWidgets.QPushButton("Refresh runs")
        self.refresh.clicked.connect(self.refresh_runs)
        self.view = QtWidgets.QPushButton("View run")
        self.view.clicked.connect(self.view_run)
        for control in (self.runs, self.refresh, self.view):
            saved.addWidget(control)
        layout.addLayout(saved)
        self.open_folder = QtWidgets.QPushButton("Open run folder")
        self.open_folder.clicked.connect(self.open_run_folder)
        layout.addWidget(self.open_folder)
        for button in (self.browse_dataset, self.browse_output, self.view):
            style_button(button, "#1769aa")
        style_button(self.run, "#287348")
        style_button(self.cancel, "#a8473b")
        self.refresh_runs()

    def choose_dataset(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose processed dataset",
                    str(self.dataset_root or Path(__file__).resolve().parents[2]))
        if folder:
            self.dataset_root = Path(folder)
            self.dataset.setText(folder)
            self.refresh_samples()

    def schedule_samples(self, *_):
        self.sample_timer.start()

    def refresh_samples(self):
        self.sample_timer.stop()
        self.samples = []
        self.classes = {}
        if self.dataset_root is None:
            return
        try:
            folder, self.classes = discover_split(self.dataset_root, self.split.currentData())
            self.actual_split = folder.name
            self.samples = sample_split(self.classes, self.percentage.value(), self.seed.value())
            total = sum(map(len, self.classes.values()))
            self.sample_summary.setText(f"{len(self.samples)}/{total} photos ({len(self.samples) / total:.1%}) "
                                        f"in {len(self.classes)} classes • {folder.name}")
            self.photos.set_samples(self.samples, {label: len(paths) for label, paths in self.classes.items()},
                                    f"{folder} • seed {self.seed.value()}")
            self.workspace.setCurrentWidget(self.photos)
        except (OSError, ValueError) as exc:
            self.sample_summary.setText(str(exc))
            self.photos.set_samples([], {}, str(exc))

    def choose_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose validation output folder", str(self.output_root))
        if folder:
            self.output_root = Path(folder)
            self.output.setText(folder)
            self.settings.setValue("output_root", folder)
            self.refresh_runs()

    def start_validation(self):
        if self.worker is not None:
            return
        self.refresh_samples()
        candidates = self.selector.selected_candidates()
        if not self.samples or not candidates:
            self.status.setText("Choose a valid processed dataset and check at least one visible model.")
            return
        self.active_run = None
        options = dict(root=self.dataset_root, split=self.actual_split, classes=self.classes,
                       samples=list(self.samples), candidates=candidates, percentage=self.percentage.value(),
                       seed=self.seed.value(), output_root=self.output_root)
        self.worker = ValidationWorker(options, self)
        self.worker.progress.connect(self.update_progress)
        self.worker.model_ready.connect(self.model_finished)
        self.worker.completed.connect(self.completed)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.worker_finished)
        self.set_busy(True)
        self.progress.setRange(0, len(self.samples) * len(candidates))
        self.progress.setValue(0)
        self.status.setText("Starting validation…")
        self.worker.start()

    def set_busy(self, busy):
        for control in (self.browse_dataset, self.split, self.percentage, self.seed, self.selector,
                        self.browse_output, self.run, self.runs, self.refresh, self.view):
            control.setEnabled(not busy)
        self.cancel.setEnabled(busy)

    def update_progress(self, done, total, message):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(message)

    def cancel_validation(self):
        if self.worker is not None:
            self.worker.stop.set()
            self.cancel.setEnabled(False)
            self.status.setText("Cancelling after the current operation; partial results will be saved.")

    def model_finished(self, directory):
        self.active_run = Path(directory).parent
        try:
            self.results.load_run(self.active_run)
            self.workspace.setCurrentWidget(self.results)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Results saved; preview could not open: {exc}")

    def completed(self, directory):
        self.active_run = Path(directory)
        self.refresh_runs()
        self.runs.setCurrentIndex(self.runs.findData(str(self.active_run)))
        self.view_run()
        run = json.loads((self.active_run / "run.json").read_text(encoding="utf-8"))
        self.status.setText(f"Validation {run['status']}. Automatically saved to {directory}")

    def failed(self, message):
        self.status.setText(message)
        self.refresh_runs()

    def worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.set_busy(False)

    def refresh_runs(self):
        previous = self.runs.currentData()
        self.runs.clear()
        for path in sorted((self.output_root / "runs").glob("*/run.json"), reverse=True):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                self.runs.addItem(f"{path.parent.name} • {run['split']} • {run['status']}", str(path.parent))
            except (OSError, ValueError, KeyError):
                continue
        self.runs.setCurrentIndex(max(0, self.runs.findData(previous)))
        self.runs.setToolTip(self.runs.currentText())

    def view_run(self):
        directory = self.runs.currentData()
        if not directory:
            return
        try:
            self.active_run = Path(directory)
            run = self.results.load_run(directory)
            with (self.active_run / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
                samples = [(row["label"], Path(row["path"])) for row in csv.DictReader(stream)]
            self.photos.set_samples(samples, run["class_counts"], f"Saved run {self.active_run.name} • {run['split']}")
            self.workspace.setCurrentWidget(self.results)
            self.status.setText(f"Viewing {directory}")
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Could not open saved run: {exc}")

    def open_run_folder(self):
        folder = self.active_run or self.output_root
        if folder.is_dir():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
