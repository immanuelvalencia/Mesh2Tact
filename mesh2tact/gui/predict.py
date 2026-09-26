"""Compare saved torchvision models on live tactile images or photo folders."""

from __future__ import annotations

from pathlib import Path
import csv
import re
from dataclasses import replace

import numpy as np
from PIL import Image, ImageOps

from .qt import QtCore, QtGui, QtWidgets
from ..prediction import ModelCandidate, SUPPORTED_ARCHITECTURES, discover_models, predict_classifier
from ..outputs import DATA_ROOT
from .model_types import ImageTypeDelegate, model_type_store


def model_filter_metadata(checkpoint):
    """Read naming hints without opening potentially large checkpoint files."""
    architecture = "Unknown"
    # Prefer the filename, then the nearest run folder. Match whole names so
    # resnet18 does not also match an unrelated resnet180 filename.
    for part in (checkpoint.stem, *(parent.name for parent in checkpoint.parents)):
        for name in SUPPORTED_ARCHITECTURES:
            if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", part.lower()):
                architecture = name
                break
        if architecture != "Unknown":
            break
    tokens = re.split(r"[^a-z0-9]+", checkpoint.stem.lower())
    kind = "Best" if "best" in tokens else "Last" if "last" in tokens else "Other"
    return architecture, kind


def style_button(button, color):
    """Keep action colors consistent with the main workspace."""
    button.setStyleSheet(f"""
        QPushButton, QToolButton {{
            background-color: {color}; color: white;
            border: 2px solid transparent; border-radius: 5px;
            padding: 6px 12px; font-weight: 600;
        }}
        QToolButton {{ padding-right: 26px; }}
        QPushButton:hover, QToolButton:hover {{ border-color: #b9d9ed; }}
        QPushButton:pressed, QToolButton:pressed {{ border-color: #172331; }}
        QPushButton:focus, QToolButton:focus {{ border-color: #f5d45c; }}
        QPushButton:disabled, QToolButton:disabled {{
            background-color: #d8dde2; color: #626b75;
        }}
    """)


class PredictionWorker(QtCore.QThread):
    result_ready = QtCore.Signal(object)
    progress = QtCore.Signal(int, int, str)

    def __init__(self, images, candidates, parent=None):
        super().__init__(parent)
        self.images = images
        self.candidates = candidates

    def run(self):
        results = []
        total = len(self.images) * len(self.candidates)
        for source, image in self.images:
            self.progress.emit(len(results), total, f"Loading {Path(source).name}…")
            try:
                if isinstance(image, Path):
                    with Image.open(image) as photo:
                        image = np.array(ImageOps.exif_transpose(photo).convert("RGB"))
            except Exception as exc:
                results.extend((source, candidate, None, str(exc)) for candidate in self.candidates)
                self.progress.emit(len(results), total, f"Could not load {Path(source).name}")
                continue
            for candidate in self.candidates:
                self.progress.emit(len(results), total, f"Running {candidate.checkpoint.name} on {Path(source).name}…")
                try:
                    results.append((source, candidate, predict_classifier(image, candidate), None))
                except Exception as exc:
                    results.append((source, candidate, None, str(exc)))
                self.progress.emit(len(results), total, f"Processed {len(results)}/{total} predictions")
        self.result_ready.emit(results)


class PhotoPreview(QtWidgets.QLabel):
    def __init__(self):
        super().__init__("Select a photo to preview")
        self.original = QtGui.QPixmap()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

    def show_photo(self, path):
        reader = QtGui.QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(1600, 1600, QtCore.Qt.KeepAspectRatio))
        self.original = QtGui.QPixmap.fromImage(reader.read())
        self.refresh()

    def refresh(self):
        if self.original.isNull():
            self.setText("Preview unavailable")
        else:
            self.setPixmap(self.original.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.original.isNull():
            self.refresh()


class PhotoBrowser(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Photos for prediction — check photos to include; click to preview"))
        self.select_all = QtWidgets.QCheckBox("Select all photos")
        self.select_all.clicked.connect(self.toggle_all)
        layout.addWidget(self.select_all)
        self.summary = QtWidgets.QLabel("No photos loaded")
        layout.addWidget(self.summary)
        self.photos = QtWidgets.QListWidget()
        self.photos.setViewMode(QtWidgets.QListView.IconMode)
        self.photos.setResizeMode(QtWidgets.QListView.Adjust)
        self.photos.setMovement(QtWidgets.QListView.Static)
        self.photos.setIconSize(QtCore.QSize(100, 80))
        self.photos.setGridSize(QtCore.QSize(160, 120))
        self.photos.setWordWrap(True)
        self.photos.itemChanged.connect(self.sync_selection)
        self.photos.currentItemChanged.connect(self.preview_item)
        self.preview = PhotoPreview()
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.photos)
        split.addWidget(self.preview)
        layout.addWidget(split, 1)
        self.caption = QtWidgets.QLabel()
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

    def set_photos(self, paths):
        self.photos.blockSignals(True)
        self.photos.clear()
        self.preview.original = QtGui.QPixmap()
        self.preview.setText("Select a photo to preview")
        self.caption.clear()
        for path in paths:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, path)
            item.setToolTip(str(path))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            reader = QtGui.QImageReader(str(path))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                reader.setScaledSize(size.scaled(100, 80, QtCore.Qt.KeepAspectRatio))
            item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(reader.read())))
            self.photos.addItem(item)
        self.photos.blockSignals(False)
        self.sync_selection()
        if paths:
            self.photos.setCurrentRow(0)

    def selected_paths(self):
        return [self.photos.item(i).data(QtCore.Qt.UserRole) for i in range(self.photos.count())
                if self.photos.item(i).checkState() == QtCore.Qt.Checked]

    def toggle_all(self, checked):
        self.photos.blockSignals(True)
        for i in range(self.photos.count()):
            self.photos.item(i).setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self.photos.blockSignals(False)
        self.sync_selection()

    def sync_selection(self, *_):
        count, total = len(self.selected_paths()), self.photos.count()
        self.select_all.setEnabled(total > 0)
        self.select_all.setCheckState(QtCore.Qt.Unchecked if not count else
                                     QtCore.Qt.Checked if count == total else QtCore.Qt.PartiallyChecked)
        self.summary.setText(f"{count}/{total} photos selected for prediction")

    def preview_item(self, item, *_):
        if item is not None:
            path = item.data(QtCore.Qt.UserRole)
            self.preview.show_photo(path)
            self.caption.setText(str(path))


class PredictPanel(QtWidgets.QWidget):
    """Scan checkpoint folders, select several models and compare predictions."""

    browse_source_changed = QtCore.Signal(bool)

    def __init__(self, window, *, selector_only=False):
        super().__init__()
        self.window = window
        self.selection_purpose = "validation" if selector_only else "prediction"
        self.root: Path | None = None
        self.candidates: list[ModelCandidate] = []
        self.type_store = model_type_store()
        self.type_store.changed.connect(self.refresh_image_type)
        self.worker: PredictionWorker | None = None
        self.image_root: Path | None = None
        self.result_rows = []
        self.photo_browser = PhotoBrowser()
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Use the current tactile RGB image or test photos from a folder. Select a model folder to scan it and every "
            "subfolder for .pth checkpoints and nearby labels.txt, classes.txt, labels.json or classes.json files."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        image_row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.addItems(("Current tactile image", "Browse photos"))
        self.source.currentIndexChanged.connect(self.update_source)
        self.image_folder = QtWidgets.QLineEdit()
        self.image_folder.setReadOnly(True)
        self.image_folder.setPlaceholderText("Choose photos to test (includes subfolders)")
        self.browse_photo = QtWidgets.QPushButton("Browse photo…")
        self.browse_photo.clicked.connect(self.choose_photo)
        self.browse_images = QtWidgets.QPushButton("Browse folder…")
        self.browse_images.clicked.connect(self.choose_image_folder)
        image_row.addWidget(self.source)
        image_row.addWidget(self.image_folder, 1)
        image_row.addWidget(self.browse_photo)
        image_row.addWidget(self.browse_images)
        layout.addLayout(image_row)
        if selector_only:
            intro.hide()
            for control in (self.source, self.image_folder, self.browse_photo, self.browse_images):
                control.hide()
        folder_row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit()
        self.folder.setReadOnly(True)
        self.folder.setPlaceholderText("Choose a folder containing trained models")
        self.browse = QtWidgets.QPushButton("Browse models…")
        self.browse.clicked.connect(self.choose_folder)
        self.rescan = QtWidgets.QPushButton("Rescan")
        self.rescan.clicked.connect(self.scan)
        self.rescan.setEnabled(False)
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.browse)
        folder_row.addWidget(self.rescan)
        layout.addLayout(folder_row)
        filter_row = QtWidgets.QHBoxLayout()
        self.folder_filter = QtWidgets.QComboBox()
        self.folder_filter.addItem("All folders", "")
        self.folder_filter.setToolTip("Filter by folder, including its subfolders")
        self.folder_filter.currentIndexChanged.connect(self.apply_model_filters)
        self.text_filter = QtWidgets.QLineEdit()
        self.text_filter.setPlaceholderText("Filter checkpoints: e.g. resnet tactile")
        self.text_filter.setClearButtonEnabled(True)
        self.text_filter.setToolTip("Case-insensitive path search. Every space-separated word must match.")
        self.text_filter.textChanged.connect(self.apply_model_filters)
        self.clear_filters = QtWidgets.QPushButton("Clear filters")
        self.clear_filters.clicked.connect(self.reset_model_filters)
        filter_row.addWidget(self.folder_filter)
        filter_row.addWidget(self.text_filter, 1)
        filter_row.addWidget(self.clear_filters)
        layout.addLayout(filter_row)
        metadata_row = QtWidgets.QHBoxLayout()
        self.architecture_filter = QtWidgets.QComboBox()
        self.architecture_filter.addItem("All architectures", "")
        self.architecture_filter.setToolTip("Architecture inferred from checkpoint and folder names; weights are not loaded.")
        self.checkpoint_filter = QtWidgets.QComboBox()
        self.checkpoint_filter.addItem("All checkpoints", "")
        for kind in ("Best", "Last", "Other"):
            self.checkpoint_filter.addItem(kind, kind)
        self.checkpoint_filter.setToolTip("Best and Last match words in the checkpoint filename; all other names are Other.")
        self.labels_filter = QtWidgets.QComboBox()
        self.labels_filter.addItem("All label statuses", "")
        self.labels_filter.addItem("Usable labels", "ready")
        self.labels_filter.addItem("Missing / invalid labels", "missing")
        for label, control in (("Architecture", self.architecture_filter),
                               ("Checkpoint", self.checkpoint_filter),
                               ("Labels", self.labels_filter)):
            metadata_row.addWidget(QtWidgets.QLabel(label))
            metadata_row.addWidget(control, 1)
            control.currentIndexChanged.connect(self.apply_model_filters)
        layout.addLayout(metadata_row)
        self.filter_summary = QtWidgets.QLabel()
        layout.addWidget(self.filter_summary)
        self.select_all = QtWidgets.QCheckBox("Select all visible models")
        self.select_all.setToolTip("Only checked, visible models are used. Hidden checks are retained until filters are cleared.")
        self.select_all.clicked.connect(self.toggle_all_models)
        layout.addWidget(self.select_all)
        self.models = QtWidgets.QTreeWidget()
        self.models.setHeaderLabels(("Use", "Checkpoint", "Image type"))
        self.models.setItemDelegate(ImageTypeDelegate(2, self.models))
        self.models.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.models.itemChanged.connect(self.sync_select_all)
        self.models.itemChanged.connect(self.edit_image_type)
        self.models.setRootIsDecorated(False)
        self.models.setAlternatingRowColors(True)
        self.models.setMinimumHeight(180)
        self.models.header().setStretchLastSection(False)
        self.configure_columns(self.models, self.models.header(), "models", (60, 320, 120), layout, version=2)
        layout.addWidget(self.models, 1)
        if selector_only:
            # Reuse the same discovery, filters and selection behavior in Validation.
            self.predict = QtWidgets.QPushButton(self)
            self.predict.hide()
            self.status = QtWidgets.QLabel("Choose a model folder and check the models to evaluate.")
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            style_button(self.browse, "#1769aa")
            for button in (self.rescan, self.clear_filters):
                style_button(button, "#526176")
            return
        self.predict = QtWidgets.QPushButton("Predict current tactile image")
        self.predict.clicked.connect(self.run_prediction)
        self.predict.setEnabled(False)
        layout.addWidget(self.predict)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFormat("%v / %m predictions (%p%)")
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.export = QtWidgets.QPushButton("Export results to CSV…")
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_csv)
        layout.addWidget(self.export)
        self.results = QtWidgets.QTableWidget(0, 6)
        self.results.setHorizontalHeaderLabels(("Image", "Model", "Prediction", "Confidence", "Top 3 / Error", "Image type"))
        self.results.horizontalHeader().moveSection(5, 2)
        self.configure_columns(self.results, self.results.horizontalHeader(), "results", (160, 200, 130, 100, 250, 120), layout, version=2)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results.setMinimumHeight(150)
        layout.addWidget(self.results, 1)
        self.status = QtWidgets.QLabel("Choose a model folder. PyTorch is used from the environment that launches Mesh2Tact.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for button in (self.browse_photo, self.browse_images, self.browse):
            style_button(button, "#1769aa")
        style_button(self.predict, "#287348")
        style_button(self.export, "#14777b")
        for button in (self.rescan, self.clear_filters):
            style_button(button, "#526176")
        self.update_source()

    @staticmethod
    def configure_columns(table, header, key, widths, layout, *, settings_group="PredictPanel", version=1):
        settings = QtCore.QSettings("Mesh2Tact", settings_group)
        state_key = f"columns/{key}/v{version}"
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setSectionsMovable(True)
        for column, width in enumerate(widths):
            header.resizeSection(column, width)
        state = settings.value(state_key)
        if state is not None:
            header.restoreState(state)
        button = QtWidgets.QToolButton()
        button.setText("Model columns" if key == "models" else "Result columns")
        button.setToolTip("Show or hide columns. Drag header edges to resize; drag headers to reorder. Changes are saved automatically.")
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        style_button(button, "#526176")
        button.setMinimumWidth(button.sizeHint().width() + 28)
        menu = QtWidgets.QMenu(button)
        button.setMenu(menu)

        def save(*_):
            settings.setValue(state_key, header.saveState())

        def populate():
            menu.clear()
            visible = sum(not table.isColumnHidden(i) for i in range(header.count()))
            for column in range(header.count()):
                label = table.model().headerData(column, QtCore.Qt.Horizontal)
                action = menu.addAction(str(label))
                action.setCheckable(True)
                action.setChecked(not table.isColumnHidden(column))
                action.setEnabled(table.isColumnHidden(column) or visible > 1)
                def toggle(checked, column=column):
                    table.setColumnHidden(column, not checked)
                    save()
                action.triggered.connect(toggle)
            menu.addSeparator()
            reset = menu.addAction("Reset columns")
            def reset_columns():
                for column, width in enumerate(widths):
                    table.setColumnHidden(column, False)
                    header.moveSection(header.visualIndex(column), column)
                    header.resizeSection(column, width)
                save()
            reset.triggered.connect(reset_columns)

        menu.aboutToShow.connect(populate)
        header.sectionResized.connect(save)
        header.sectionMoved.connect(save)
        layout.addWidget(button)
        return button

    def edit_image_type(self, item, column):
        if column == 2:
            candidate = item.data(0, QtCore.Qt.UserRole)
            if candidate is not None and item.text(2) != candidate.resolved_image_type:
                self.type_store.set(candidate.checkpoint, item.text(2))

    def refresh_image_type(self, path_key, value):
        self.models.blockSignals(True)
        for row in range(self.models.topLevelItemCount()):
            item = self.models.topLevelItem(row)
            candidate = item.data(0, QtCore.Qt.UserRole)
            if self.type_store.path_key(candidate.checkpoint) == path_key:
                item.setData(0, QtCore.Qt.UserRole, replace(candidate, image_type=value))
                item.setText(2, value)
        self.models.blockSignals(False)
        self.candidates = [replace(candidate, image_type=value)
                           if self.type_store.path_key(candidate.checkpoint) == path_key else candidate
                           for candidate in self.candidates]
        if hasattr(self, "results"):
            for row, (source, candidate, prediction, error) in enumerate(self.result_rows):
                if self.type_store.path_key(candidate.checkpoint) == path_key:
                    self.result_rows[row] = (source, replace(candidate, image_type=value), prediction, error)
                    self.results.setItem(row, 5, QtWidgets.QTableWidgetItem(value))

    def update_source(self):
        browsing = bool(self.source.currentIndex())
        for control in (self.image_folder, self.browse_photo, self.browse_images):
            control.setVisible(browsing)
        self.predict.setText("Predict selected photos" if browsing else "Predict current tactile image")
        self.browse_source_changed.emit(browsing)

    def choose_photo(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose photo", str(self.image_root or DATA_ROOT),
            "Photos (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if filename:
            path = Path(filename)
            self.image_root = path.parent
            self.image_folder.setText(filename)
            self.photo_browser.set_photos([path])
            self.source.setCurrentIndex(1)

    def choose_image_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose photo folder", str(self.image_root or DATA_ROOT))
        if folder:
            try:
                extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                paths = sorted(path for path in Path(folder).rglob("*")
                               if path.is_file() and path.suffix.lower() in extensions)
            except OSError as exc:
                self.status.setText(f"Could not read photo folder: {exc}")
                return
            self.image_root = Path(folder)
            self.image_folder.setText(folder)
            self.photo_browser.set_photos(paths)
            self.source.setCurrentIndex(1)
            self.status.setText(f"Loaded {len(paths)} photos. Check the photos to include in prediction.")

    def toggle_all_models(self, checked):
        for row in range(self.models.topLevelItemCount()):
            item = self.models.topLevelItem(row)
            if not item.isHidden() and item.data(0, QtCore.Qt.UserRole).ready:
                item.setCheckState(0, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self.sync_select_all()

    def sync_select_all(self, *_):
        visible = [self.models.topLevelItem(row) for row in range(self.models.topLevelItemCount())
                   if not self.models.topLevelItem(row).isHidden()]
        valid = sum(item.data(0, QtCore.Qt.UserRole).ready for item in visible)
        count = len(self.selected_candidates())
        self.select_all.setEnabled(valid > 0)
        self.select_all.setCheckState(QtCore.Qt.Unchecked if not count else
                                     QtCore.Qt.Checked if count == valid else QtCore.Qt.PartiallyChecked)
        self.filter_summary.setText(f"Showing {len(visible)}/{len(self.candidates)} models; {count} selected for {self.selection_purpose}.")

    def reset_model_filters(self):
        self.folder_filter.setCurrentIndex(0)
        self.architecture_filter.setCurrentIndex(0)
        self.checkpoint_filter.setCurrentIndex(0)
        self.labels_filter.setCurrentIndex(0)
        self.text_filter.clear()

    def apply_model_filters(self, *_):
        folder = self.folder_filter.currentData() or ""
        architecture = self.architecture_filter.currentData()
        kind = self.checkpoint_filter.currentData()
        labels = self.labels_filter.currentData()
        terms = self.text_filter.text().casefold().replace("\\", "/").split()
        for row in range(self.models.topLevelItemCount()):
            item = self.models.topLevelItem(row)
            candidate = item.data(0, QtCore.Qt.UserRole)
            relative = candidate.checkpoint.relative_to(self.root).as_posix()
            in_folder = not folder or relative.startswith(folder + "/")
            model_architecture, model_kind = item.data(1, QtCore.Qt.UserRole)
            matches = (in_folder and all(term in relative.casefold() for term in terms)
                       and (not architecture or architecture == model_architecture)
                       and (not kind or kind == model_kind)
                       and (not labels or candidate.ready == (labels == "ready")))
            item.setHidden(not matches)
        self.sync_select_all()

    def choose_folder(self):
        project_root = Path(__file__).resolve().parents[2]
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose model folder", str(self.root or project_root / "train")
        )
        if folder:
            self.root = Path(folder)
            self.folder.setText(str(self.root))
            self.rescan.setEnabled(True)
            self.scan()

    def scan(self):
        if self.root is None:
            return
        try:
            self.candidates = [replace(candidate, image_type=self.type_store.get(candidate.checkpoint, candidate.resolved_image_type))
                               for candidate in discover_models(self.root)]
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.models.clear()
        for candidate in self.candidates:
            item = QtWidgets.QTreeWidgetItem(("", str(candidate.checkpoint.relative_to(self.root)), candidate.resolved_image_type))
            item.setToolTip(2, "Image type from training metadata, path, or your saved override. Double-click to change it.")
            flags = item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEditable
            if candidate.ready:
                item.setFlags(flags)
                item.setCheckState(0, QtCore.Qt.Unchecked)
                item.setToolTip(1, f"{len(candidate.labels)} labels from {candidate.labels_path}")
            else:
                item.setFlags(flags & ~QtCore.Qt.ItemIsEnabled)
                item.setToolTip(1, "A usable labels.txt/classes.txt/labels.json/classes.json file was not found")
            item.setData(0, QtCore.Qt.UserRole, candidate)
            metadata = model_filter_metadata(candidate.checkpoint)
            item.setData(1, QtCore.Qt.UserRole, metadata)
            item.setToolTip(1, f"{item.toolTip(1)}\nArchitecture (name): {metadata[0]}\nCheckpoint: {metadata[1]}")
            self.models.addTopLevelItem(item)
        valid = sum(candidate.ready for candidate in self.candidates)
        previous = self.folder_filter.currentData()
        folders = set()
        for candidate in self.candidates:
            parts = candidate.checkpoint.relative_to(self.root).parts
            if len(parts) > 1:
                folders.add(parts[0])
        self.folder_filter.blockSignals(True)
        self.folder_filter.clear()
        self.folder_filter.addItem("All folders", "")
        for folder in sorted(folders, key=str.casefold):
            self.folder_filter.addItem(folder, folder)
        self.folder_filter.setCurrentIndex(max(0, self.folder_filter.findData(previous)))
        self.folder_filter.blockSignals(False)
        previous_architecture = self.architecture_filter.currentData()
        architectures = {self.models.topLevelItem(row).data(1, QtCore.Qt.UserRole)[0]
                         for row in range(self.models.topLevelItemCount())}
        self.architecture_filter.blockSignals(True)
        self.architecture_filter.clear()
        self.architecture_filter.addItem("All architectures", "")
        for architecture in sorted(architectures, key=str.casefold):
            self.architecture_filter.addItem(architecture, architecture)
        self.architecture_filter.setCurrentIndex(max(0, self.architecture_filter.findData(previous_architecture)))
        self.architecture_filter.blockSignals(False)
        self.apply_model_filters()
        self.predict.setEnabled(valid > 0)
        self.status.setText(f"Found {len(self.candidates)} checkpoint(s); {valid} have usable labels. Select one or more to compare.")

    def selected_candidates(self):
        selected = []
        for row in range(self.models.topLevelItemCount()):
            item = self.models.topLevelItem(row)
            if not item.isHidden() and item.checkState(0) == QtCore.Qt.Checked:
                selected.append(item.data(0, QtCore.Qt.UserRole))
        return selected

    def run_prediction(self):
        selected = self.selected_candidates()
        if not selected:
            self.status.setText("Select at least one model with labels.")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        if self.source.currentIndex():
            images = [(str(path), path) for path in self.photo_browser.selected_paths()]
            if not images:
                self.status.setText("Browse a photo or folder, then check at least one photo to predict.")
                return
        elif not self.window._has_object:
            self.status.setText("Load an object first so there is a tactile image to classify.")
            return
        else:
            try:
                images = [("Current tactile image", self.window.current_tactile_image())]
            except Exception as exc:
                self.status.setText(f"Could not render the tactile image: {exc}")
                return
        self.set_busy(True)
        self.progress_bar.setRange(0, len(selected) * len(images))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status.setText(f"Running {len(selected)} model(s) on {len(images)} image(s)…")
        self.worker = PredictionWorker(images, selected, self)
        self.worker.progress.connect(self.update_progress)
        self.worker.result_ready.connect(self.show_results)
        self.worker.finished.connect(lambda: self.set_busy(False))
        self.worker.start()

    def update_progress(self, completed, total, message):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.status.setText(message)

    def show_results(self, rows):
        self.result_rows = rows
        self.export.setEnabled(bool(rows))
        self.results.setRowCount(0)
        succeeded = 0
        for source, candidate, prediction, error in rows:
            row = self.results.rowCount()
            self.results.insertRow(row)
            image_item = QtWidgets.QTableWidgetItem(Path(source).name)
            image_item.setToolTip(source)
            self.results.setItem(row, 0, image_item)
            model_item = QtWidgets.QTableWidgetItem(str(candidate.checkpoint.relative_to(self.root)) if self.root else candidate.checkpoint.name)
            model_item.setToolTip(str(candidate.checkpoint))
            self.results.setItem(row, 1, model_item)
            self.results.setItem(row, 5, QtWidgets.QTableWidgetItem(candidate.resolved_image_type))
            if prediction is None:
                self.results.setItem(row, 2, QtWidgets.QTableWidgetItem("Failed"))
                self.results.setItem(row, 3, QtWidgets.QTableWidgetItem(""))
                self.results.setItem(row, 4, QtWidgets.QTableWidgetItem(error))
                continue
            succeeded += 1
            self.results.setItem(row, 2, QtWidgets.QTableWidgetItem(prediction.label))
            self.results.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{prediction.confidence:.1%}"))
            top = "  |  ".join(f"{label}: {confidence:.1%}" for label, confidence in prediction.top_k)
            self.results.setItem(row, 4, QtWidgets.QTableWidgetItem(top))
        self.status.setText(f"Completed {succeeded}/{len(rows)} predictions successfully.")

    def set_busy(self, busy):
        for control in (self.predict, self.browse, self.rescan, self.source, self.browse_images,
                        self.models, self.select_all, self.folder_filter, self.text_filter, self.clear_filters,
                        self.architecture_filter, self.checkpoint_filter, self.labels_filter,
                        self.browse_photo, self.photo_browser):
            control.setEnabled(not busy)
        if not busy:
            self.rescan.setEnabled(self.root is not None)
            self.sync_select_all()

    def export_csv(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export prediction results", "predictions.csv", "CSV files (*.csv)")
        if not filename:
            return
        if not Path(filename).suffix:
            filename += ".csv"
        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(("Image", "Checkpoint", "Prediction", "Confidence", "Top 3", "Error", "Image type"))
                for source, candidate, prediction, error in self.result_rows:
                    writer.writerow((source, str(candidate.checkpoint), prediction.label if prediction else "",
                                     prediction.confidence if prediction else "",
                                     " | ".join(f"{label}: {confidence:.8g}" for label, confidence in prediction.top_k) if prediction else "",
                                     error or "", candidate.resolved_image_type))
        except OSError as exc:
            self.status.setText(f"Could not export results: {exc}")
            return
        self.status.setText(f"Exported {len(self.result_rows)} prediction(s) to {filename}")
