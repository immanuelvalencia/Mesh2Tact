"""Predict-tab controls for comparing saved torchvision models on the live tactile RGB image."""

from __future__ import annotations

from pathlib import Path

from .qt import QtCore, QtWidgets
from ..prediction import ModelCandidate, discover_models, predict_classifier


class PredictionWorker(QtCore.QThread):
    result_ready = QtCore.Signal(object)

    def __init__(self, image, candidates, parent=None):
        super().__init__(parent)
        self.image = image
        self.candidates = candidates

    def run(self):
        results = []
        for candidate in self.candidates:
            try:
                results.append((candidate, predict_classifier(self.image, candidate), None))
            except Exception as exc:
                results.append((candidate, None, str(exc)))
        self.result_ready.emit(results)


class PredictPanel(QtWidgets.QWidget):
    """Scan checkpoint folders, select several models and compare predictions."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.root: Path | None = None
        self.candidates: list[ModelCandidate] = []
        self.worker: PredictionWorker | None = None
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Use the current tactile RGB image from the main workspace. Select a folder to scan it and every "
            "subfolder for .pth checkpoints and nearby labels.txt, classes.txt, labels.json or classes.json files."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
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
        self.models = QtWidgets.QTreeWidget()
        self.models.setHeaderLabels(("Use", "Checkpoint", "Labels"))
        self.models.setRootIsDecorated(False)
        self.models.setAlternatingRowColors(True)
        self.models.setMinimumHeight(180)
        self.models.header().setStretchLastSection(False)
        self.models.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.models.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(self.models, 1)
        self.predict = QtWidgets.QPushButton("Predict current tactile image")
        self.predict.clicked.connect(self.run_prediction)
        self.predict.setEnabled(False)
        layout.addWidget(self.predict)
        self.results = QtWidgets.QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(("Model", "Prediction", "Confidence", "Top 3"))
        self.results.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.results.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.results.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.results.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results.setMinimumHeight(150)
        layout.addWidget(self.results, 1)
        self.status = QtWidgets.QLabel("Choose a model folder. PyTorch is used from the environment that launches Mesh2Tact.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def choose_folder(self):
        project_root = Path(__file__).resolve().parents[2]
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose model folder", str(self.root or project_root)
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
            self.candidates = discover_models(self.root)
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.models.clear()
        for candidate in self.candidates:
            item = QtWidgets.QTreeWidgetItem(("", str(candidate.checkpoint.relative_to(self.root)), candidate.relative_label_name))
            flags = item.flags() | QtCore.Qt.ItemIsUserCheckable
            if candidate.ready:
                item.setFlags(flags)
                item.setCheckState(0, QtCore.Qt.Unchecked)
                item.setToolTip(2, f"{len(candidate.labels)} labels from {candidate.labels_path}")
            else:
                item.setFlags(flags & ~QtCore.Qt.ItemIsEnabled)
                item.setToolTip(2, "A usable labels.txt/classes.txt/labels.json/classes.json file was not found")
            item.setData(0, QtCore.Qt.UserRole, candidate)
            self.models.addTopLevelItem(item)
        valid = sum(candidate.ready for candidate in self.candidates)
        self.predict.setEnabled(valid > 0)
        self.status.setText(f"Found {len(self.candidates)} checkpoint(s); {valid} have usable labels. Select one or more to compare.")

    def selected_candidates(self):
        selected = []
        for row in range(self.models.topLevelItemCount()):
            item = self.models.topLevelItem(row)
            if item.checkState(0) == QtCore.Qt.Checked:
                selected.append(item.data(0, QtCore.Qt.UserRole))
        return selected

    def run_prediction(self):
        selected = self.selected_candidates()
        if not selected:
            self.status.setText("Select at least one model with labels.")
            return
        if not self.window._has_object:
            self.status.setText("Load an object first so there is a tactile image to classify.")
            return
        try:
            image = self.window.current_tactile_image()
        except Exception as exc:
            self.status.setText(f"Could not render the tactile image: {exc}")
            return
        self.predict.setEnabled(False)
        self.status.setText(f"Running {len(selected)} model(s)…")
        self.worker = PredictionWorker(image, selected, self)
        self.worker.result_ready.connect(self.show_results)
        self.worker.finished.connect(lambda: self.predict.setEnabled(True))
        self.worker.start()

    def show_results(self, rows):
        self.results.setRowCount(0)
        succeeded = 0
        for candidate, prediction, error in rows:
            row = self.results.rowCount()
            self.results.insertRow(row)
            self.results.setItem(row, 0, QtWidgets.QTableWidgetItem(candidate.checkpoint.name))
            if prediction is None:
                self.results.setItem(row, 1, QtWidgets.QTableWidgetItem("Failed"))
                self.results.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
                self.results.setItem(row, 3, QtWidgets.QTableWidgetItem(error))
                continue
            succeeded += 1
            self.results.setItem(row, 1, QtWidgets.QTableWidgetItem(prediction.label))
            self.results.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{prediction.confidence:.1%}"))
            top = "  |  ".join(f"{label}: {confidence:.1%}" for label, confidence in prediction.top_k)
            self.results.setItem(row, 3, QtWidgets.QTableWidgetItem(top))
        self.status.setText(f"Completed {succeeded}/{len(rows)} model(s). Results use the current tactile RGB image.")
