"""Batch model queue with the same controls as single-object gathering."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import Event

from .qt import QtCore, QtWidgets
from .gather import GatherPanel
from ..batch_gather import gather_batch, model_entries
from ..outputs import model_name


class BatchWorker(QtCore.QThread):
    sample_ready = QtCore.Signal(object)
    result_ready = QtCore.Signal(object)

    def __init__(self, sim, paths, directory, settings, units, scale, parent=None):
        super().__init__(parent)
        self.arguments = sim, paths, directory, settings, units, scale
        self.stop_event = Event()

    def run(self):
        try:
            result = gather_batch(*self.arguments, cancelled=self.stop_event.is_set,
                                  progress=self.sample_ready.emit)
        except Exception as exc:
            result = dict(status='failed', completed=0, completed_models=0, results=[], error=str(exc))
        self.result_ready.emit(result)


class ModelChecklist(QtWidgets.QListWidget):
    """Check state controls collection; the current row controls preview/delete."""
    def addItem(self, label, path, checked=True):
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, path)
        item.setToolTip(path)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        super().addItem(item)

    def itemData(self, index):
        return self.item(index).data(QtCore.Qt.UserRole)

    def currentData(self):
        return self.currentItem().data(QtCore.Qt.UserRole) if self.currentItem() else None

    def setCurrentIndex(self, index):
        self.setCurrentRow(index)

    def findData(self, value):
        return next((i for i in range(self.count()) if self.itemData(i) == value), -1)


class BatchGatherPanel(GatherPanel):
    def __init__(self, window):
        self.worker = None
        super().__init__(window)
        self.capture.hide()
        self.start.clicked.disconnect()
        self.stop.clicked.disconnect()
        self.start.setText('Start batch auto gather')
        self.start.clicked.connect(self.start_batch)
        self.stop.clicked.connect(self.stop_batch)
        self.queue_controls = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(self.queue_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel('Check the models to gather in list order. Samples below are per checked model. Select a row and click Preview selected to view it. Labels use lowercase filenames.')
        hint.setWordWrap(True)
        controls.addWidget(hint)
        self.selector = ModelChecklist()
        self.selector.setMaximumHeight(160)
        controls.addWidget(self.selector)
        row = QtWidgets.QHBoxLayout()
        self.add = QtWidgets.QPushButton('Add 3D models…')
        self.delete = QtWidgets.QPushButton('Delete selected')
        self.preview = QtWidgets.QPushButton('Preview selected')
        self.copy_settings = QtWidgets.QPushButton('Copy Data gathering settings')
        for button in (self.add, self.delete, self.preview):
            row.addWidget(button)
        controls.addLayout(row)
        controls.addWidget(self.copy_settings)
        self.layout().insertWidget(0, self.queue_controls)
        self.add.clicked.connect(self.add_dialog)
        self.delete.clicked.connect(self.delete_selected)
        self.preview.clicked.connect(self.preview_selected)
        self.copy_settings.clicked.connect(self.copy_from_single)
        self.selector.currentRowChanged.connect(self.select_model)
        self.save_layout.setCurrentIndex(self.save_layout.findData('object_label'))
        self.save_layout.setEnabled(False)
        self.object_label.setReadOnly(True)
        self.object_label.setPlaceholderText('Automatic lowercase model filename')
        self.delete.setEnabled(False)
        self.preview.setEnabled(False)
        self.update_destination()
        self.status.setText('Add models to start. Uses current sensor, appearance, import units and object quality.')

    def update_destination(self):
        super().update_destination()
        if not self.object_label.text().strip():
            self.destination.setText('Add a model to preview its automatic lowercase label and indexed destination.')

    def paths(self):
        return [self.selector.itemData(i) for i in range(self.selector.count())]

    def checked_paths(self):
        return [self.selector.itemData(i) for i in range(self.selector.count())
                if self.selector.item(i).checkState() == QtCore.Qt.Checked]

    def settings(self):
        settings = super().settings()
        settings.save_layout = 'object_label'
        path = self.selector.currentData() if hasattr(self, 'selector') else None
        settings.object_label = model_name(path).lower() if path else 'batch'
        return settings

    def add_paths(self, paths):
        previous = set(self.paths())
        checked = set(self.checked_paths())
        selected = self.selector.currentData()
        entries = model_entries(self.paths() + list(paths))
        self.selector.blockSignals(True)
        self.selector.clear()
        for path, label in entries:
            self.selector.addItem(label, path, checked=path not in previous or path in checked)
        self.selector.setCurrentRow(max(0, self.selector.findData(selected)))
        self.selector.blockSignals(False)
        self.select_model()

    def add_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add 3D models',
            str(Path(__file__).resolve().parents[2] / 'assets'), 'Meshes (*.stl *.obj *.ply *.glb)')
        if paths:
            try:
                self.add_paths(paths)
            except Exception as exc:
                self.status.setText(str(exc))

    def delete_selected(self):
        # Remove only from the queue; never delete the CAD file.
        self.selector.takeItem(self.selector.currentRow())
        self.select_model()

    def select_model(self, *_):
        path = self.selector.currentData()
        self.delete.setEnabled(bool(path))
        self.preview.setEnabled(bool(path))
        self.object_label.setText(model_name(path).lower() if path else '')

    def preview_selected(self):
        path = self.selector.currentData()
        if not path or self.worker is not None or not hasattr(self.window, 'plot'):
            return
        try:
            window = self.window
            snapshot = deepcopy(window.sim)
            snapshot.load(path, units=window.units.currentText(), scale=window.scale.value())
            snapshot.set_quality(window.quality.currentData(), window.mesh_smoothing.value())
            snapshot.set_base_rotation(window.base_flip.currentData())
            window.sim = snapshot
            window._has_object = True
            window.model_label.setText(Path(path).name)
            window._reset_camera = True
            window.refresh()
            self.status.setText(f'{self.selector.count()} models queued. Selected: {path}')
        except Exception as exc:
            self.status.setText(f'Cannot preview {path}: {exc}')

    def copy_from_single(self):
        self.restore_settings(asdict(self.window.gather_panel.settings()),
                              self.window.gather_panel.range_mode.currentData())
        self.save_layout.setCurrentIndex(self.save_layout.findData('object_label'))
        self.select_model()

    def data(self):
        values = asdict(self.settings())
        values['object_label'] = values['object_label'] or 'batch'
        return dict(paths=self.paths(), settings=values, range_mode=self.range_mode.currentData(),
                    checked_paths=self.checked_paths(),
                    selected=self.selector.currentData(), position_overlay=self.position_overlay.isChecked(),
                    rotation_overlay=self.rotation_overlay.isChecked())

    def restore(self, data):
        self.restore_settings(data['settings'], data.get('range_mode', 'exact'))
        self.save_layout.setCurrentIndex(self.save_layout.findData('object_label'))
        self.selector.blockSignals(True)
        self.selector.clear()
        checked = set(data.get('checked_paths', data['paths']))
        for path in data['paths']:
            self.selector.addItem(Path(path).stem.lower(), path, checked=path in checked)
        selected = self.selector.findData(data.get('selected'))
        if selected >= 0:
            self.selector.setCurrentIndex(selected)
        elif self.selector.count():
            self.selector.setCurrentRow(0)
        self.selector.blockSignals(False)
        # Keep missing entries visible so the user can delete or replace them.
        self.object_label.setText(model_name(self.selector.currentData()).lower() if self.selector.count() else '')
        self.position_overlay.setChecked(data.get('position_overlay', True))
        self.rotation_overlay.setChecked(data.get('rotation_overlay', False))
        self.select_model()

    def start_batch(self):
        window = self.window
        if self.worker is not None or window.gather_worker is not None:
            self.status.setText('Wait for the current gathering run to finish.')
            return
        if window.calibration.alignment_active:
            self.status.setText('Set or cancel calibration alignment before gathering.')
            return
        try:
            paths = self.checked_paths()
            entries = model_entries(paths)
            if not entries:
                raise ValueError('Check at least one 3D model')
            settings = self.settings()
            settings.object_label = entries[0][1]
            settings.validate()
            window.apply_controls()
            window.gather_autosave_path.parent.mkdir(parents=True, exist_ok=True)
            window.save_preset(window.gather_autosave_path)
            snapshot = deepcopy(window.sim)
            snapshot.quality_level = window.quality.currentData()
            snapshot.smoothing_iterations = window.mesh_smoothing.value()
            snapshot.cfg.camera.width, snapshot.cfg.camera.height = window.capture_resolution
            self.worker = BatchWorker(snapshot, paths, str(window.data_root), settings,
                                      window.units.currentText(), window.scale.value(), self)
            self.worker.sample_ready.connect(self.batch_progress)
            self.worker.result_ready.connect(self.batch_result)
            self.worker.finished.connect(self.batch_finished)
            self.progress.setRange(0, len(entries)*settings.count)
            self.progress.setValue(0)
            self._samples_per_model = settings.count
            self.set_running(True)
            self.queue_controls.setEnabled(False)
            window.gather_panel.setEnabled(False)
            self.status.setText('Starting batch…')
            self.worker.start()
        except Exception as exc:
            if self.worker is not None and not self.worker.isRunning():
                self.worker.deleteLater()
                self.worker = None
                self.set_running(False)
                self.queue_controls.setEnabled(True)
                window.gather_panel.setEnabled(window._has_object)
            self.status.setText(f'Cannot start batch: {exc}')

    def stop_batch(self):
        if self.worker is not None:
            self.worker.stop_event.set()
            self.stop.setEnabled(False)
            self.status.setText('Stopping after the current sample. Completed captures are kept.')

    def batch_progress(self, record):
        prefix = f"Model {record['model_number']}/{record['model_count']}: {record['label']}"
        if record.get('loading'):
            self.status.setText(prefix + ' — loading…')
        elif 'model_result' in record:
            result = record['model_result']
            self.status.setText(prefix + f" — {result['status']}: {result['completed']} saved. " + result.get('error', ''))
        else:
            completed = record.get('completed', 0)
            self.progress.setValue((record['model_number']-1)*self._samples_per_model + completed)
            self.status.setText(prefix + f" — {completed}/{self._samples_per_model} saved; {record.get('skipped', 0)} rejected.")

    def batch_result(self, result):
        self.last_result = result
        error = result.get('error', '') or next((r.get('error', '') for r in result['results'] if r['status'] != 'complete'), '')
        self.status.setText(f"{result['status'].capitalize()}: {result['completed_models']} models complete, {result['completed']} samples saved.\n{error}")

    def batch_finished(self):
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        self.set_running(False)
        self.queue_controls.setEnabled(True)
        self.window.gather_panel.setEnabled(self.window._has_object)
        if self.window._close_after_gather:
            self.window.close()
