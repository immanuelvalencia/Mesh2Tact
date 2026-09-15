"""Calibration workspace: independent references, alignment, fitting and review."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_erosion

from .qt import QtCore, QtGui, QtWidgets
from ..calibration import (appearance, make_sim, render_float, fit_references,
                           save_session, load_session, Cancelled)

SESSION_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'calibration_sessions'


class Picture(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 360)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = self.scene().addPixmap(QtGui.QPixmap())
        self.setBackgroundBrush(QtGui.QColor('#15212b'))
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.fitted = True
        self.image = None

    def show_array(self, rgb):
        a = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
        self.image = QtGui.QPixmap.fromImage(QtGui.QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QtGui.QImage.Format_RGB888).copy())
        changed_size = self.item.pixmap().size() != self.image.size()
        self.item.setPixmap(self.image)
        self.setSceneRect(self.item.boundingRect())
        if changed_size or self.fitted:
            self.fit_image()

    def clear(self):
        self.image = None
        self.item.setPixmap(QtGui.QPixmap())
        self.resetTransform()
        self.fitted = True

    def fit_image(self):
        self.fitted = True
        if self.image is not None:
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def zoom(self, factor):
        if self.image is None:
            return
        scale = self.transform().m11()
        target = min(32., max(.05, scale*factor))
        self.scale(target/scale, target/scale)
        self.fitted = False

    def wheelEvent(self, event):
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1/1.2)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitted:
            self.fit_image()


class FitWorker(QtCore.QThread):
    message = QtCore.Signal(str)
    result = QtCore.Signal(object)
    failure = QtCore.Signal(str)

    def __init__(self, settings, refs, options, parent):
        super().__init__(parent)
        self.settings, self.refs, self.options = deepcopy(settings), deepcopy(refs), options
        self.stop = Event()

    def run(self):
        try:
            result = fit_references(self.settings, self.refs, cancelled=self.stop.is_set,
                                   progress=self.message.emit, **self.options)
            self.message.emit('Saving calibration run, metrics, images and sensor config…')
            try:
                from ..calibration_archive import save_run
                result['run_directory'] = str(save_run(SESSION_DIR/'runs', self.settings, self.refs, result))
            except Exception as exc:
                result['autosave_error'] = str(exc)
            self.result.emit(result)
        except Cancelled:
            self.failure.emit('Cancelled. The original configuration is unchanged.')
        except Exception as exc:
            self.failure.emit(str(exc))


class PreviewWorker(QtCore.QThread):
    ready = QtCore.Signal(object)
    failure = QtCore.Signal(str)

    def __init__(self, reference, settings, fitted, size, effects, cache, mesh_key, parent):
        super().__init__(parent)
        self.reference, self.settings = deepcopy(reference), deepcopy(settings)
        self.fitted, self.size, self.effects = deepcopy(fitted), size, effects
        self.cache, self.mesh_key = cache, mesh_key

    def run(self):
        try:
            settings = self.fitted or self.settings
            key = (self.mesh_key, self.size, json.dumps(settings, sort_keys=True))
            sim = self.cache.get(key)
            if sim is None:
                sim = make_sim(self.reference, settings, self.size)
                sim.effects.blur_px *= self.size[0]/settings['sensor']['camera']['width']
                if len(self.cache) >= 2:
                    self.cache.clear()
                self.cache[key] = sim
            sim.rotation = tuple(self.reference['rotation'])
            sim.offset = tuple(self.reference['offset'])
            sim.cut_depth = -1 if self.reference['blank'] else self.reference['cut_depth']
            rgb = render_float(sim, stochastic=self.effects)
            mask = sim.raw_depth > 0
            real = np.asarray(Image.fromarray(self.reference['photo']).resize(
                self.size, Image.Resampling.LANCZOS))/255
            self.ready.emit((real, rgb, mask))
        except Exception as exc:
            self.failure.emit(str(exc))


class CalibrationPanel(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.window, self.references, self.settings = window, [], None
        self.worker = self.result = None
        self.preview_cache = None
        self.loading = False
        self.alignment_active = False
        self.preview_worker = None
        self.preview_pending = False
        self.preview_revision = 0
        self.alignment_snapshot = None
        self.render_cache = {}
        self.live_pose_key = None
        self.preview_detail = True
        self.preview_geometry = None
        self.geometry_generation = 0
        self.session_path = None
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel('Load STL → Upload reference → Name the shape → Align and save reference. Repeat for each shape, then calibrate all saved references together. Use No shape for an empty sensor image.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.editor = QtWidgets.QWidget()
        editor_layout = self.editor_layout = QtWidgets.QVBoxLayout(self.editor)
        layout.addWidget(self.editor)
        self.session_controls = QtWidgets.QWidget()
        buttons = QtWidgets.QGridLayout(self.session_controls)
        for index, (title, action) in enumerate([('New collection', self.new), ('Open collection…', self.open), ('Save collection as…', self.save), ('Upload reference…', self.add), ('Remove', self.remove)]):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(action)
            buttons.addWidget(button, index//2, index%2)
        editor_layout.addWidget(self.session_controls)
        sync = self.sync_button = QtWidgets.QPushButton('Use current sensor settings for this session')
        sync.clicked.connect(self.sync_settings)
        editor_layout.addWidget(sync)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(115)
        self.list.currentRowChanged.connect(self.select)
        editor_layout.addWidget(self.list)
        self.reference_count = QtWidgets.QLabel('0 saved references')
        editor_layout.addWidget(self.reference_count)
        editor_layout.addWidget(QtWidgets.QLabel('Shape shown in the reference'))
        self.shape_name = QtWidgets.QLineEdit()
        self.shape_name.setPlaceholderText('For example: cube, cylinder, or object name')
        self.shape_name.editingFinished.connect(self.changed)
        editor_layout.addWidget(self.shape_name)
        self.shape_label = QtWidgets.QLabel('No reference selected')
        self.shape_label.setWordWrap(True)
        editor_layout.addWidget(self.shape_label)
        self.align_button = QtWidgets.QPushButton('Align reference')
        self.align_button.clicked.connect(self.reference_action)
        editor_layout.addWidget(self.align_button)
        row = QtWidgets.QVBoxLayout()
        self.blank = QtWidgets.QCheckBox('No shape (empty sensor reference)')
        self.validation = QtWidgets.QCheckBox('Validation only')
        self.lock = QtWidgets.QCheckBox('Lock alignment')
        for widget in (self.blank, self.validation, self.lock):
            row.addWidget(widget)
            widget.toggled.connect(self.changed)
        self.lock.hide()  # Saving a reference locks its mesh and pose.
        editor_layout.addLayout(row)
        repeat_row = QtWidgets.QHBoxLayout()
        repeat_row.addWidget(QtWidgets.QLabel('Repeat group (optional)'))
        self.repeat_group = QtWidgets.QLineEdit()
        self.repeat_group.setPlaceholderText('Same name for repeated photos of an unchanged contact')
        self.repeat_group.editingFinished.connect(self.changed)
        repeat_row.addWidget(self.repeat_group)
        editor_layout.addLayout(repeat_row)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Opacity overlay', 'Contact outline', 'Absolute error ×4', 'Simulation only'])
        self.mode.currentIndexChanged.connect(self.preview)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setToolTip('Simulation opacity in the overlay')
        self.opacity.valueChanged.connect(self.preview)
        self.fit_options = QtWidgets.QWidget()
        row = QtWidgets.QGridLayout(self.fit_options)
        self.background = QtWidgets.QCheckBox('Fit background')
        self.lighting = QtWidgets.QCheckBox('Fit light colour / intensity')
        self.directions = QtWidgets.QCheckBox('Fit light directions / edge geometry')
        self.filters = QtWidgets.QCheckBox('Fit brightness, blur, vignette, contrast and gamma')
        self.noise = QtWidgets.QCheckBox('Estimate read, shot and speckle noise')
        self.texture = QtWidgets.QCheckBox('Fit texture strength and size')
        self.blur = QtWidgets.QCheckBox('Refine blur')
        self.background.setChecked(True)
        self.lighting.setChecked(True)
        for widget in (self.directions, self.filters, self.noise, self.texture):
            widget.setChecked(True)
        for index, widget in enumerate((self.background, self.lighting, self.directions, self.filters, self.noise, self.texture, self.blur)):
            row.addWidget(widget, index, 0, 1, 2)
        self.lighting.toggled.connect(self.directions.setEnabled)
        self.filters.toggled.connect(lambda value: self.blur.setEnabled(not value))
        self.blur.setEnabled(False)
        self.search_quality = QtWidgets.QComboBox()
        self.search_quality.addItems(['Standard search', 'Thorough search (slower)'])
        row.addWidget(self.search_quality, 7, 0, 1, 2)
        row.addWidget(QtWidgets.QLabel('Calibration search seed'), 8, 0)
        self.search_seed = QtWidgets.QSpinBox()
        self.search_seed.setRange(0, 1000000)
        row.addWidget(self.search_seed, 8, 1)
        self.algorithm = QtWidgets.QComboBox()
        self.algorithm.addItem('Current — robust least squares', 'least_squares')
        self.algorithm.addItem('AI-assisted — learned surrogate (experimental)', 'ai_surrogate')
        self.algorithm.setToolTip('AI-assisted fitting learns from local trial renders, then refines the best settings. It may take longer; no model download or cloud service is used.')
        row.addWidget(QtWidgets.QLabel('Fitting algorithm'), 9, 0)
        row.addWidget(self.algorithm, 10, 0, 1, 2)
        layout.addWidget(self.fit_options)
        self.show_effects = QtWidgets.QCheckBox('Show fitted noise and texture in preview (fixed seed)')
        self.show_effects.setChecked(True)
        self.show_effects.toggled.connect(self.preview)
        layout.addWidget(self.show_effects)
        note = QtWidgets.QLabel('Pose, contact softness, glow positions and exposure stay fixed. Noise is estimated from native-pixel statistics; repeated unchanged contacts improve reliability. Crop photos to the same sensing area. Noise/texture fitting enables image effects; pixel error metrics remain deterministic.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QVBoxLayout()
        self.run_button = QtWidgets.QPushButton('Calibrate all saved references')
        self.run_button.clicked.connect(self.run_fit)
        self.cancel = QtWidgets.QPushButton('Cancel fitting')
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(lambda: self.worker.stop.set() if self.worker else None)
        self.save_result = QtWidgets.QPushButton('Save fitted sensor config…')
        self.save_result.setEnabled(False)
        self.save_result.clicked.connect(self.export)
        for button in (self.run_button, self.cancel, self.save_result):
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel('Start a session using the current sensor settings.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(160)
        self.report.setPlaceholderText('After fitting: before/after errors for every reference and parameter changes.')
        layout.addWidget(self.report)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.preview)
        self.settle_timer = QtCore.QTimer(self)
        self.settle_timer.setSingleShot(True)
        self.settle_timer.timeout.connect(self.refine_preview)

    def current(self):
        i = self.list.currentRow()
        return self.references[i] if 0 <= i < len(self.references) else None

    def invalidate(self):
        self.preview_revision += 1
        self.preview_cache = None
        self.result = None
        self.save_result.setEnabled(False)
        self.report.clear()
        self.update_reference_list()

    def update_reference_list(self):
        for i, ref in enumerate(self.references):
            item = self.list.item(i)
            if item is not None:
                shape = 'No shape' if ref['blank'] else ref.get('shape_name', ref.get('shape', 'Object'))
                item.setText(f"{'Saved' if ref['locked'] else 'Draft'} | {shape} | {ref['name']}")
        saved = sum(r['locked'] for r in self.references)
        self.reference_count.setText(f'{saved} saved / {len(self.references)} references')

    def checkpoint(self):
        if self.session_path is None:
            self.session_path = SESSION_DIR / ('references-'+uuid4().hex[:12]+'.npz')
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        save_session(self.session_path, self.settings, self.references, self.result)

    def reference_action(self):
        if self.current() is None:
            self.status.setText('Upload a reference image first.')
        elif self.blank.isChecked():
            self.save_empty_reference()
        else:
            self.begin_alignment()

    def save_empty_reference(self):
        ref = self.current()
        if ref is None or not ref['blank'] or self.worker is not None:
            return
        ref['locked'] = True
        self.invalidate()
        try:
            self.checkpoint()
        except Exception as exc:
            ref['locked'] = False
            self.update_reference_list()
            self.status.setText(f'Reference could not be saved: {exc}')
            return
        self.select()
        self.status.setText(f'Empty reference saved to {self.session_path.name}. Upload the next reference or calibrate the collection.')

    def new(self):
        self.end_alignment()
        self.session_path = None
        self.window.apply_controls()
        self.settings = appearance(self.window.sim)
        self.settings['sensor']['camera']['width'], self.settings['sensor']['camera']['height'] = self.window.capture_resolution
        self.references = []
        self.list.clear()
        self.invalidate()
        self.status.setText('Session uses a snapshot of the current sensor settings. Add reference photos.')

    def sync_settings(self):
        self.window.apply_controls()
        updated = appearance(self.window.sim)
        updated['sensor']['camera']['width'], updated['sensor']['camera']['height'] = self.window.capture_resolution
        changed_geometry = self.settings is not None and any(
            self.settings['sensor']['gel'][key] != updated['sensor']['gel'][key]
            for key in ('size_x', 'size_y'))
        self.settings = updated
        if changed_geometry:
            for ref in self.references:
                ref['locked'] = False
        self.invalidate()
        self.select()
        self.status.setText('Starting sensor settings updated; reference shapes and poses retained.' +
                            (' Sensor area changed: recheck and lock alignments.' if changed_geometry else ''))

    def add(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Reference photos', '', 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not paths:
            return
        if self.settings is None:
            self.new()
        self.window.apply_controls()
        try:
            for path in paths:
                with Image.open(path) as image:
                    photo = np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
                sim = self.window.sim
                self.references.append(dict(name=Path(path).name, photo=photo, vertices=np.asarray(sim.mesh.vertices).copy(),
                    faces=np.asarray(sim.mesh.faces).copy(), shape=sim.source, rotation=list(sim.rotation),
                    offset=list(sim.offset), cut_depth=sim.cut_depth, blank=False, validation=False, locked=False,
                    shape_name=sim.source.split(':')[-1] if sim.source.startswith('primitive:') else Path(sim.source).stem))
                self.list.addItem(Path(path).name)
            self.invalidate()
            self.list.setCurrentRow(len(self.references)-1)
        except Exception as exc:
            self.status.setText(str(exc))

    def remove(self):
        self.end_alignment()
        i = self.list.currentRow()
        if i >= 0:
            self.references.pop(i)
            self.list.takeItem(i)
            self.invalidate()
            self.select()
            if self.session_path is not None:
                try:
                    self.checkpoint()
                except Exception as exc:
                    self.status.setText(f'Collection update could not be saved: {exc}')

    def select(self, *_):
        ref = self.current()
        if ref is None:
            self.shape_label.setText('No reference selected')
            if hasattr(self.window, 'alignment_reference'):
                self.window.alignment_reference.clear()
                self.window.alignment_overlay.clear()
            self.end_alignment()
            return
        self.loading = True
        self.blank.setChecked(ref['blank'])
        self.validation.setChecked(ref['validation'])
        self.lock.setChecked(ref['locked'])
        self.repeat_group.setText(ref.get('repeat_group', ''))
        self.shape_name.setText(ref.get('shape_name', Path(ref.get('shape', 'Object')).stem))
        self.shape_label.setText(f"Shape: {ref.get('shape', 'mesh')}; {len(ref['faces']):,} faces")
        self.window.alignment_title.setText('Calibration: '+ref['name'])
        self.window.alignment_reference.show_array(ref['photo'])
        self.loading = False
        self.lock_controls()
        if self.alignment_active:
            self.load_alignment_reference()
        self.preview()

    def begin_alignment(self):
        ref = self.current()
        if ref is None or self.settings is None:
            self.status.setText('Add a reference photo and select it first.')
            return
        if self.worker is not None:
            return
        if ref['locked']:
            self.lock.setChecked(False)
        w = self.window
        if not self.alignment_active:
            w.timer.stop()
            w.apply_controls()
            self.alignment_snapshot = (deepcopy(w.sim), w._has_object,
                {key: getattr(w, key).value() for key in
                 ('rx', 'ry', 'rz', 'x', 'y', 'z', 'max_penetration', 'mesh_smoothing', 'scale')},
                 w.model_label.text(), w.quality.currentIndex(), w.units.currentIndex())
            self.alignment_active = True
        for widget in (self.session_controls, self.sync_button, self.list, self.blank,
                       self.lock, self.validation, self.repeat_group, self.shape_name, self.align_button):
            widget.setEnabled(False)
        self.run_button.setEnabled(False)
        self.fit_options.setEnabled(False)
        w.alignment_hint.setText('Match the reference using the main controls and 3D handles, then Save reference to keep the STL and pose.')
        w.cancel_alignment_button.setEnabled(True)
        w.settings_window.hide()
        w.settings_action.setEnabled(False)
        w.set_alignment_button.setEnabled(True)
        w.right_stack.setCurrentIndex(1)
        w.tabs.setCurrentIndex(0)
        self.load_alignment_reference()
        w.show()
        w.raise_()

    def load_alignment_reference(self):
        import trimesh
        ref, w = self.current(), self.window
        if ref is None:
            self.end_alignment()
            return
        w.sim.set_mesh(trimesh.Trimesh(ref['vertices'], ref['faces'], process=False), ref.get('shape', ref['name']))
        self.sync_mesh_controls()
        w.scale.blockSignals(True)
        w.scale.setValue(1.)
        w.scale.blockSignals(False)
        w._has_object = not ref['blank']
        for key, value in zip(('rx', 'ry', 'rz', 'x', 'y'),
                list(ref['rotation']) + [v*1000 for v in ref['offset']]):
            control = getattr(w, key)
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        w.max_penetration.setValue(max(w.max_penetration.value(), ref['cut_depth']*1000))
        w.sync_slider(ref['cut_depth']*1000)
        w.model_label.setText(f"Calibration: {ref.get('shape', ref['name'])}")
        w.alignment_title.setText('Calibration: '+ref['name'])
        w.alignment_reference.show_array(ref['photo'])
        w._scene_key = None
        w._reset_camera = True
        self.preview_revision += 1
        self.preview_cache = None
        w.refresh()

    def set_alignment(self):
        ref, w = self.current(), self.window
        if not self.alignment_active or ref is None or ref['locked']:
            return
        w.apply_controls()
        previous = dict(ref)
        ref.update(vertices=np.asarray(w.sim.mesh.vertices).copy(), faces=np.asarray(w.sim.mesh.faces).copy(),
            shape=w.sim.source, rotation=list(w.sim.rotation), offset=list(w.sim.offset),
            cut_depth=w.sim.cut_depth, locked=True)
        self.invalidate()
        try:
            self.checkpoint()
        except Exception as exc:
            ref.clear()
            ref.update(previous)
            self.update_reference_list()
            self.status.setText(f'Reference could not be saved: {exc}')
            w.statusBar().showMessage(self.status.text())
            return
        self.end_alignment()
        self.select()
        w.tabs.setCurrentWidget(w.calibration_scroll)
        self.status.setText(f'Reference saved to {self.session_path.name}. Load the next STL and upload its image, or calibrate all saved references.')

    def end_alignment(self):
        if not self.alignment_active:
            return
        w = self.window
        self.alignment_active = False
        self.settle_timer.stop()
        self.preview_detail = True
        self.live_pose_key = None
        self.preview_revision += 1
        self.preview_cache = None
        w.timer.stop()
        w.sim, w._has_object, values, label, quality, units = self.alignment_snapshot
        self.alignment_snapshot = None
        for key, value in values.items():
            control = getattr(w, key)
            control.blockSignals(True)
            if key == 'z':
                control.setRange(-1000000, 1000000)
            control.setValue(value)
            control.blockSignals(False)
        w.model_label.setText(label)
        for control, value in ((w.quality, quality), (w.units, units)):
            control.blockSignals(True)
            control.setCurrentIndex(value)
            control.blockSignals(False)
        for widget in (self.session_controls, self.sync_button, self.list, self.blank,
                       self.lock, self.validation, self.repeat_group, self.shape_name, self.align_button):
            widget.setEnabled(True)
        self.lock_controls()
        self.run_button.setEnabled(True)
        self.fit_options.setEnabled(True)
        w.settings_action.setEnabled(True)
        w.set_alignment_button.setEnabled(False)
        w.cancel_alignment_button.setEnabled(False)
        w.alignment_hint.setText('Review the reference and simulation. Unlock the reference and choose Align reference to change its object or pose.')
        w.right_stack.setCurrentIndex(0)
        w._scene_key = None
        w.refresh()
        w.tabs.setCurrentWidget(w.calibration_scroll)
        w.right_stack.setCurrentIndex(1)
        self.preview()

    def refresh_alignment(self):
        sim = self.window.sim
        pose = (id(sim.mesh), *sim.rotation, *sim.offset, sim.cut_depth)
        if pose == self.live_pose_key:
            return
        self.live_pose_key = pose
        self.preview_detail = False
        self.preview_revision += 1
        self.settle_timer.start(180)
        self.preview()

    def refine_preview(self):
        if self.alignment_active:
            self.preview_detail = True
            self.preview_revision += 1
            self.preview()

    def preview_key(self):
        ref = self.current()
        geometry = ((self.window.sim.mesh.vertices, self.window.sim.mesh.faces) if self.alignment_active
                    else (ref['vertices'], ref['faces']) if ref else (None, None))
        if self.preview_geometry is None or any(a is not b for a, b in zip(geometry, self.preview_geometry)):
            self.geometry_generation += 1
            self.preview_geometry = geometry
        mesh = self.geometry_generation
        return (id(ref), id(self.result), self.show_effects.isChecked(), mesh,
                self.preview_size(),
                self.preview_detail, self.preview_revision)

    def lock_controls(self):
        self.blank.setEnabled(not self.alignment_active)
        self.shape_name.setEnabled(not self.blank.isChecked() and not self.alignment_active)
        ref = self.current()
        if ref is not None:
            self.shape_label.setText('No shape — empty sensor image' if self.blank.isChecked() else
                                    f"STL: {ref.get('shape', 'mesh')}; {len(ref['faces']):,} faces")
        self.align_button.setText('Save reference' if self.blank.isChecked() else
                                 'Edit reference alignment' if self.lock.isChecked() else 'Align reference')

    def changed(self, *_):
        if self.loading or self.current() is None:
            return
        ref = self.current()
        changed_content = (ref['blank'] != self.blank.isChecked() or
                           ref.get('shape_name', '') != self.shape_name.text().strip())
        ref.update(blank=self.blank.isChecked(), validation=self.validation.isChecked(),
                   locked=False if changed_content else self.lock.isChecked(), shape_name=self.shape_name.text().strip())
        self.lock.blockSignals(True)
        self.lock.setChecked(ref['locked'])
        self.lock.blockSignals(False)
        ref['repeat_group'] = self.repeat_group.text().strip()
        self.invalidate()
        self.lock_controls()
        self.timer.start(180)

    def sync_mesh_controls(self):
        w = self.window
        for widget in (w.quality, w.mesh_smoothing):
            widget.blockSignals(True)
        w.quality.setCurrentIndex(w.quality.findData(0))
        w.mesh_smoothing.setValue(0)
        for widget in (w.quality, w.mesh_smoothing):
            widget.blockSignals(False)

    def preview(self, *_):
        ref = self.current()
        if ref is None or self.settings is None or self.worker is not None:
            return
        try:
            key = self.preview_key()
            if self.preview_cache is not None and self.preview_cache[0][:-2] == key[:-2]:
                self.display_preview(self.preview_cache[1:])
            if self.preview_cache is None or self.preview_cache[0] != key:
                if self.preview_worker is not None:
                    self.preview_pending = True
                    return
                draft = dict(ref)
                if self.alignment_active:
                    sim = self.window.sim
                    draft.update(vertices=np.asarray(sim.mesh.vertices), faces=np.asarray(sim.mesh.faces),
                        rotation=list(sim.rotation), offset=list(sim.offset), cut_depth=sim.cut_depth)
                worker = PreviewWorker(draft, self.settings,
                    self.result['settings'] if self.result else None,
                    self.preview_size(),
                    self.show_effects.isChecked(), self.render_cache, key[3], self)
                self.preview_worker = worker
                def ready(arrays):
                    current_key = self.preview_key()
                    if key[:-2] == current_key[:-2]:
                        self.preview_cache = (key, *arrays)
                        self.display_preview(arrays)
                def finished():
                    self.preview_worker = None
                    worker.deleteLater()
                    if self.preview_pending:
                        self.preview_pending = False
                        self.preview()
                worker.ready.connect(ready)
                worker.failure.connect(self.status.setText)
                worker.finished.connect(finished)
                worker.start()
                return
        except Exception as exc:
            self.status.setText(str(exc))

    def display_preview(self, arrays):
        try:
            real, after, mask = arrays
            if self.result:
                i = self.list.currentRow()
                self.status.setText(json.dumps(self.result['metrics'][i], indent=2))
            mixed = real*(1-self.opacity.value()/100) + after*self.opacity.value()/100
            if self.mode.currentIndex() == 1:
                mixed = real.copy()
                mixed[mask & ~binary_erosion(mask)] = (0, 1, 0)
            elif self.mode.currentIndex() == 2:
                mixed = np.clip(np.abs(real-after)*4, 0, 1)
            elif self.mode.currentIndex() == 3:
                mixed = after
            self.window.alignment_overlay.show_array(mixed*255)
        except Exception as exc:
            self.status.setText(str(exc))

    def preview_size(self):
        return (self.window.output_w.value(), self.window.output_h.value())

    def run_fit(self):
        if self.alignment_active:
            self.status.setText('Save the current reference or cancel alignment before fitting.')
            return
        if self.settings is None:
            self.status.setText('Add and align reference photos first.')
            return
        if not self.references or any(not ref['locked'] for ref in self.references):
            self.status.setText('Save every reference first. Draft references are marked in the list.')
            return
        self.invalidate()
        self.timer.stop()
        size = self.preview_size()
        self.settings['sensor']['camera']['width'], self.settings['sensor']['camera']['height'] = size
        self.worker = FitWorker(self.settings, self.references, dict(size=size,
            algorithm=self.algorithm.currentData(),
            fit_background=self.background.isChecked(), fit_lighting=self.lighting.isChecked(), fit_blur=self.blur.isChecked(),
            fit_directions=self.directions.isChecked(), fit_filters=self.filters.isChecked(), fit_noise=self.noise.isChecked(),
            fit_texture=self.texture.isChecked(), calibration_seed=self.search_seed.value(),
            search_starts=4 if self.search_quality.currentIndex() else 2,
            max_nfev=60 if self.search_quality.currentIndex() else 35), self)
        self.worker.message.connect(self.status.setText)
        self.worker.failure.connect(self.status.setText)
        self.worker.result.connect(self.completed)
        self.worker.finished.connect(self.finished)
        self.editor.setEnabled(False)
        self.fit_options.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel.setEnabled(True)
        self.worker.start()

    def completed(self, result):
        self.result = result
        self.update_report()
        self.status.setText(('Run autosave failed: '+result['autosave_error']) if result.get('autosave_error')
                            else 'Run saved to '+result.get('run_directory', '')+'. Load the sensor config manually to apply it.')
        try:
            self.checkpoint()
        except Exception as exc:
            self.status.setText(f'Fit completed, but the collection could not be saved: {exc}')

    def update_report(self):
        if not self.result:
            return
        lines = ['Errors in RGB [0, 1]; deterministic preview at '+str(self.result['size'])]
        if self.result.get('run_directory'):
            lines.append('Autosaved run: '+self.result['run_directory'])
        if self.result.get('autosave_error'):
            lines.append('Run autosave failed: '+self.result['autosave_error'])
        for row in self.result['metrics']:
            lines.append(f"{row['name']} ({'validation' if row['validation'] else 'fit'}): MAE {row['before_mae']:.4f} → {row['after_mae']:.4f}; RMSE {row['before_rmse']:.4f} → {row['after_rmse']:.4f}")
        def changes(before, after, prefix=''):
            if isinstance(before, dict):
                for key in before:
                    changes(before[key], after[key], prefix+'.'+key)
            elif isinstance(before, (list, tuple)):
                for i, (a, b) in enumerate(zip(before, after)):
                    changes(a, b, prefix+f'[{i}]')
            elif before != after:
                lines.append(f'{prefix.lstrip(".")}: {before} → {after}')
        changes(self.settings, self.result['settings'])
        lines.append(self.result.get('metric_note', ''))
        for stage in self.result['stages']:
            lines.append(stage['stage']+': '+stage.get('algorithm', stage.get('message', '')))
            if 'initial_cost' in stage:
                lines.append(f"  Objective: {stage['initial_cost']:.6g} → {stage['final_cost']:.6g}")
            lines.extend(stage.get('warnings', []))
        self.report.setPlainText('\n'.join(lines))

    def finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.editor.setEnabled(True)
        self.fit_options.setEnabled(True)
        self.run_button.setEnabled(True)
        self.cancel.setEnabled(False)
        verification = next((s for s in self.result['stages'] if s['stage'] == 'Production verification'), None) if self.result else None
        self.save_result.setEnabled(self.result is not None and (verification is None or verification['accepted']))
        if self.result:
            self.preview()

    def save(self):
        if self.settings is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save reference collection', str(self.session_path or SESSION_DIR), 'Calibration (*.npz)')
        if path:
            try:
                destination = Path(path if path.endswith('.npz') else path+'.npz')
                save_session(destination, self.settings, self.references, self.result)
                self.session_path = destination
                self.status.setText('Session saved with photos, meshes, alignments, settings and fitting results.')
            except Exception as exc:
                self.status.setText(str(exc))

    def open(self):
        self.end_alignment()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open reference collection', str(SESSION_DIR), 'Calibration (*.npz)')
        if path:
            try:
                settings, refs, result = load_session(path, include_result=True)
                self.settings, self.references = settings, refs
                self.session_path = Path(path)
                self.list.clear()
                self.list.addItems([r['name'] for r in refs])
                self.invalidate()
                self.result = result
                options = result.get('options', {}) if result else {}
                for key, widget in [('fit_background', self.background), ('fit_lighting', self.lighting),
                    ('fit_directions', self.directions), ('fit_filters', self.filters), ('fit_blur', self.blur),
                    ('fit_noise', self.noise), ('fit_texture', self.texture)]:
                    if key in options:
                        widget.setChecked(options[key])
                self.search_seed.setValue(options.get('calibration_seed', 0))
                algorithm_index = self.algorithm.findData(options.get('algorithm', 'least_squares'))
                self.algorithm.setCurrentIndex(max(0, algorithm_index))
                self.search_quality.setCurrentIndex(1 if options.get('max_nfev', 35) >= 60 else 0)
                self.save_result.setEnabled(result is not None)
                self.update_report()
                self.list.setCurrentRow(0)
                self.select()
            except Exception as exc:
                self.status.setText(str(exc))

    def export(self):
        if self.result is None:
            return
        verification = next((stage for stage in self.result['stages']
                             if stage['stage'] == 'Production verification'), None)
        training = [m for m in self.result['metrics'] if not m['validation']]
        if (verification and not verification['accepted']) or (training and
                np.mean([m['after_mae'] for m in training]) > np.mean([m['before_mae'] for m in training])+1e-7):
            self.status.setText('This fit did not pass production verification. Recheck alignment and fit again before exporting.')
            return
        name, ok = QtWidgets.QInputDialog.getText(self, 'Save fitted sensor config', 'New configuration name')
        if not ok or not name.strip():
            return
        try:
            import re
            name = re.sub(r'[^\w .-]', '_', name).strip(' .')
            if not name:
                raise ValueError('Enter a valid configuration name')
            path = self.window.sensor_configs.directory / (name+'.json')
            if self.window.sensor_configs.is_protected(path):
                raise ValueError('Built-in sensor configs are read-only. Choose a new configuration name.')
            data = deepcopy(self.result['settings'])
            from ..calibration_archive import sensor_config
            config = sensor_config(self.result)
            for key, value in config['controls'].items():
                self.window.validate_widget(getattr(self.window, key), value, key)
            for key, widget in self.window.effect_widgets.items():
                self.window.validate_widget(widget, data['effects'][key], key)
            from ..config import SensorConfig
            from ..lighting import validate_lighting
            validate_lighting(SensorConfig.from_dict(data['sensor']).optics)
            with path.open('x', encoding='utf-8') as stream:
                json.dump(config, stream, indent=2, allow_nan=False)
            self.window.sensor_configs.scan(force=True)
            self.status.setText(f'Saved {path.name}. Select it in Saved sensor configs to apply all fitted settings, including selected noise/texture effects.')
        except Exception as exc:
            self.status.setText(str(exc))
