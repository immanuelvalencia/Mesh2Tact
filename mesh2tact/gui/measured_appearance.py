"""Visible, editable measured appearance data with reversible mode switches."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .qt import QtCore, QtGui, QtWidgets
from ..config import OpticsConfig, SensorConfig
from ..lighting import validate_lighting
from ..render.optical import GelSightRenderer


class CoefficientDialog(QtWidgets.QDialog):
    """Edit signed RGB coefficients without rounding values on an unrelated edit."""
    def __init__(self, title, values, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(650, 590)
        self.rows = rows
        self.result_values = deepcopy(values)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Signed RGB coefficients. Changes apply only when you click Apply. Clearing removes this fitted response; other settings are preserved.')
        note.setWordWrap(True)
        layout.addWidget(note)
        actions = QtWidgets.QHBoxLayout()
        for label, callback in [('Import JSON…', self.import_json), ('Export JSON…', self.export_json),
                                ('Clear response', self.clear_response)]:
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); actions.addWidget(button)
        layout.addLayout(actions)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Red', 'Green', 'Blue'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.error = QtWidgets.QLabel(); self.error.setWordWrap(True); layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.populate(values if values is not None else np.zeros((max(rows), 3)).tolist())

    def populate(self, values):
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[0] not in self.rows or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError('Expected finite RGB coefficients with '+str(self.rows)+' rows and 3 columns.')
        self.table.setRowCount(len(array))
        spatial = ['constant', 'x', 'y', 'x²', 'xy', 'y²', 'centre']
        normals = ['nx', 'ny', 'nz − 1', 'nx ny', 'nx² − ny²']
        labels = spatial if len(array) == 7 else [s+' × '+n for s in spatial for n in normals][:len(array)]
        self.table.setVerticalHeaderLabels(labels)
        for row, channels in enumerate(array):
            for col, value in enumerate(channels):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(repr(float(value))))

    def values(self):
        values = [[float(self.table.item(r, c).text()) for c in range(3)] for r in range(self.table.rowCount())]
        if not np.isfinite(values).all():
            raise ValueError('Every coefficient must be a finite number.')
        return values

    def apply(self):
        try:
            self.result_values = self.values(); self.accept()
        except (ValueError, AttributeError) as exc:
            self.error.setText(str(exc))

    def clear_response(self):
        self.result_values = None; self.accept()

    def import_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import coefficients', '', 'JSON (*.json)')
        if path:
            try:
                self.populate(json.loads(Path(path).read_text(encoding='utf-8')))
                self.error.clear()
            except (OSError, ValueError, TypeError) as exc:
                self.error.setText(str(exc))

    def export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export coefficients', 'coefficients.json', 'JSON (*.json)')
        if path:
            try:
                Path(path).write_text(json.dumps(self.values(), indent=2, allow_nan=False)+'\n', encoding='utf-8')
            except (OSError, ValueError, AttributeError) as exc:
                self.error.setText(str(exc))


class MeasuredAppearancePanel(QtWidgets.QGroupBox):
    changed = QtCore.Signal(object)

    def __init__(self, snapshot=None):
        super().__init__('Background and measured lighting')
        self.optics = OpticsConfig()
        self.loading = False
        self.snapshot = snapshot
        layout = QtWidgets.QVBoxLayout(self)
        self.background_enabled = QtWidgets.QCheckBox('Use measured background image')
        self.background_enabled.setToolTip('Switch off to use manual gel backgrounds and gradients. The embedded image is retained when saving.')
        self.background_enabled.toggled.connect(self.edit)
        layout.addWidget(self.background_enabled)
        self.background_status = QtWidgets.QLabel(); self.background_status.setWordWrap(True)
        layout.addWidget(self.background_status)
        self.preview = QtWidgets.QLabel(); self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(160); self.preview.setMaximumHeight(250)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.preview.setStyleSheet('background: #17232c; color: #d4dfe5; padding: 8px;')
        layout.addWidget(self.preview)
        actions = QtWidgets.QVBoxLayout()
        self.import_background_button = QtWidgets.QPushButton('Import background image…')
        self.export_background_button = QtWidgets.QPushButton('Export embedded image…')
        self.clear_background_button = QtWidgets.QPushButton('Remove embedded image')
        self.import_background_button.clicked.connect(self.import_background_dialog)
        self.export_background_button.clicked.connect(self.export_background_dialog)
        self.clear_background_button.clicked.connect(self.clear_background)
        for button in (self.import_background_button, self.export_background_button, self.clear_background_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        note = QtWidgets.QLabel('Imported images are embedded in the saved sensor config at up to 256 pixels per side. Manual gradients apply when the measured background is off. Brightness, contrast, hue, saturation and gamma in Gel background and gradients still adjust the whole image.')
        note.setWordWrap(True); layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.model = QtWidgets.QComboBox()
        for label, value in [('Analytic LED lighting', 'analytic'), ('Measured surface response', 'measured'), ('Calibration lookup table', 'lookup')]:
            self.model.addItem(label, value)
        self.model.currentIndexChanged.connect(self.edit)
        form.addRow('Surface colour model', self.model)
        self.model_status = QtWidgets.QLabel(); self.model_status.setWordWrap(True)
        form.addRow(self.model_status)
        self.slope_damping = self.spin(0, 8, .1)
        form.addRow('Steep-slope damping', self.slope_damping)
        self.depth_enabled = QtWidgets.QCheckBox('Apply fitted flat-face colour')
        self.depth_enabled.toggled.connect(self.edit)
        form.addRow(self.depth_enabled)
        self.depth_scale = self.spin(.01, 20, .05)
        self.depth_scale.setSuffix(' mm')
        form.addRow('Flat-face depth scale', self.depth_scale)
        layout.addLayout(form)
        self.matrix_labels, self.matrix_buttons = {}, {}
        for key, title in [('spatial_response', 'Surface response'), ('boundary_response', 'Boundary response'),
                           ('contact_depth_response', 'Flat-face response')]:
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(); label.setWordWrap(True); self.matrix_labels[key] = label
            row.addWidget(label, 1)
            button = QtWidgets.QPushButton('Edit / import…')
            button.clicked.connect(lambda _=False, k=key, t=title: self.edit_matrix(k, t))
            self.matrix_buttons[key] = button; row.addWidget(button); layout.addLayout(row)
        self.lookup_status = QtWidgets.QLabel(); self.lookup_status.setWordWrap(True)
        layout.addWidget(self.lookup_status)
        row = QtWidgets.QHBoxLayout()
        self.lookup_load = QtWidgets.QPushButton('Load lookup table…')
        self.lookup_clear = QtWidgets.QPushButton('Remove lookup path')
        self.lookup_load.clicked.connect(self.import_lookup_dialog)
        self.lookup_clear.clicked.connect(self.clear_lookup)
        row.addWidget(self.lookup_load); row.addWidget(self.lookup_clear); layout.addLayout(row)
        self.all_values = QtWidgets.QPushButton('Inspect all saved sensor settings…')
        self.all_values.clicked.connect(self.show_snapshot); layout.addWidget(self.all_values)
        self.message = QtWidgets.QLabel(); self.message.setWordWrap(True); layout.addWidget(self.message)
        self.set_optics(self.optics)

    def spin(self, lo, hi, step):
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(lo, hi); control.setDecimals(6); control.setSingleStep(step)
        control.valueChanged.connect(self.edit)
        return control

    def set_optics(self, optics):
        self.loading = True
        self.optics = deepcopy(optics)
        self.background_enabled.setChecked(optics.calibrated_background_enabled)
        grid = optics.calibrated_background
        if grid is not None:
            array = np.ascontiguousarray(np.clip(np.asarray(grid)*255, 0, 255).astype('uint8'))
            h, w = array.shape[:2]
            im = QtGui.QImage(array.data, w, h, array.strides[0], QtGui.QImage.Format_RGB888).copy()
            self.preview.setPixmap(QtGui.QPixmap.fromImage(im).scaled(360, 220, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.background_status.setText(f'Embedded background: {w} × {h} RGB. '+('Active; original JPEG is not needed.' if optics.calibrated_background_enabled else 'Stored but inactive; manual background controls apply.'))
        else:
            self.preview.clear(); self.preview.setText('No embedded background image')
            self.background_status.setText('No embedded image. '+('The lookup file may supply a background.' if optics.uses_lookup and optics.calibrated_background_enabled else 'Manual gel background / analytic reference tint is used.'))
        self.export_background_button.setEnabled(grid is not None)
        self.clear_background_button.setEnabled(grid is not None)
        mode = 'lookup' if optics.uses_lookup else 'measured' if optics.uses_spatial_response else 'analytic'
        self.model.setCurrentIndex(self.model.findData(mode))
        self.model_status.setText('LED direction, colour and reflection controls apply.' if mode == 'analytic' else 'Fitted data supplies the surface colours. LED controls are inactive; boundary shading and image adjustments remain separate.')
        self.slope_damping.setValue(optics.spatial_slope_damping)
        self.slope_damping.setEnabled(mode == 'measured')
        self.depth_enabled.setChecked(optics.contact_depth_enabled)
        self.depth_scale.setValue(optics.contact_depth_scale_mm)
        self.depth_scale.setEnabled(optics.contact_depth_enabled and optics.contact_depth_response is not None)
        for key, label in self.matrix_labels.items():
            values = getattr(optics, key)
            active = mode == 'measured' if key == 'spatial_response' else optics.boundary_enabled if key == 'boundary_response' else optics.contact_depth_enabled
            title = {'spatial_response':'Surface response','boundary_response':'Boundary response','contact_depth_response':'Flat-face response'}[key]
            label.setText(title+': '+('none; fallback shading' if values is None and key == 'boundary_response' else 'none' if values is None else f'{len(values)} × 3 coefficients — '+('active' if active else 'stored, inactive')))
        self.lookup_status.setText('Lookup table: '+(optics.calibration_path or 'none')+(' (active)' if mode == 'lookup' else ''))
        self.lookup_clear.setEnabled(bool(optics.calibration_path))
        self.loading = False

    def publish(self, candidate):
        validate_lighting(candidate)
        self.set_optics(candidate)
        self.message.setText('Applied to the preview. Use Saved sensor configs to save these changes.')
        self.changed.emit(deepcopy(candidate))

    def edit(self, *_):
        if self.loading: return
        candidate = deepcopy(self.optics)
        mode = self.model.currentData()
        if (mode == 'measured' and candidate.spatial_response is None) or (mode == 'lookup' and not candidate.calibration_path):
            self.set_optics(candidate)
            self.message.setText('Import or create the required response data before selecting this model.')
            return
        candidate.spatial_response_enabled = mode == 'measured'
        candidate.calibration_enabled = mode == 'lookup'
        candidate.calibrated_background_enabled = self.background_enabled.isChecked()
        candidate.contact_depth_enabled = self.depth_enabled.isChecked()
        if self.sender() is self.slope_damping:
            candidate.spatial_slope_damping = self.slope_damping.value()
        if self.sender() is self.depth_scale:
            candidate.contact_depth_scale_mm = self.depth_scale.value()
        self.publish(candidate)

    def set_matrix(self, key, values):
        if key not in self.matrix_labels: raise ValueError('Unknown response matrix')
        candidate = deepcopy(self.optics); setattr(candidate, key, deepcopy(values))
        if values is not None:
            if key == 'spatial_response': candidate.spatial_response_enabled=True; candidate.calibration_enabled=False
            if key == 'contact_depth_response': candidate.contact_depth_enabled=True
        self.publish(candidate)

    def edit_matrix(self, key, title):
        dialog = CoefficientDialog(title, getattr(self.optics, key), (7,) if key == 'contact_depth_response' else (30,35) if key == 'spatial_response' else (35,), self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.set_matrix(key, dialog.result_values)

    def import_background(self, path):
        with Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert('RGB')
            im.thumbnail((256,256), Image.Resampling.BOX)
            candidate = deepcopy(self.optics)
            candidate.calibrated_background = (np.asarray(im,dtype=float)/255).tolist()
        candidate.calibrated_background_enabled = True
        self.publish(candidate)
        self.message.setText(f'Imported {Path(path).name}. Image pixels are embedded when you save the sensor config.')

    def import_background_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import empty sensor image', '', 'Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)')
        if path:
            try: self.import_background(path)
            except (OSError, ValueError) as exc: self.message.setText(str(exc))

    def export_background(self, path):
        if self.optics.calibrated_background is None: raise ValueError('No embedded image to export')
        array = np.rint(np.clip(np.asarray(self.optics.calibrated_background),0,1)*255).astype('uint8')
        Image.fromarray(array).save(path)

    def export_background_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export embedded background', 'sensor_background.png', 'PNG (*.png)')
        if path:
            try: self.export_background(path if Path(path).suffix else path+'.png')
            except (OSError, ValueError) as exc: self.message.setText(str(exc))

    def clear_background(self):
        candidate=deepcopy(self.optics);candidate.calibrated_background=None;self.publish(candidate)

    def import_lookup(self, path):
        candidate=deepcopy(self.optics);candidate.calibration_path=str(Path(path).resolve())
        candidate.calibration_enabled=True;candidate.spatial_response_enabled=False
        renderer=GelSightRenderer(SensorConfig(optics=candidate))
        if renderer._table.ndim != 3 or renderer._table.shape[2] != 3 or min(renderer._table.shape[:2]) < 2 or not np.isfinite(renderer._table).all() or not np.isfinite(renderer._table_range) or renderer._table_range <= 0:
            raise ValueError('Lookup table needs a finite Gx × Gy × 3 table and a positive gradient range.')
        self.publish(candidate)

    def import_lookup_dialog(self):
        path, _=QtWidgets.QFileDialog.getOpenFileName(self,'Load calibration lookup','','NumPy lookup (*.npz)')
        if path:
            try:self.import_lookup(path)
            except (OSError,ValueError,KeyError) as exc:self.message.setText(str(exc))

    def clear_lookup(self):
        candidate=deepcopy(self.optics);candidate.calibration_path=None;self.publish(candidate)

    def show_snapshot(self):
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('All saved sensor settings');dialog.resize(850,650)
        layout=QtWidgets.QVBoxLayout(dialog)
        note=QtWidgets.QLabel('Read-only snapshot of current settings. Appearance controls are on the Settings pages. Material, solver and marker fields are retained for compatibility; the geometric renderer does not use them. Array data can be edited/imported in Background and measured lighting.')
        note.setWordWrap(True);layout.addWidget(note)
        tree=QtWidgets.QTreeWidget();tree.setHeaderLabels(['Setting','Current value']);layout.addWidget(tree)
        def add(parent,key,value):
            item=QtWidgets.QTreeWidgetItem(parent,[str(key),''])
            if isinstance(value,dict):
                for k,v in value.items():add(item,k,v)
            elif isinstance(value,list) and value and isinstance(value[0],list):
                shape=np.asarray(value).shape;item.setText(1,' × '.join(map(str,shape))+' values; available in the image/coefficient editor')
            elif isinstance(value,list) and value and isinstance(value[0],dict):
                for k,v in enumerate(value):add(item,k,v)
            else:item.setText(1,json.dumps(value,ensure_ascii=False))
        data=self.snapshot() if self.snapshot else {'optics':asdict(self.optics)}
        for key,value in data.items():add(tree,key,value)
        tree.expandToDepth(1);tree.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        close=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close);close.rejected.connect(dialog.reject);layout.addWidget(close)
        dialog.exec_()
