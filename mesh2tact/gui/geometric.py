"""Desktop STL slicing workbench, independent of the physics application."""
from copy import deepcopy
from dataclasses import asdict
from .appearance_presets import AppearancePresets
from datetime import datetime
from pathlib import Path
import sys
import warnings

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial.transform import Rotation

from .qt import QtCore, QtGui, QtWidgets
from .lighting import LightingPanel
from .measured_appearance import MeasuredAppearancePanel
from .gather import GatherPanel, GatherWorker
from .batch_gather import BatchGatherPanel
from .gel import GelLightingPanel
from .presets import PresetMixin
from ..config import SensorConfig
from ..geometric import GeometricSim
from ..geometry.mesh import primitive
from ..render.optical import GelSightRenderer
from ..render.effects import ImageEffects
from ..outputs import DATA_ROOT
from ..gather import GatherSettings, gather
from .sensor_configs import SensorConfigPanel
from .settings_window import SettingsWindow
from .calibration import CalibrationPanel, Picture
from .predict import PredictPanel
from .validation import ValidationPanel
from ..preview import linked_capture


def style_action(button, color):
    button.setStyleSheet(f'''
        QPushButton {{ background: {color}; color: white; border: 2px solid transparent;
                       border-radius: 5px; padding: 6px 10px; font-weight: 600; }}
        QPushButton:hover {{ border-color: #b9d9ed; }}
        QPushButton:pressed {{ border-color: #172331; padding-top: 7px; padding-bottom: 5px; }}
        QPushButton:focus {{ border-color: #f5d45c; }}
        QPushButton:disabled {{ background: #d8dde2; color: #626b75; }}
    ''')


class ImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 170)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: #151c25")
        self.original = None

    def set_image(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        self.original = QtGui.QPixmap.fromImage(QtGui.QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy())
        self._fit()

    def _fit(self):
        if self.original is not None:
            self.setPixmap(self.original.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class GeometricWindow(PresetMixin, QtWidgets.QMainWindow):
    def __init__(self, sensor=None, mesh=None, gather_autosave_path=None):
        super().__init__()
        self.gather_autosave_path = Path(gather_autosave_path) if gather_autosave_path else Path(__file__).resolve().parents[2] / 'configs' / 'last_auto_gather.json'
        cfg = SensorConfig.from_yaml(sensor) if sensor else SensorConfig()
        self.sim = GeometricSim(cfg)
        self._has_object = bool(mesh)
        self.capture_resolution = (cfg.camera.width, cfg.camera.height)
        self.sim.cfg.camera.width, self.sim.cfg.camera.height = 328, 246
        self.setWindowTitle("Mesh2Tact — Geometric surface slicing")
        self.resize(1450, 900)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        self.gather_worker = None
        self.data_root = DATA_ROOT
        self._close_after_gather = False
        self.gizmo = None
        self.object_actor = None
        self._actor_mesh = None
        navigation = self.addToolBar("Navigation")
        navigation.setMovable(False)
        self.settings_action = navigation.addAction("Settings")
        self.settings_action.triggered.connect(self.open_settings)
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        tabs.setMinimumWidth(400)
        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(tabs)
        splitter.addWidget(sidebar)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(controls)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        scroll.setWidget(controls)
        tabs.addTab(scroll, "General object")
        general_form = form
        self.load_button = QtWidgets.QPushButton("Load STL…")
        style_action(self.load_button, '#1769aa')
        self.load_button.clicked.connect(self.load_dialog)
        form.addRow(self.load_button)
        self.quality = QtWidgets.QComboBox()
        for name, level in [("Very low", -3), ("Low", -2), ("Medium", -1), ("Original", 0),
                            ("High", 1), ("Ultra", 2), ("Ultra high", 3)]:
            self.quality.addItem(name, level)
        self.quality.setCurrentIndex(self.quality.findData(2))
        form.addRow("Object quality", self.quality)
        self.mesh_smoothing = self.integer_control(form, 'Shape smoothing iterations', 0, 100, 0, schedule=False)
        self.model_label = QtWidgets.QLabel("No STL loaded")
        self.model_label.setWordWrap(True)
        form.addRow(self.model_label)
        self.units = QtWidgets.QComboBox()
        self.units.addItems(["mm", "cm", "m", "in"])
        form.addRow("STL source units", self.units)
        self.scale = self.control(form, "Import scale", .001, 1000, 1, .1, schedule=False)
        self.scale.setToolTip('Resize the loaded object: 1 = original size, 2 = twice as large, 0.5 = half size. Also used for the next import.')
        self.scale.setKeyboardTracking(False)
        self.scale.valueChanged.connect(self.resize_object)
        self.base_flip = QtWidgets.QComboBox()
        for label, angles in [('Original', (0., 0., 0.)), ('Flip X — 180°', (180., 0., 0.)),
                              ('Flip Y — 180°', (0., 180., 0.)), ('Flip Z — 180°', (0., 0., 180.))]:
            self.base_flip.addItem(label, angles)
        self.base_flip.setToolTip('Global base orientation for every loaded shape and both gathering tabs. Applied before current or random pose rotations. Source files are unchanged.')
        form.addRow('Global model base orientation', self.base_flip)
        self.base_flip.currentIndexChanged.connect(self.change_base_flip)
        self.rx = self.control(form, "Rotate X (°)", -180, 180, 0, 5)
        self.ry = self.control(form, "Rotate Y (°)", -180, 180, 0, 5)
        self.rz = self.control(form, "Rotate Z (°)", -180, 180, 0, 5)
        self.x = self.control(form, "Offset X (mm)", -1000, 1000, 0, .1)
        self.y = self.control(form, "Offset Y (mm)", -1000, 1000, 0, .1)
        self.cut = QtWidgets.QDoubleSpinBox(self)  # Legacy preset/calibration depth adapter.
        self.cut.setRange(-1000, 1000)
        self.cut.setDecimals(3)
        self.cut.setValue(1)
        self.cut.hide()
        self.z = self.control(form, 'Object Z (mm)', -1000, 1000, 3, .05)
        self.max_penetration = self.control(form, 'Maximum indentation depth (mm)', 0, 1000, cfg.camera.max_depth*1000, .1)
        self.max_penetration.setToolTip('Maximum distance below the fixed Z=0 sensor plane. Independent of the depth colour scale.')
        self.slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider.setRange(-1000, 1000)
        self.slider.setValue(0)
        self.slider.setSingleStep(5)
        self.slider.setPageStep(100)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider.setTickInterval(1000)
        self.slider.valueChanged.connect(self.move_z_slider)
        self.z.valueChanged.connect(self.sync_z_bounds)
        self.max_penetration.valueChanged.connect(self.sync_z_bounds)
        self.cut.valueChanged.connect(self.sync_slider)
        self.move_step = self.control(form, "Move step (mm)", .001, 10, .1, .05, schedule=False)
        move_row = QtWidgets.QGridLayout()
        for row_index, axis in enumerate("XYZ"):
            for column_index, direction in enumerate((-1, 1)):
                button = QtWidgets.QPushButton(axis + ("−" if direction < 0 else "+"))
                style_action(button, {'X': '#ad4040', 'Y': '#287348', 'Z': '#2866aa'}[axis])
                button.clicked.connect(lambda checked=False, a=axis, d=direction: self.nudge(a, d))
                move_row.addWidget(button, row_index, column_index)
        form.addRow(move_row)
        self.press_button = QtWidgets.QPushButton('Press to maximum depth')
        style_action(self.press_button, '#2866aa')
        self.press_button.clicked.connect(self.press_to_maximum)
        form.addRow(self.press_button)
        fit = QtWidgets.QPushButton("Fit 3D view")
        style_action(fit, '#14777b')
        fit.clicked.connect(self.fit_view)
        form.addRow(fit)
        reset = QtWidgets.QPushButton("Reset orientation / position")
        style_action(reset, '#986018')
        reset.clicked.connect(self.reset_pose)
        form.addRow(reset)
        hint = QtWidgets.QLabel("Sensor plane stays at Z=0. Z moves the object origin: down penetrates, up releases. "
                               "The indentation limit clamps motion, including rotation and dragging. Softness is image smoothing. "
                               "Depth previews use a fixed scale. No forces or elasticity are modeled.")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.settings_window = SettingsWindow(self)
        sensor_layout = None
        self.sensor_configs = SensorConfigPanel(self)
        self.settings_window.add_page("Saved sensor configs", self.sensor_configs)
        geometry = QtWidgets.QGroupBox("Sensor area and depth")
        form = QtWidgets.QFormLayout(geometry)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.softness = self.control(form, "Edge softness σ (mm)", 0, 2, 0, .01)
        self.width = self.control(form, "Sensor width (mm)", .1, 1000, cfg.gel.size_x*1000, .1)
        self.height = self.control(form, "Sensor height (mm)", .1, 1000, cfg.gel.size_y*1000, .1)
        self.depth_range = self.control(form, "Depth display maximum (mm)", .001, 1000, cfg.camera.max_depth*1000, .1)
        self.add_sensor_section(sensor_layout, "Sensor area and depth", geometry, False)
        self.gel_panel = GelLightingPanel()
        self.gel_panel.changed.connect(self.change_gel)
        self.add_sensor_section(sensor_layout, "Gel background and gradients", self.gel_panel, True)
        self.measured_appearance = MeasuredAppearancePanel(snapshot=self.sensor_configs.data)
        self.measured_appearance.set_optics(cfg.optics)
        self.measured_appearance.changed.connect(self.change_lighting)
        self.add_sensor_section(sensor_layout, "Background and measured lighting", self.measured_appearance, True)
        self.lighting = LightingPanel(show_presets=False)
        self.lighting.set_optics(deepcopy(cfg.optics))
        self.lighting.changed.connect(self.change_lighting)
        self.add_sensor_section(sensor_layout, "Directional lighting", self.lighting, False)
        self.add_sensor_section(sensor_layout, "Shadows", self.lighting.shadow_panel, False)
        self.build_image_controls(general_form, sensor_layout)
        from .preset_guard import PresetEditGuard, edit_controls
        appearance_controls = (edit_controls(self.gel_panel, (self.gel_panel.selector,))
                               + edit_controls(self.measured_appearance, (self.measured_appearance.export_background_button, self.measured_appearance.all_values))
                               + edit_controls(self.lighting, (self.lighting.selector,))
                               + edit_controls(self.lighting.shadow_panel))
        appearance_controls = [w for w in appearance_controls if not
            (isinstance(w, QtWidgets.QPushButton) and w.text() in ('Create preset…', 'Update preset', 'Delete preset'))]
        self.sensor_edit_guard = PresetEditGuard(self,
            [self.softness, self.width, self.height, self.depth_range,
             *edit_controls(self.effects_enabled.parentWidget()), *appearance_controls],
            self.sensor_configs.is_protected, self.sensor_configs.create_copy)
        self.calibration = CalibrationPanel(self)
        self.calibration_scroll = QtWidgets.QScrollArea()
        self.calibration_scroll.setWidgetResizable(True)
        self.calibration_scroll.setWidget(self.calibration)
        tabs.addTab(self.calibration_scroll, 'Calibration')
        self.predict_panel = PredictPanel(self)
        tabs.addTab(self.predict_panel, 'Predict')
        self.validation_panel = ValidationPanel(self)
        tabs.addTab(self.validation_panel, 'Validation')
        self.quality.currentIndexChanged.connect(self.apply_quality)
        self.mesh_smoothing.valueChanged.connect(self.apply_quality)
        self.settings_window.navigation.setCurrentRow(2)
        self.gather_panel = GatherPanel(self)
        tabs.addTab(self.gather_panel, "Data gathering")
        self.batch_gather_panel = BatchGatherPanel(self)
        tabs.addTab(self.batch_gather_panel, 'Batch gathering')
        tabs.setMinimumWidth(max(400, tabs.tabBar().sizeHint().width()+8))
        scene = QtWidgets.QWidget()
        scene_layout = QtWidgets.QHBoxLayout(scene)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = QtInteractor(scene)
        scene_layout.addWidget(self.plot.interactor, 1)
        cut_panel = QtWidgets.QWidget()
        cut_layout = QtWidgets.QVBoxLayout(cut_panel)
        cut_layout.addWidget(QtWidgets.QLabel("Object Z\n(mm)"))
        self.cut_readout = QtWidgets.QLabel("1.00")
        cut_layout.addWidget(self.cut_readout)
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(self.slider)
        midpoint = QtWidgets.QLabel('0\ncontact')
        midpoint.setToolTip('Zero indentation / first contact; not necessarily object-origin Z=0.')
        midpoint.setAlignment(QtCore.Qt.AlignCenter)
        slider_row.addWidget(midpoint)
        cut_layout.addLayout(slider_row, 1)
        self.slider.setToolTip('Middle = first contact (zero indentation). Lower half controls indentation; upper half lifts up to twice the rotated object height.')
        scene_layout.addWidget(cut_panel)
        self.scene_stack = QtWidgets.QStackedWidget()
        self.scene_stack.addWidget(scene)
        self.scene_stack.addWidget(self.predict_panel.photo_browser)
        self.scene_stack.addWidget(self.validation_panel.workspace)
        splitter.addWidget(self.scene_stack)
        def update_prediction_view(*_):
            browsing = tabs.currentWidget() is self.predict_panel and bool(self.predict_panel.source.currentIndex())
            validating = tabs.currentWidget() is self.validation_panel
            self.scene_stack.setCurrentIndex(2 if validating else 1 if browsing else 0)
            self.right_stack.setVisible(not (browsing or validating))
        tabs.currentChanged.connect(update_prediction_view)
        tabs.currentChanged.connect(self.gather_panel.queue_position_overlay)
        self.predict_panel.browse_source_changed.connect(update_prediction_view)
        self.plot.set_background("#202936")
        self.plot.add_axes()
        from .preview_grid import PreviewGrid
        panel = self.preview_grid = PreviewGrid(ImageView)
        panel.checkboxes['contact'].toggled.connect(self.enable_contact_mask_output)
        if panel.checkboxes['contact'].isChecked():
            self.enable_contact_mask_output(True)
        self.depth_view = panel.views['depth']
        self.depth_label = QtWidgets.QLabel()
        panel.layout().addWidget(self.depth_label)
        self.rgb_view = panel.views['rgb']
        panel.applied.connect(self.schedule)
        self.right_stack = QtWidgets.QStackedWidget()
        self.right_stack.addWidget(panel)
        alignment = QtWidgets.QWidget()
        alignment_layout = QtWidgets.QVBoxLayout(alignment)
        self.alignment_title = QtWidgets.QLabel('Calibration alignment')
        self.alignment_title.setWordWrap(True)
        alignment_layout.addWidget(self.alignment_title)
        hint = self.alignment_hint = QtWidgets.QLabel('Select a reference photo, then choose Align reference to position the object.')
        hint.setWordWrap(True)
        alignment_layout.addWidget(hint)
        self.alignment_reference, self.alignment_overlay = Picture(), Picture()
        for title, view in [('Uploaded reference', self.alignment_reference), ('Live overlay', self.alignment_overlay)]:
            alignment_layout.addWidget(QtWidgets.QLabel(title))
            view.setMinimumSize(140, 100)
            alignment_layout.addWidget(view, 1)
        overlay_controls = QtWidgets.QHBoxLayout()
        overlay_controls.addWidget(self.calibration.mode)
        overlay_controls.addWidget(self.calibration.opacity)
        alignment_layout.addLayout(overlay_controls)
        buttons = QtWidgets.QHBoxLayout()
        self.set_alignment_button = QtWidgets.QPushButton('Save reference')
        self.set_alignment_button.setEnabled(False)
        style_action(self.set_alignment_button, '#287348')
        self.set_alignment_button.clicked.connect(self.calibration.set_alignment)
        buttons.addWidget(self.set_alignment_button)
        cancel_alignment = self.cancel_alignment_button = QtWidgets.QPushButton('Cancel alignment')
        cancel_alignment.setEnabled(False)
        cancel_alignment.clicked.connect(self.calibration.end_alignment)
        buttons.addWidget(cancel_alignment)
        alignment_layout.addLayout(buttons)
        fit_images = QtWidgets.QPushButton('Fit reference and overlay to panels')
        fit_images.clicked.connect(lambda: (self.alignment_reference.fit_image(), self.alignment_overlay.fit_image()))
        alignment_layout.addWidget(fit_images)
        self.right_stack.addWidget(alignment)
        tabs.currentChanged.connect(lambda *_: self.right_stack.setCurrentIndex(
            1 if self.calibration.alignment_active or tabs.currentWidget() is self.calibration_scroll else 0))
        splitter.addWidget(self.right_stack)
        splitter.setSizes([460, 550, 440])
        self._scene_key = None
        self._reset_camera = True
        if mesh:
            self.sim.load(mesh, self.units.currentText(), self.scale.value())
            self.sim.set_quality(self.quality.currentData())
            self.model_label.setText(str(mesh))
        if sensor is None and not self.sensor_configs.restore_selection():
            default_path = self.sensor_configs.directory / "GelSightV1.json"
            relief_path = self.sensor_configs.directory / "GelSightV1 depth relief.json"
            if relief_path.exists():
                default_path = relief_path
            if default_path.exists():
                self.sensor_configs.load(default_path)
                self.sensor_configs.selector.blockSignals(True)
                self.sensor_configs.selector.setCurrentIndex(
                    self.sensor_configs.selector.findData(str(default_path)))
                self.sensor_configs.selector.blockSignals(False)
                self.sensor_configs.sync_protection()
        self.sync_appearance_panels()
        self.sync_slider(self.cut.value())
        self.refresh()

        if sensor is None and mesh is None and self.gather_autosave_path.is_file():
            try:
                self.load_preset(self.gather_autosave_path)
                self.gather_panel.status.setText('Restored settings from the last auto gather.')
            except Exception as exc:
                self.gather_panel.status.setText(f'Could not restore last auto gather settings: {exc}')

    def open_settings(self):
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def nudge(self, axis, direction):
        control = {"X": self.x, "Y": self.y, "Z": self.z}[axis]
        # The sensor is fixed at Z=0. Moving up releases the indentation.
        control.setValue(control.value()+direction*self.move_step.value())

    def add_sensor_section(self, layout, title, widget, expanded):
        self.settings_window.add_page(title, widget)

    def fit_view(self):
        self.plot.reset_camera()
        self.plot.render()

    def toggle_gizmo(self, enabled):
        if self.gizmo is not None:
            self.gizmo.remove()
            self.gizmo = None
        if enabled and self.object_actor is not None:
            try:
                self.gizmo = self.plot.add_affine_transform_widget(
                    self.object_actor, release_callback=self.gizmo_moved,
                    interact_callback=self.gizmo_dragged)
            except Exception as exc:
                self.statusBar().showMessage(f"Drag controls unavailable: {exc}. Use XYZ buttons or pose fields.")

    def gizmo_dragged(self, matrix):
        if self.calibration.alignment_active:
            self.gizmo_moved(matrix, live=True)
        else:
            # VTK moves the actor before invoking this callback. Constrain the
            # visible actor too, even when the stored pose is already at the limit.
            self.constrain_drag_pose(matrix)

    def constrain_drag_pose(self, matrix):
        pose = np.asarray(matrix, dtype=float).copy()
        bottom = float((self.sim.mesh.vertices @ pose[:3, :3].T)[:, 2].min()*1000)
        pose[2, 3] = max(pose[2, 3], -bottom-self.max_penetration.value())
        if self.object_actor is not None:
            self.object_actor.user_matrix = pose
        return pose

    def gizmo_moved(self, matrix, live=False):
        matrix = self.constrain_drag_pose(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # equivalent Euler pose at gimbal lock
            angles = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
        self.z.blockSignals(True)
        self.z.setRange(-1000000, 1000000)
        self.z.blockSignals(False)
        for control, value in zip((self.rx, self.ry, self.rz, self.x, self.y, self.z),
                                  (*angles, matrix[0, 3], matrix[1, 3], matrix[2, 3])):
            control.blockSignals(True)
            control.setValue(float(value))
            control.blockSignals(False)
        self.sync_z_bounds()
        # A drag can change the VTK actor without changing the clamped simulation
        # pose. Invalidate the scene cache so release always restores that pose.
        self._scene_key = None
        if live:
            self.apply_controls()
            self.calibration.refresh_alignment()
            return
        # Refresh after VTK completes its release callback; recreates the widget
        # with the new pose so later drags and numeric changes share one origin.
        self.schedule()

    def capture_to_folder(self):
        self.start_gather(GatherSettings(count=1, random_rotation=False,
                          random_cut=False, outputs=self.gather_panel.outputs(),
                          save_layout=self.gather_panel.save_layout.currentData(),
                          object_label=self.gather_panel.object_label.text().strip(),
                          require_contact=self.gather_panel.require_contact.isChecked(),
                          min_contact_pixels=self.gather_panel.min_contact_pixels.value(),
                          still_capture=True))

    def start_gather(self, settings=None):
        if self.batch_gather_panel.worker is not None:
            self.gather_panel.status.setText('Wait for batch gathering to finish.')
            return
        if self.calibration.alignment_active:
            self.statusBar().showMessage('Set or cancel calibration alignment before capturing data.')
            return
        if not self._has_object:
            return
        if self.gather_worker is not None:
            return
        try:
            self.sync_z_bounds()
            settings = settings if isinstance(settings, GatherSettings) else self.gather_panel.settings()
            settings.validate()
            self.timer.stop()
            self.apply_controls()
            if not settings.still_capture:
                self.gather_autosave_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_preset(self.gather_autosave_path)
            snapshot = deepcopy(self.sim)
            snapshot.cfg.camera.width, snapshot.cfg.camera.height = self.capture_resolution
            self.gather_worker = GatherWorker(snapshot, str(self.data_root), settings, self)
            self.gather_worker.sample_ready.connect(self.gather_progress)
            self.gather_worker.result_ready.connect(self.gather_result)
            self.gather_worker.finished.connect(self.gather_finished)
            self.gather_panel.progress.setRange(0, settings.count)
            self.gather_panel.progress.setValue(0)
            self.gather_panel.status.setText("Gathering with the current object, lighting, effects and capture resolution…")
            self.gather_panel.set_running(True)
            self.batch_gather_panel.setEnabled(False)
            self.gather_worker.start()
        except Exception as exc:
            self.gather_worker = None
            self.gather_panel.set_running(False)
            self.gather_panel.status.setText(f"Cannot start gathering: {exc}")

    def stop_gather(self):
        if self.gather_worker is not None:
            self.gather_worker.stop_event.set()
            self.gather_panel.stop.setEnabled(False)
            self.gather_panel.status.setText("Stopping after the current sample; completed captures will be kept…")

    def gather_progress(self, record):
        if record.get('rejected'):
            self.gather_panel.progress.setValue(record['completed'])
            self.gather_panel.status.setText(f"Saved {record['completed']}; skipped {record['skipped']} masks. "
                                           f"Latest mask: {record['contact_pixels']} contact pixels. Retrying…")
            return
        self.gather_panel.progress.setValue(record.get('completed', record['index']))
        self.gather_panel.status.setText(f"Saved {record.get('completed', record['index'])} samples; skipped {record.get('skipped', 0)}. "
            f"Rotation XYZ: {tuple(round(v, 3) for v in record['rotation_xyz_deg'])}°. "
            f"Indentation: {record['cut_depth_m']*1000:.3f} mm; "
            f"contact: {record['contact_fraction']*100:.1f}%\n{record['directory']}")

    def gather_result(self, result):
        self.gather_panel.progress.setValue(result["completed"])
        self.gather_panel.status.setText(f"{result['status'].capitalize()}: {result['completed']} samples saved.\n"
                                       f"Skipped by contact checker: {result.get('rejected', 0)}.\n"
                                       f"Sampling seed: {result.get('sampling_seed', 'unavailable')}.\n"
                                       f"Geometry: {result.get('geometry_backend', 'cpu').upper()}.\n"
                                       f"{result['directory']}\n{result.get('error', '')}")

    def gather_finished(self):
        worker = self.gather_worker
        self.gather_worker = None
        if worker is not None:
            worker.deleteLater()
        self.gather_panel.set_running(False)
        self.batch_gather_panel.setEnabled(True)
        self.gather_panel.update_destination()
        if self._close_after_gather:
            self.close()

    def build_image_controls(self, general_form, sensor_layout):
        quality_box = QtWidgets.QGroupBox("Capture and preview resolution")
        form = QtWidgets.QFormLayout(quality_box)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        general_form.addRow(quality_box)
        self.resolution_preset = QtWidgets.QComboBox()
        self.resolution_preset.addItem('Custom', None)
        for width in (640, 984, 1280, 1920, 2048):
            self.resolution_preset.addItem(f'{width} px', width)
        form.addRow("Capture preset", self.resolution_preset)
        self.output_w = self.integer_control(form, "Capture width (px)", 64, 2048, self.capture_resolution[0])
        self.output_h = self.integer_control(form, "Capture height (px)", 64, 2048, self.capture_resolution[1])
        self.capture_geometry_label = QtWidgets.QLabel()
        self.capture_geometry_label.setWordWrap(True)
        form.addRow(self.capture_geometry_label)
        self.output_w.setToolTip('Changes resolution. Height follows the physical sensor aspect ratio.')
        self.output_h.setToolTip('Changes resolution. Width follows the physical sensor aspect ratio.')
        self.preview_scale = self.control(form, "Preview scale", 10, 100, 33.333, 5)
        self.preview_scale.setSuffix(" %")
        self.preview_scale.setToolTip('Resize the finished capture image for display. Filters always use capture pixels.')
        self.preview_size_label = QtWidgets.QLabel()
        form.addRow("Preview size", self.preview_size_label)
        self.resolution_preset.currentIndexChanged.connect(self.select_resolution)
        self.output_w.valueChanged.connect(lambda: self.sync_capture('width', custom=True))
        self.output_h.valueChanged.connect(lambda: self.sync_capture('height', custom=True))
        self.width.valueChanged.connect(lambda: self.sync_capture())
        self.height.valueChanged.connect(lambda: self.sync_capture())
        self.sync_capture()
        self.quality.setToolTip('Changes apply automatically. Imported mesh refinement adds triangles; smoothing may alter shape. Select Original and set smoothing to zero to use the original geometry.')
        effect_widget = QtWidgets.QGroupBox("Image processing effects")
        form = QtWidgets.QFormLayout(effect_widget)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.add_sensor_section(sensor_layout, "Image processing effects", effect_widget, False)
        self.effects_enabled = QtWidgets.QCheckBox("Enable image effects")
        self.effects_enabled.setChecked(True)
        self.effects_enabled.toggled.connect(self.schedule)
        form.addRow(self.effects_enabled)
        self.effect_widgets = {}
        definitions = [
            ("read_noise", "Gaussian read noise", 0, .2, 0, .005),
            ("shot_noise", "Photon noise", 0, .2, 0, .005),
            ("speckle", "Multiplicative speckle", 0, .5, 0, .01),
            ("texture", "Gel texture strength", 0, .5, 0, .01),
            ("texture_scale_mm", "Texture size (mm)", .01, 2, .15, .01),
            ("hd_texture", "HD gel texture strength", 0, .5, 0, .005),
            ("hd_texture_scale_mm", "HD grain size (mm)", .005, .2, .025, .005),
            ("blur_px", "Optical blur σ (output px)", 0, 5, 0, .1),
            ("vignette", "Corner darkening", 0, 1, 0, .05),
            ("contrast", "Contrast", 0, 3, 1, .05),
            ("gamma", "Gamma", .1, 3, 1, .05),
        ]
        for name, label, low, high, value, step in definitions:
            self.effect_widgets[name] = self.control(form, label, low, high, value, step)
        self.effect_widgets['hd_texture'].setToolTip('Fine fixed gel grain. Zero disables it; 0.035 is a subtle starting point. Applied before optical blur.')
        self.effect_widgets['hd_texture_scale_mm'].setToolTip('Smaller values produce finer grain on a 2048-pixel source grid. This is an appearance scale, not measured surface roughness.')
        fine_texture = QtWidgets.QPushButton('Use fine gel texture')
        fine_texture.clicked.connect(self.use_fine_gel_texture)
        form.addRow(fine_texture)
        self.effect_seed = self.integer_control(form, "Pattern / noise seed", 0, 1_000_000, 0)
        new_seed = QtWidgets.QPushButton("New noise / texture sample")
        new_seed.clicked.connect(lambda: self.effect_seed.setValue((self.effect_seed.value()+1)%1_000_001))
        form.addRow(new_seed)
        reset_effects = QtWidgets.QPushButton("Reset image effects")
        reset_effects.clicked.connect(self.reset_effects)
        form.addRow(reset_effects)
        note = QtWidgets.QLabel("Noise strengths use 0–1 intensity. The seed fixes each sample for repeatable captures. "
                               "Texture is fixed to the gel area. Effects change RGB only; depth stays geometric. "
                               "Disable effects for clean RGB, including no lighting-panel noise.")
        note.setWordWrap(True)
        form.addRow(note)

        self.effect_presets = AppearancePresets(
            Path(__file__).resolve().parents[2] / "configs" / "effect_presets.json",
            lambda: asdict(ImageEffects(enabled=self.effects_enabled.isChecked(), seed=self.effect_seed.value(),
                **{name: widget.value() for name, widget in self.effect_widgets.items()})),
            self.apply_effect_preset, asdict(ImageEffects()), self.validate_effect_preset)
        form.insertRow(0, self.effect_presets)

    def validate_effect_preset(self, raw):
        value = ImageEffects(**raw)
        if not isinstance(value.enabled, bool) or not isinstance(value.seed, int) or not 0 <= value.seed <= 1_000_000:
            raise ValueError("Invalid effects enable flag or seed")
        for name, widget in self.effect_widgets.items():
            number = getattr(value, name)
            if not isinstance(number, (int, float)) or not np.isfinite(number) or not widget.minimum() <= number <= widget.maximum():
                raise ValueError(f"Invalid effect: {name}")

    def apply_effect_preset(self, raw):
        self.validate_effect_preset(raw)
        value = ImageEffects(**raw)
        for name, widget in self.effect_widgets.items():
            widget.setValue(getattr(value, name))
        self.effect_seed.setValue(value.seed)
        self.effects_enabled.setChecked(value.enabled)
        self.schedule()

    def integer_control(self, form, label, low, high, value, schedule=True):
        control = QtWidgets.QSpinBox()
        control.setRange(low, high)
        control.setValue(value)
        control.setKeyboardTracking(False)
        if schedule:
            control.valueChanged.connect(self.schedule)
        form.addRow(label, control)
        return control

    def select_resolution(self, index):
        target = self.resolution_preset.itemData(index)
        if target is None:
            return
        self.output_w.blockSignals(True)
        self.output_w.setValue(target)
        self.output_w.blockSignals(False)
        self.sync_capture()
        self.schedule()

    def sync_capture(self, anchor='width', custom=False, strict=False):
        try:
            width, height = linked_capture(self.output_w.value(), self.output_h.value(),
                                           self.width.value(), self.height.value(), anchor)
            for control, value in ((self.output_w, width), (self.output_h, height)):
                control.blockSignals(True)
                control.setValue(value)
                control.blockSignals(False)
            self.resolution_preset.blockSignals(True)
            for index in range(1, self.resolution_preset.count()):
                target = self.resolution_preset.itemData(index)
                w, h = linked_capture(target, height, self.width.value(), self.height.value())
                label = f'{w} × {h}' if w == target else f'{target} px target ({w} × {h})'
                self.resolution_preset.setItemText(index, label)
            if custom:
                self.resolution_preset.setCurrentIndex(0)
            self.resolution_preset.blockSignals(False)
            self.capture_geometry_label.setText(
                f'Sensor {self.width.value():g} × {self.height.value():g} mm → {width} × {height} px\n'
                f'Pixel size: {self.width.value()*1000/width:.2f} × {self.height.value()*1000/height:.2f} µm. '
                'Width and height follow the sensor aspect ratio.')
        except ValueError as exc:
            self.capture_geometry_label.setText(str(exc))
            if strict:
                raise

    def apply_quality(self, *_):
        if not self._has_object:
            return
        try:
            self.sim.set_quality(self.quality.currentData(), self.mesh_smoothing.value())
            self._scene_key = None
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not refine object", str(exc))

    def reset_quality(self):
        self.quality.blockSignals(True)
        self.mesh_smoothing.blockSignals(True)
        self.quality.setCurrentIndex(self.quality.findData(0))
        self.mesh_smoothing.setValue(0)
        self.quality.blockSignals(False)
        self.mesh_smoothing.blockSignals(False)
        self.apply_quality()

    def reset_effects(self):
        defaults = ImageEffects()
        for name, widget in self.effect_widgets.items():
            widget.setValue(getattr(defaults, name))
        self.effect_seed.setValue(0)
        self.effects_enabled.setChecked(True)
        self.schedule()

    def use_fine_gel_texture(self):
        self.effect_widgets['hd_texture'].setValue(.035)
        self.effect_widgets['hd_texture_scale_mm'].setValue(.025)
        self.effects_enabled.setChecked(True)
        self.schedule()

    def control(self, layout, label, low, high, value, step, schedule=True):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(3)
        box.setSingleStep(step)
        box.setValue(value)
        if schedule:
            box.valueChanged.connect(self.schedule)
        layout.addRow(label, box)
        return box

    def schedule(self, *_):
        if hasattr(self, 'calibration') and self.calibration.alignment_active:
            if not self.timer.isActive():
                self.timer.start(16)
        else:
            self.timer.start(75)

    def sync_slider(self, value):
        bottom = self.rotated_z_bounds()[0]
        self.z.blockSignals(True)
        self.z.setRange(-1000000, 1000000)
        self.z.setValue(-bottom-value)
        self.z.blockSignals(False)
        self.sync_z_bounds()
        self.schedule()

    def rotated_z_bounds(self):
        angles = (self.rx.value(), self.ry.value(), self.rz.value())
        key = (id(self.sim.mesh), *angles)
        cached = getattr(self, '_rotated_bounds_cache', None)
        if cached is None or cached[0] != key:
            z = self.sim.mesh.vertices @ Rotation.from_euler('xyz', angles, degrees=True).as_matrix()[2]
            self._rotated_bounds_cache = (key, (float(z.min()*1000), float(z.max()*1000)))
        return self._rotated_bounds_cache[1]

    def sync_z_bounds(self, *_):
        bottom, top = self.rotated_z_bounds()
        limit = self.max_penetration.value()
        clearance = max(2*(top-bottom), .1)
        lower, upper = -bottom-limit, -bottom+clearance
        value = min(upper, max(lower, self.z.value()))
        self.z.blockSignals(True)
        self.z.setRange(lower, upper)
        self.z.setValue(value)
        self.z.blockSignals(False)
        self.cut.blockSignals(True)
        self.cut.setValue(-bottom-self.z.value())
        self.cut.blockSignals(False)
        self.slider.blockSignals(True)
        offset = self.z.value()+bottom
        span = clearance if offset >= 0 else limit
        self.slider.setValue(round(1000*offset/span) if span > 0 else 0)
        self.slider.blockSignals(False)
        if hasattr(self, 'cut_readout'):
            self.cut_readout.setText(f'{self.z.value():.2f}\nContact: {-bottom:.2f}')
        if hasattr(self, 'gather_panel'):
            panels = [self.gather_panel]
            if hasattr(self, 'batch_gather_panel'):
                panels.append(self.batch_gather_panel)
            for panel in panels:
                for index, control in enumerate(panel.cut_range):
                    followed_limit = index == 1 and control.value() == control.maximum()
                    control.setRange(0, limit)
                    if followed_limit:
                        control.setValue(limit)

    def press_to_maximum(self):
        if self._has_object:
            self.sync_slider(self.max_penetration.value())
            self.refresh()

    def move_z_slider(self, position):
        bottom, top = self.rotated_z_bounds()
        span = max(2*(top-bottom), .1) if position >= 0 else self.max_penetration.value()
        self.z.setValue(-bottom+position/1000*span)
        self.sync_z_bounds()

    def reset_pose(self):
        for control in (self.rx, self.ry, self.rz, self.x, self.y):
            control.setValue(0)
        self.sync_slider(0)
        self._reset_camera = True
        self.schedule()

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load object",
            str(Path(__file__).resolve().parents[2] / 'assets'), "Meshes (*.stl *.obj *.ply *.glb)")
        if path:
            try:
                self.sim.load(path, self.units.currentText(), self.scale.value())
                self.sim.set_base_rotation(self.base_flip.currentData())
                self._has_object = True
                self.sim.set_quality(self.quality.currentData())
                self.mesh_smoothing.setValue(0)
                self.model_label.setText(Path(path).name)
                self.reset_pose()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not load mesh", str(exc))

    def load_demo(self, kind):
        self.sim.set_mesh(primitive(kind, .004), "primitive:"+kind)
        self.sim.set_base_rotation(self.base_flip.currentData())
        self._has_object = True
        self.sim.set_quality(self.quality.currentData())
        self.mesh_smoothing.setValue(0)
        self.sim.units, self.sim.scale = "m", 1.0
        self.scale.blockSignals(True)
        self.scale.setValue(1.)
        self.scale.blockSignals(False)
        self.model_label.setText("Demo " + kind + " • size 4 mm")
        self.reset_pose()

    def resize_object(self, value):
        if not self._has_object:
            return
        self.sim.set_scale(value)
        self.schedule()

    def change_base_flip(self, *_):
        self.sim.set_base_rotation(self.base_flip.currentData())
        # Replace mesh identities so the 3D actor and range overlays refresh.
        self.sim.mesh = self.sim.mesh.copy()
        self._scene_key = None
        self.schedule()

    def change_lighting(self, optics):
        self.sim.cfg.optics = deepcopy(optics)
        if self.sender() is not self.lighting:
            self.lighting.set_optics(optics)
        self.sync_appearance_panels()
        self.schedule()

    def sync_appearance_panels(self):
        optics = self.sim.cfg.optics
        self.measured_appearance.set_optics(optics)
        background_active = optics.uses_measured_background
        if optics.uses_lookup and optics.calibrated_background_enabled and not background_active:
            # A legacy lookup may carry its own no-contact frame.
            try:
                with np.load(optics.calibration_path, allow_pickle=False) as archive:
                    background_active = 'background' in archive
            except (OSError, ValueError):
                pass
        self.gel_panel.set_measured_background_active(background_active)

    def change_gel(self, settings):
        self.sim.gel_lighting = deepcopy(settings)
        self.schedule()

    def apply_controls(self):
        self.sync_capture(strict=True)
        s = self.sim
        s.rotation = (self.rx.value(), self.ry.value(), self.rz.value())
        s.offset = (self.x.value()/1000, self.y.value()/1000)
        self.sync_z_bounds()
        s.max_penetration = self.max_penetration.value()/1000
        s.object_z = self.z.value()/1000
        s.softness = self.softness.value()/1000
        s.cfg.gel.size_x, s.cfg.gel.size_y = self.width.value()/1000, self.height.value()/1000
        s.cfg.camera.max_depth = self.depth_range.value()/1000
        self.capture_resolution = (self.output_w.value(), self.output_h.value())
        s.cfg.camera.width, s.cfg.camera.height = self.preview_dimensions()
        self.preview_size_label.setText(f"{s.cfg.camera.width} × {s.cfg.camera.height} px")
        s.effects = ImageEffects(enabled=self.effects_enabled.isChecked(), seed=self.effect_seed.value(),
                                 **{name: control.value() for name, control in self.effect_widgets.items()})
        if self.calibration.alignment_active:
            geometry = self.calibration.settings['sensor']['gel']
            s.cfg.gel.size_x, s.cfg.gel.size_y = geometry['size_x'], geometry['size_y']

    def refresh(self):
        try:
            self.apply_controls()
            if self.calibration.alignment_active:
                self.gather_panel.setEnabled(False)
                self.slider.setEnabled(self._has_object)
                self.press_button.setEnabled(self._has_object)
                self.calibration.refresh_alignment()
                self.update_scene()
                self.statusBar().showMessage('Calibration mode | Match the image, then Save reference | Preview updates in background')
                return
            self.gather_panel.update_destination()
            self.gather_panel.setEnabled(self._has_object and self.batch_gather_panel.worker is None)
            self.slider.setEnabled(self._has_object)
            self.press_button.setEnabled(self._has_object)
            depth, rgb = self.render_capture_preview()
            self.depth_view.set_image(GelSightRenderer.depth_colormap(depth, self.sim.cfg.camera.max_depth))
            self.rgb_view.set_image(rgb)
            selected = self.preview_grid.selected()
            if 'clean' in selected:
                self.preview_grid.views['clean'].set_image(self.sim.clean_rgb)
            if 'raw' in selected:
                self.preview_grid.views['raw'].set_image(GelSightRenderer.depth_colormap(self.sim.raw_depth, self.sim.cfg.camera.max_depth))
            if 'contact' in selected:
                mask = (self.sim.raw_depth > 0).astype(np.uint8)*255
                self.preview_grid.views['contact'].set_image(np.repeat(mask[..., None], 3, axis=2))
            if 'normals' in selected:
                normals = self.sim.sampler.normals(-depth)
                self.preview_grid.views['normals'].set_image(np.clip((normals+1)*127.5, 0, 255).astype(np.uint8))
            # Downsample finished pixels, not filter inputs: blur and seeded
            # camera noise must have the same meaning as in exported images.
            target = QtCore.QSize(*self.preview_dimensions())
            for name in set(selected) | {'rgb', 'depth'}:
                view = self.preview_grid.views.get(name)
                if view is not None and view.original is not None:
                    view.original = view.original.scaled(target, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                    view._fit()
            self.depth_label.setText(f"0–{self.depth_range.value():.3f} mm scale | peak {depth.max()*1000:.3f} mm")
            self.update_scene()
            contact = np.count_nonzero(self.sim.raw_depth)/depth.size*100
            self.statusBar().showMessage(f"Geometric mode | {len(self.sim.mesh.faces):,} triangles | contact {contact:.1f}% | "
                                        f"preview {target.width()} × {target.height()}")
        except Exception as exc:
            self.statusBar().showMessage(f"Render failed: {exc}")

    def render_capture_preview(self):
        """Reuse one full-resolution render while only display settings change."""
        s = self.sim
        s.cfg.camera.width, s.cfg.camera.height = self.capture_resolution
        key = (id(s), id(s.mesh), repr(asdict(s.cfg)), tuple(s.rotation), tuple(s.offset),
               s.cut_depth if self._has_object else 0, s.max_penetration, s.softness,
               repr(asdict(s.effects)), repr(asdict(s.gel_lighting)), self._has_object)
        cached = getattr(self, '_capture_preview_cache', None)
        if cached is None or cached[0] != key:
            saved_cut = s.cut_depth
            try:
                if not self._has_object:
                    s.cut_depth = 0
                depth, rgb = s.render()
            finally:
                s.cut_depth = saved_cut
            self._capture_preview_cache = (key, depth, rgb, s.raw_depth, s.clean_rgb)
        _, depth, rgb, s.raw_depth, s.clean_rgb = self._capture_preview_cache
        return depth, rgb

    def current_tactile_image(self):
        """Render the current tactile image at the configured capture resolution.

        Prediction deliberately uses capture dimensions, never the reduced preview
        grid, so the classifier receives the same input size saved by gathering.
        """
        if not self._has_object:
            raise ValueError('Load an object before requesting a tactile image')
        self.apply_controls()
        _, tactile = self.render_capture_preview()
        return tactile

    def preview_dimensions(self):
        factor = self.preview_scale.value()/100
        return (max(2, round(self.output_w.value()*factor)), max(2, round(self.output_h.value()*factor)))

    def update_scene(self):
        s = self.sim
        self.update_gather_overlay(render=False)
        if not self._has_object:
            self.toggle_gizmo(False)
            self.plot.remove_actor('object', render=False)
            self.plot.remove_actor('cut', render=False)
            self.object_actor = self._actor_mesh = None
            self.plot.add_mesh(pv.Plane(i_size=s.cfg.gel.size_x*1000, j_size=s.cfg.gel.size_y*1000),
                               name="plane", color="#4abcca", opacity=.35)
            self.plot.camera_position = "iso"
            return
        key = (id(s.mesh), *s.rotation, *s.offset, s.cfg.gel.size_x, s.cfg.gel.size_y, s.cut_depth)
        if key == self._scene_key:
            self.plot.render()
            return
        self.toggle_gizmo(False)
        faces = np.column_stack((np.full(len(s.mesh.faces), 3), s.mesh.faces)).ravel()
        if self._actor_mesh is not s.mesh:
            local_poly = pv.PolyData(s.mesh.vertices*1000, faces)
            self.object_actor = self.plot.add_mesh(local_poly, name="object", color="#c9ae7b", opacity=.45,
                                                  smooth_shading=True, split_sharp_edges=True, reset_camera=False)
            self._actor_mesh = s.mesh
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", s.rotation, degrees=True).as_matrix()
        pose[:3, 3] = [s.offset[0]*1000, s.offset[1]*1000, s.object_z*1000]
        self.object_actor.user_matrix = pose
        vertices = Rotation.from_euler('xyz', s.rotation, degrees=True).apply(s.mesh.vertices)*1000
        vertices[:, :2] += np.asarray(s.offset)*1000
        vertices[:, 2] += s.object_z*1000
        poly = pv.PolyData(vertices, faces)
        imprint = poly.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=True)
        self.plot.remove_actor("cut", render=False)
        if imprint.n_points:
            self.plot.add_mesh(imprint, name="cut", color="#ffa638", reset_camera=False)
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                         i_size=s.cfg.gel.size_x*1000, j_size=s.cfg.gel.size_y*1000)
        self.plot.add_mesh(plane, name="plane", color="#4abcca", opacity=.35, show_edges=True, reset_camera=False)
        if self._reset_camera:
            self.plot.camera_position = "iso"
            self.plot.reset_camera()
            self._reset_camera = False
        self.plot.render()
        self._scene_key = key
        self.toggle_gizmo(True)

    def update_gather_overlay(self, render=True):
        """Preview gathering bounds using display-only copies of the object."""
        panel = self.batch_gather_panel if self.tabs.currentWidget() is self.batch_gather_panel else self.gather_panel
        visible = (self._has_object and self.tabs.currentWidget() is panel
                   and panel.position_overlay.isChecked()
                   and not self.calibration.alignment_active)
        positions = panel.xy_preview_positions() if visible else []
        s = self.sim
        rotation_visible = (self._has_object and self.tabs.currentWidget() is panel
                            and panel.rotation_overlay.isChecked() and not self.calibration.alignment_active)
        rotations = panel.rotation_preview_poses(s.rotation) if rotation_visible else []
        key = (id(s.mesh), tuple(s.rotation), tuple(s.offset), s.object_z, tuple(positions), tuple(rotations))
        if getattr(self, '_gather_overlay_key', None) == key:
            return
        for name in getattr(self, '_gather_overlay_names', []):
            self.plot.remove_actor(name, render=False)
        self._gather_overlay_names = []
        if positions or rotations:
            faces = np.column_stack((np.full(len(s.mesh.faces), 3), s.mesh.faces)).ravel()
            poly = pv.PolyData(s.mesh.vertices*1000, faces)
        if positions:
            rotation = Rotation.from_euler('xyz', s.rotation, degrees=True).as_matrix()
            for index, (x, y) in enumerate(positions):
                name = f'gather_xy_{index}'
                actor = self.plot.add_mesh(poly, name=name, color='#8ce8ff', opacity=.45,
                                           show_edges=True, edge_color='#8ce8ff', lighting=False,
                                           pickable=False, reset_camera=False, render=False)
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = (x, y, s.object_z*1000)
                actor.user_matrix = pose
                actor.SetUseBounds(False)
                self._gather_overlay_names.append(name)
            xs = sorted(set(x for x, _ in positions))
            ys = sorted(set(y for _, y in positions))
            points = []
            for x in xs:
                points.extend(((x, ys[0], 0), (x, ys[-1], 0)))
            for y in ys:
                points.extend(((xs[0], y, 0), (xs[-1], y, 0)))
            grid = self.plot.add_lines(np.asarray(points), connected=False,
                                       name='gather_xy_grid', color='#8ce8ff', width=1)
            grid.SetPickable(False)
            grid.SetUseBounds(False)
            self._gather_overlay_names.append('gather_xy_grid')
            self.plot.add_text(f'XY range preview - {len(positions)} positions\n'
                               f'X: {xs[0]:g} to {xs[-1]:g} mm; Y: {ys[0]:g} to {ys[-1]:g} mm\n'
                               'Min / midpoint / max; current rotation and Z',
                               name='gather_xy_caption', position='upper_left', font_size=10,
                               color='#8ce8ff')
            self._gather_overlay_names.append('gather_xy_caption')
        for index, (axis, angles) in enumerate(rotations):
            name = f'gather_rotation_{index}'
            color = ('#ff7070', '#75ef92', '#719dff')[axis]
            actor = self.plot.add_mesh(poly, name=name, color=color, opacity=.35,
                                       show_edges=True, edge_color=color, lighting=False,
                                       pickable=False, reset_camera=False, render=False)
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
            pose[:3, 3] = (*np.asarray(s.offset)*1000, s.object_z*1000)
            actor.user_matrix = pose
            actor.SetUseBounds(False)
            self._gather_overlay_names.append(name)
        if rotations:
            self.plot.add_text('Rotation ranges: X red, Y green, Z blue\n'
                               'Min / midpoint / max - one axis at a time; fixed object Z',
                               name='gather_rotation_caption', position='lower_left', font_size=10,
                               color='white')
            self._gather_overlay_names.append('gather_rotation_caption')
        self._gather_overlay_key = key
        if render:
            self.plot.render()

    def enable_contact_mask_output(self, enabled):
        if enabled:
            self.gather_panel.output_checks['contact'].setChecked(True)

    def focus_gather_overlay(self):
        self.update_gather_overlay(render=False)
        actors = [self.plot.renderer.actors[name] for name in self._gather_overlay_names
                  if name.startswith('gather_xy_') and name.rsplit('_', 1)[-1].isdigit()]
        if not actors:
            return
        bounds = np.asarray([actor.bounds for actor in actors])
        combined = tuple(value for axis in range(3)
                         for value in (bounds[:, 2*axis].min(), bounds[:, 2*axis+1].max()))
        self.plot.view_xy(render=False)
        self.plot.reset_camera(bounds=combined, render=False)
        self.plot.render()

    def save_capture(self):
        self.capture_to_folder()

    def closeEvent(self, event):
        if self.validation_panel.worker is not None:
            self.validation_panel.cancel_validation()
            self.statusBar().showMessage("Saving partial validation results. Close again once validation has stopped.")
            event.ignore()
            return
        self.timer.stop()
        self.calibration.timer.stop()
        self.calibration.settle_timer.stop()
        if self.calibration.preview_worker is not None:
            self.calibration.preview_pending = False
            self.calibration.timer.stop()
            self.calibration.preview_worker.finished.connect(self.close)
            self.statusBar().showMessage('Finishing calibration preview before closing…')
            event.ignore()
            return
        if self.calibration.worker is not None:
            self.calibration.worker.stop.set()
            self.statusBar().showMessage("Cancelling calibration. Close again once fitting has stopped.")
            event.ignore()
            return
        if self.batch_gather_panel.worker is not None:
            self._close_after_gather = True
            self.batch_gather_panel.stop_batch()
            event.ignore()
            return
        if self.gather_worker is not None:
            self._close_after_gather = True
            self.stop_gather()
            event.ignore()
            return
        self.timer.stop()
        self.sensor_configs.scan_timer.stop()
        self.settings_window.close()
        self.toggle_gizmo(False)
        self.plot.close()
        super().closeEvent(event)


def run(sensor=None, mesh=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Mesh2Tact")
    window = GeometricWindow(sensor=sensor, mesh=mesh)
    window.show()
    return app.exec_()
