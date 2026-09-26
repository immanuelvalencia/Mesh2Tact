"""GelSight collection window using the Mesh2Tact desktop controls."""
import json
import queue
import sys
import threading
from pathlib import Path
import cv2
import numpy as np
from .qt import QtCore, QtGui, QtWidgets
from .image_widgets import ImageView, style_action
from ..gelsight import DEFAULTS, RawImageWriter, contact_mask, label_path

SETTINGS = Path(__file__).resolve().parents[2] / 'configs' / 'gelsight_collection.json'


class CameraWorker(QtCore.QThread):
    def __init__(self):
        super().__init__()
        self.commands = queue.Queue()
        self.latest = None
        self.lock = threading.Lock()

    def publish(self, raw, mask, message, armed, reference):
        with self.lock:
            self.latest = (raw, mask, message, armed, reference)

    def run(self):
        cap = None
        raw = reference = writer = None
        cfg = dict(DEFAULTS)
        message = 'Connect the sensor, then capture an untouched reference.'
        try:
            while not self.isInterruptionRequested():
                try:
                    while True:
                        action, value = self.commands.get_nowait()
                        if action in ('connect', 'settings'):
                            reset_reference = any(cfg[key] != value[key] for key in
                                ('source', 'frame_scale', 'abs_thresh', 'abs_blur', 'close_size', 'open_size'))
                            cfg = value
                            writer = None
                            if action == 'connect':
                                if cap is not None:
                                    cap.release()
                                raw = reference = None
                                message = 'Opening camera…'
                                self.publish(None, None, message, False, False)
                                cap = cv2.VideoCapture(cfg['source'])
                                if not cap.isOpened():
                                    cap.release()
                                    cap = None
                                    message = 'Camera unavailable. Check the source and click Refresh / connect.'
                            elif reset_reference:
                                raw = reference = None
                                message = 'Sensor settings changed. Capture a new untouched reference.'
                        elif action == 'reference' and raw is not None:
                            reference = raw.copy()
                            writer = None
                            message = 'Reference captured. Ready to arm.'
                        elif action == 'arm':
                            if reference is not None:
                                writer = RawImageWriter(label_path(cfg['dataset_dir'], cfg['label']))
                                message = 'Armed — waiting for contact.'
                        elif action == 'stop':
                            writer = None
                            message = 'Stopped.'
                except queue.Empty:
                    pass
                mask = None
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.release()
                        cap = None
                        raw = reference = writer = None
                        message = 'Camera disconnected. Refresh / connect, then capture a new reference.'
                    else:
                        scale = cfg['frame_scale']
                        raw = cv2.resize(frame, (max(1, round(frame.shape[1]*scale)),
                                                max(1, round(frame.shape[0]*scale))),
                                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                        if reference is not None:
                            if reference.shape != raw.shape:
                                reference = writer = None
                                message = 'Camera resolution changed. Capture a new reference.'
                            else:
                                mask = contact_mask(raw, reference, cfg['abs_thresh'], cfg['abs_blur'],
                                                    cfg['close_size'], cfg['open_size'])
                                if writer is not None:
                                    try:
                                        saved = writer.accept(raw, mask, cfg['auto_capture_threshold'])
                                        message = (f'Saved {writer.total} images • {saved}' if saved else
                                                   f'Armed — waiting for contact. Saved {writer.total} images.')
                                    except Exception as exc:
                                        writer = None
                                        message = f'Capture stopped: {exc}'
                self.publish(raw, mask, message, writer is not None, reference is not None)
                self.msleep(15 if cap is not None else 50)
        except Exception as exc:
            self.publish(None, None, f'Camera stopped: {exc}. Close and reopen this window.', False, False)
        finally:
            if cap is not None:
                cap.release()


class GelSightWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_path=None):
        super().__init__()
        self.settings_path = Path(settings_path or SETTINGS)
        self.setWindowTitle('Mesh2Tact — GelSight collection')
        self.resize(1180, 740)
        cfg = dict(DEFAULTS)
        load_error = ''
        if self.settings_path.exists():
            try:
                saved = json.loads(self.settings_path.read_text(encoding='utf-8'))
                cfg.update({key: saved[key] for key in cfg if key in saved})
            except Exception as exc:
                load_error = f'Could not restore settings: {exc}'
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter)
        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(410)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        heading = QtWidgets.QLabel('GelSight collection')
        heading.setStyleSheet('font-size: 20px; font-weight: 600; padding: 6px;')
        sidebar_layout.addWidget(heading)
        self.tabs = QtWidgets.QTabWidget()
        sidebar_layout.addWidget(self.tabs, 1)
        splitter.addWidget(sidebar)
        tab_layouts = []
        for title in ('Sensor && mask', 'Data gathering'):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            panel = QtWidgets.QWidget()
            scroll.setWidget(panel)
            tab_layouts.append(QtWidgets.QVBoxLayout(panel))
            self.tabs.addTab(scroll, title)
        sensor_controls, gather_controls = tab_layouts
        self.fields = {}
        for title, target, specs in (
            ('Camera', sensor_controls, [('source', 'Camera index', 0, 32, 0),
                                        ('frame_scale', 'Resolution scale', .05, 2, 2)]),
            ('Contact mask', sensor_controls, [('abs_thresh', 'Difference threshold', 0, 1, 3),
                              ('abs_blur', 'Gaussian blur', 1, 51, 0),
                              ('close_size', 'Close gaps', 1, 51, 0),
                              ('open_size', 'Remove specks', 1, 51, 0)]),
            ('Capture condition', gather_controls, [('auto_capture_threshold', 'Minimum contact pixels', 1, 100000000, 0)])):
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            target.addWidget(box)
            for key, label, low, high, decimals in specs:
                spin = QtWidgets.QDoubleSpinBox() if decimals else QtWidgets.QSpinBox()
                if decimals:
                    spin.setDecimals(decimals)
                    spin.setSingleStep(.01)
                spin.setRange(low, high)
                try:
                    spin.setValue(float(cfg[key]) if decimals else int(cfg[key]))
                except (ValueError, TypeError, OverflowError):
                    spin.setValue(DEFAULTS[key])
                form.addRow(label, spin)
                self.fields[key] = spin
        dataset = QtWidgets.QGroupBox('Dataset')
        dataset_layout = QtWidgets.QVBoxLayout(dataset)
        dataset_layout.addWidget(QtWidgets.QLabel('Class / object label'))
        self.class_choice = QtWidgets.QComboBox()
        self.class_choice.setEditable(True)
        root = Path(DEFAULTS['dataset_dir'])
        self.class_choice.addItems(sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else [])
        self.class_choice.setCurrentText(str(cfg['label']))
        self.fields['label'] = self.class_choice.lineEdit()
        dataset_layout.addWidget(self.class_choice)
        self.destination = QtWidgets.QLabel()
        self.destination.setWordWrap(True)
        self.destination.setMinimumWidth(0)
        self.destination.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.destination.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        dataset_layout.addWidget(self.destination)
        gather_controls.insertWidget(0, dataset)
        self.connect_button = QtWidgets.QPushButton('Refresh / connect camera')
        self.gather_refresh = QtWidgets.QPushButton('Refresh camera  ·  F5')
        self.reference_button = QtWidgets.QPushButton('Capture reference (untouched)')
        self.gather_reference = QtWidgets.QPushButton('Capture reference (untouched)')
        self.arm_button = QtWidgets.QPushButton('Start auto gathering')
        self.stop_button = QtWidgets.QPushButton('Stop gathering')
        for button, color, target in (
                (self.connect_button, '#1769aa', sensor_controls),
                (self.reference_button, '#14777b', sensor_controls),
                (self.arm_button, '#287348', gather_controls),
                (self.stop_button, '#986018', gather_controls),
                (self.gather_refresh, '#1769aa', gather_controls),
                (self.gather_reference, '#14777b', gather_controls)):
            style_action(button, color)
            target.addWidget(button)
        for target, text in (
            (sensor_controls, 'Remove contact before capturing the reference. Difference threshold uses a 0–1 range. Blur and cleanup sizes round up to odd kernels.'),
            (gather_controls, 'Images save automatically while the mask meets the minimum pixel count. Only raw RGB images are saved directly in the class folder. Indexing continues across contacts and restarts. The mask is used for detection and preview only.')):
            note = QtWidgets.QLabel(text)
            note.setWordWrap(True)
            note.setMinimumWidth(0)
            note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            target.addWidget(note)
            target.addStretch()
        self.settings_status = QtWidgets.QLabel(load_error or 'Settings restore automatically.')
        self.settings_status.setWordWrap(True)
        sidebar_layout.addWidget(self.settings_status)
        preview_panel = QtWidgets.QWidget()
        previews = QtWidgets.QVBoxLayout(preview_panel)
        previews.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 730])
        self.views = []
        for title in ('Raw GelSight', 'Contact mask'):
            box = QtWidgets.QGroupBox(title)
            box_layout = QtWidgets.QVBoxLayout(box)
            view = ImageView()
            view.setStyleSheet('background: #151c25; color: #dbe5ef')
            view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            box_layout.addWidget(view)
            previews.addWidget(box, 1)
            self.views.append(view)
        self.metrics = QtWidgets.QLabel('No frames')
        previews.addWidget(self.metrics)
        self.status = QtWidgets.QLabel('Refresh / connect to begin.')
        self.status.setWordWrap(True)
        previews.addWidget(self.status)
        self.refresh_action = QtWidgets.QAction('Refresh camera', self)
        self.refresh_action.setShortcut(QtGui.QKeySequence('F5'))
        self.addAction(self.refresh_action)
        self.refresh_action.triggered.connect(self.refresh_camera)
        self.gather_refresh.clicked.connect(self.refresh_camera)
        self.gather_reference.clicked.connect(lambda: self.send('reference'))
        self.update_destination()
        self.worker = CameraWorker()
        self.worker.commands.put(('settings', self.settings()))
        self.worker.start()
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.connect_button.clicked.connect(self.refresh_camera)
        self.reference_button.clicked.connect(lambda: self.send('reference'))
        self.arm_button.clicked.connect(self.arm)
        self.stop_button.clicked.connect(lambda: self.send('stop'))
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.persist)
        for field in self.fields.values():
            signal = field.textChanged if isinstance(field, QtWidgets.QLineEdit) else field.valueChanged
            signal.connect(self.changed)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(50)

    def settings(self):
        return dict(dataset_dir=DEFAULTS['dataset_dir'], **{
            key: field.text() if isinstance(field, QtWidgets.QLineEdit) else field.value()
            for key, field in self.fields.items()})

    def send(self, action, value=None):
        self.worker.commands.put((action, value))

    def changed(self, *_):
        self.update_destination()
        self.send('settings', self.settings())
        self.save_timer.start(300)

    def persist(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.settings_path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.settings(), indent=2), encoding='utf-8')
            temp.replace(self.settings_path)
            self.settings_status.setText('Settings saved automatically.')
        except OSError as exc:
            self.settings_status.setText(f'Settings could not be saved: {exc}')

    def update_destination(self):
        try:
            destination = label_path(DEFAULTS['dataset_dir'], self.fields['label'].text())
            self.destination.setText(f'Automatic save location:\ndata/gelsight/{destination.name}/')
            self.destination.setToolTip(str(destination))
        except ValueError as exc:
            self.destination.setText(str(exc))

    def refresh_camera(self):
        self.send('connect', self.settings())

    def arm(self):
        try:
            cfg = self.settings()
            label_path(cfg['dataset_dir'], cfg['label'])
            self.persist()
            self.send('arm')
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Data gathering', str(exc))

    def update_preview(self):
        with self.worker.lock:
            snapshot = self.worker.latest
        if snapshot is None:
            return
        raw, mask, message, armed, reference = snapshot
        for view, pixels in zip(self.views, (raw, mask)):
            if pixels is None:
                view.original = None
                view.clear()
                view.setText('Waiting for camera' if raw is None else 'Capture a reference')
            else:
                rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB if pixels.ndim == 3 else cv2.COLOR_GRAY2RGB)
                view.set_image(rgb)
        self.status.setText(message)
        self.metrics.setText('No frames' if raw is None else
            f'{raw.shape[1]} × {raw.shape[0]} px • Contact: {np.count_nonzero(mask) if mask is not None else 0:,} / {self.fields["auto_capture_threshold"].value():,} px')
        self.arm_button.setEnabled(reference and not armed)
        self.reference_button.setEnabled(raw is not None and not armed)
        self.stop_button.setEnabled(armed)
        for field in self.fields.values():
            field.setEnabled(not armed)
        self.class_choice.setEnabled(not armed)
        self.gather_reference.setEnabled(raw is not None and not armed)

    def closeEvent(self, event):
        self.timer.stop()
        self.save_timer.stop()
        self.shutdown()
        event.accept()

    def shutdown(self):
        self.persist()
        self.worker.requestInterruption()
        self.worker.wait()


def run():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = GelSightWindow()
    window.show()
    return app.exec_()
