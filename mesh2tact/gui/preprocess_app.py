"""Selection, aligned inspection, recovery and export for preprocessing."""
from pathlib import Path
import json
from types import SimpleNamespace

from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageOps

from preprocess import (VARIANTS, scan_variants, prepare_variants, sample_inventory,
                        mask_contact_pixels, quarantine_samples, restore_quarantine,
                        scan_dataset, prepare_dataset, raw_inventory)


class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, task, options, parent=None):
        super().__init__(parent)
        self.task, self.options = task, options

    def run(self):
        try:
            if self.options.get('gelsight'):
                opts = dict(self.options)
                root = opts.pop('root')
                if self.task == 'scan':
                    labels, records, skipped = scan_dataset(root, opts['group_by_subfolder'],
                                                           opts['selected_labels'], gelsight=True)
                    inventory = raw_inventory(root, labels)
                    usable = {(r.label, r.source.relative_to(Path(root).resolve()/r.label).as_posix()) for r in records}
                    excluded = [dict(label=key[0], key=key[1], reason='Duplicate or empty image')
                                for key in inventory if key not in usable]
                    scan = SimpleNamespace(labels=labels, summary=dict(
                        excluded_samples=excluded, matched_total=len(records), issues=skipped, issue_examples=[]))
                    self.done.emit((scan, inventory, {key: 'Not applicable' for key in inventory}))
                else:
                    summary = prepare_dataset(source=root, progress=self.progress.emit, **opts)
                    summary['exported_total'] = summary['total_images']
                    self.done.emit(summary)
            elif self.task == 'scan':
                scan = scan_variants(**self.options, progress=self.progress.emit)
                inventory = sample_inventory(self.options['root'], self.options['variants'], self.options['selected_labels'])
                masks = sample_inventory(self.options['root'], ('mask',), self.options['selected_labels'])
                contacts = {}
                for key in inventory:
                    paths = masks.get(key, {}).get('mask', [])
                    try:
                        contacts[key] = mask_contact_pixels(paths[0]) if len(paths) == 1 else 'Missing / ambiguous mask'
                    except (OSError, ValueError) as exc:
                        contacts[key] = f'Invalid mask: {exc}'
                self.done.emit((scan, inventory, contacts))
            else:
                self.done.emit(prepare_variants(**self.options, progress=self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class Preview(QtWidgets.QLabel):
    def __init__(self):
        super().__init__('Select a sample')
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(180, 130)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet('background: #18202a; color: #eeeeee;')
        self.original = None

    def load(self, paths):
        self.original = None
        self.clear()
        if len(paths) != 1:
            self.setText('Missing' if not paths else 'Ambiguous: multiple files')
            return
        try:
            with Image.open(paths[0]) as raw:
                image = ImageOps.exif_transpose(raw).convert('RGB')
                data = image.tobytes()
                qimage = QtGui.QImage(data, image.width, image.height, image.width*3, QtGui.QImage.Format_RGB888).copy()
                self.original = QtGui.QPixmap.fromImage(qimage)
            self.setToolTip(str(paths[0]) + '\nDouble-click for full resolution')
            self.fit()
        except (OSError, ValueError) as exc:
            self.setText(f'Unreadable image\n{exc}')

    def fit(self):
        if self.original is not None:
            self.setPixmap(self.original.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def mouseDoubleClickEvent(self, event):
        if self.original is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Full-resolution image')
        dialog.resize(1000, 750)
        layout = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea()
        label = QtWidgets.QLabel()
        label.setPixmap(self.original)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.exec_()


class PreprocessWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mesh2Tact — aligned dataset preparation and inspection')
        self.resize(1250, 940)
        self.worker = None
        self.scan_result = None
        self.inventory, self.contacts = {}, {}
        self.keys, self.undo_stack = [], []
        layout = QtWidgets.QVBoxLayout(self)
        self.inputs = QtWidgets.QWidget()
        top = QtWidgets.QVBoxLayout(self.inputs)
        top.setContentsMargins(0, 0, 0, 0)
        self.dataset_kind = QtWidgets.QComboBox()
        self.dataset_kind.addItem('Mesh2Tact  -  aligned image branches', 'aligned')
        self.dataset_kind.addItem('GelSight  -  one folder of classes (raw RGB)', 'gelsight')
        top.addWidget(self.dataset_kind)
        self.source, self.output = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for title, field, handler in [('Source data folder', self.source, self.pick_source),
                                      ('New output folder', self.output, self.pick_output)]:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(title))
            row.addWidget(field, 1)
            button = QtWidgets.QPushButton('Browse…')
            button.clicked.connect(handler)
            row.addWidget(button)
            top.addLayout(row)
        self.source.editingFinished.connect(self.discover)
        self.source.textChanged.connect(self.invalidate)
        row = QtWidgets.QHBoxLayout()
        self.variant_checks = {}
        row.addWidget(QtWidgets.QLabel('Dataset types:'))
        for name in VARIANTS:
            box = QtWidgets.QCheckBox(name.title())
            box.setEnabled(False)
            box.toggled.connect(self.branch_changed)
            row.addWidget(box)
            self.variant_checks[name] = box
        row.addStretch()
        top.addLayout(row)
        classes_row = QtWidgets.QHBoxLayout()
        classes_row.addWidget(QtWidgets.QLabel('Classes to process:'))
        self.classes = QtWidgets.QListWidget()
        self.classes.setMaximumHeight(90)
        self.classes.itemChanged.connect(self.invalidate)
        classes_row.addWidget(self.classes, 1)
        for title, checked in [('All', True), ('None', False)]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _, value=checked: self.check_classes(value))
            classes_row.addWidget(button)
        top.addLayout(classes_row)
        options = QtWidgets.QHBoxLayout()
        self.group = QtWidgets.QCheckBox('Keep trial/run subfolders in one split')
        self.group.setChecked(True)
        self.contact = QtWidgets.QCheckBox('Exclude samples failing mask contact check')
        self.pixels = QtWidgets.QSpinBox()
        self.pixels.setRange(1, 100000000)
        self.pixels.setValue(1)
        self.pixels.setPrefix('Min pixels: ')
        for widget in (self.group, self.contact, self.pixels):
            options.addWidget(widget)
        self.group.toggled.connect(self.invalidate)
        self.contact.toggled.connect(self.invalidate)
        self.pixels.valueChanged.connect(self.invalidate)
        top.addLayout(options)
        split_row = QtWidgets.QHBoxLayout()
        self.ratios = []
        for title, value in [('Train % ', 70), ('Val % ', 15), ('Test % ', 15)]:
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 98)
            spin.setValue(value)
            spin.setPrefix(title)
            self.ratios.append(spin)
            split_row.addWidget(spin)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        self.seed.setPrefix('Seed: ')
        split_row.addWidget(self.seed)
        self.balance = QtWidgets.QCheckBox('Balance classes per split')
        split_row.addWidget(self.balance)
        top.addLayout(split_row)
        note = QtWidgets.QLabel('Only complete, usable matches are exported. Every selected type uses identical sample IDs, labels and splits. Masks remain binary. Inspecting one class is allowed; training export needs at least two classes and three independent groups per class.')
        note.setWordWrap(True)
        self.dataset_note = note
        top.addWidget(note)
        layout.addWidget(self.inputs)
        actions = QtWidgets.QHBoxLayout()
        self.scan_button = QtWidgets.QPushButton('Scan selected datasets')
        self.scan_button.clicked.connect(self.scan)
        self.prepare_button = QtWidgets.QPushButton('Export aligned datasets')
        self.prepare_button.clicked.connect(self.prepare)
        self.prepare_button.setEnabled(False)
        actions.addWidget(self.scan_button)
        actions.addWidget(self.prepare_button)
        layout.addLayout(actions)
        split = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText('Filter by class, filename or inspection status')
        self.filter.textChanged.connect(self.filter_rows)
        left_layout.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Class', 'Sample filename / ID', 'Contact pixels', 'Status'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.currentCellChanged.connect(self.show_sample)
        left_layout.addWidget(self.table)
        nav = QtWidgets.QHBoxLayout()
        for title, step in [('Previous', -1), ('Next', 1)]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _, value=step: self.navigate(value))
            nav.addWidget(button)
        left_layout.addLayout(nav)
        self.delete_button = QtWidgets.QPushButton('Remove selected samples from ALL types')
        self.delete_button.setToolTip('Moves all matching image types, including unchecked types, to .inspection-trash. Recoverable with Undo.')
        self.delete_button.clicked.connect(self.remove_selected)
        self.undo_button = QtWidgets.QPushButton('Undo last removal')
        self.undo_button.clicked.connect(self.undo)
        left_layout.addWidget(self.delete_button)
        left_layout.addWidget(self.undo_button)
        split.addWidget(left)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.identity = QtWidgets.QLabel('Choose a source, check datasets, then scan.')
        self.identity.setWordWrap(True)
        right_layout.addWidget(self.identity)
        grid = QtWidgets.QGridLayout()
        self.cards, self.views = {}, {}
        for i, name in enumerate((*VARIANTS, "gelsight")):
            card = QtWidgets.QGroupBox(name.title())
            card_layout = QtWidgets.QVBoxLayout(card)
            view = Preview()
            card_layout.addWidget(view)
            grid.addWidget(card, i//2, i%2)
            self.cards[name], self.views[name] = card, view
        right_layout.addLayout(grid, 1)
        split.addWidget(right)
        split.setSizes([490, 700])
        layout.addWidget(split, 1)
        self.bar = QtWidgets.QProgressBar()
        layout.addWidget(self.bar)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(110)
        layout.addWidget(self.status)
        self.dataset_kind.currentIndexChanged.connect(self.discover)
        self.cards['gelsight'].hide()

    def is_gelsight(self):
        return self.dataset_kind.currentData() == 'gelsight'

    def selected(self):
        if self.is_gelsight():
            return ['gelsight']
        return [name for name, box in self.variant_checks.items() if box.isChecked() and box.isEnabled()]

    def labels(self):
        return [self.classes.item(i).text() for i in range(self.classes.count())
                if self.classes.item(i).checkState() == QtCore.Qt.Checked]

    def invalidate(self, *_):
        self.scan_result = None
        if hasattr(self, 'prepare_button'):
            self.prepare_button.setEnabled(False)
            self.table.setRowCount(0)
            self.keys = []
            self.inventory = {}
            for view in self.views.values():
                view.load([])
            self.identity.setText('Selection changed — scan to refresh inspection.')

    def discover(self, *_):
        root = Path(self.source.text())
        for name, box in self.variant_checks.items():
            box.blockSignals(True)
            exists = not self.is_gelsight() and (root / name).is_dir()
            box.setVisible(not self.is_gelsight())
            box.setEnabled(exists)
            box.setChecked(exists)
            box.blockSignals(False)
        raw = self.is_gelsight()
        self.contact.setChecked(not raw and (root / 'mask').is_dir())
        self.contact.setEnabled(not raw)
        self.pixels.setEnabled(not raw)
        self.prepare_button.setText('Export GelSight dataset' if raw else 'Export aligned datasets')
        self.delete_button.setText('Remove selected raw images' if raw else 'Remove selected samples from ALL types')
        self.dataset_note.setText(
            'Select data/gelsight (the folder containing classes). No image branches or masks are required. '
            'Flat images are split individually; recording/session independence cannot be recovered from numbered filenames. '
            'Use trial subfolders and grouped splits when available.' if raw else
            'Selected image types share sample IDs, classes and splits. Masks remain binary.')
        self.branch_changed()

    def branch_changed(self, *_):
        root = Path(self.source.text())
        branches = [root] if self.is_gelsight() else [root/name for name in self.selected()]
        labels = sorted({p.name for branch in branches if branch.is_dir() for p in branch.iterdir()
                         if p.is_dir() and not p.name.startswith('.')})
        previous = {self.classes.item(i).text(): self.classes.item(i).checkState() for i in range(self.classes.count())}
        self.classes.blockSignals(True)
        self.classes.clear()
        for label in labels:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(previous.get(label, QtCore.Qt.Checked))
            self.classes.addItem(item)
        self.classes.blockSignals(False)
        self.invalidate()
        for name, card in self.cards.items():
            card.setVisible(name in self.selected())

    def check_classes(self, enabled):
        for i in range(self.classes.count()):
            self.classes.item(i).setCheckState(QtCore.Qt.Checked if enabled else QtCore.Qt.Unchecked)

    def pick_source(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Folder containing classes' if self.is_gelsight() else 'Folder containing clean, default, tactile, mask')
        if folder:
            self.source.setText(folder)
            self.output.setText(str(Path(folder).parent / ('ml_gelsight' if self.is_gelsight() else 'ml_dataset')))
            self.classes.clear()
            self.discover()

    def pick_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose output parent (a new ml_dataset folder will be created)')
        if folder:
            self.output.setText(str(Path(folder) / 'ml_dataset'))

    def options(self):
        if self.is_gelsight():
            return dict(root=self.source.text(), selected_labels=self.labels(),
                        group_by_subfolder=self.group.isChecked(), gelsight=True)
        return dict(root=self.source.text(), variants=self.selected(), selected_labels=self.labels(),
                    group_by_subfolder=self.group.isChecked(), require_contact=self.contact.isChecked(),
                    min_contact_pixels=self.pixels.value())

    def launch(self, task, options):
        if self.worker is not None:
            return
        self.inputs.setEnabled(False)
        for button in (self.scan_button, self.prepare_button, self.delete_button, self.undo_button):
            button.setEnabled(False)
        self.bar.setRange(0, 0)
        self.status.setPlainText('Scanning…' if task == 'scan' else 'Exporting the same samples and splits for each selected dataset…')
        self.worker = Worker(task, options, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_scan if task == 'scan' else self.on_export)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def scan(self):
        if not self.selected() or not self.labels():
            self.status.setPlainText('Select at least one dataset type and class.')
            return
        self.launch('scan', self.options())

    def on_scan(self, result):
        self.scan_result, self.inventory, self.contacts = result
        self.keys = list(self.inventory)
        excluded = {(item['label'], item['key']): item['reason'] for item in self.scan_result.summary['excluded_samples']}
        self.table.setRowCount(len(self.keys))
        for row, key in enumerate(self.keys):
            pixels = self.contacts[key]
            reason = excluded.get(key, 'Ready')
            if isinstance(pixels, int) and pixels < self.pixels.value() and reason == 'Ready':
                reason = 'Low contact (filter off)'
            for column, value in enumerate((*key, pixels, reason)):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.filter_rows()
        self.navigate(1, start=-1)
        summary = self.scan_result.summary
        if self.is_gelsight():
            self.status.setPlainText(f"{len(self.keys)} raw images inspected; {summary['matched_total']} usable. "
                                     f"{len(summary['excluded_samples'])} duplicates or empty images excluded.")
            return
        self.status.setPlainText(f"{len(self.keys)} sample IDs inspected; {summary['matched_total']} usable matches. "
                                 f"{len(summary['excluded_samples'])} excluded from EVERY selected output.\n" +
                                 json.dumps({'issues': summary['issues'], 'examples': summary['issue_examples']}, indent=2))

    def show_sample(self, row, *_):
        if not 0 <= row < len(self.keys):
            return
        key = self.keys[row]
        detail = 'Raw RGB image' if self.is_gelsight() else f'Mask contact pixels: {self.contacts.get(key)}'
        self.identity.setText(f'{key[0]} / {key[1]}\n{detail}')
        for name, view in self.views.items():
            view.load(self.inventory[key].get(name, []))

    def filter_rows(self, *_):
        term = self.filter.text().casefold()
        for row in range(self.table.rowCount()):
            text = ' '.join(self.table.item(row, col).text() if self.table.item(row, col) else '' for col in range(4))
            self.table.setRowHidden(row, term not in text.casefold())

    def navigate(self, step, start=None):
        row = self.table.currentRow() if start is None else start
        for candidate in range(row+step, self.table.rowCount() if step > 0 else -1, step):
            if not self.table.isRowHidden(candidate):
                self.table.setCurrentCell(candidate, 0)
                self.table.selectRow(candidate)
                break

    def remove_selected(self):
        keys = [self.keys[index.row()] for index in self.table.selectionModel().selectedRows()
                if not self.table.isRowHidden(index.row())]
        if not keys or self.worker is not None:
            return
        try:
            manifest = quarantine_samples(self.source.text(), keys, raw=self.is_gelsight())
            self.undo_stack.append(manifest)
            self.invalidate()
            self.scan()
        except Exception as exc:
            self.on_error(str(exc))

    def undo(self):
        if not self.undo_stack or self.worker is not None:
            return
        try:
            restore_quarantine(self.undo_stack[-1])
            self.undo_stack.pop()
            self.invalidate()
            self.scan()
        except Exception as exc:
            self.on_error(str(exc))

    def prepare(self):
        if self.scan_result is None:
            return
        options = self.options()
        options.update(output=self.output.text(), ratios=tuple(spin.value() for spin in self.ratios),
                       seed=self.seed.value(), matched_only=True, balance_classes=self.balance.isChecked(),
                       scan=self.scan_result)
        if self.is_gelsight():
            options.pop('scan')
            options.pop('matched_only')
        self.launch('prepare', options)

    def on_export(self, summary):
        unit = 'raw RGB images' if self.is_gelsight() else 'matching samples per dataset'
        self.status.setPlainText(f"Exported {summary['exported_total']} {unit} to {self.output.text()}\n" +
                                 json.dumps(summary['counts'], indent=2))

    def on_error(self, message):
        self.invalidate()
        self.status.setPlainText(message)

    def on_progress(self, completed, total):
        self.bar.setRange(0, total)
        self.bar.setValue(completed)

    def finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.inputs.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.delete_button.setEnabled(bool(self.keys))
        self.undo_button.setEnabled(bool(self.undo_stack))
        valid = self.scan_result is not None and len(self.scan_result.labels) >= 2
        if valid:
            valid = not any(self.scan_result.summary['issues'].get(key) for key in
                            ('missing_class', 'class_case_collision', 'invalid_label', 'insufficient_groups'))
        self.prepare_button.setEnabled(valid)
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 1)

    def closeEvent(self, event):
        if self.worker is not None:
            self.status.setPlainText('Wait for the current scan or export to finish before closing.')
            event.ignore()
        else:
            super().closeEvent(event)


def run(gelsight=False):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PreprocessWindow()
    if gelsight:
        window.dataset_kind.setCurrentIndex(1)
        window.source.setText(str(Path(__file__).resolve().parents[2] / 'data' / 'gelsight'))
        window.discover()
    window.show()
    app.exec_()
