import json
import os
from pathlib import Path
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from mesh2tact.gui.qt import QtWidgets, QtCore
from PyQt5.QtTest import QTest


def test_sensor_protection_and_copy_before_edit(tmp_path, monkeypatch):
    from mesh2tact.gui import sensor_configs
    from mesh2tact.gui.geometric import GeometricWindow
    monkeypatch.setattr(sensor_configs, 'CONFIG_DIR', tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        panel = window.sensor_configs
        path = tmp_path / 'GelSightV1.json'
        path.write_text(json.dumps(panel.data()))
        original = path.read_bytes()
        panel.scan(force=True)
        panel.selector.setCurrentIndex(panel.selector.findData(str(path)))
        prompts = []
        def cancel(*args):
            prompts.append(args)
            return '', False
        monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', cancel)
        panel.update_selected()
        panel.delete_selected()
        width = window.width.value()
        QTest.keyClick(window.width, QtCore.Qt.Key_Up)
        assert window.width.value() == width and len(prompts) == 3
        assert path.read_bytes() == original
        # Workspace controls stay editable with a built-in sensor selected.
        for control in (window.max_penetration, window.output_w, window.output_h, window.preview_scale):
            previous = control.value()
            QTest.keyClick(control, QtCore.Qt.Key_Up)
            assert control.value() > previous
        preset_index = window.resolution_preset.currentIndex()
        QTest.keyClick(window.resolution_preset, QtCore.Qt.Key_Down)
        assert window.resolution_preset.currentIndex() != preset_index
        assert len(prompts) == 3
        assert path.read_bytes() == original
        with pytest.raises(ValueError, match='read-only'):
            panel.write(path)
        with pytest.raises(ValueError, match='read-only'):
            window.save_preset(path)
        monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', lambda *a: ('My sensor', True))
        QTest.keyClick(window.width, QtCore.Qt.Key_Up)
        assert Path(panel.selector.currentData()).stem == 'My sensor'
        assert window.width.value() == width
        QTest.keyClick(window.width, QtCore.Qt.Key_Up)
        assert window.width.value() > width
        assert path.read_bytes() == original
    finally:
        window.close()


def test_background_and_directional_edit_cancel_and_copy(tmp_path, monkeypatch):
    from mesh2tact.gui.gel import GelLightingPanel
    from mesh2tact.gui.lighting import LightingPanel
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gel = GelLightingPanel(tmp_path / 'backgrounds.json')
    light = LightingPanel()
    light.presets.path = tmp_path / 'lights.json'
    try:
        gel.background_preset.setCurrentText('GelSight')
        light.presets.selector.setCurrentText('Default')
        monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', lambda *a: ('', False))
        brightness, intensity = gel.brightness.value(), light.intensity.value()
        QTest.keyClick(gel.brightness, QtCore.Qt.Key_Up)
        QTest.keyClick(light.intensity, QtCore.Qt.Key_Up)
        gel.delete_preset()
        light.presets.update()
        assert gel.brightness.value() == brightness and light.intensity.value() == intensity
        assert not gel.preset_path.exists() and not light.presets.path.exists()
        monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', lambda *a: ('My copy', True))
        QTest.keyClick(gel.brightness, QtCore.Qt.Key_Up)
        QTest.keyClick(light.intensity, QtCore.Qt.Key_Up)
        assert gel.background_preset.currentText() == light.presets.selector.currentText() == 'My copy'
        assert 'GelSight' not in json.loads(gel.preset_path.read_text())
        assert 'Default' not in json.loads(light.presets.path.read_text())
    finally:
        gel.close()
        light.close()
