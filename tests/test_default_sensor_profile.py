import json
import os
from pathlib import Path

from mesh2tact.config import SensorConfig
from mesh2tact.render.gel import GelLighting


def test_default_sensor_profile_is_editable_and_has_no_calibration_lookup():
    from mesh2tact.gui.sensor_configs import PROTECTED_NAMES
    path = Path(__file__).resolve().parents[1] / 'configs' / 'sensors' / 'Default.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    config = SensorConfig.from_dict(data['sensor'])
    lighting = GelLighting.from_dict(data['gel_lighting'])

    assert data['format'] == 'mesh2tact-sensor-config'
    assert config.name == 'default_uncalibrated'
    assert config.optics.spatial_response is None
    assert config.optics.calibration_path is None
    assert not config.optics.reference_background
    assert not config.optics.side_lighting
    lighting.validate()
    assert 'default' not in PROTECTED_NAMES


def test_gather_output_defaults_persist(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from mesh2tact.gui.gather import GatherPanel
    from mesh2tact.gui.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    names = ('tactile', 'clean', 'depth_image')
    source = GatherPanel.__new__(GatherPanel)
    source.output_preferences_path = tmp_path / 'gather_output_preferences.json'
    source.output_defaults_status = QtWidgets.QLabel()
    source.output_checks = {name: QtWidgets.QCheckBox() for name in names}
    source.output_checks['tactile'].setChecked(True)
    source.output_checks['clean'].setChecked(False)
    source.output_checks['depth_image'].setChecked(True)

    assert source.save_output_preferences()
    loaded = GatherPanel.__new__(GatherPanel)
    loaded.output_preferences_path = source.output_preferences_path
    loaded.output_defaults_status = QtWidgets.QLabel()
    loaded.output_checks = {name: QtWidgets.QCheckBox() for name in names}
    loaded.load_output_preferences()

    assert {name: checkbox.isChecked() for name, checkbox in loaded.output_checks.items()} == {
        'tactile': True, 'clean': False, 'depth_image': True}


def test_default_profile_tactile_output(tmp_path):
    from mesh2tact.geometric import GeometricSim
    from mesh2tact.outputs import SaveOptions

    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
    outputs = SaveOptions(**{name: name == 'default_tactile' for name in vars(SaveOptions())})
    out = sim.export(tmp_path / 'sample', outputs=outputs)

    assert (out / 'tactile_default.png').is_file()
    assert not (out / 'tactile.png').exists()


def test_default_tactile_uses_the_profile_pad_geometry(tmp_path, monkeypatch):
    from mesh2tact import geometric
    from mesh2tact.geometric import GeometricSim

    profile = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'sensors' / 'Default.json').read_text())
    profile['sensor']['gel']['size_x'] = .0093
    profile['sensor']['gel']['size_y'] = .00715
    path = tmp_path / 'Default.json'
    path.write_text(json.dumps(profile), encoding='utf-8')
    monkeypatch.setattr(geometric, 'DEFAULT_SENSOR_PROFILE_PATH', path)
    geometric.invalidate_default_sensor_profile()
    try:
        sim = GeometricSim()
        sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
        depth, active = sim.render()
        default = sim.render_default_profile(depth)

        assert default.shape == active.shape == (48, 64, 3)
        assert not (default == active).all()
    finally:
        geometric.invalidate_default_sensor_profile()
