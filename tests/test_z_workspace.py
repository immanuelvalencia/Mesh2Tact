import json
import os
import numpy as np
import pytest

from mesh2tact.geometric import GeometricSim
from mesh2tact.gather import GatherSettings, gather


def test_fixed_plane_z_limit_and_export(tmp_path):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 80, 64
    sim.max_penetration = .0006
    sim.rotation = (30, 20, 10)
    sim.object_z = -1
    depth, _ = sim.render()
    assert sim.plane_z == 0
    assert sim.cut_depth == .0006
    assert 0 < depth.max() <= .00060001
    assert sim.object_z+sim.first_contact == pytest.approx(-.0006)
    out = sim.export(tmp_path/'sample')
    meta = json.loads((out/'settings.json').read_text())
    assert meta['plane_z_m'] == 0 and meta['object_z_m'] == sim.object_z
    sim.object_z += .001
    assert not sim.depth().any()
    with pytest.raises(ValueError, match='exceeds'):
        gather(sim, tmp_path/'invalid', GatherSettings(count=1, cut_max_mm=2))
    assert not (tmp_path/'invalid').exists()


def test_z_slider_preview_grid_and_preset_roundtrip(tmp_path, monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from mesh2tact.gui import sensor_configs
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    monkeypatch.setattr(sensor_configs, 'CONFIG_DIR', tmp_path/'sensors')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        window.load_demo('box')
        window.max_penetration.setValue(.6)
        window.z.setValue(-100)
        window.refresh()
        assert window.sim.plane_z == 0
        assert window.sim.raw_depth.max() <= .00060001
        assert window.cut.value() == pytest.approx(.6)
        z = window.z.value()
        window.nudge('Z', 1)
        assert window.z.value() > z
        window.ry.setValue(45)
        window.refresh()
        assert window.sim.cut_depth <= .00060001
        assert (window.slider.minimum(), window.slider.maximum()) == (-1000, 1000)
        bottom, top = window.rotated_z_bounds()
        window.slider.setValue(0)
        assert window.z.value() == pytest.approx(-bottom, abs=.0005)
        window.slider.setValue(-500)
        assert window.cut.value() == pytest.approx(.3, abs=.001)
        window.slider.setValue(-1000)
        assert window.sim.plane_z == 0
        assert window.cut.value() == pytest.approx(.6, abs=.001)
        window.slider.setValue(1000)
        assert window.z.value() == pytest.approx(-bottom+2*(top-bottom), abs=.0005)
        window.preview_grid.set_selected(list(window.preview_grid.labels))
        window.refresh()
        assert all(view.original is not None for view in window.preview_grid.views.values())
        assert window.preview_grid.grid.count() == 6
        target = tmp_path/'workspace.json'
        window.save_preset(target)
        pose = window.sim.object_z
        window.max_penetration.setValue(.2)
        window.preview_grid.set_selected([])
        assert window.preview_grid.grid.count() == 0
        window.load_preset(target)
        assert window.max_penetration.value() == .6
        assert window.sim.object_z == pytest.approx(pose, abs=5e-7)
        assert len(window.preview_grid.selected()) == 6
        # Opening a scene changes that scene's panels, not the user's saved defaults.
        import json
        assert json.loads(window.preview_grid.preferences_path.read_text())['panels'] == []
        titles = [button.text() for button in window.findChildren(QtWidgets.QPushButton)]
        assert 'Apply object quality' not in titles
        assert 'Restore original object' not in titles
        window.max_penetration.setValue(.4)
        assert window.gather_panel.cut_range[1].maximum() == .4
        window.max_penetration.setValue(.8)
        assert window.gather_panel.cut_range[1].value() == .8
        window.press_button.click()
        assert window.sim.cut_depth == pytest.approx(.0008, abs=1e-6)
        assert window.sim.plane_z == 0
    finally:
        window.close()
