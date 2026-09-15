from copy import deepcopy
from dataclasses import asdict
import json
import os

import numpy as np
import pytest

from mesh2tact.geometric import GeometricSim
from mesh2tact.render.gel import GelLighting, GelGlow


def test_background_and_gradient_are_spatial_and_independent_of_depth():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 80, 60
    sim.cut_depth = 0
    sim.gel_lighting = GelLighting(background=(.2, .3, .4), vignette=0,
                                   glows=[GelGlow(strength=0)])
    depth, image = sim.render()
    assert not depth.any()
    assert np.max(np.abs(image.astype(int)-np.array([51, 76, 102]))) <= 1
    sim.gel_lighting.glows[0] = GelGlow(color=(1, 0, 0), strength=1, x=.5, y=.5, width=.1, height=.1)
    new_depth, glow = sim.render()
    assert np.array_equal(depth, new_depth)
    assert glow[30, 40, 0] > glow[0, 0, 0] + 100
    sim.gel_lighting.glows[0].x = .1
    moved = sim.render()[1]
    assert moved[30, 8, 0] > moved[30, 40, 0]
    sim.cut_depth = .001
    assert sim.render()[0].max() > 0
    with_contact = sim.render()[1]
    sim.cfg.optics.leds[0].intensity = 0
    assert not np.array_equal(with_contact, sim.render()[1])


def test_gel_round_trip_and_capture_metadata(tmp_path):
    gel = GelLighting()
    restored = GelLighting.from_dict(json.loads(json.dumps(asdict(gel))))
    assert np.array_equal(gel.image((73, 98)), restored.image((73, 98)))
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 80, 60
    out = sim.export(tmp_path / "capture")
    assert json.loads((out / "settings.json").read_text())["gel_lighting"] == json.loads(json.dumps(asdict(gel)))
    broken = asdict(gel)
    broken["glows"][0]["width"] = 0
    with pytest.raises(ValueError):
        GelLighting.from_dict(broken)


def test_master_brightness_scales_lighting_without_changing_depth():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 80, 60
    sim.effects.enabled = False
    sim.cut_depth = .001
    assert len(sim.gel_lighting.glows) == 2
    depth, original = sim.render()
    sim.gel_lighting.brightness = .5
    half_depth, half = sim.render()
    assert np.array_equal(depth, half_depth)
    assert np.max(np.abs(half.astype(float)-original.astype(float)*.5)) <= 1
    sim.gel_lighting.brightness = 0
    assert not sim.render()[1].any()
    old = asdict(GelLighting())
    old.pop("brightness")
    assert GelLighting.from_dict(old).brightness == 1


def test_background_library_and_empty_gradients(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from mesh2tact.gui.gel import GelLightingPanel, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = tmp_path / "backgrounds.json"
    panel = GelLightingPanel(path)
    panel.remove()
    panel.remove()
    assert panel.settings.glows == []
    panel.settings.validate()
    panel.color_controls["saturation"].setValue(0)
    rgb = panel.settings.adjust(np.array([[[.2, .5, .8]]]))
    assert np.allclose(rgb[..., 0], rgb[..., 1])
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a: ("My gel", True))
    panel.create_preset()
    panel.color_controls["hue"].setValue(45)
    panel.update_preset()
    restored = GelLightingPanel(path)
    restored.background_preset.setCurrentText("My gel")
    assert restored.settings.hue == 45 and restored.settings.glows == []
    restored.delete_preset()
    assert json.loads(path.read_text()) == {}
    restored.reset()
    assert restored.settings == GelLighting()
    assert restored.update_button.isEnabled()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a: ('', False))
    restored.delete_preset()
    assert restored.background_preset.findText("GelSight") >= 0
    panel.close()
    restored.close()


def test_three_tabs_complete_preset_round_trip_and_bad_load(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyvistaqt")
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == ["General object", "Data gathering"]
        window.load_demo("box")
        window.quality.setCurrentIndex(window.quality.findData(1))
        window.mesh_smoothing.setValue(2)
        window.apply_quality()
        window.rx.setValue(25)
        window.x.setValue(.3)
        window.cut.setValue(.7)
        window.width.setValue(16)
        window.height.setValue(12)
        window.output_w.setValue(512)
        window.output_h.setValue(384)
        window.preview_scale.setValue(18.75)
        window.effect_widgets["texture"].setValue(.08)
        window.effect_seed.setValue(123)
        window.lighting.intensity.setValue(.7)
        window.gel_panel.settings.background = (.17, .25, .29)
        window.gel_panel.controls["x"].setValue(.7)
        window.gel_panel.controls["strength"].setValue(.6)
        window.gel_panel.background_preset.setCurrentText("GelSight")
        assert window.gel_panel.settings.background == (.19, .28, .29)
        window.gel_panel.brightness_slider.setValue(75)
        assert window.gel_panel.settings.brightness == .75
        window.gather_panel.count.setValue(12)
        window.gather_panel.cut_range[1].setValue(1.1)
        window.data_root = tmp_path / "dataset"
        window.refresh()
        before = window.sim.render()[1]
        preset = tmp_path / "all_settings.json"
        window.save_preset(preset)
        saved = json.loads(preset.read_text())
        window.load_demo("cone")
        window.gel_panel.reset()
        window.reset_effects()
        window.load_preset(preset)
        assert np.array_equal(before, window.sim.render()[1])
        assert window.sim.source == "primitive:box"
        assert window.sim.quality_level == 1
        assert window.sim.smoothing_iterations == 2
        assert window.gather_panel.count.value() == 12
        assert window.gather_panel.cut_range[1].value() == 1.1
        assert window.output_w.value() == 512
        window.save_preset(tmp_path / "again.json")
        assert json.loads((tmp_path / "again.json").read_text()) == saved
        bad = deepcopy(saved)
        bad["gel_lighting"]["glows"][0]["width"] = -1
        invalid = tmp_path / "invalid.json"
        invalid.write_text(json.dumps(bad))
        previous = window.sim
        with pytest.raises(ValueError):
            window.load_preset(invalid)
        assert window.sim is previous
        assert np.array_equal(before, window.sim.render()[1])
        bad = deepcopy(saved)
        bad["object"]["source"] = "missing_model.stl"
        invalid.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="missing"):
            window.load_preset(invalid)
        assert window.sim is previous
    finally:
        window.close()
