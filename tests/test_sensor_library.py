import json
import os
from dataclasses import asdict

import numpy as np
import pytest

from mesh2tact.geometric import GeometricSim
from mesh2tact.outputs import SaveOptions, model_name
from mesh2tact.gather import GatherSettings, gather


def test_output_selection_and_flat_model_layout(tmp_path):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
    outputs = SaveOptions(**{name: name == "tactile" for name in asdict(SaveOptions())})
    result = gather(sim, tmp_path, GatherSettings(count=2, outputs=outputs))
    from pathlib import Path
    out = Path(result["directory"])
    assert out.parent == tmp_path / "sphere"
    assert result["status"] == "complete"
    assert {p.name for p in out.iterdir()} == {"run.json", "manifest.jsonl", "sample_000001_tactile.png", "sample_000002_tactile.png"}
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["files"] == {"tactile.png": "sample_000001_tactile.png"}
    none = SaveOptions(**{name: False for name in asdict(SaveOptions())})
    with pytest.raises(ValueError):
        gather(sim, tmp_path, GatherSettings(outputs=none))
    assert model_name("primitive:sphere") == "sphere"
    assert model_name("CON.stl") == "model_CON"
    assert "/" not in model_name("primitive:../../unsafe")


def test_sensor_config_crud_scan_and_object_preservation(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyvistaqt")
    from mesh2tact.gui import sensor_configs
    monkeypatch.setattr(sensor_configs, "CONFIG_DIR", tmp_path / "sensors")
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        assert not hasattr(window.gather_panel, "folder")
        assert not any("settings JSON" in button.text() for button in window.findChildren(QtWidgets.QPushButton))
        window.load_demo("box")
        window.rx.setValue(20)
        window.cut.setValue(.8)
        window.effect_widgets["texture"].setValue(.04)
        window.refresh()
        pose = (window.sim.rotation, window.sim.offset, window.sim.cut_depth)
        rgb = window.sim.render()[1]
        panel = window.sensor_configs
        path = panel.add("My sensor")
        assert panel.selector.count() == 2
        saved = json.loads(path.read_text())
        assert "object" not in saved
        window.effect_widgets["texture"].setValue(.2)
        window.refresh()
        panel.load(path)
        assert window.sim.source == "primitive:box"
        assert (window.sim.rotation, window.sim.offset, window.sim.cut_depth) == pose
        assert np.array_equal(rgb, window.sim.render()[1])
        window.effect_widgets["texture"].setValue(.1)
        panel.update_selected()
        assert json.loads(path.read_text())["effects"]["texture"] == .1
        external = panel.directory / "External.json"
        external.write_text(json.dumps(saved))
        panel.scan()
        assert panel.selector.count() == 3
        bad = panel.directory / "Broken.json"
        bad.write_text('{"format":"wrong"}')
        previous = window.sim
        with pytest.raises(ValueError):
            panel.load(bad)
        assert window.sim is previous
        panel.delete_selected()
        assert not path.exists()
        assert external.exists()
        assert window.sim is previous
    finally:
        window.close()
