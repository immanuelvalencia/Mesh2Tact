import json
import os
import numpy as np
from mesh2tact.render.gel import GelLighting, GelGlow


def test_gradient_overlays_shaded_object():
    gel = GelLighting(vignette=0, glows=[GelGlow(color=(1, 0, 0), strength=1, x=.5, y=.5)])
    shaded = np.ones((3, 3, 3)) * .4
    result = gel.overlay(shaded)
    assert np.allclose(result[1, 1], [1, 0, 0])
    gel.glows = []
    assert np.allclose(gel.overlay(shaded), shaded)


def test_named_presets_crud_persistence_and_protected_default(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from mesh2tact.gui.appearance_presets import AppearancePresets, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    live = {"value": 2}
    path = tmp_path / "presets.json"
    panel = AppearancePresets(path, lambda: dict(live), lambda raw: live.update(raw),
                              {"value": 1}, lambda raw: None)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a: ("Mine", True))
    panel.create()
    live["value"] = 3
    panel.update()
    assert json.loads(path.read_text())["Mine"]["value"] == 3
    second = AppearancePresets(path, lambda: dict(live), lambda raw: live.update(raw),
                               {"value": 1}, lambda raw: None)
    second.selector.setCurrentText("Mine")
    assert live["value"] == 3
    second.delete()
    assert live["value"] == 3 and json.loads(path.read_text()) == {}
    second.selector.setCurrentText("Default")
    assert live["value"] == 1 and second.update_button.isEnabled()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a: ('', False))
    second.delete()
    assert second.selector.findText("Default") >= 0
    panel.close()
    second.close()
