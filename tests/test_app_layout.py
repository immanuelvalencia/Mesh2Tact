import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from mesh2tact.geometric import GeometricSim


def test_imported_quality_levels_and_restore():
    sim = GeometricSim()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=.004)
    sim.set_mesh(mesh, "imported.stl")
    counts = []
    for level in (-3, -2, -1, 0, 1, 2, 3):
        sim.set_quality(level)
        counts.append(len(sim.mesh.faces))
        assert len(sim.mesh.faces) > 0
        assert np.isfinite(sim.mesh.vertices).all()
    assert counts == sorted(counts)
    assert counts[0] < counts[3] < counts[-1]
    sim.set_quality(0)
    assert np.array_equal(sim.mesh.vertices, mesh.vertices)
    assert np.array_equal(sim.original_mesh.vertices, mesh.vertices)


def test_navigation_settings_and_vertical_cut():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        assert window.quality.currentText() == "Ultra"
        assert not hasattr(window, "demo") and not hasattr(window, "gizmo_check")
        assert window.load_button.parent().layout().itemAt(0).widget() is window.load_button
        assert window.tabs.count() == 3
        assert window.tabs.tabText(1) == 'Calibration'
        assert not window._has_object and not window.gather_panel.isEnabled()
        assert window.slider.orientation() == QtCore.Qt.Vertical
        window.preview_scale.setValue(50)
        window.width.setValue(16)
        window.height.setValue(12)
        window.resolution_preset.setCurrentIndex(window.resolution_preset.findData(1280))
        assert (window.output_w.value(), window.output_h.value()) == (1280, 960)
        assert (window.width.value(), window.height.value()) == (16, 12)
        window.output_w.setValue(1280)
        window.output_h.setValue(960)
        window.refresh()
        assert window.preview_dimensions() == (640, 480)
        window.height.setValue(9)
        assert (window.output_w.value(), window.output_h.value()) == (1280, 720)
        window.output_w.setValue(1920)
        window.output_h.setValue(1080)
        window.refresh()
        assert window.preview_dimensions() == (960, 540)
        window.preview_scale.setValue(100)
        assert window.preview_dimensions() == (1920, 1080)
        window.preview_scale.setValue(10)
        window.settings_action.trigger()
        app.processEvents()
        assert window.settings_window.isVisible()
        flags = window.settings_window.windowFlags()
        assert flags & QtCore.Qt.WindowMaximizeButtonHint
        assert not flags & QtCore.Qt.WindowContextHelpButtonHint
        assert window.settings_window.pages.count() == 5
        window.settings_window.showMaximized()
        app.processEvents()
        assert window.settings_window.isMaximized()
        window.settings_window.showNormal()
        window.settings_window.navigation.setCurrentRow(4)
        assert window.settings_window.pages.currentIndex() == 4
        window.effect_widgets["texture"].setValue(.12)
        window.settings_window.close()
        window.settings_action.trigger()
        assert window.effect_widgets["texture"].value() == .12
        assert window.settings_window.navigation.currentRow() == 4
        window.load_demo("box")  # private fixture helper; no demo control in UI
        window.refresh()
        assert window.gizmo is not None
        assert window.sim.quality_level == 2
        window.slider.setValue(0)
        assert window.cut.value() == 0
        window.cut.setValue(12)
        assert window.cut.value() == window.max_penetration.value()
        assert window.slider.value() == -1000
        assert window.cut_readout.text().startswith(f'{window.z.value():.2f}')
    finally:
        window.close()


def test_physics_runtime_is_outside_geometric_package():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "mesh2tact/physics").exists()
    assert not (root / "mesh2tact/sim.py").exists()
    assert (root / "physics_simulator/mesh2tact/physics/mpm_gel.py").is_file()
    assert (root / "physics_simulator/environment.yml").is_file()
    assert "taichi" not in (root / "environment.yml").read_text().lower()
