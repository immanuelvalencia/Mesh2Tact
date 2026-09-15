import json
import os
import subprocess
import sys

import numpy as np
import pytest
import trimesh

from mesh2tact.config import SensorConfig
from mesh2tact.geometric import GeometricSim


def test_batched_rasterization_matches_full_frame_barycentric_reference():
    from mesh2tact.geometric import rasterize_surface
    rng = np.random.default_rng(1701)
    # Mix many tiny triangles, large overlapping faces, and off-sensor faces.
    triangles = rng.uniform(-.0004, .0004, (120, 3, 3))
    triangles += rng.uniform(-.015, .015, (120, 1, 3))
    triangles[:5] = rng.uniform(-.02, .02, (5, 3, 3))
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    actual = rasterize_surface(vertices, faces, 64, 48, .0186, .0143)
    projected = triangles.copy()
    projected[..., 0] = (projected[..., 0]/.0186+.5)*64-.5
    projected[..., 1] = (projected[..., 1]/.0143+.5)*48-.5
    expected = np.full((48, 64), np.inf)
    yy, xx = np.mgrid[:48, :64]
    for a, b, c in projected:
        denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-12:
            continue
        u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        v = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w = 1-u-v
        inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
        expected = np.minimum(expected, np.where(inside, u*a[2]+v*b[2]+w*c[2], np.inf))
    np.testing.assert_array_equal(actual, expected)


def small_sim():
    cfg = SensorConfig()
    cfg.camera.width, cfg.camera.height = 160, 120
    return GeometricSim(cfg)


def test_sphere_matches_analytic_depth_and_contact_growth():
    sim = small_sim()
    sim.set_mesh(trimesh.creation.icosphere(subdivisions=4, radius=.004))
    sim.cut_depth = .0005
    shallow = sim.depth()
    cached = sim.surface
    sim.cut_depth = .001
    deep = sim.depth()
    assert sim.surface is cached
    assert np.count_nonzero(deep) > np.count_nonzero(shallow)
    x = ((np.arange(160)+.5)/160-.5)*sim.cfg.gel.size_x
    y = ((np.arange(120)+.5)/120-.5)*sim.cfg.gel.size_y
    rr = x[None, :]**2 + y[:, None]**2
    expected = np.maximum(np.sqrt(np.maximum(.004**2-rr, 0))-.003, 0)
    assert np.max(np.abs(expected-deep)) < 8e-6
    sim.cut_depth = 0
    assert not sim.depth().any()
    sim.cut_depth = -.001
    assert not sim.depth().any()


def test_nearest_surface_wins_and_outside_sensor_is_empty():
    sim = small_sim()
    near = trimesh.creation.box(extents=(.004, .004, .002))
    far = near.copy()
    far.apply_translation([0, 0, .005])
    sim.set_mesh(trimesh.util.concatenate([far, near]))
    depth = sim.depth()
    assert depth[60, 80] == pytest.approx(.001)
    assert depth[0, 0] == 0
    sim.offset = (.05, 0)
    assert not sim.depth().any()


def test_tilted_plane_gradient_and_softness():
    sim = small_sim()
    sim.set_mesh(trimesh.creation.box(extents=(.01, .01, .002)))
    sim.rotation = (0, 20, 0)
    sim.cut_depth = .004
    raw = sim.depth()
    # Lower face is a plane with dz/dx = -tan(20 degrees).
    dx = sim.cfg.gel.size_x / sim.cfg.camera.width
    assert (raw[60, 85]-raw[60, 75])/(10*dx) == pytest.approx(np.tan(np.deg2rad(20)), abs=1e-5)
    sim.softness = .0002
    soft = sim.depth()
    assert np.isfinite(soft).all() and soft.min() >= 0
    assert soft.max() <= raw.max()
    assert np.array_equal(sim.raw_depth, raw)
    assert not np.array_equal(soft, raw)


def test_stl_units_export_and_lighting(tmp_path):
    mesh = trimesh.creation.box(extents=(4, 4, 4))
    path = tmp_path / "box.stl"
    mesh.export(path)
    sim = small_sim()
    sim.load(path, units="mm")
    assert np.allclose(sim.mesh.extents, .004)
    depth, rgb = sim.render()
    sim.cfg.optics.leds[0].intensity = 0
    assert not np.array_equal(rgb, sim.render()[1])
    out = sim.export(tmp_path / "capture")
    assert np.array_equal(np.load(out / "depth_m.npy"), depth)
    metadata = json.loads((out / "settings.json").read_text())
    assert metadata["mode"] == "geometric"
    assert metadata["source_units"] == "mm"
    import cv2
    assert cv2.imread(str(out / "tactile.png")).shape == (120, 160, 3)
    with pytest.raises(FileExistsError):
        sim.export(out)


def test_no_physics_import():
    result = subprocess.run([sys.executable, "-c",
        "from mesh2tact.geometric import GeometricSim; import sys; "
        "s=GeometricSim(); s.cfg.camera.width=32; s.cfg.camera.height=24; "
        "s.render(); assert 'taichi' not in sys.modules; assert 'mesh2tact.physics.mpm_gel' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_effects_reproducible_and_preserve_depth():
    from mesh2tact.render.effects import ImageEffects
    sim = small_sim()
    before, clean = sim.render()
    sim.effects = ImageEffects(seed=42, read_noise=.02, shot_noise=.01, speckle=.03,
                               texture=.12, blur_px=.5, vignette=.2, contrast=1.1, gamma=.9)
    after, altered = sim.render()
    assert np.array_equal(before, after)
    assert not np.array_equal(clean, altered)
    assert np.array_equal(altered, sim.render()[1])
    sim.effects.seed += 1
    assert not np.array_equal(altered, sim.render()[1])
    sim.effects.enabled = False
    assert np.array_equal(sim.render()[1], sim.clean_rgb)


@pytest.mark.parametrize("name,value", [("read_noise", .1), ("shot_noise", .1),
    ("speckle", .1), ("texture", .2), ("blur_px", 2), ("vignette", .5), ("contrast", 1.5), ("gamma", 2)])
def test_individual_effect_changes_image(name, value):
    from mesh2tact.render.effects import ImageEffects
    effects = ImageEffects(**{name: value})
    rng = np.random.default_rng(12)
    rgb = rng.uniform(.2, .8, (60, 80, 3)).astype(np.float32)
    original = rgb.copy()
    result = effects.apply(rgb, .0186, .0143)
    assert np.array_equal(rgb, original)
    assert not np.allclose(result, rgb)
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1


def test_quality_regeneration_and_restore(tmp_path):
    sim = small_sim()
    original = sim.mesh.vertices.copy()
    faces = len(sim.mesh.faces)
    sim.set_quality(1)
    assert len(sim.mesh.faces) == faces*4
    assert np.allclose(np.linalg.norm(sim.mesh.vertices, axis=1), .004)
    sim.set_quality(0)
    assert np.array_equal(sim.mesh.vertices, original)
    path = tmp_path / "original.stl"
    trimesh.creation.box(extents=(4, 4, 4)).export(path)
    source_bytes = path.read_bytes()
    sim.load(path)
    original = sim.mesh.copy()
    sim.set_quality(1, 5)
    assert not np.allclose(sim.mesh.extents, original.extents)
    assert path.read_bytes() == source_bytes
    sim.set_quality(0)
    assert np.array_equal(sim.mesh.vertices, original.vertices)


def test_geometric_window_controls_and_capture(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyvistaqt")
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        window.load_demo("sphere")
        window.cut.setValue(0)
        window.refresh()
        assert not window.sim.raw_depth.any()
        window.slider.setValue(150)
        window.refresh()
        assert window.sim.raw_depth.max() > .0014
        window.ry.setValue(30)
        window.refresh()
        assert window.sim.rotation[1] == 30
        assert window.rgb_view.original is not None
        assert "Render failed" not in window.statusBar().currentMessage()
        monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
        messages = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a: messages.append(a[1]))
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: pytest.fail(str(a)))
        window.data_root = tmp_path
        window.save_capture()
        from capture_helpers import wait_capture
        wait_capture(window)
        assert "Complete: 1" in window.gather_panel.status.text()
        capture = next((tmp_path / "sphere").iterdir())
        assert np.load(capture / "sample_000001_depth_m.npy").shape == (757, 984)
        assert window.sim.cfg.camera.width == 328
        window.show()
        app.processEvents()
        window.grab().save(str(tmp_path / "geometric_window.png"))
        window.width.setValue(16)
        window.height.setValue(12)
        window.resolution_preset.setCurrentIndex(window.resolution_preset.findData(1280))
        window.preview_scale.setValue(31.25)
        window.effect_widgets["texture"].setValue(.2)
        window.effect_widgets["read_noise"].setValue(.05)
        window.effect_seed.setValue(11)
        window.refresh()
        assert window.sim.render()[0].shape == (300, 400)
        window.save_capture()
        wait_capture(window)
        latest = sorted((tmp_path / "sphere").iterdir())[-1]
        assert np.load(latest / "sample_000001_depth_m.npy").shape == (960, 1280)
        metadata = json.loads((latest / "sample_000001_settings.json").read_text())
        assert metadata["image_effects"]["texture"] == .2
        assert metadata["image_effects"]["seed"] == 11
        assert (latest / "processed_mesh.ply").is_file()
        assert (latest / "sample_000001_tactile_clean.png").is_file()
        window.lighting.intensity.setValue(4)
        window.lighting.reset_button.click()
        assert window.sim.cfg.optics.leds[0].intensity == 1
        window.reset_effects()
        window.refresh()
        assert window.sim.effects.texture == 0
        window.quality.setCurrentIndex(window.quality.findData(1))
        window.apply_quality()
        assert window.sim.quality_level == 1
        window.reset_quality()
        assert window.sim.quality_level == 0
    finally:
        window.close()
