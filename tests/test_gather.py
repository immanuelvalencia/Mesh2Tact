from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mesh2tact.geometric import GeometricSim
from mesh2tact.gather import GatherSettings, gather
from mesh2tact.outputs import reserve_labelled_capture


def small_sim():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
    return sim


def test_seeded_gather_ranges_and_metadata(tmp_path):
    settings = GatherSettings(count=3, seed=7, rotation_min=(-10, 20, 30), rotation_max=(10, 30, 40),
                              cut_min_mm=.2, cut_max_mm=.4, random_xy=True,
                              xy_min_mm=(-1, -2), xy_max_mm=(1, 2), random_effect_seed=True)
    first = gather(small_sim(), tmp_path, settings)
    second = gather(small_sim(), tmp_path, settings)
    assert first["status"] == second["status"] == "complete"
    assert first["directory"] != second["directory"]
    path, other = Path(first["directory"]), Path(second["directory"])
    assert len(rows) == 3 and (path / "processed_mesh.ply").is_file()
    for a, b in zip(rows, rows2):
        assert np.all(np.array(a["rotation_xyz_deg"]) >= settings.rotation_min)
        assert np.all(np.array(a["rotation_xyz_deg"]) <= settings.rotation_max)
        assert .0002 <= a["cut_depth_m"] <= .0004
        assert a["rotation_xyz_deg"] == b["rotation_xyz_deg"]
        assert a["effect_seed"] == b["effect_seed"]
        metadata = json.loads((path / a["files"]["settings.json"]).read_text())
        assert metadata["rotation_xyz_deg"] == a["rotation_xyz_deg"]
        assert metadata["image_effects"]["seed"] == a["effect_seed"]
        assert (path / metadata["processed_mesh"]).resolve().is_file()
        assert np.load(path / a["files"]["depth_m.npy"]).shape == (48, 64)
        assert (path / a["files"]["tactile.png"]).read_bytes() == (other / b["files"]["tactile.png"]).read_bytes()


def test_labelled_capture_reservations_increment_without_subfolders(tmp_path):
    first, start, locks = reserve_labelled_capture(tmp_path, 'blue sphere', 2)
    assert first == tmp_path/'blue sphere'
    assert start == 1 and all(lock.is_file() for lock in locks)
    for lock in locks:
        lock.unlink()
    (first/'sample_000003_tactile.png').write_bytes(b'existing data')
    second, start, locks = reserve_labelled_capture(tmp_path, 'blue sphere', 1)
    assert second == first and start == 4
    assert (first/'sample_000003_tactile.png').read_bytes() == b'existing data'
    for lock in locks:
        lock.unlink()
    with pytest.raises(ValueError, match='object label'):
        reserve_labelled_capture(tmp_path, '', 1)


def test_labelled_gather_records_settings_and_increments(tmp_path):
    settings = GatherSettings(count=1, random_rotation=False, random_cut=False,
                              save_layout='object_label', object_label='cube test')
    rows, rows2 = [], []
    first = gather(small_sim(), tmp_path, settings, progress=rows.append)
    second = gather(small_sim(), tmp_path, settings, progress=rows2.append)
    assert Path(first['directory']) == tmp_path/'tactile'/'cube test'
    assert Path(second['directory']) == tmp_path/'tactile'/'cube test'
    assert (tmp_path/'tactile'/'cube test'/'sample_000001_tactile.png').is_file()
    assert (tmp_path/'tactile'/'cube test'/'sample_000002_tactile.png').is_file()
    assert (tmp_path/'clean'/'cube test'/'sample_000001_tactile_clean.png').is_file()
    assert (tmp_path/'default'/'cube test'/'sample_000001_tactile_default.png').is_file()
    assert not [path for path in (tmp_path/'tactile'/'cube test').iterdir() if path.is_dir()]
    assert not list((tmp_path/'tactile'/'cube test').glob('run_*.json'))
    assert not list((tmp_path/'tactile'/'cube test').glob('manifest_*.jsonl'))
    with pytest.raises(ValueError, match='object label'):
        GatherSettings(save_layout='object_label', object_label='').validate()


def test_cancel_preserves_frames_and_fixed_pose(tmp_path):
    sim = small_sim()
    sim.rotation, sim.offset, sim.cut_depth = (10, 20, 30), (.001, -.001), .0005
    saved = []
    result = gather(sim, tmp_path, GatherSettings(count=5, random_rotation=False, random_cut=False),
                    cancelled=lambda: len(saved) == 2, progress=saved.append)
    assert result["status"] == "cancelled" and result["completed"] == 2
    path = Path(result["directory"])
    assert len(saved) == 2
    assert saved[0]["rotation_xyz_deg"] == [10, 20, 30]
    assert saved[0]["cut_depth_m"] == .0005
    assert saved[0]["offset_xy_m"] == [.001, -.001]
    assert len(list(path.glob("sample_*_settings.json"))) == 2


def test_failed_frame_keeps_only_the_partial_directory(tmp_path, monkeypatch):
    sim = small_sim()
    export = sim.export
    calls = []
    def fail_second(path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            path.mkdir()
            raise OSError("simulated disk failure")
        return export(path, **kwargs)
    monkeypatch.setattr(sim, "export", fail_second)
    result = gather(sim, tmp_path, GatherSettings(count=3))
    path = Path(result["directory"])
    assert result["status"] == "failed" and result["completed"] == 1
    assert (path / "sample_000001_tactile.png").is_file()
    assert (path / "sample_000002.partial").exists()
    assert not (path / "sample_000002").exists()
    assert not list(path.glob("run*.json"))
    assert not list(path.glob("manifest*.jsonl"))


def test_invalid_ranges_do_not_create_dataset(tmp_path):
    with pytest.raises(ValueError):
        gather(small_sim(), tmp_path, GatherSettings(cut_min_mm=3, cut_max_mm=1))
    assert not list(tmp_path.iterdir())


def test_gizmo_pose_and_background_collection(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyvistaqt")
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    try:
        window.load_demo("sphere")
        window.refresh()
        assert window.gizmo is not None
        before = window.cut.value()
        window.nudge("Z", -1)
        window.refresh()
        assert window.cut.value() > before
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
        rotated = window.sim.mesh.vertices @ transform[:3, :3].T
        transform[:3, 3] = [1.2, -.5, -rotated[:, 2].min()*1000-.8]
        window.gizmo_moved(transform)
        window.refresh()
        assert np.allclose(window.sim.rotation, [20, 30, 40], atol=.001)
        assert np.allclose(window.sim.offset, [.0012, -.0005])
        assert window.sim.cut_depth == pytest.approx(.0008, abs=5e-7)
        assert np.allclose(window.object_actor.user_matrix, transform, atol=.001)
        old_gizmo = window.gizmo
        window.load_demo("box")
        window.refresh()
        assert window.gizmo is not None and window.gizmo is not old_gizmo
        window.output_w.setValue(64)
        window.output_h.setValue(64)
        panel = window.gather_panel
        window.data_root = tmp_path
        panel.count.setValue(2)
        window.apply_controls()
        preview_pose = (window.sim.rotation, window.sim.offset, window.sim.cut_depth)
        window.start_gather()
        assert window.gather_worker is not None
        loop = QtCore.QEventLoop()
        window.gather_worker.finished.connect(loop.quit)
        timeout = QtCore.QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(15000)
        loop.exec_()
        timeout.stop()
        app.processEvents()
        assert window.gather_worker is None
        assert panel.progress.value() == 2
        assert "Complete: 2" in panel.status.text()
        assert (window.sim.rotation, window.sim.offset, window.sim.cut_depth) == preview_pose
        run = next((tmp_path / "tactile" / "box").iterdir())
        assert np.load(run / "sample_000001_depth_m.npy").shape == (window.output_h.value(), window.output_w.value())
        window.capture_to_folder()
        assert window.gather_worker is not None
        assert not panel.capture.isEnabled()
        from capture_helpers import wait_capture
        wait_capture(window)
        assert panel.capture.isEnabled()
        assert len(list((tmp_path / "tactile" / "box").iterdir())) == 2
    finally:
        if window.gather_worker is not None:
            window.stop_gather()
            window.gather_worker.wait(15000)
            app.processEvents()
        window.close()
