from copy import deepcopy
import json
import numpy as np
import pytest

from mesh2tact.geometric import GeometricSim
from mesh2tact.calibration import (appearance, render_float, fit_references,
                               save_session, load_session, Cancelled)


def reference(sim, blank=False, validation=False):
    sim.cut_depth = -1 if blank else .0008
    return dict(name='sphere', shape='sphere', vertices=np.asarray(sim.mesh.vertices),
                faces=np.asarray(sim.mesh.faces), rotation=list(sim.rotation), offset=list(sim.offset),
                cut_depth=sim.cut_depth, blank=blank, validation=validation, locked=True,
                photo=(render_float(sim)*255).astype(np.uint8))


@pytest.mark.parametrize('algorithm', ['least_squares', 'ai_surrogate'])
def test_background_recovery_and_validation_exclusion(tmp_path, algorithm):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 48, 36
    sim.gel_lighting.glows = []
    sim.gel_lighting.vignette = 0
    sim.gel_lighting.background = (.2, .4, .3)
    ref = reference(sim, blank=True)
    settings = appearance(sim)
    settings['gel_lighting']['background'] = [.35, .25, .45]
    heldout = deepcopy(ref)
    heldout['validation'] = True
    heldout['photo'] = np.full_like(ref['photo'], 230)
    result = fit_references(settings, [ref, heldout], size=(48, 36), fit_lighting=False, max_nfev=20, algorithm=algorithm)
    assert result['metrics'][0]['after_mae'] < .01
    assert result['metrics'][0]['after_mae'] < result['metrics'][0]['before_mae']/5
    assert settings['gel_lighting']['background'] == [.35, .25, .45]
    path = tmp_path/'session.npz'
    save_session(path, settings, [ref, heldout], result)
    restored, refs = load_session(path)
    assert restored == json.loads(json.dumps(settings))
    _, _, reopened_result = load_session(path, include_result=True)
    assert reopened_result['options']['algorithm'] == algorithm
    assert result['stages'][0]['surrogate_trials'] == (20 if algorithm == 'ai_surrogate' else 0)
    np.testing.assert_array_equal(reopened_result['final'][0], result['final'][0])
    np.testing.assert_array_equal(refs[0]['photo'], ref['photo'])
    assert refs[1]['validation']


def test_lighting_recovery_and_production_render_parity():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 48, 36
    sim.effects.enabled = False
    ref = reference(sim)
    _, rgb = sim.render()
    np.testing.assert_array_equal(rgb, ref['photo'])
    settings = appearance(sim)
    for led in settings['sensor']['optics']['leds']:
        led['intensity'] *= .5
    result = fit_references(settings, [ref], size=(48, 36), fit_background=False, fit_directions=False, max_nfev=20)
    assert result['metrics'][0]['after_contact_mae'] < result['metrics'][0]['before_contact_mae']*.5


def test_spatial_glow_recovery_and_stage_export(tmp_path):
    from mesh2tact.render.gel import GelGlow
    from mesh2tact.calibration_archive import save_run
    from PIL import Image
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 48, 36
    sim.effects.enabled = False
    sim.gel_lighting.vignette = 0
    sim.gel_lighting.glows = [GelGlow((.8,.2,.3), .65, .3,.65,.13,.24,25)]
    ref = reference(sim, blank=True)
    settings = appearance(sim)
    settings['gel_lighting']['glows'][0].update(x=.48, y=.5, width=.23, height=.3, angle=0)
    result = fit_references(settings, [ref], size=(48,36), fit_lighting=False, max_nfev=50)
    assert result['metrics'][0]['after_mae'] < result['metrics'][0]['before_mae']*.2
    fitted = result['settings']['gel_lighting']['glows'][0]
    assert abs(fitted['x']-.3) < .03 and abs(fitted['y']-.65) < .03
    result['settings']['sensor']['camera'].update(width=96, height=72)
    run = save_run(tmp_path, settings, [ref], result)
    stage = np.asarray(Image.open(run/'reference_001/stage_01_background.png'))
    np.testing.assert_array_equal(stage, (np.clip(result['final'][0],0,1)*255).astype(np.uint8))


def test_calibrated_background_does_not_erase_contact_response():
    from mesh2tact.render.gel import GelLighting, GelGlow
    gel = GelLighting(background=(.25,.3,.35), preserve_contact=True,
                      glows=[GelGlow((.5,.4,.5), .95, .5,.5,.4,.4)])
    base = np.broadcast_to(gel.background, (20,24,3)).copy()
    delta = np.zeros_like(base)
    delta[8:12,9:15] = [.04,-.02,.03]
    np.testing.assert_allclose(gel.overlay(base+delta)-gel.image((20,24)), delta, atol=1e-7)
    assert GelLighting.from_dict(__import__('dataclasses').asdict(gel)).preserve_contact


def test_calibration_preserves_configured_resolution_through_all_stages():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 800, 96
    sim.gel_lighting.glows = []
    ref = reference(sim, blank=True)
    result = fit_references(appearance(sim), [ref], fit_lighting=False,
                            fit_filters=True, max_nfev=1)
    assert result['size'] == (800, 96)
    for key in ('initial','final','photos'):
        assert result[key][0].shape == (96,800,3)
    assert all(s['resolution'] == (800,96) for s in result['stages'] if 'resolution' in s)


def test_calibration_preview_uses_live_general_object_resolution():
    from types import SimpleNamespace
    from mesh2tact.gui.calibration import CalibrationPanel
    class Value:
        def __init__(self, value): self.number = value
        def value(self): return self.number
    panel = SimpleNamespace(window=SimpleNamespace(output_w=Value(984),output_h=Value(739)))
    assert CalibrationPanel.preview_size(panel) == (984,739)
    panel.window.output_w.number = 1200
    panel.window.output_h.number = 900
    assert CalibrationPanel.preview_size(panel) == (1200,900)


def test_background_sampling_keeps_all_rgb_channels_at_calibration_resolution():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 192, 148
    sim.gel_lighting.glows = []
    sim.gel_lighting.vignette = 0
    sim.gel_lighting.background = (.2, .4, .3)
    ref = reference(sim, blank=True)
    settings = appearance(sim)
    settings['gel_lighting']['background'] = [.35, .95, .05]
    result = fit_references(settings, [ref], size=(192, 148), fit_lighting=False, max_nfev=20)
    # Previously the flattened stride was 15, fitting red while ignoring G/B.
    np.testing.assert_allclose(result['settings']['gel_lighting']['background'],
                               [.2, .4, .3], atol=.006)
    assert result['stages'][-1]['accepted']
    assert result['metrics'][0]['after_mae'] < .004


def test_full_production_verification_rejects_a_biased_sample_fit():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 192, 148
    sim.gel_lighting.glows = []
    sim.gel_lighting.vignette = 0
    sim.gel_lighting.background = (.2, .2, .2)
    ref = reference(sim, blank=True)
    # A periodic image can fool the bounded pixel sample. The full render
    # must reject the bright fit instead of exporting a degraded background.
    ref['photo'].reshape(-1, 3)[::15] = 204
    settings = appearance(sim)
    result = fit_references(settings, [ref], size=(192, 148), fit_lighting=False, max_nfev=20)
    assert not result['stages'][-1]['accepted']
    assert result['stages'][-1]['candidate_mae'] > result['stages'][-1]['before_mae']
    assert result['settings'] == settings
    np.testing.assert_array_equal(result['final'][0], result['initial'][0])


def test_cancellation_and_unlocked_references():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 32, 24
    ref = reference(sim)
    with pytest.raises(Cancelled):
        fit_references(appearance(sim), [ref], cancelled=lambda: True)
    ref['locked'] = False
    with pytest.raises(ValueError, match='Lock'):
        fit_references(appearance(sim), [ref])


def test_blur_fit_from_zero_at_configured_resolution():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 96, 72
    sim.effects.blur_px = 2
    ref = reference(sim)
    settings = appearance(sim)
    settings['effects']['blur_px'] = 0
    result = fit_references(settings, [ref], fit_background=False,
                            fit_lighting=False, fit_blur=True, max_nfev=15)
    assert result['size'] == (96, 72)
    assert abs(result['settings']['effects']['blur_px'] - 2) < .3
    assert result['metrics'][0]['after_mae'] < result['metrics'][0]['before_mae']
    assert result['metrics'][0]['after_mae'] < 1/255  # Quantized reference floor.


@pytest.mark.parametrize('algorithm', ['least_squares', 'ai_surrogate'])
def test_calibration_panel_worker_and_config_roundtrip(tmp_path, monkeypatch, algorithm):
    import os
    import time
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    panel = window.calibration
    try:
        window.sensor_configs.directory = tmp_path
        window.output_w.setValue(128)
        window.output_h.setValue(96)
        panel.new()
        sim = GeometricSim()
        sim.cfg.camera.width, sim.cfg.camera.height = 128, 96
        sim.gel_lighting.glows = []
        ref = reference(sim, blank=True)
        panel.settings = appearance(sim)
        panel.references = [ref]
        panel.list.addItem('Blank reference')
        panel.list.setCurrentRow(0)
        panel.lighting.setChecked(False)
        panel.filters.setChecked(False)
        panel.noise.setChecked(True)
        panel.texture.setChecked(False)
        assert panel.algorithm.currentData() == 'least_squares'
        panel.algorithm.setCurrentIndex(panel.algorithm.findData(algorithm))
        panel.run_fit()
        expected_size = (window.output_w.value(), window.output_h.value())
        assert panel.worker.options['size'] == expected_size
        assert panel.worker.settings['sensor']['camera']['width'] == expected_size[0]
        assert panel.worker.settings['sensor']['camera']['height'] == expected_size[1]
        deadline = time.monotonic()+30
        while panel.worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert panel.worker is None
        assert panel.result is not None, panel.status.text()
        assert any(stage['stage'] == 'Noise' for stage in panel.result['stages'])
        from pathlib import Path
        from PIL import Image
        run = Path(panel.result['run_directory'])
        assert (run/'collection.npz').exists()
        assert (run/'metrics.csv').exists()
        report = json.loads((run/'metrics.json').read_text())
        assert report['metrics'] == panel.result['metrics']
        np.testing.assert_array_equal(np.asarray(Image.open(run/'reference_001'/'after.png')),
            (np.clip(panel.result['final'][0], 0, 1)*255).astype(np.uint8))
        assert (run/'reference_001'/'before.png').exists()
        assert (run/'reference_001'/'reference_original.png').exists()
        from mesh2tact.calibration_archive import save_run
        second = save_run(run.parent, panel.settings, panel.references, panel.result)
        assert second != run and (run/'sensor_config.json').exists()
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *a: (str(run/'sensor_config.json'), 'JSON'))
        window.sensor_configs.load_file_dialog()
        assert window.sim.gel_lighting.brightness == panel.result['settings']['gel_lighting']['brightness']
        assert panel.save_result.isEnabled()
        monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', lambda *a: ('Test calibration', True))
        panel.export()
        path = tmp_path/'Test calibration.json'
        assert path.exists(), panel.status.text()
        saved = json.loads(path.read_text())
        assert saved['calibration']['options']['fit_noise']
        assert saved['calibration']['options']['algorithm'] == algorithm
        assert saved['sensor']['optics']['noise_sigma'] == 0
        window.sensor_configs.load(path)
        assert window.sim.cfg.camera.width >= 2
        deadline = time.monotonic()+30
        while (panel.preview_worker is not None or panel.settle_timer.isActive()) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        window.tabs.setCurrentWidget(window.calibration_scroll)
        window.show()
        app.processEvents()
        window.grab().save(str(tmp_path/'calibrate.png'))
        view = window.alignment_overlay
        view.fit_image()
        fitted_scale = view.transform().m11()
        view.zoom(2)
        assert view.transform().m11() == pytest.approx(fitted_scale*2)
        panel.opacity.setValue(65)
        assert view.transform().m11() == pytest.approx(fitted_scale*2)
        view.fit_image()
        assert view.fitted
        window.width.setValue(window.width.value()+1)
        panel.sync_settings()
        assert len(panel.references) == 1
        assert not panel.references[0]['locked']
        assert panel.result is None
    finally:
        if panel.worker:
            panel.worker.stop.set()
            panel.worker.wait(10000)
            app.processEvents()
        if panel.preview_worker:
            panel.preview_pending = False
            panel.preview_worker.wait(30000)
            app.processEvents()
        window.close()


def test_reference_collection_save_reopen_and_joint_fit(tmp_path, monkeypatch):
    import os
    import time
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from mesh2tact.gui import sensor_configs
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    monkeypatch.setattr(sensor_configs, 'CONFIG_DIR', tmp_path/'sensors')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    panel = window.calibration
    try:
        window.output_w.setValue(128)
        window.output_h.setValue(96)
        panel.new()
        sim = GeometricSim()
        sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
        panel.settings = appearance(sim)
        for name, blank in [('Sphere', False), ('Box', False), ('Empty pad', True)]:
            if name == 'Box':
                import trimesh
                sim.set_mesh(trimesh.creation.box(extents=[.004, .006, .008]), 'box.stl')
            ref = reference(sim, blank=blank)
            ref.update(name=name+'.png', shape_name=name, locked=False)
            panel.references.append(ref)
            panel.list.addItem(ref['name'])
            panel.list.setCurrentRow(len(panel.references)-1)
            panel.reference_action()
            if not blank:
                window.set_alignment_button.click()
            assert ref['locked'], panel.status.text()
            assert panel.session_path.is_file()
        saved_path = panel.session_path
        settings, refs = load_session(saved_path)
        assert len(refs) == 3
        assert refs[0]['shape_name'] == 'Sphere'
        assert refs[1]['shape_name'] == 'Box' and len(refs[1]['faces']) != len(refs[0]['faces'])
        assert refs[2]['blank'] and all(r['locked'] for r in refs)
        assert panel.reference_count.text() == '3 saved / 3 references'
        panel.new()
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *a: (str(saved_path), 'Calibration'))
        panel.open()
        assert panel.session_path == saved_path
        assert len(panel.references) == 3
        panel.directions.setChecked(False)
        panel.filters.setChecked(False)
        panel.noise.setChecked(False)
        panel.texture.setChecked(False)
        panel.run_fit()
        assert len(panel.worker.refs) == 3
        deadline = time.monotonic()+30
        while panel.worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert panel.result is not None, panel.status.text()
        assert len(panel.result['metrics']) == 3
        _, _, result = load_session(saved_path, include_result=True)
        assert result is not None and len(result['metrics']) == 3
        result['options']['algorithm'] = 'ai_surrogate'
        save_session(saved_path, panel.settings, panel.references, result)
        panel.open()
        assert panel.algorithm.currentData() == 'ai_surrogate'
    finally:
        panel.timer.stop()
        panel.settle_timer.stop()
        panel.preview_pending = False
        if panel.worker:
            panel.worker.stop.set()
            panel.worker.wait(30000)
            app.processEvents()
        if panel.preview_worker:
            panel.preview_pending = False
            panel.preview_worker.wait(30000)
            app.processEvents()
        window.close()


def test_main_window_alignment_commits_only_on_set_and_restores_scene(tmp_path, monkeypatch):
    import os
    import time
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
    from mesh2tact.gui import sensor_configs
    monkeypatch.setattr(sensor_configs, 'CONFIG_DIR', tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GeometricWindow()
    panel = window.calibration
    try:
        panel.new()
        sim = GeometricSim()
        sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
        ref = reference(sim)
        ref['locked'] = False
        panel.settings = appearance(sim)
        panel.references = [ref]
        panel.list.addItem('Reference')
        panel.list.setCurrentRow(0)
        original_pose = window.x.value(), window.z.value()
        original_mesh = window.sim.mesh.vertices.copy()
        panel.begin_alignment()
        assert panel.alignment_active
        assert window.right_stack.currentIndex() == 1
        assert window.tabs.currentIndex() == 0
        window.x.setValue(1.2)
        window.rz.setValue(23)
        # Pose changes and the 3D actor must not rasterize on the GUI thread.
        monkeypatch.setattr(window.sim, 'prepare', lambda: pytest.fail('GUI rasterization'))
        window.refresh()
        assert ref['offset'] == [0., 0.]
        assert not ref['locked']
        # A second request replaces the pending pose, without blocking Qt.
        window.x.setValue(1.7)
        window.refresh()
        deadline = time.monotonic()+30
        while (panel.preview_worker is not None or panel.settle_timer.isActive()) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert panel.preview_worker is None
        assert panel.preview_cache[0][-1] == panel.preview_revision
        assert window.alignment_overlay.image is not None
        from mesh2tact.calibration import make_sim
        expected = make_sim(dict(ref, offset=[.0017, 0.], rotation=[0., 0., 23.],
                                 cut_depth=window.sim.cut_depth), panel.settings, panel.preview_size())
        expected.effects.blur_px *= panel.preview_size()[0]/panel.settings['sensor']['camera']['width']
        np.testing.assert_allclose(panel.preview_cache[2], render_float(expected, stochastic=True))
        panel.set_alignment()
        assert ref['locked']
        assert ref['offset'][0] == pytest.approx(.0017)
        assert ref['rotation'][2] == 23
        assert not panel.alignment_active
        assert (window.x.value(), window.z.value()) == original_pose
        np.testing.assert_array_equal(window.sim.mesh.vertices, original_mesh)
        panel.lock.setChecked(False)
        panel.begin_alignment()
        window.x.setValue(3)
        panel.end_alignment()
        assert ref['offset'][0] == pytest.approx(.0017)
        assert not ref['locked']
        from PIL import Image
        photo_path = tmp_path/'main_object_reference.png'
        Image.fromarray(ref['photo']).save(photo_path)
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileNames',
                            lambda *a: ([str(photo_path)], 'Images'))
        panel.add()
        panel.align_button.click()
        assert panel.alignment_active
        assert window.tabs.currentIndex() == 0
        assert panel.current()['name'] == photo_path.name
        np.testing.assert_array_equal(window.sim.mesh.vertices, original_mesh)
        assert window.load_button.isEnabled() and not window.load_button.isHidden()
        assert not window.units.isHidden() and not window.scale.isHidden()
        removed = {'Use primitive', 'Import mesh…', 'Use app shape',
                   'Calibrate current object…', 'Open calibration in main window'}
        assert not removed.intersection(button.text() for button in window.findChildren(QtWidgets.QPushButton))
        assert not any(action.text() == 'Calibration' for action in window.findChildren(QtWidgets.QToolBar)[0].actions())
        stored_vertices = panel.current()['vertices'].copy()
        import trimesh
        path = tmp_path/'reference_box.stl'
        trimesh.creation.box(extents=[2, 4, 6]).export(path)
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *a: (str(path), 'Meshes'))
        window.units.setCurrentText('mm')
        window.load_button.click()
        window.refresh()
        np.testing.assert_allclose(window.sim.mesh.extents, [.002, .004, .006])
        window.scale.setValue(2.)
        window.refresh()
        np.testing.assert_allclose(window.sim.mesh.extents, [.004, .008, .012])
        actor = window.object_actor
        transform = np.eye(4)
        transform[:3, 3] = [.6, .4, window.z.value()]
        window.gizmo_dragged(transform)
        assert window.sim.offset == pytest.approx((.0006, .0004))
        assert window.object_actor is actor
        panel.end_alignment()
        np.testing.assert_array_equal(panel.current()['vertices'], stored_vertices)
        np.testing.assert_array_equal(window.sim.mesh.vertices, original_mesh)
        panel.begin_alignment()
        window.load_button.click()
        window.scale.setValue(.5)
        window.refresh()
        panel.set_alignment()
        np.testing.assert_allclose(np.ptp(panel.current()['vertices'], axis=0), [.001, .002, .003])
    finally:
        panel.timer.stop()
        panel.settle_timer.stop()
        panel.preview_pending = False
        if panel.preview_worker:
            panel.preview_worker.wait(30000)
            app.processEvents()
        window.close()
