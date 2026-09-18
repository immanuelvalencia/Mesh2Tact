import numpy as np
import pytest

from mesh2tact.config import SensorConfig
from mesh2tact.geometric import GeometricSim
from mesh2tact.lighting import load_lighting, save_lighting
from mesh2tact.render.optical import GelSightRenderer


def test_flat_face_has_absolute_depth_cue_and_empty_pad_is_unchanged():
    cfg = SensorConfig()
    cfg.camera.width, cfg.camera.height = 40, 30
    cfg.optics.side_lighting = False
    cfg.optics.noise_sigma = 0
    sim = GeometricSim(cfg)
    sim.prepare()
    renderer = GelSightRenderer(cfg)
    renderer.set_background(np.full((30, 40, 3), .5))
    blank = np.zeros((30, 40))
    before = renderer.render(sim.sampler, blank, as_uint8=False)
    cfg.optics.depth_shading = .35
    np.testing.assert_array_equal(before, renderer.render(sim.sampler, blank, as_uint8=False))
    shallow = renderer.render(sim.sampler, blank-.0005, as_uint8=False)
    deep = renderer.render(sim.sampler, blank-.0015, as_uint8=False)
    assert np.all(deep < shallow)
    assert np.all(shallow < before)
    cfg.camera.max_depth *= 10
    np.testing.assert_array_equal(deep, renderer.render(sim.sampler, blank-.0015, as_uint8=False))


def test_depth_shading_preserves_geometry_and_roundtrips(tmp_path):
    cfg = SensorConfig()
    cfg.camera.width, cfg.camera.height = 40, 30
    sim = GeometricSim(cfg)
    original_depth, original_rgb = sim.render()
    original_raw = sim.raw_depth.copy()
    sim.cfg.optics.depth_shading = .35
    sim.cfg.optics.shadow_strength = .65
    depth, rgb = sim.render()
    np.testing.assert_array_equal(depth, original_depth)
    np.testing.assert_array_equal(sim.raw_depth, original_raw)
    assert not np.array_equal(rgb, original_rgb)
    path = tmp_path / 'lighting.json'
    save_lighting(path, sim.cfg.optics)
    assert load_lighting(path).depth_shading == .35


@pytest.mark.parametrize('strength', [-1, float('nan'), float('inf'), 3])
def test_invalid_depth_strength_is_rejected(strength, tmp_path):
    cfg = SensorConfig()
    cfg.optics.depth_shading = strength
    with pytest.raises(ValueError, match='Depth shading'):
        save_lighting(tmp_path / 'invalid.json', cfg.optics)


def test_directional_shadow_tracks_occlusion_and_light_direction():
    cfg = SensorConfig()
    cfg.optics.shadow_strength = .7
    cfg.optics.shadow_azimuth = 0
    cfg.optics.shadow_elevation = 15
    renderer = GelSightRenderer(cfg)
    x = (np.arange(160)+.5)/160*cfg.gel.size_x-cfg.gel.size_x/2
    height = -np.broadcast_to(np.maximum(.0015*(1-(x/.003)**2), 0), (80, 160)).copy()
    original = height.copy()
    right = renderer.depth_shadow(height)
    cfg.optics.shadow_azimuth = 180
    left = renderer.depth_shadow(height)
    assert right.min() < .5
    assert not np.allclose(right, left)
    np.testing.assert_allclose(right, left[:, ::-1], atol=2e-5)
    np.testing.assert_array_equal(height, original)
    np.testing.assert_array_equal(right[height == 0], 1)
    np.testing.assert_array_equal(renderer.depth_shadow(np.zeros_like(height)), 1)
    cfg.optics.shadow_strength = 0
    np.testing.assert_array_equal(renderer.depth_shadow(height), 1)


def test_shadow_controls_roundtrip(tmp_path):
    cfg = SensorConfig()
    cfg.optics.shadow_strength = .65
    cfg.optics.shadow_azimuth = 120
    cfg.optics.shadow_elevation = 15
    path = tmp_path / 'shadows.json'
    save_lighting(path, cfg.optics)
    loaded = load_lighting(path)
    assert (loaded.shadow_strength, loaded.shadow_azimuth, loaded.shadow_elevation) == (.65, 120, 15)


def test_relief_lifts_centre_shades_rim_and_preserves_empty_pad():
    cfg = SensorConfig()
    cfg.optics.depth_relief = .8
    renderer = GelSightRenderer(cfg)
    x = (np.arange(161)-80)*cfg.gel.size_x/161
    y = (np.arange(121)-60)*cfg.gel.size_y/121
    depth = np.maximum(.001*(1-(x[None,:]**2+y[:,None]**2)/.003**2), 0)
    original = depth.copy()
    gain = renderer.relief_gain(-depth)
    assert gain[60,80] > 1.1
    assert gain[(depth > .0001) & (depth < .0002)].mean() < 1
    np.testing.assert_array_equal(gain[depth == 0], 1)
    np.testing.assert_array_equal(depth, original)
    cfg.camera.max_depth *= 10
    np.testing.assert_array_equal(gain, renderer.relief_gain(-depth))
