from copy import deepcopy
import numpy as np
from mesh2tact.config import SensorConfig
from mesh2tact.render.optical import GelSightRenderer
from mesh2tact.geometric import GeometricSim


def test_tiled_shading_matches_full_frame_without_seams():
    cfg = SensorConfig()
    renderer = GelSightRenderer(cfg)
    rng = np.random.default_rng(12)
    height = -rng.uniform(0, .001, (139, 301)).astype(np.float32)
    normals = rng.normal(size=(*height.shape, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    for azimuth in (0, 90, 180, 270):
        cfg.optics.leds[0].azimuth = azimuth
        actual = renderer.shade(normals, height=height)
        full = renderer._shade_edges(normals, cfg.optics, height, full_height=height.shape[0])
        assert np.array_equal(actual, full)


def test_flat_cache_invalidates_for_settings_and_resolution():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 80, 64
    sim.cut_depth = .001
    sim.render()
    cached = next(iter(sim._flat_lighting_cache.values()))
    sim.cut_depth = .0005
    actual = sim.render()[1]
    assert next(iter(sim._flat_lighting_cache.values())) is cached
    fresh = deepcopy(sim)
    fresh._flat_lighting_cache.clear()
    assert np.array_equal(actual, fresh.render()[1])
    for change in ('light', 'size', 'resolution'):
        previous = next(iter(sim._flat_lighting_cache.values()))
        if change == 'light':
            sim.cfg.optics.leds[0].elevation += 10
        elif change == 'size':
            sim.cfg.gel.size_x *= 1.1
        else:
            sim.cfg.camera.width += 3
        actual = sim.render()[1]
        assert len(sim._flat_lighting_cache) == 1
        assert next(iter(sim._flat_lighting_cache.values())) is not previous
        fresh = deepcopy(sim)
        fresh._flat_lighting_cache.clear()
        assert np.array_equal(actual, fresh.render()[1])
