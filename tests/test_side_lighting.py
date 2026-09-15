import numpy as np
import json
from pathlib import Path

from mesh2tact.config import SensorConfig, LEDConfig
from mesh2tact.render.optical import GelSightRenderer


def test_real_reference_config_and_edge_color_order():
    from mesh2tact.lighting import validate_lighting
    from mesh2tact.sensor.heightmap import SurfaceSampler
    data = json.loads((Path(__file__).resolve().parents[1]/'configs/sensors/Real GelSight reference.json').read_text())
    cfg = SensorConfig.from_dict(data['sensor'])
    validate_lighting(cfg.optics)
    cfg.camera.width, cfg.camera.height = 60, 50
    sampler = SurfaceSampler(cfg, np.array([-cfg.gel.size_x/2,-cfg.gel.size_y/2,0]),
                              np.array([cfg.gel.size_x/2,cfg.gel.size_y/2,0]))
    y = (np.arange(50)-24.5)*cfg.gel.size_y/50
    height = -np.broadcast_to(.0002*np.exp(-.5*(y/.0004)**2)[:,None], (50,60))
    renderer = GelSightRenderer(cfg)
    normal = sampler.normals(height)
    flat = np.broadcast_to([0.,0.,1.],normal.shape)
    delta = renderer.shade(normal,height=height)-renderer.shade(flat,height=np.zeros_like(height))
    assert delta[23,30,0] > delta[26,30,0]
    assert delta[26,30,2] > delta[23,30,2]


def test_side_light_changes_with_position_depth_and_surface_normal():
    cfg = SensorConfig()
    cfg.optics.leds = [LEDConfig(color=(1, 0, 0), azimuth=0, elevation=25)]
    cfg.optics.ambient = (0, 0, 0)
    cfg.optics.specular_gain = 0
    renderer = GelSightRenderer(cfg)
    height = np.zeros((30, 40))
    normals = np.broadcast_to([0., 0., 1.], (30, 40, 3))
    flat = renderer.shade(normals, height=height)
    assert flat[15, -1, 0] > flat[15, 0, 0]
    assert not np.allclose(flat, renderer.shade(normals, height=height-.001))
    tilted = np.broadcast_to([.6, 0, .8], normals.shape)
    assert renderer.shade(tilted, height=height)[15, 20, 0] > flat[15, 20, 0]
    cfg.optics.side_lighting = False
    directional = renderer.shade(normals, height=height)
    assert np.allclose(directional, directional[0, 0])


def test_full_edge_matches_dense_line_integral_not_center_point():
    cfg = SensorConfig()
    cfg.optics.leds = [LEDConfig(color=(1, 0, 0), azimuth=0, elevation=25)]
    cfg.optics.ambient = (0, 0, 0)
    cfg.optics.specular_gain = 0
    cfg.optics.side_falloff = 0
    renderer = GelSightRenderer(cfg)
    shape = (20, 24)
    height = np.zeros(shape)
    normals = np.broadcast_to([0., 0., 1.], (*shape, 3))
    result = renderer.shade(normals, height=height)[..., 0]
    sx, sy = cfg.gel.size_x, cfg.gel.size_y
    x, y = np.meshgrid((np.arange(shape[1])+.5)/shape[1]*sx-sx/2,
                       (np.arange(shape[0])+.5)/shape[0]*sy-sy/2)
    radius = sx/2*cfg.optics.side_distance
    z = radius*np.tan(np.deg2rad(25))
    dense = np.zeros(shape)
    for offset in ((np.arange(1024)+.5)/1024-.5)*sy:
        dense += z/np.sqrt((radius-x)**2+(offset-y)**2+z*z)/1024
    dense *= cfg.optics.diffuse_gain
    assert np.max(np.abs(result-dense)) < .002
    center_point = cfg.optics.diffuse_gain*z/np.sqrt((radius-x)**2+y*y+z*z)
    assert np.max(np.abs(result-center_point)) > .02
    assert np.allclose(result, result[::-1], atol=1e-6)
    cfg.optics.leds[0].azimuth = 180
    opposite = renderer.shade(normals, height=height)[..., 0]
    assert np.allclose(result[:, ::-1], opposite, atol=1e-6)
