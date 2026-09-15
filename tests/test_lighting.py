from dataclasses import asdict
import json

import numpy as np
import pytest

from mesh2tact.config import SensorConfig
from mesh2tact.lighting import load_lighting, save_lighting
from mesh2tact.render.optical import GelSightRenderer
from mesh2tact.sensor.heightmap import SurfaceSampler


def test_lighting_changes_flat_pad_and_contact_and_round_trips(tmp_path):
    cfg = SensorConfig()
    cfg.optics.noise_sigma = 0
    lo = np.zeros(3)
    hi = np.array([cfg.gel.size_x, cfg.gel.size_y, cfg.gel.thickness])
    sampler = SurfaceSampler(cfg, lo, hi)
    flat = np.full((48, 64), cfg.gel.thickness)
    y, x = np.mgrid[-1:1:48j, -1:1:64j]
    contact = flat - .0005 * np.exp(-10 * (x*x + y*y))
    renderer = GelSightRenderer(cfg)
    baseline = renderer.render(sampler, flat)
    cfg.optics.leds[0].intensity = 0
    assert not np.array_equal(baseline, renderer.render(sampler, flat))
    cfg.optics.leds[1].color = (1, .2, .05)
    before = renderer.render(sampler, contact)
    cfg.optics.leds[1].azimuth += 90
    assert not np.array_equal(before, renderer.render(sampler, contact))
    expected = renderer.render(sampler, contact)
    path = tmp_path / "lighting.json"
    save_lighting(path, cfg.optics)
    restored = load_lighting(path)
    assert asdict(restored) == asdict(cfg.optics)
    cfg.optics = restored
    assert np.array_equal(expected, GelSightRenderer(cfg).render(sampler, contact))
    for led in cfg.optics.leds:
        led.intensity = 0
    cfg.optics.ambient = (0, 0, 0)
    assert not GelSightRenderer(cfg).render(sampler, flat).any()


@pytest.mark.parametrize("value", [-1, float("nan"), 9])
def test_invalid_presets_are_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    save_lighting(path, SensorConfig().optics)
    raw = json.loads(path.read_text())
    raw["optics"]["leds"][0]["intensity"] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Intensity"):
        load_lighting(path)
