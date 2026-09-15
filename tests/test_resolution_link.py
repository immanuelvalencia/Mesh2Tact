import numpy as np
import pytest

from mesh2tact.preview import linked_capture, link_capture_controls
from mesh2tact.render.effects import ImageEffects, TextureCache


def test_capture_tracks_physical_aspect_and_limits():
    assert linked_capture(984, 739, 18.6, 14.3) == (984, 757)
    assert linked_capture(984, 960, 16, 12, 'height') == (1280, 960)
    assert linked_capture(984, 739, 10, 10) == (984, 984)
    assert linked_capture(2048, 64, 10, 20) == (1024, 2048)
    assert linked_capture(64, 64, 20, 10) == (128, 64)
    with pytest.raises(ValueError):
        linked_capture(984, 739, 1000, .1)
    old = dict(output_w=984, output_h=739, width=18.6, height=14.3)
    assert link_capture_controls(old)['output_h'] == 757
    assert old['output_h'] == 739


def test_texture_cache_preserves_seeded_pixels_and_is_bounded():
    cache = TextureCache()
    for seed in range(7):
        for scale, shape, sx in ((.15, (48, 64), .0186), (.3, (32, 48), .02)):
            effects = ImageEffects(seed=seed, texture=.12, texture_scale_mm=scale,
                                   read_noise=.01, shot_noise=.02, speckle=.03)
            rgb = np.full((*shape, 3), .4, dtype=np.float32)
            expected = effects.apply(rgb, sx, .0143)
            for _ in range(2):
                assert np.array_equal(expected, effects.apply(rgb, sx, .0143, texture_cache=cache))
    assert len(cache.sources) <= 4 and len(cache.fields) <= 4
