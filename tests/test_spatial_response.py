from dataclasses import asdict
import numpy as np
import pytest
from mesh2tact.config import SensorConfig
from mesh2tact.render.optical import GelSightRenderer
from mesh2tact.lighting import validate_lighting


def test_spatial_response_has_flat_invariance_and_position_dependence():
    cfg=SensorConfig()
    coefficients=np.zeros((30,3))
    coefficients[0,0]=.2  # nx
    coefficients[5,0]=.15  # x * nx
    cfg.optics.spatial_response=coefficients.tolist()
    restored=SensorConfig.from_dict(asdict(cfg))
    validate_lighting(restored.optics)
    renderer=GelSightRenderer(restored)
    flat=np.broadcast_to([0.,0.,1.],(18,24,3))
    np.testing.assert_allclose(renderer.shade(flat),np.broadcast_to(cfg.optics.ambient,flat.shape),atol=1e-7)
    tilted=np.broadcast_to([.6,0.,.8],flat.shape)
    rgb=renderer.shade(tilted)
    assert rgb[9,-1,0]>rgb[9,0,0]+.1
    np.testing.assert_allclose(rgb[...,1:],renderer.shade(flat)[...,1:])


def test_invalid_spatial_response_is_rejected():
    cfg=SensorConfig()
    cfg.optics.spatial_response=[[1,2,3]]
    with pytest.raises(ValueError,match='30 by 3'):validate_lighting(cfg.optics)
