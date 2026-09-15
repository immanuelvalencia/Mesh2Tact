import numpy as np
import pytest
import trimesh

from mesh2tact.geometric import GeometricSim


def test_resize_import_then_quality_and_restore(tmp_path):
    path = tmp_path / 'box.stl'
    trimesh.creation.box(extents=[2, 4, 6]).export(path)
    sim = GeometricSim()
    sim.load(path, units='mm', scale=2.)
    sim.set_quality(1)
    sim.prepare()
    sim.set_scale(.5)
    assert sim._key is None
    np.testing.assert_allclose(sim.mesh.extents, [.001, .002, .003])
    sim.set_quality(2)
    np.testing.assert_allclose(sim.mesh.extents, [.001, .002, .003])
    sim.set_scale(1.)
    np.testing.assert_allclose(sim.mesh.extents, [.002, .004, .006])
    assert sim.units == 'mm'
    for invalid in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            sim.set_scale(invalid)


def test_demo_quality_retains_scale():
    sim = GeometricSim()
    sim.set_scale(2.)
    sim.set_quality(1)
    np.testing.assert_allclose(sim.mesh.extents, [.016, .016, .016])
