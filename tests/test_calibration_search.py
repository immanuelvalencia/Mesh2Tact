import numpy as np
import pytest

from mesh2tact.calibration import Cancelled
from mesh2tact.calibration_search import surrogate_search


def test_surrogate_learns_reproducibly_and_keeps_best_evaluated_trial():
    def run():
        trials = []
        def objective(x):
            assert np.all((x >= 0) & (x <= 1))
            trials.append(x.copy())
            return np.sum((x - .7)**2)
        result = surrogate_search(objective, np.array([.1, .1]), 24,
            np.random.default_rng(7), lambda: None, lambda message: None)
        assert len(trials) == result[2] == 24
        assert any(np.array_equal(result[0], trial) for trial in trials)
        assert result[1] == min(objective(x) for x in trials.copy())
        assert result[1] < .05
        return result[0]
    np.testing.assert_array_equal(run(), run())


def test_surrogate_cancels_before_render():
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        surrogate_search(lambda x: pytest.fail('Rendered after cancellation'),
            np.array([.5]), 8, np.random.default_rng(0), cancel, lambda message: None)
