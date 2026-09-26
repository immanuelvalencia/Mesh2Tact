"""Seeded Gaussian-process search over normalized calibration parameters."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def surrogate_search(objective, start, budget, rng, check_cancel, progress):
    """Learn render error online; return the best actually evaluated setting.

    A fixed RBF kernel and lower-confidence-bound acquisition provide a small,
    dependency-free Bayesian search before the production solver refines it.
    """
    points, values = [], []
    dimension = len(start)

    def kernel(a, b):
        distance = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-distance / (.18 * max(1, dimension)))

    for index in range(max(1, budget)):
        check_cancel()
        if index == 0:
            candidate = start.copy()
        elif index < min(6, budget):
            candidate = np.clip(start + rng.normal(0, .15, dimension), 0., 1.)
        else:
            x, y = np.asarray(points), np.asarray(values)
            best = x[np.argmin(y)]
            pool = np.vstack((rng.uniform(0, 1, (128, dimension)),
                              np.clip(best + rng.normal(0, .08, (128, dimension)), 0, 1)))
            covariance = kernel(x, x) + np.eye(len(x)) * 1e-6
            factor = cho_factor(covariance)
            cross = kernel(pool, x)
            normalized = (y - y.mean()) / max(float(y.std()), 1e-8)
            mean = cross @ cho_solve(factor, normalized)
            variance = np.maximum(0, 1 - np.sum(cross * cho_solve(factor, cross.T).T, axis=1))
            acquisition = mean - 1.5 * np.sqrt(variance)
            candidate = pool[np.argmin(acquisition)]
        value = float(objective(candidate))
        if not np.isfinite(value):
            raise ValueError('Non-finite calibration error during surrogate search')
        points.append(candidate.copy())
        values.append(value)
        progress(f'AI surrogate: trial {index+1}/{budget}')
    best_index = int(np.argmin(values))
    return points[best_index], values[best_index], len(values)
