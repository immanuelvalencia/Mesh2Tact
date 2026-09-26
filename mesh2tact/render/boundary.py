"""Metric appearance field; never modifies exported contact geometry.

Each connected contact has its own indentation-dependent smoothing scale.
Signed slopes naturally reverse at holes. Finite support preserves the remote
background. This is an empirical optical model, not an elastic solver.
"""
import numpy as np
from scipy import ndimage


def boundary_height(depth, size_x, size_y, width_mm, growth):
    depth = np.maximum(np.asarray(depth, dtype=np.float32), 0)
    h, w = depth.shape
    spacing = (size_y/h, size_x/w)
    labels, count = ndimage.label(depth > 1e-8)
    result = np.zeros_like(depth)
    if not count:
        return result
    peaks = ndimage.maximum(depth, labels, np.arange(1, count+1))
    for index, (box, peak) in enumerate(zip(ndimage.find_objects(labels), peaks), 1):
        sigma_m = (width_mm+growth*float(peak)*1000)/1000
        sigma = tuple(sigma_m/s for s in spacing)
        padding = tuple(int(np.ceil(3*s))+1 for s in sigma)
        region = tuple(slice(max(0, b.start-p), min(n, b.stop+p))
                       for b, p, n in zip(box, padding, depth.shape))
        local = np.where(labels[region] == index, depth[region], 0)
        result[region] += ndimage.gaussian_filter(local, sigma, mode='nearest', truncate=3)
    return -result
