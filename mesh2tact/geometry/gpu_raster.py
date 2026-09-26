"""Bounded CUDA minimum-Z rasterization using the installed PyTorch runtime."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, OSError):
        return False


def rasterize_surface_cuda(vertices, faces, width, height, size_x, size_y):
    import torch
    if not cuda_available():
        raise RuntimeError('CUDA geometry requires a CUDA-enabled PyTorch installation and NVIDIA GPU. Choose CPU or Auto.')
    if width < 2 or height < 2 or min(size_x, size_y) <= 0:
        raise ValueError('Sensor dimensions must be positive and resolution at least 2x2')
    # Match the CPU projection, bounding boxes and degeneracy threshold exactly.
    projected = np.asarray(vertices, dtype=float).copy()
    projected[:, 0] = (projected[:, 0] / size_x + .5) * width - .5
    projected[:, 1] = (projected[:, 1] / size_y + .5) * height - .5
    triangles = projected[np.asarray(faces)]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominator = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    lower = np.maximum(np.ceil(triangles[:, :, :2].min(axis=1)).astype(np.int64), 0)
    upper = np.minimum(np.floor(triangles[:, :, :2].max(axis=1)).astype(np.int64), [width-1, height-1])
    visible = (np.abs(denominator) >= 1e-12) & np.all(lower <= upper, axis=1)
    if not visible.any():
        return np.full((height, width), np.inf)
    lower, upper = lower[visible], upper[visible]
    widths = upper[:, 0]-lower[:, 0]+1
    counts = widths*(upper[:, 1]-lower[:, 1]+1)
    ends = np.cumsum(counts)
    with torch.inference_mode():
        device = 'cuda'
        triangle = torch.as_tensor(triangles[visible], device=device)
        denom = torch.as_tensor(denominator[visible], device=device)
        lo = torch.as_tensor(lower, device=device)
        box_width = torch.as_tensor(widths, device=device)
        stops = torch.as_tensor(ends, device=device)
        starts = torch.as_tensor(ends-counts, device=device)
        surface = torch.full((height*width,), float('inf'), dtype=torch.float64, device=device)
        # Bound intermediate storage independently of triangle count/resolution.
        for start in range(0, int(ends[-1]), 262144):
            position = torch.arange(start, min(start+262144, int(ends[-1])), device=device)
            face = torch.searchsorted(stops, position, right=True)
            offset = position-starts[face]
            x = lo[face, 0] + offset % box_width[face]
            y = lo[face, 1] + offset // box_width[face]
            a, b, c = triangle[face, 0], triangle[face, 1], triangle[face, 2]
            u = ((b[:, 1]-c[:, 1])*(x-c[:, 0]) + (c[:, 0]-b[:, 0])*(y-c[:, 1])) / denom[face]
            v = ((c[:, 1]-a[:, 1])*(x-c[:, 0]) + (a[:, 0]-c[:, 0])*(y-c[:, 1])) / denom[face]
            w = 1-u-v
            inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
            z = u*a[:, 2] + v*b[:, 2] + w*c[:, 2]
            surface.scatter_reduce_(0, y*width+x, torch.where(inside, z, torch.inf), reduce='amin', include_self=True)
        return surface.reshape(height, width).cpu().numpy()
