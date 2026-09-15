"""Regular image-grid gradients and normals; no particle simulation."""
from dataclasses import dataclass
import numpy as np
from ..config import SensorConfig

@dataclass
class SurfaceSampler:
    cfg: SensorConfig
    gel_lo: np.ndarray
    gel_hi: np.ndarray
    rest_height: float | None = None

    def __post_init__(self):
        self.width, self.height = self.cfg.camera.width, self.cfg.camera.height
        self.footprint = (self.gel_hi-self.gel_lo)[:2]
        self.pixel_size = self.footprint / [self.width, self.height]

    def gradients(self, height):
        gy, gx = np.gradient(height, self.pixel_size[1], self.pixel_size[0])
        return gx, gy

    def normals(self, height):
        gx, gy = self.gradients(height)
        normals = np.stack([-gx, -gy, np.ones_like(height)], axis=-1)
        return normals / np.linalg.norm(normals, axis=-1, keepdims=True)
