"""Seeded camera/gel appearance effects. These never modify geometric depth."""
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class TextureCache:
    """Small, per-workload cache; no global arrays or unbounded seed growth."""
    def __init__(self):
        self.sources = OrderedDict()
        self.fields = OrderedDict()

    @staticmethod
    def remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > 4:
            cache.popitem(last=False)
        return value

    def field(self, seed, scale_mm, size_x, size_y, h, w):
        key = (seed, scale_mm, size_x, size_y, h, w)
        if key in self.fields:
            self.fields.move_to_end(key)
            return self.fields[key]
        if seed not in self.sources:
            self.remember(self.sources, seed, np.random.default_rng(seed+1).normal(size=(512, 512)).astype(np.float32))
        self.sources.move_to_end(seed)
        field = _texture_field(self.sources[seed], scale_mm, size_x, size_y, h, w)
        return self.remember(self.fields, key, field)

    def hd_field(self, seed, scale_mm, size_x, size_y, h, w):
        key = ('hd', seed, scale_mm, size_x, size_y, h, w)
        if key in self.fields:
            self.fields.move_to_end(key)
            return self.fields[key]
        source_key = ('hd', seed)
        if source_key not in self.sources:
            source = np.random.default_rng(seed+2).standard_normal((2048, 2048), dtype=np.float32)
            self.remember(self.sources, source_key, source)
        self.sources.move_to_end(source_key)
        source = self.sources[source_key]
        sigma = (scale_mm/(size_y*1000)*2048, scale_mm/(size_x*1000)*2048)
        # Band-limited grain: retain fine structure without broad illumination blotches.
        grain = gaussian_filter(source, sigma, mode='reflect')
        grain -= gaussian_filter(source, tuple(3*s for s in sigma), mode='reflect')
        grain -= grain.mean()
        grain /= max(float(grain.std()), 1e-6)
        # Suppress subpixel structure before sampling a smaller capture.
        aa = (.5*max(2048/h-1, 0), .5*max(2048/w-1, 0))
        if max(aa):
            grain = gaussian_filter(grain, aa, mode='reflect')
        yy, xx = np.meshgrid((np.arange(h)+.5)*2048/h-.5,
                             (np.arange(w)+.5)*2048/w-.5, indexing='ij')
        field = map_coordinates(grain, [yy, xx], order=1, mode='nearest')
        return self.remember(self.fields, key, field)


def _texture_field(source, scale_mm, size_x, size_y, h, w):
    sigma = (scale_mm/(size_y*1000)*512, scale_mm/(size_x*1000)*512)
    field = gaussian_filter(source, sigma, mode='reflect')
    field = (field-field.mean())/max(float(field.std()), 1e-6)
    yy, xx = np.meshgrid((np.arange(h)+.5)*512/h-.5, (np.arange(w)+.5)*512/w-.5, indexing='ij')
    return map_coordinates(field, [yy, xx], order=1, mode='nearest')


@dataclass
class ImageEffects:
    enabled: bool = True
    seed: int = 0
    read_noise: float = 0.0       # additive Gaussian sigma in 0..1 intensity
    shot_noise: float = 0.0       # sigma at intensity 1; Poisson photon noise
    speckle: float = 0.0          # multiplicative Gaussian sigma
    texture: float = 0.0          # fixed gel reflectance modulation strength
    texture_scale_mm: float = .15
    hd_texture: float = 0.0       # fine, fixed achromatic gel grain; off for legacy presets
    hd_texture_scale_mm: float = .025  # spatial correlation scale, not measured roughness
    blur_px: float = 0.0          # Gaussian optical blur, output pixels
    vignette: float = 0.0         # fractional corner darkening
    contrast: float = 1.0
    gamma: float = 1.0

    def apply(self, rgb, size_x, size_y, *, texture_cache=None):
        if not self.enabled:
            return np.clip(rgb, 0, 1)
        if not all(np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Image effects must be finite")
        if min(self.read_noise, self.shot_noise, self.speckle, self.texture,
               self.hd_texture, self.blur_px, self.vignette, self.contrast) < 0 or self.gamma <= 0 or min(self.texture_scale_mm, self.hd_texture_scale_mm) <= 0:
            raise ValueError("Effects must be nonnegative; gamma and texture scale must be positive")
        rng = np.random.default_rng(self.seed)
        out = np.asarray(rgb, dtype=np.float32).copy()
        h, w = out.shape[:2]
        if self.texture:
            # Fixed reference grid gives the same gel pattern across resolutions.
            if texture_cache is None:
                source = np.random.default_rng(self.seed+1).normal(size=(512, 512)).astype(np.float32)
                texture = _texture_field(source, self.texture_scale_mm, size_x, size_y, h, w)
            else:
                texture = texture_cache.field(self.seed, self.texture_scale_mm, size_x, size_y, h, w)
            out *= np.maximum(0, 1+self.texture*texture[..., None])
        if self.hd_texture:
            cache = texture_cache if texture_cache is not None else TextureCache()
            grain = cache.hd_field(self.seed, self.hd_texture_scale_mm, size_x, size_y, h, w)
            out *= np.maximum(0, 1+self.hd_texture*grain[..., None])
        if self.vignette:
            y = (np.arange(h)+.5)/h*2-1
            x = (np.arange(w)+.5)/w*2-1
            radius = (y[:, None]**2+x[None, :]**2)/2
            out *= np.maximum(0, 1-self.vignette*radius[..., None])
        if self.blur_px:
            out = gaussian_filter(out, (self.blur_px, self.blur_px, 0), mode="reflect")
        if self.shot_noise:
            photons = 1/max(self.shot_noise**2, 1e-10)
            out = rng.poisson(np.clip(out, 0, 1)*photons).astype(np.float32)/photons
        if self.speckle:
            out *= 1+rng.normal(0, self.speckle, (h, w, 1))
        if self.read_noise:
            out += rng.normal(0, self.read_noise, out.shape)
        out = np.clip((out-.5)*self.contrast+.5, 0, 1)
        return np.clip(out**(1/self.gamma), 0, 1).astype(np.float32)
