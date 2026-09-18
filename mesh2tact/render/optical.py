"""Optical model: height field -> the RGB image a GelSight-style camera returns.

Two modes are supported.

*Analytic* (default).  The gel's inner surface is lit by several coloured LEDs at
low elevation angles and imaged from directly below.  Each pixel's colour is a
Lambertian + Blinn-Phong response of the local surface normal to each light.
This is the same family of model TACTO uses and it needs no real hardware.

*Calibrated* (Taxim-style).  If you have a real sensor, press a ball of known
radius into it, recover the ground-truth surface gradients analytically, and fit
a per-channel polynomial / lookup table from (dz/dx, dz/dy) to pixel intensity.
Point ``optics.calibration_path`` at the resulting ``.npz`` and the analytic
shading is replaced by that table, which is what buys photorealism.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import json

import numpy as np
from scipy import ndimage

from ..config import SensorConfig, OpticsConfig


def _light_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def spatial_features(normals):
    """Position-conditioned normal response, exactly zero for a flat surface."""
    h,w = normals.shape[:2]
    y,x = np.meshgrid(2*(np.arange(h)+.5)/h-1, 2*(np.arange(w)+.5)/w-1, indexing='ij')
    nx,ny,nz = np.moveaxis(normals,-1,0)
    spatial = np.stack((np.ones_like(x),x,y,x*x,x*y,y*y),axis=-1)
    normal = np.stack((nx,ny,nz-1,nx*ny,nx*nx-ny*ny),axis=-1)
    return (spatial[..., :,None]*normal[...,None,:]).reshape(h,w,30)


class GelSightRenderer:
    """Turns a surface height field into a tactile image."""

    def __init__(self, cfg: SensorConfig) -> None:
        self.cfg = cfg
        self.optics = cfg.optics
        self.lights = [
            (_light_direction(led.azimuth, led.elevation),
             np.asarray(led.color, dtype=float) * led.intensity)
            for led in self.optics.leds
        ]
        self._table = None
        self._table_range = None
        self._background = None
        self._reference_corrections = {}
        self._flat_cache = {}
        if self.optics.calibration_path:
            self.load_calibration(self.optics.calibration_path)
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    def load_calibration(self, path: str | Path) -> None:
        """Load a Taxim-style LUT: ``table`` (Gx, Gy, 3) plus gradient ``range``."""
        data = np.load(path)
        self._table = data["table"].astype(np.float64)
        self._table_range = float(data["grad_range"]) if "grad_range" in data else 2.0
        if "background" in data:
            self._background = data["background"].astype(np.float64) / 255.0

    def set_background(self, image: np.ndarray) -> None:
        """Use a real no-contact frame as the base image (recommended if you have one)."""
        self._background = np.asarray(image, dtype=np.float64)
        if self._background.max() > 1.5:
            self._background /= 255.0

    # ------------------------------------------------------------------ #
    def shade(self, normals: np.ndarray, optics=None, height=None) -> np.ndarray:
        """Analytic multi-LED shading of a normal map, float RGB in [0, 1]."""
        optics = optics or self.optics
        normals = np.asarray(normals, dtype=np.float32)
        if optics.spatial_response is not None:
            coefficients = np.asarray(optics.spatial_response, dtype=float)
            if coefficients.shape != (30,3) or not np.isfinite(coefficients).all():
                raise ValueError('spatial_response must be a finite 30 by 3 matrix')
            return (spatial_features(normals) @ coefficients + np.asarray(optics.ambient)).astype(np.float32)
        if optics.side_lighting and height is not None:
            return self._shade_edges(normals, optics, height)
        lights = [(_light_direction(led.azimuth, led.elevation).astype(np.float32),
                   np.asarray(led.color, dtype=np.float32) * led.intensity) for led in optics.leds]
        rgb = np.zeros(normals.shape[:2] + (3,), dtype=np.float32)
        rgb += np.asarray(optics.ambient, dtype=np.float32)

        view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for led, (direction, color) in zip(optics.leds, lights):
            attenuation = 1.0
            directions = [(direction, attenuation)]
            for direction, attenuation in directions:
                lambert = np.clip(np.sum(normals*direction, axis=-1), 0.0, None)*attenuation
                for channel in range(3):
                    rgb[..., channel] += (optics.diffuse_gain * color[channel]) * lambert

                if optics.specular_gain > 0.0:
                    half = direction + view
                    half /= np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-9)
                    spec = np.clip(np.sum(normals*half, axis=-1), 0.0, None) ** optics.shininess * attenuation
                    for channel in range(3):
                        rgb[..., channel] += (optics.specular_gain * color[channel]) * spec
        return rgb

    def _shade_edges(self, normals, optics, height, row_start=0, full_height=None):
        """Same 12-point line integral, without temporary H×W×3 vectors.

        Coordinate grids are shared by the lights, and each light's height and
        fixed-axis offsets are shared by its samples. This reduces allocation
        and strided reductions in both rendering and calibration Jacobians.
        """
        h, w = height.shape
        # Keep intermediate light vectors in a small CPU-cache-friendly block.
        # Global row coordinates preserve the exact full-resolution lighting.
        rows = max(1, 16384//w)
        if full_height is None and h > rows:
            rgb = np.empty((h, w, 3), dtype=np.float32)
            for start in range(0, h, rows):
                rgb[start:start+rows] = self._shade_edges(normals[start:start+rows], optics,
                    height[start:start+rows], start, h)
            return rgb
        full_height = h if full_height is None else full_height
        sx, sy = self.cfg.gel.size_x, self.cfg.gel.size_y
        x, y = np.meshgrid((np.arange(w)+.5)/w*sx-sx/2,
                           (np.arange(row_start, row_start+h)+.5)/full_height*sy-sy/2)
        nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb += np.asarray(optics.ambient, dtype=np.float32)
        for led in optics.leds:
            color = np.asarray(led.color, dtype=np.float32)*led.intensity
            side = int(np.floor((led.azimuth % 360+45)/90)) % 4
            radius = (sx if side in (0, 2) else sy)/2*optics.side_distance
            z = radius*np.tan(np.deg2rad(min(led.elevation, 85)))
            dz = np.asarray(z-height, dtype=np.float32)
            zz = dz*dz
            if side in (0, 2):
                fixed = np.asarray((radius if side == 0 else -radius)-x, dtype=np.float32)
                span = sy
            else:
                fixed = np.asarray((radius if side == 1 else -radius)-y, dtype=np.float32)
                span = sx
            for offset in ((np.arange(12)+.5)/12-.5)*span:
                if side in (0, 2):
                    dx, dy = fixed, np.asarray(offset-y, dtype=np.float32)
                else:
                    dx, dy = np.asarray(offset-x, dtype=np.float32), fixed
                distance = np.maximum(np.sqrt(dx*dx+dy*dy+zz), 1e-9)
                lx, ly, lz = dx/distance, dy/distance, dz/distance
                reference_distance = np.sqrt(radius*radius+offset*offset+z*z)
                attenuation = (reference_distance/distance)**optics.side_falloff/12
                lambert = np.clip(nx*lx+ny*ly+nz*lz, 0., None)*attenuation
                for channel in range(3):
                    rgb[..., channel] += (optics.diffuse_gain*color[channel])*lambert
                if optics.specular_gain > 0:
                    hz = lz+np.float32(1.)
                    norm = np.maximum(np.sqrt(lx*lx+ly*ly+hz*hz), 1e-9)
                    spec = np.clip(nx*(lx/norm)+ny*(ly/norm)+nz*(hz/norm), 0., None)**optics.shininess*attenuation
                    for channel in range(3):
                        rgb[..., channel] += (optics.specular_gain*color[channel])*spec
        return rgb

    @staticmethod
    def reference_illumination(shape) -> np.ndarray:
        """Smooth blue-green illumination inspired by the supplied sensor frame.

        This is an approximate empty-pad appearance, not a calibrated background:
        the contact visible in the reference photograph is deliberately not copied.
        """
        y, x = np.mgrid[-1:1:complex(shape[0]), -1:1:complex(shape[1])]
        base = np.empty((*shape, 3))
        base[:] = [0.27, 0.38, 0.40]
        blue = np.exp(-((x / 0.9) ** 2 + ((y + 0.5) / 0.8) ** 2))
        green = np.exp(-((x + 0.98) / 0.16) ** 2)
        warm = np.exp(-((x / 0.7) ** 2 + ((y - 0.45) / 0.8) ** 2))
        base += blue[..., None] * [-0.03, 0.01, 0.10]
        base += green[..., None] * [-0.06, 0.10, -0.03]
        base += warm[..., None] * [0.10, 0.04, 0.015]
        vignette = 1 - 0.28 * np.clip((x*x + y*y - 0.6) / 1.4, 0, 1)
        return base * vignette[..., None]

    def shade_calibrated(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Look the pixel colour up from a calibrated gradient table."""
        assert self._table is not None
        n_bins = self._table.shape[0]
        rng = self._table_range or 2.0
        fx = np.clip((gx + rng) / (2 * rng), 0, 1) * (n_bins - 1)
        fy = np.clip((gy + rng) / (2 * rng), 0, 1) * (self._table.shape[1] - 1)
        out = np.empty(gx.shape + (3,), dtype=np.float64)
        for c in range(3):
            out[..., c] = ndimage.map_coordinates(
                self._table[..., c], [fx.ravel(), fy.ravel()], order=1, mode="nearest"
            ).reshape(gx.shape)
        return out

    # ------------------------------------------------------------------ #
    def render(
        self,
        sampler,
        height: np.ndarray,
        markers=None,
        displacement: np.ndarray | None = None,
        as_uint8: bool = True,
        use_gpu: bool = True,
    ) -> np.ndarray:
        """Full pipeline: height field (+ optional markers) -> tactile image."""
        if self._table is not None:
            gx, gy = sampler.gradients(height)
            rgb = self.shade_calibrated(gx, gy)
        else:
            rgb = self.shade(sampler.normals(height), height=height)

        if self._background is None and self._table is None and self.optics.reference_background:
            # Fixed appearance correction: do not subtract the *edited* flat
            # response, which would cancel intensity and ambient changes.
            correction = self._reference_corrections.get(height.shape)
            if correction is None:
                flat = self.shade(np.broadcast_to([0.0, 0.0, 1.0], (1, 1, 3)), OpticsConfig())
                correction = self.reference_illumination(height.shape) / np.maximum(flat, 1e-9)
                self._reference_corrections[height.shape] = correction
            rgb *= correction

        if self._background is not None:
            base = self._background
            if base.shape[:2] != rgb.shape[:2]:
                zoom = (rgb.shape[0] / base.shape[0], rgb.shape[1] / base.shape[1], 1)
                base = ndimage.zoom(base, zoom, order=1)
            # contact modulates the real background rather than replacing it
            key = (height.shape, self.cfg.gel.size_x, self.cfg.gel.size_y,
                   json.dumps(asdict(self.optics), sort_keys=True))
            flat = self._flat_cache.get(key) if self._table is None else None
            if flat is None:
                flat = (self.shade_calibrated(np.zeros_like(height), np.zeros_like(height))
                        if self._table is not None else self.shade(np.broadcast_to(
                            np.array([0.0, 0.0, 1.0]), rgb.shape), height=np.zeros_like(height)))
                if self._table is None:
                    self._flat_cache.clear()
                    self._flat_cache[key] = flat
            rgb = np.clip(base + (rgb - flat), 0.0, None)

        if markers is not None and displacement is not None:
            rgb = markers.draw(rgb, sampler, displacement)

        # Use absolute indentation, never the frame peak or display colour range.
        # This empirical attenuation supplies a cue even on constant-slope faces.
        strength = self.optics.depth_shading
        if not np.isfinite(strength) or not 0 <= strength <= 2:
            raise ValueError('Depth shading must be finite and between 0 and 2')
        if strength:
            rgb *= np.exp(-strength * np.maximum(-height, 0) * 1000)[..., None]

        if self.optics.shadow_strength:
            rgb *= self.depth_shadow(height)[..., None]

        if self.optics.depth_relief:
            rgb *= self.relief_gain(height)[..., None]

        rgb *= self.optics.exposure
        if self.optics.noise_sigma > 0:
            rgb = rgb + self._rng.normal(0.0, self.optics.noise_sigma, rgb.shape)

        rgb = np.clip(rgb, 0.0, 1.0)
        if as_uint8:
            return (rgb * 255.0).astype(np.uint8)
        return rgb

    # ------------------------------------------------------------------ #
    def relief_gain(self, height):
        """Empirical centre lift and perimeter shade from metric depth alone.

        A fixed physical smoothing scale measures local slopes without changing
        the input geometry. No per-image peak normalization or virtual light.
        """
        strength = self.optics.depth_relief
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Depth relief must be between 0 and 1')
        depth = np.maximum(-height, 0)
        h, w = depth.shape
        dx, dy = self.cfg.gel.size_x/w, self.cfg.gel.size_y/h
        smooth = ndimage.gaussian_filter(depth, (.00012/dy, .00012/dx), mode='nearest')
        gy, gx = np.gradient(smooth, dy, dx)
        slope = np.hypot(gx, gy)
        # Fade to zero at first contact and retain an unchanged empty pad.
        contact = -np.expm1(-depth/.00006)
        centre = -np.expm1(-depth/.0006)
        rim = slope/(slope+.25)
        return 1 + strength*contact*(.30*centre-.45*rim)

    def depth_shadow(self, height):
        """Approximate directional visibility of a single-valued gel surface.

        A separate virtual light supplies an optional achromatic appearance cue,
        including for fitted RGB responses. This is not LED transport calibration.
        Outside the sampled pad is unknown and does not cast shadows.
        """
        strength = self.optics.shadow_strength
        azimuth, elevation = self.optics.shadow_azimuth, self.optics.shadow_elevation
        if (not np.isfinite([strength, azimuth, elevation]).all()
                or not 0 <= strength <= 1 or not -360 <= azimuth <= 360
                or not 5 <= elevation <= 85):
            raise ValueError('Invalid depth shadow controls')
        h, w = height.shape
        relief = float(np.ptp(height))
        if not strength or relief <= 0:
            return np.ones_like(height)
        dx, dy = self.cfg.gel.size_x/w, self.cfg.gel.size_y/h
        angle = np.deg2rad(azimuth)
        slope = np.tan(np.deg2rad(elevation))
        # Beyond this distance even the highest surface cannot block the ray.
        reach = min(relief/slope, np.hypot(self.cfg.gel.size_x, self.cfg.gel.size_y))
        step = min(dx, dy)
        if reach < step:
            return np.ones_like(height)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        obstruction = np.zeros_like(height)
        for distance in np.linspace(step, reach, min(128, int(np.ceil(reach/step)))):
            sampled = ndimage.map_coordinates(height,
                [yy + distance*np.sin(angle)/dy, xx + distance*np.cos(angle)/dx],
                order=1, mode='constant', cval=-np.inf, prefilter=False)
            obstruction = np.maximum(obstruction, sampled-height-distance*slope)
        # Fixed 0.1 mm transition limits jagged sampled shadow boundaries.
        shadow = np.clip(obstruction/.0001, 0, 1)
        shadow = shadow*shadow*(3-2*shadow)
        return 1-strength*shadow

    @staticmethod
    def depth_colormap(depth: np.ndarray, max_depth: float) -> np.ndarray:
        """A quick false-colour view of indentation depth, for debugging."""
        norm = np.clip(depth / max(max_depth, 1e-9), 0.0, 1.0)
        import cv2
        rgb = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return rgb[:, :, ::-1].copy()
