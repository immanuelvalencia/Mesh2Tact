"""Manually adjustable, spatial gel illumination inspired by reference photos."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GelGlow:
    color: tuple = (.65, .40, .51)
    strength: float = .35
    x: float = .52
    y: float = .53
    width: float = .18
    height: float = .16
    angle: float = 0.


@dataclass
class GelLighting:
    enabled: bool = True
    preserve_contact: bool = False
    background: tuple = (.19, .28, .29)
    vignette: float = .35
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    hue: float = 0.0
    gamma: float = 1.0
    glows: list = field(default_factory=lambda: [
        GelGlow((.35, .51, .65), .45, .50, .35, .29, .28),
        GelGlow((.68, .40, .52), .40, .51, .54, .19, .17),
    ])

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        glows = [GelGlow(**value) for value in raw.pop("glows")]
        result = cls(glows=glows, **raw)
        result.validate()
        return result

    def validate(self):
        def number(value, lo, hi):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Gel-lighting value must be between {lo} and {hi}")
        def color(rgb):
            if len(rgb) != 3:
                raise ValueError("Gel colors require three RGB values")
            for value in rgb:
                number(value, 0, 1)
        if not isinstance(self.enabled, bool) or not 0 <= len(self.glows) <= 8:
            raise ValueError("Gel lighting needs a boolean enabled flag and 0–8 glows")
        if not isinstance(self.preserve_contact, bool):
            raise ValueError('preserve_contact must be boolean')
        color(self.background)
        number(self.vignette, 0, 1)
        number(self.brightness, 0, 3)
        number(self.contrast, 0, 3)
        number(self.saturation, 0, 3)
        number(self.hue, -180, 180)
        number(self.gamma, .1, 3)
        for glow in self.glows:
            color(glow.color)
            for value in (glow.strength, glow.x, glow.y):
                number(value, 0, 1)
            for value in (glow.width, glow.height):
                number(value, .01, 1)
            number(glow.angle, -180, 180)

    def adjust(self, rgb):
        """Apply global color controls before camera noise and texture."""
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        self.validate()
        out = np.clip(rgb, 0, 1)
        if self.hue or self.saturation != 1:
            hsv = rgb_to_hsv(out)
            hsv[..., 0] = (hsv[..., 0] + self.hue / 360) % 1
            hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 1)
            out = hsv_to_rgb(hsv)
        out = np.clip((out - .5) * self.contrast + .5, 0, 1)
        return np.clip(out ** (1 / self.gamma) * self.brightness, 0, 1)

    def image(self, shape):
        out = np.empty((*shape, 3), dtype=np.float32)
        out[:] = self.background
        return self.overlay(out)

    def overlay(self, rgb):
        """Composite gel color layers over the shaded contact image."""
        self.validate()
        h, w = rgb.shape[:2]
        yy, xx = np.meshgrid((np.arange(h)+.5)/h, (np.arange(w)+.5)/w, indexing="ij")
        out = np.asarray(rgb, dtype=np.float32).copy()
        transmission = np.ones((h,w,1), dtype=np.float32)
        for glow in self.glows:
            angle = np.deg2rad(glow.angle)
            dx, dy = xx-glow.x, yy-glow.y
            u = dx*np.cos(angle) + dy*np.sin(angle)
            v = -dx*np.sin(angle) + dy*np.cos(angle)
            alpha = (glow.strength*np.exp(-.5*((u/glow.width)**2+(v/glow.height)**2)))[..., None]
            out = out*(1-alpha)+np.asarray(glow.color)*alpha
            transmission *= 1-alpha
        radius = ((xx-.5)**2+(yy-.5)**2)*2
        vignette = 1-self.vignette*radius[..., None]
        out *= vignette
        if self.preserve_contact:
            # Calibrated glows describe the empty pad, not an opaque layer over
            # contact shading. Preserve the shader's signed RGB contact response.
            out += (np.asarray(rgb)-np.asarray(self.background))*(1-transmission*vignette)
        return np.clip(out, 0, 1).astype(np.float32)
