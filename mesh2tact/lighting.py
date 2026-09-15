"""Portable JSON presets for manual tactile lighting (no physics settings)."""
from dataclasses import asdict
from pathlib import Path
import json
import math

from .config import LEDConfig, OpticsConfig


def validate_lighting(optics):
    if optics.spatial_response is not None:
        import numpy as np
        values = np.asarray(optics.spatial_response, dtype=float)
        if values.shape != (30,3) or not np.isfinite(values).all():
            raise ValueError('spatial_response must be a finite 30 by 3 matrix')
    def number(value, lo, hi, name):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")

    def color(value, name):
        if len(value) != 3:
            raise ValueError(f"{name} needs three RGB components")
        for channel in value:
            number(channel, 0, 1, name)

    if not 1 <= len(optics.leds) <= 16:
        raise ValueError("Use between 1 and 16 lights")
    for led in optics.leds:
        color(led.color, "Light color")
        number(led.azimuth, -360, 360, "Azimuth")
        number(led.elevation, 0, 90, "Elevation")
        number(led.intensity, 0, 5, "Intensity")
    color(optics.ambient, "Ambient color")
    number(optics.side_distance, 1, 3, "Side distance")
    number(optics.side_falloff, 0, 2, "Side falloff")
    if not isinstance(optics.side_lighting, bool):
        raise ValueError("side_lighting must be true or false")
    for key, lo, hi in [("exposure", 0, 5), ("diffuse_gain", 0, 3),
                        ("specular_gain", 0, 3), ("shininess", 1, 256), ("noise_sigma", 0, .1)]:
        number(getattr(optics, key), lo, hi, key)
    if not isinstance(optics.reference_background, bool):
        raise ValueError("reference_background must be true or false")
    return optics


def save_lighting(path, optics):
    validate_lighting(optics)
    data = asdict(optics)
    data.pop("calibration_path", None)
    Path(path).write_text(json.dumps({"version": 1, "optics": data}, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_lighting(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("optics"), dict):
        raise ValueError("Not a Mesh2Tact lighting preset (version 1)")
    raw = dict(data["optics"])
    if "calibration_path" in raw:
        raise ValueError("Manual lighting presets cannot contain a calibration lookup table")
    leds = [LEDConfig(**item) for item in raw.pop("leds")]
    return validate_lighting(OpticsConfig(leds=leds, **raw))
