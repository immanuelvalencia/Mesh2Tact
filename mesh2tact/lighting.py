"""Portable JSON presets for manual tactile lighting (no physics settings)."""
from dataclasses import asdict
from pathlib import Path
import json
import math

from .config import LEDConfig, OpticsConfig


def validate_lighting(optics):
    for key in ('spatial_response_enabled', 'contact_depth_enabled',
                'calibrated_background_enabled', 'calibration_enabled'):
        if not isinstance(getattr(optics, key), bool):
            raise ValueError(key+' must be true or false')
    if optics.calibration_path is not None and not isinstance(optics.calibration_path, str):
        raise ValueError('calibration_path must be a path string or null')
    if optics.contact_depth_response is not None:
        import numpy as np
        values = np.asarray(optics.contact_depth_response, dtype=float)
        if values.shape != (7, 3) or not np.isfinite(values).all():
            raise ValueError('contact_depth_response must be a finite 7 by 3 matrix')
    if optics.boundary_response is not None:
        import numpy as np
        values = np.asarray(optics.boundary_response, dtype=float)
        if values.shape != (35, 3) or not np.isfinite(values).all():
            raise ValueError('boundary_response must be a finite 35 by 3 matrix')
    if optics.calibrated_background is not None:
        import numpy as np
        grid = np.asarray(optics.calibrated_background, dtype=float)
        if (grid.ndim != 3 or grid.shape[2] != 3 or min(grid.shape[:2]) < 2
                or max(grid.shape[:2]) > 256 or not np.isfinite(grid).all()
                or np.any((grid < 0) | (grid > 1))):
            raise ValueError('calibrated_background must be a finite 2..256 by 2..256 RGB grid in [0, 1]')
    if optics.spatial_response is not None:
        import numpy as np
        values = np.asarray(optics.spatial_response, dtype=float)
        if values.shape not in ((30,3), (35,3)) or not np.isfinite(values).all():
            raise ValueError('spatial_response must be a finite 30 by 3 or 35 by 3 matrix')
    def number(value, lo, hi, name):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")

    def color(value, name):
        if len(value) != 3:
            raise ValueError(f"{name} needs three RGB components")
        for channel in value:
            number(channel, 0, 1, name)

    number(optics.contact_depth_scale_mm, .01, 20, 'Contact depth scale')
    number(optics.spatial_slope_damping, 0, 8, 'Spatial slope damping')
    if not isinstance(optics.boundary_enabled, bool):
        raise ValueError('boundary_enabled must be true or false')
    number(optics.boundary_width_mm, .01, 2, 'Boundary width')
    number(optics.boundary_growth, 0, 2, 'Boundary growth')
    number(optics.boundary_gain, 0, 2, 'Boundary strength')

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
    number(optics.depth_shading, 0, 2, "Depth shading")
    number(optics.depth_relief, 0, 1, "Depth relief")
    number(optics.shadow_strength, 0, 1, "Shadow strength")
    number(optics.shadow_azimuth, -360, 360, "Shadow direction")
    number(optics.shadow_elevation, 5, 85, "Shadow elevation")
    if not isinstance(optics.perimeter_shadow_enabled, bool):
        raise ValueError('perimeter_shadow_enabled must be true or false')
    number(optics.perimeter_shadow_opacity, 0, 1, 'Perimeter shadow opacity')
    number(optics.perimeter_shadow_width_mm, .01, 2, 'Perimeter shadow width')
    number(optics.perimeter_shadow_growth, 0, 2, 'Perimeter shadow depth growth')
    number(optics.perimeter_shadow_softness, .2, 2, 'Perimeter shadow softness')
    if not isinstance(optics.depth_overlay_enabled, bool) or not isinstance(optics.depth_overlay_invert, bool):
        raise ValueError('Depth overlay switches must be true or false')
    number(optics.depth_overlay_strength, 0, 2, 'Depth overlay strength')
    number(optics.depth_overlay_range_mm, .01, 20, 'Depth overlay range')
    number(optics.depth_overlay_falloff, .2, 5, 'Depth overlay falloff')
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
    Path(path).write_text(json.dumps({"version": 1, "optics": data}, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_lighting(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("optics"), dict):
        raise ValueError("Not a Mesh2Tact lighting preset (version 1)")
    raw = dict(data["optics"])
    leds = [LEDConfig(**item) for item in raw.pop("leds")]
    return validate_lighting(OpticsConfig(leds=leds, **raw))
