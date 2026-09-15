"""Preview scaling and migration of older independent dimension settings."""
import math


def linked_capture(width, height, size_x, size_y, anchor='width', minimum=64, maximum=2048):
    """Fit a square-pixel capture grid to the physical sensor aspect ratio.

    Physical dimensions never change. Round the dependent pixel dimension;
    clamp both together at capture limits, rather than stretching the image.
    """
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and value > 0 for value in (width, height, size_x, size_y)):
        raise ValueError('Capture and sensor dimensions must be finite and positive')
    ratio = size_x/size_y
    if not minimum/maximum <= ratio <= maximum/minimum:
        raise ValueError(f'Sensor aspect ratio cannot fit {minimum}–{maximum} px per axis. Adjust the sensor width/height.')
    if anchor == 'width':
        low, high = max(minimum, math.ceil(minimum*ratio)), min(maximum, math.floor(maximum*ratio))
        w = min(high, max(low, round(width)))
        return w, min(maximum, max(minimum, round(w/ratio)))
    if anchor == 'height':
        low, high = max(minimum, math.ceil(minimum/ratio)), min(maximum, math.floor(maximum/ratio))
        h = min(high, max(low, round(height)))
        return min(maximum, max(minimum, round(h*ratio))), h
    raise ValueError('Capture anchor must be width or height')


def link_capture_controls(controls):
    """Normalize old presets in memory; do not rewrite their source files."""
    controls = dict(controls)
    for key in ('output_w', 'output_h'):
        value = controls[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 64 <= value <= 2048 or int(value) != value:
            raise ValueError(f'{key} must be an integer between 64 and 2048')
    controls['output_w'], controls['output_h'] = linked_capture(
        controls['output_w'], controls['output_h'], controls['width'], controls['height'])
    return controls

def migrate_preview(controls, full=False):
    controls = dict(controls)
    if "preview_scale" not in controls:
        if full:
            controls["preview_scale"] = 100.
        else:
            controls["preview_scale"] = round(max(10., min(100.,
                100*min(controls["preview_w"]/controls["output_w"],
                        controls["preview_h"]/controls["output_h"]))), 3)
    controls.pop("preview_w", None)
    controls.pop("preview_h", None)
    return controls

def preview_size(controls):
    factor = controls["preview_scale"]/100
    return tuple(max(2, round(controls[name]*factor)) for name in ("output_w", "output_h"))
