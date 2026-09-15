"""Reproducible multi-position appearance fit. Source photographs are read-only.

Contact outlines are estimated from differences to blank.jpg. Sphere dimensions
and indentation are proxies, not measurements of depth, pressure or force.
"""
from pathlib import Path
import sys
import json
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation, label
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mesh2tact.geometric import GeometricSim
from mesh2tact.calibration import appearance, fit_references, make_sim, render_float, save_session
from mesh2tact.calibration_archive import save_run, sensor_config
from mesh2tact.render.gel import GelGlow
from mesh2tact.config import LEDConfig

DATA = ROOT/'benchmarks/Camera Roll'
OUT = ROOT/'benchmarks/camera_roll_calibration'
SIZE = (164, 123)

def read(path, size=None):
    with Image.open(path) as im:
        im = im.convert('RGB')
        if size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(im).copy()

def outline(photo, blank):
    delta = gaussian_filter((photo.astype(float)-blank.astype(float))/255, (1.2, 1.2, 0))
    delta -= np.median(delta.reshape(-1, 3), axis=0)
    strength = np.linalg.norm(delta, axis=-1)
    strength[:8] = strength[-8:] = 0
    strength[:, :8] = strength[:, -8:] = 0
    regions, count = label(strength > max(.018, strength.max()*.20))
    sizes = np.bincount(regions.ravel()); sizes[0] = 0
    mask = binary_fill_holes(regions == sizes.argmax())
    yy, xx = np.where(mask)
    if len(xx) < 15:
        raise ValueError('Cannot identify a contact outline')
    cx, cy = (xx.min()+xx.max())/2, (yy.min()+yy.max())/2
    radius = ((xx.max()-xx.min())+(yy.max()-yy.min()))/4
    return (cx+.5)/photo.shape[1], (cy+.5)/photo.shape[0], radius/photo.shape[1]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 328, 246
    sim.cfg.gel.size_y = sim.cfg.gel.size_x*2464/3280
    sim.cfg.optics.noise_sigma = 0
    sim.cfg.optics.specular_gain = 0
    sim.cfg.optics.diffuse_gain = 1
    sim.cfg.optics.leds = [LEDConfig(color=c, azimuth=a, intensity=.5) for c,a in
                          [((1,0,0),90), ((0,1,0),180), ((0,0,1),270)]]
    sim.effects.enabled = True
    sim.effects.blur_px = 1
    sim.softness = .000025
    sim.gel_lighting.preserve_contact = True
    # Third field allows the narrow green edge to be fitted separately.
    sim.gel_lighting.glows.append(GelGlow((.1,.5,.3), .25, .03,.5,.04,.8))
    initial = appearance(sim)
    refs, alignments = [], []
    blank = read(DATA/'blank.jpg', (656,493))
    names = ['blank'] + [f'{p}{i}' for p in ('top_left','top_right','center_left','center','center_right','low_left','low_right') for i in (1,2,3)] + ['ball1','ball2']
    for name in names:
        photo = read(DATA/(name+'.jpg'))
        empty = name == 'blank'
        cx, cy, fraction = (.5,.5,.02) if empty else outline(read(DATA/(name+'.jpg'), (656,493)), blank)
        # Shared proxy radius for the large-sphere series; separate small ball.
        radius = .001 if name.startswith('ball') else .0045
        contact = min(fraction*sim.cfg.gel.size_x, radius*.95)
        depth = radius-np.sqrt(radius**2-contact**2)
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        refs.append(dict(name=name, photo=photo, vertices=mesh.vertices, faces=mesh.faces,
            rotation=[0.,0.,0.], offset=[(cx-.5)*sim.cfg.gel.size_x,(cy-.5)*sim.cfg.gel.size_y],
            cut_depth=float(depth), blank=empty, validation=name.startswith('ball'), locked=True,
            shape_name='Sphere (approximate geometry)', repeat_group=''))
        alignments.append(dict(name=name, center_uv=[cx,cy], contact_radius_fraction=fraction,
                               assumed_sphere_radius_m=radius, inferred_cut_depth_m=float(depth)))
    (OUT/'alignment.json').write_text(json.dumps(alignments, indent=2))
    save_session(OUT/'initial_collection.npz', initial, refs)
    print('Fitting empty-pad gradients', flush=True)
    bg = fit_references(initial, refs[:1], size=SIZE, fit_lighting=False, max_nfev=65,
                        progress=lambda m: print(m, flush=True))
    print('Fitting shared position-dependent contact lighting', flush=True)
    result = fit_references(bg['settings'], refs, size=SIZE, fit_background=False,
        fit_directions=True, fit_blur=True, max_nfev=30, search_starts=1,
        progress=lambda m: print(m, flush=True))
    # Preserve the complete baseline and the background stage in the final run.
    result['stages'] = bg['stages'][:-1]+result['stages']
    result['initial'] = []
    for ref, photo, metric in zip(refs, result['photos'], result['metrics']):
        before_sim = make_sim(ref, initial, result['size'])
        before_sim.effects.blur_px *= result['size'][0]/initial['sensor']['camera']['width']
        before = render_float(before_sim)
        result['initial'].append(before)
        before_sim.depth()
        mask = binary_dilation(before_sim.raw_depth > 0, iterations=3)
        metric['before_mae'] = float(np.abs(before-photo).mean())
        metric['before_rmse'] = float(np.sqrt(np.mean((before-photo)**2)))
        metric['before_contact_mae'] = float(np.abs(before-photo)[mask].mean()) if mask.any() else None
    result['options']['fit_background'] = True
    result['metric_note'] = ('Appearance fit with image-estimated contact outlines and assumed sphere radii. '
        'Numbered presses are different depths, not repeat exposures. Ball frames excluded from lighting fit; '
        'their geometry is still image-estimated, so this is not independent geometric validation.')
    run = save_run(OUT/'runs', initial, refs, result)
    preset = sensor_config(result)
    preset['sensor']['name'] = 'GelSight Camera Roll spatial lighting'
    preset['calibration']['geometry_note'] = result['metric_note']
    target = ROOT/'configs/sensors/GelSight Camera Roll spatial.json'
    target.write_text(json.dumps(preset, indent=2)+'\n')
    # Comparison sheet uses the archived production images, not retouched photos.
    columns = [('Real capture','reference.png'),('Before','before.png'),('Background','stage_01_background.png'),
               ('Spatial lighting','stage_02_lighting.png'),('After blur','after.png')]
    chosen = ['blank','top_left3','top_right3','center_left3','center3','center_right3','low_left3','low_right3','ball2']
    w,h = 246,185
    sheet = Image.new('RGB',(len(columns)*w, len(chosen)*(h+22)+30),'white')
    draw = ImageDraw.Draw(sheet)
    for col,(title,_) in enumerate(columns): draw.text((col*w+8,8),title,fill='black')
    for row,name in enumerate(chosen):
        folder = run/f'reference_{names.index(name)+1:03d}'
        for col,(_,file) in enumerate(columns):
            y = 30+row*(h+22)
            draw.text((col*w+8,y+3), name+(' (excluded)' if name.startswith('ball') else ''),fill='black')
            with Image.open(folder/file) as im: sheet.paste(im.resize((w,h)),(col*w,y+22))
    sheet.save(OUT/'stage_comparison.jpg', quality=95)
    (OUT/'latest_run.json').write_text(json.dumps(dict(run=str(run), config=str(target)),indent=2))
    print('SAVED '+str(run), flush=True)
    print(json.dumps(result['metrics'],indent=2),flush=True)
    from fit_spatial_response import main as fit_spatial
    fit_spatial()
    import runpy
    runpy.run_path(str(ROOT/'tools/review_camera_roll.py'), run_name='__main__')

if __name__ == '__main__': main()
