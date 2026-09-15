"""Single-column methodology figures from production rendering and archived fit.

Run: conda run -n vtsim python publication/scripts/build_methodology_revision.py
No refitting or changes to source photographs, configurations or datasets.
"""
from pathlib import Path
import sys, json, hashlib
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mesh2tact.geometric import GeometricSim
from mesh2tact.config import SensorConfig
from mesh2tact.render.gel import GelLighting
from mesh2tact.render.effects import ImageEffects

OUT = ROOT/'publication/figures/methodology_revision'
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVE = ROOT/'publication/figures/paper_figures_20260914_002838/generated_dataset/pyramid_square/2026-09-14_00-29-00_437637/sample_000001_settings.json'
PHOTO = ROOT.parent/'Touchlab-VTS/dataset/GelSight/hemisphere/video/sequence_003/sequence_003_frame_0012_raw.png'
saved = json.loads(ARCHIVE.read_text())
W, H = 984, 739  # Retain the historical reference registration; do not stretch it.
radius = .0045
contact_radius = 175/984*.0186
indent = radius-np.sqrt(radius**2-contact_radius**2)
offset = ((480/984-.5)*.0186, (350/739-.5)*.0143)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})

def sim(fitted=True):
    s = GeometricSim(SensorConfig.from_dict(saved['sensor'])) if fitted else GeometricSim()
    s.cfg.camera.width, s.cfg.camera.height = W, H
    s.cfg.gel.size_x, s.cfg.gel.size_y = .0186, .0143
    s.set_mesh(trimesh.creation.icosphere(subdivisions=5, radius=radius), 'figure:sphere_r4.5mm')
    s.offset, s.cut_depth, s.softness = offset, float(indent), .00003
    if fitted:
        s.gel_lighting = GelLighting.from_dict(saved['gel_lighting'])
    s.effects = ImageEffects(enabled=False)
    return s

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=600, facecolor='white')
    plt.close(fig)

def panel(ax, rgb, title, **kwargs):
    ax.imshow(rgb, **kwargs)
    ax.set_title(title, loc='left', fontsize=7, pad=4)
    ax.axis('off')

s = sim()
depth, clean = s.render()
raw = s.raw_depth.copy()
default = sim(False).render()[1]
s.effects = ImageEffects(**saved['image_effects'])
s.effects.seed = 17
filtered = s.render()[1]
real = np.asarray(Image.open(PHOTO).convert('RGB'))
if real.shape != (H,W,3):
    raise ValueError(f'Reference dimensions changed: {real.shape}')

# Each effect starts from identical geometry/lighting; values are deliberate
# illustrations, not claimed calibration estimates or task-optimal settings.
variants = [
    ('texture', .12, 'Texture: 0.12'),
    ('vignette', .6, 'Vignette: 0.60'),
    ('blur_px', 5., 'RGB blur: 5 px'),
    ('shot_noise', .06, 'Shot noise: 0.06'),
    ('speckle', .12, 'Speckle: 0.12'),
    ('read_noise', .04, 'Read noise: 0.04'),
    ('contrast', 1.5, 'Contrast: 1.50'),
    ('gamma', 1.5, 'Gamma: 1.50'),
]
fig, axes = plt.subplots(4,3,figsize=(3.5,4.75))
fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.96,wspace=.10,hspace=.32)
panel(axes.flat[0],raw*1000,'(a) Raw depth',cmap='viridis',vmin=0,vmax=1.5)
panel(axes.flat[1],depth*1000,'(b) Depth blur: 30 µm',cmap='viridis',vmin=0,vmax=1.5)
panel(axes.flat[2],clean,'(c) RGB baseline')
arrays = {'raw_depth_m':raw, 'smoothed_depth_m':depth, 'clean_rgb':clean,
          'default_rgb':default, 'filtered_rgb':filtered, 'real_rgb':real}
for k,(key,value,title) in enumerate(variants,3):
    s.effects = ImageEffects(seed=17, **{key:value})
    d, rgb = s.render()
    assert np.array_equal(d, depth)
    arrays[key] = rgb
    panel(axes.flat[k],rgb,f'({chr(97+k)}) {title}')
combined = ImageEffects(seed=17)
for key,value,_ in variants:
    setattr(combined,key,value)
s.effects = combined
arrays['combined_rgb'] = s.render()[1]
panel(axes.flat[11],arrays['combined_rgb'],'(l) Combined effects')
fig.text(.025,.009,'Depth scale: 0–1.5 mm. RGB panels share the same display range.',fontsize=6.5)
save(fig,'fig02_individual_filters')

# Compact calibration overview, with validation excluded from fitting.
from process_grid import calibration_grid
fig = calibration_grid()
save(fig,'fig03_calibration_process')

fig, axes = plt.subplots(2,2,figsize=(3.5,3.18))
fig.subplots_adjust(left=.025,right=.975,bottom=.065,top=.95,wspace=.09,hspace=.29)
for ax,rgb,title in zip(axes.flat,[default,clean,filtered,real],
        ['(a) Default appearance','(b) Reference-fitted appearance',
         '(c) Fitted + image effects','(d) Actual GelSight reference']):
    panel(ax,rgb,title)
fig.text(.025,.019,'Same sphere geometry in (a–c); (d) was used for appearance fitting.',fontsize=6.5)
save(fig,'fig04_sphere_stages')
np.savez_compressed(OUT/'figure_arrays.npz',**arrays)
provenance = {
    'column_width_mm':88.9, 'resolution':[W,H], 'sensor_area_mm':[18.6,14.3],
    'sphere_radius_mm':radius*1000,'assumed_contact_radius_mm':contact_radius*1000,
    'indentation_mm':float(indent*1000),'offset_xy_mm':[v*1000 for v in offset],
    'softness_sigma_mm':.03,'sphere_subdivisions':5,'seed':17,
    'default_definition':'GeometricSim() appearance defaults; geometry, depth smoothing and reference grid held fixed across stages',
    'historical_grid':'984x739 retained for reference registration; current aspect-ratio rule gives 984x757 for 18.6x14.3 mm',
    'fit_source':str(ARCHIVE),'fit_source_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
    'reference_source':str(PHOTO),'reference_sha256':hashlib.sha256(PHOTO.read_bytes()).hexdigest(),
    'fit_scope':'Archived preliminary analytic sphere appearance fit reused with production mesh rasterization. No new calibration or held-out validation.',
    'filter_demonstration':{k:v for k,v,_ in variants},
    'filter_texture_scale_mm':.15,
    'stage_c_effects':dict(saved['image_effects'],seed=17),
    'fitted_sensor':saved['sensor'],'fitted_gel_lighting':saved['gel_lighting'],
    'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [ROOT/'mesh2tact/geometric.py',ROOT/'mesh2tact/render/effects.py',ROOT/'mesh2tact/render/gel.py',ROOT/'mesh2tact/render/optical.py']}}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print(f'Created 3 figures (PNG/PDF/SVG), source arrays and provenance in {OUT}')
