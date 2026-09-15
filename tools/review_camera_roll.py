"""Verify the saved config and export equal-geometry position probes."""
from pathlib import Path
import sys, json, os
from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mesh2tact.geometric import GeometricSim
from mesh2tact.calibration import apply_appearance
from mesh2tact.config import SensorConfig
from mesh2tact.render.gel import GelLighting
from mesh2tact.lighting import validate_lighting

OUT = ROOT/'benchmarks/camera_roll_calibration'
latest = json.loads((OUT/'latest_run.json').read_text())
run = Path(latest['run'])
data = json.loads(Path(latest['config']).read_text())
validate_lighting(SensorConfig.from_dict(data['sensor']).optics)
GelLighting.from_dict(data['gel_lighting']).validate()
sim = GeometricSim()
settings = {key: deepcopy(data[key]) for key in ('sensor','gel_lighting','effects')}
settings['softness'] = data['controls']['softness']/1000
apply_appearance(sim, settings)
sim.cfg.camera.width, sim.cfg.camera.height = 328,246
sim.cut_depth = .0005
sim.effects.read_noise = sim.effects.shot_noise = sim.effects.speckle = sim.effects.texture = 0
sheet = Image.new('RGB',(328*3,270*3),'white')
draw = ImageDraw.Draw(sheet)
patch_means=[]
for row,y in enumerate((.25,.5,.75)):
    for col,x in enumerate((.22,.5,.78)):
        sim.offset = ((x-.5)*sim.cfg.gel.size_x, (y-.5)*sim.cfg.gel.size_y)
        depth,rgb = sim.render()
        assert np.isfinite(depth).all() and depth.max() > 0
        sheet.paste(Image.fromarray(rgb),(col*328,row*270+24))
        draw.text((col*328+8,row*270+6),f'Same sphere / same depth | x={x}, y={y}',fill='black')
        patch_means.append(rgb[depth>0].mean(axis=0).tolist())
sheet.save(OUT/'same_contact_positions.jpg',quality=95)
report = json.loads((run/'metrics.json').read_text())
preview=Image.new('RGB',(328*3,270*3),'white')
pd=ImageDraw.Draw(preview)
for row,name in enumerate(('top_left3','center3','low_right3')):
    index=next(i for i,r in enumerate(report['metrics']) if r['name']==name)
    for col,(label,file) in enumerate((('Real','reference.png'),('Before','before.png'),('Calibrated','after.png'))):
        pd.text((col*328+8,row*270+5),f'{name} | {label}',fill='black')
        with Image.open(run/f'reference_{index+1:03d}'/file) as im:preview.paste(im.resize((328,246)),(col*328,row*270+24))
preview.save(OUT/'preview.jpg',quality=95)
train = [r for r in report['metrics'] if not r['validation']]
heldout = [r for r in report['metrics'] if r['validation']]
summary = {}
for name,records in [('training',train),('excluded_ball',heldout)]:
    summary[name] = {key:float(np.mean([r[key] for r in records if r[key] is not None])) for key in
        ('before_mae','after_mae','before_contact_mae','after_contact_mae')}
# Exercise the real GUI config loader and preserve the new mode through controls.
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from mesh2tact.gui.geometric import GeometricWindow, QtWidgets
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
window=GeometricWindow()
try:
    window.sensor_configs.load(Path(latest['config']))
    assert window.sim.gel_lighting.preserve_contact
    window.apply_controls()
    assert window.sim.gel_lighting.preserve_contact
    depth,rgb=window.sim.render()
    assert rgb.dtype == np.uint8
finally:
    window.close()
summary['gui_config_load_verified']=True
summary['same_contact_position_rgb_means']=patch_means
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
text = f'''# GelSight Camera Roll appearance calibration

## Saved outputs

- Config: `{latest['config']}`
- Complete run: `{run}`
- [Stage comparison](stage_comparison.jpg)
- [Same sphere and depth at nine positions](same_contact_positions.jpg)
- [Numerical summary](summary.json)
- [Estimated alignments](alignment.json)

Load **GelSight Camera Roll spatial** in **Settings → Saved sensor configurations**.
The collection in the run directory can be reopened in Calibration.

## Fit and limits

All 24 JPGs were readable. The empty image fits background glow geometry and colour.
The 21 location/depth contacts first fit shared edge lighting, then a smooth
position-conditioned normal-to-RGB response and blur. The final config uses that empirical
spatial response in place of analytic LED shading. Its coefficients are stored inline in JSON.
The two ball images were excluded from fitting. Different numbered presses were not treated
as repeat exposures. Original images were not changed.

The background and signed contact shading are now composed independently, preserving
contact colours beneath strong background glows. The final config saves this mode.
Each reference folder includes the real image, before, background, lighting, blur and after images.
Stage images are deterministic production renders, not edited photographs.
The Before column uses the script's seeded starting appearance, not a previously calibrated user preset.

Average training image MAE: {summary['training']['before_mae']:.5f} before → {summary['training']['after_mae']:.5f} after (RGB range 0–1).
Average excluded-ball image MAE: {summary['excluded_ball']['before_mae']:.5f} → {summary['excluded_ball']['after_mae']:.5f}.
Full-image scores include a large background area: inspect contacts and the per-reference metrics too.

Bed dimensions and sphere radii are assumed. Positions and contact outlines are estimated from
the photos; their inferred depths are not physical measurements. Remaining outline, texture,
housing and lighting errors are visible in the comparisons. This is an appearance calibration,
not force calibration or independent geometric validation. No sensor texture was copied into the renderer.
Behaviour outside the sampled position and normal ranges has not been validated.

The GUI config loader and a nine-position production-render sweep were verified.
'''
(OUT/'README.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2))
