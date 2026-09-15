"""Fit an appearance preset to Jim's supplied frames; never alter source data.

Sphere size/depth are approximate: this is an appearance fit, not metric calibration.
"""
from pathlib import Path
import sys
import json
from dataclasses import asdict
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares, lsq_linear
from scipy.ndimage import gaussian_filter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mesh2tact.config import SensorConfig, LEDConfig
from mesh2tact.render.optical import GelSightRenderer
from mesh2tact.render.gel import GelLighting, GelGlow
from mesh2tact.sensor.heightmap import SurfaceSampler

DATA = ROOT.parent / 'Touchlab-VTS/dataset/GelSight'
W, H = 246, 185
def read(path):
    return np.asarray(Image.open(path).convert('RGB').resize((W,H)), dtype=float)/255

def main():
    # Contacts occupy different positions; median rejects those local imprints.
    frames = [read(DATA/f'pyramid/video/sequence_{i:03}/sequence_{i:03}_frame_0000_raw.png') for i in range(220,229)]
    base = np.median(frames, axis=0)
    # Fit only the interior; the real camera's dark housing is not gel illumination.
    mask = np.zeros((H,W), bool); mask[15:-15,15:-15] = True
    def gel(p):
        return GelLighting(background=tuple(p[:3]), vignette=p[3], glows=[
            GelGlow(tuple(p[4:7]), p[7], .50, .26, .40, .32),
            GelGlow(tuple(p[8:11]), p[11], .50, .56, .25, .25)])
    fit = least_squares(lambda p: (gel(p).image((H,W))-base)[mask].ravel(),
                        [.2,.3,.3,.25,.3,.5,.65,.5,.65,.4,.5,.5], bounds=(0,1), max_nfev=80)
    appearance = gel(fit.x)
    cfg = SensorConfig(); cfg.camera.width=W; cfg.camera.height=H
    cfg.gel.size_x=.0186; cfg.gel.size_y=.0143
    cfg.optics.specular_gain=0; cfg.optics.diffuse_gain=1; cfg.optics.noise_sigma=0
    cfg.optics.reference_background=False; cfg.optics.side_falloff=.4
    sampler = SurfaceSampler(cfg, np.array([-cfg.gel.size_x/2,-cfg.gel.size_y/2,0]),
                             np.array([cfg.gel.size_x/2,cfg.gel.size_y/2,0]))
    yy,xx=np.mgrid[:H,:W]
    # Approximate outline of supplied sphere frame 0012 in original 984x739 coordinates.
    cx,cy=480/984*W,350/739*H
    r=175/984*W
    dx=(xx-cx)*cfg.gel.size_x/W; dy=(yy-cy)*cfg.gel.size_y/H
    radius=.0045; contact_radius=r*cfg.gel.size_x/W
    depth=np.maximum(np.sqrt(np.maximum(radius**2-dx*dx-dy*dy,0))-np.sqrt(radius**2-contact_radius**2),0)
    depth=gaussian_filter(depth,.8)
    normals=sampler.normals(-depth); flat=np.broadcast_to([0.,0.,1.], normals.shape)
    columns=[]
    # Red below, green left, blue above reproduce the observed opposing bands
    # for this renderer's image coordinates; these are effective source directions.
    for angle in (90,180,270):
        cfg.optics.leds=[LEDConfig(color=(1,1,1), azimuth=angle, elevation=25)]
        render=GelSightRenderer(cfg)
        columns.append((render.shade(normals,height=-depth)-render.shade(flat,height=np.zeros_like(depth)))[...,0])
    target=read(DATA/'hemisphere/video/sequence_003/sequence_003_frame_0012_raw.png')
    contact=((xx-cx)**2+(yy-cy)**2)<(r*1.12)**2
    # Overlay attenuates contact shading; include that in the fitting basis.
    alpha=appearance.overlay(np.ones((H,W,3)))-appearance.overlay(np.zeros((H,W,3)))
    weights=[]
    for channel in range(3):
        matrix=np.stack([col*alpha[...,channel] for col in columns],axis=-1)
        weights.append(lsq_linear(matrix[contact],(target-base)[...,channel][contact],bounds=(0,3)).x)
    weights=np.asarray(weights).T
    cfg.optics.leds=[LEDConfig(color=tuple(row/max(row.max(),1e-6)), intensity=float(row.max()), azimuth=angle,elevation=25)
                     for row,angle in zip(weights,(90,180,270))]
    preset=json.loads((ROOT/'configs/sensors/Photo inspired gel.json').read_text())
    preset['sensor']['optics']=asdict(cfg.optics)
    preset['gel_lighting']=asdict(appearance)
    preset['effects'].update(shot_noise=.002,speckle=0,texture=.008,blur_px=.5)
    preset['controls']['softness']=.03
    (ROOT/'configs/sensors/Real GelSight reference.json').write_text(json.dumps(preset,indent=2))
    renderer=GelSightRenderer(cfg)
    renderer.set_background(np.broadcast_to(appearance.background,(H,W,3)))
    simulated=appearance.overlay(renderer.render(sampler,-depth,as_uint8=False))
    artifact=ROOT/'publication/figures/reference_lighting'; artifact.mkdir(parents=True,exist_ok=True)
    empty=appearance.image((H,W))
    metrics={'background_interior_mae':float(np.abs(empty-base)[mask].mean()),
             'sphere_roi_pixels':int(contact.sum()),
             'sphere_roi_mae':float(np.abs(simulated-target)[contact].mean()),
             'background_only_roi_mae':float(np.abs(empty-target)[contact].mean()),
             'sphere_roi_rmse':float(np.sqrt(np.mean((simulated-target)[contact]**2))),
             'assumed_peak_depth_mm':float(depth.max()*1000),
             'effective_weights':weights.tolist(), 'gel_parameters':asdict(appearance)}
    (artifact/'metrics.json').write_text(json.dumps(metrics,indent=2))
    canvas=Image.new('RGB',(W*3,H+26),'white'); draw=ImageDraw.Draw(canvas)
    for i,(label,rgb) in enumerate([('Real sphere',target),('Approximate simulation',simulated),('Fitted empty gel',appearance.image((H,W)))]):
        canvas.paste(Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')),(i*W,26)); draw.text((i*W+5,5),label,fill='black')
    canvas.save(artifact/'comparison.png')
    sweep=Image.new('RGB',(W*3,H*2+52),'white'); labels=ImageDraw.Draw(sweep)
    for row,kind in enumerate(('Sphere','Horizontal edge')):
        for col,mm in enumerate((.1,.4,.8)):
            if kind=='Sphere':
                shape_depth=np.maximum(np.sqrt(np.maximum(radius**2-dx*dx-dy*dy,0))-(radius-mm/1000),0)
            else:
                shape_depth=np.maximum(mm/1000-np.abs(dy)*.8,0)*(np.abs(dx)<.006)
            shape_depth=gaussian_filter(shape_depth,.8)
            rgb=appearance.overlay(renderer.render(sampler,-shape_depth,as_uint8=False))
            x0,y0=col*W,row*(H+26)
            labels.text((x0+5,y0+5),f'{kind}: {mm} mm (assumed)',fill='black')
            sweep.paste(Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')),(x0,y0+26))
    sweep.save(artifact/'depth_response.png')
    print(json.dumps({'weights':weights.tolist(),'interior_background_mae':float(np.abs(appearance.image((H,W))-base)[mask].mean())}))

if __name__=='__main__': main()
