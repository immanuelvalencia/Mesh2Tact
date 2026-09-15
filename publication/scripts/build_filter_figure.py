from pathlib import Path
import sys,json,hashlib
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mesh2tact.geometric import GeometricSim
from mesh2tact.config import SensorConfig
from mesh2tact.render.gel import GelLighting
from mesh2tact.render.effects import ImageEffects
OUT=ROOT/'publication/figures/filter_comparison';OUT.mkdir(exist_ok=True)
preset_path=ROOT/'configs/sensors/Real GelSight reference.json';preset=json.loads(preset_path.read_text())
s=GeometricSim(SensorConfig.from_dict(preset['sensor']));s.cfg.camera.width=984;s.cfg.camera.height=739
s.set_mesh(trimesh.creation.icosphere(subdivisions=4,radius=.004),'filter_demo:sphere')
s.cut_depth=.0008;s.softness=.00003;s.gel_lighting=GelLighting.from_dict(preset['gel_lighting']);s.effects=ImageEffects(**preset['effects'])
d,rgb=s.render();raw=s.raw_depth.copy();clean=s.clean_rgb.copy();mask=raw>0
ys,xs=np.nonzero(mask);pad=22;y0=max(0,ys.min()-pad);y1=min(739,ys.max()+pad+1);x0=max(0,xs.min()-pad);x1=min(984,xs.max()+pad+1)
crop=(slice(y0,y1),slice(x0,x1))
np.savez_compressed(OUT/'filter_arrays.npz',raw_depth_m=raw,filtered_depth_m=d,before_effects_rgb=clean,after_effects_rgb=rgb,mask=mask)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(2,3,figsize=(3.5,3.0));fig.subplots_adjust(left=.02,right=.98,top=.86,bottom=.12,wspace=.14,hspace=.8)
for a in ax.flat:a.axis('off')
for i,title in enumerate(['Before','After','Absolute change']):ax[0,i].set_title(title,fontsize=7)
fig.text(.02,.96,'(a) Geometric Gaussian smoothing',fontsize=8,weight='bold')
fig.text(.02,.49,'(b) RGB appearance effects',fontsize=8,weight='bold')
ax[0,0].imshow(raw[crop]*1000,vmin=0,vmax=.8,cmap='viridis');ax[0,1].imshow(d[crop]*1000,vmin=0,vmax=.8,cmap='viridis')
change=np.abs(d-raw)*1000;limit=max(float(change[crop].max()),1e-8)
im=ax[0,2].imshow(change[crop],cmap='magma',vmin=0,vmax=limit)
ax[0,2].text(.5,-.12,f'0–{limit:.3f} mm',transform=ax[0,2].transAxes,ha='center',fontsize=6.5)
ax[1,0].imshow(clean[crop]);ax[1,1].imshow(rgb[crop]);err=np.abs(rgb.astype(float)-clean).mean(axis=2)/255;lim=max(float(err[crop].max()),1e-8)
im=ax[1,2].imshow(err[crop],cmap='magma',vmin=0,vmax=lim)
ax[1,2].text(.5,-.12,f'0–{lim:.3f}',transform=ax[1,2].transAxes,ha='center',fontsize=6.5)
fig.text(.02,.045,'Depth: mm (0–0.8)   RGB change: mean |ΔRGB| / 255',fontsize=6.7)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'fig_filters.{ext}',dpi=600,facecolor='white')
plt.close(fig)
(OUT/'provenance.json').write_text(json.dumps({'preset_sha256':hashlib.sha256(preset_path.read_bytes()).hexdigest(),'sphere_radius_mm':4,'subdivisions':4,'cut_depth_mm':.8,'softness_mm':.03,'resolution':[984,739],'crop_xyxy':[int(x0),int(y0),int(x1),int(y1)],'effects':preset['effects'],'comparison':'Row a raw vs smoothed depth; row b same smoothed geometry before vs after complete RGB effects. Errors from exported 8-bit RGB. No colour enhancement of RGB examples.','depth_change_max_mm':float(change.max()),'rgb_change_max':float(err.max())},indent=2))
print(OUT)
