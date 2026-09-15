"""Reproducible Mesh2Tact figures using the application's production renderer.

Does not refit or modify presets, source meshes, or real reference images.
Run: conda run -n mesh2tact python publication/scripts/build_paper_figures.py
"""
from pathlib import Path
import sys, json, time, platform, hashlib
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mesh2tact.geometric import GeometricSim
from mesh2tact.config import SensorConfig
from mesh2tact.render.gel import GelLighting
from mesh2tact.render.effects import ImageEffects
from mesh2tact.render.optical import GelSightRenderer
from mesh2tact.sensor.heightmap import SurfaceSampler
from mesh2tact.gather import gather, GatherSettings

OUT = ROOT / 'publication' / 'figures' / ('paper_figures_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True)
PRESET = json.loads((ROOT/'configs/sensors/Real GelSight reference.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':10,
                     'pdf.fonttype':42, 'svg.fonttype':'none', 'figure.facecolor':'white'})
book = PdfPages(OUT/'Mesh2Tact_figure_book.pdf')
captions = []

def save(fig, name, caption):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    book.savefig(fig, bbox_inches='tight')
    captions.append((name, caption))
    plt.close(fig)
    print(name, flush=True)

def panel(ax, data, title, **kw):
    im = ax.imshow(data, origin='upper', **kw)
    ax.set_title(title, loc='left'); ax.set_axis_off()
    return im

def sim_for(shape='pyramid_square', resolution=(328,246)):
    sim = GeometricSim(SensorConfig.from_dict(PRESET['sensor']))
    sim.cfg.camera.width, sim.cfg.camera.height = resolution
    sim.gel_lighting = GelLighting.from_dict(PRESET['gel_lighting'])
    sim.effects = ImageEffects(**PRESET['effects'])
    sim.softness = PRESET['controls']['softness']/1000
    sim.load(ROOT/'assets'/f'{shape}.stl', units='mm')
    # Explicitly standardize asset size for illustrative comparisons.
    factor = .008/max(sim.mesh.extents)
    sim.load(ROOT/'assets'/f'{shape}.stl', units='mm', scale=factor)
    sim.rotation = (155.,20.,15.) if shape=='pyramid_square' else (25.,30.,10.)
    sim.cut_depth = .0008
    return sim

def mesh_panel(ax, sim):
    sim.prepare()
    poly = Poly3DCollection(sim.vertices[sim.mesh.faces]*1000, facecolor='#6fa5b9',
                            edgecolor='#315467', linewidth=.15, alpha=.9)
    ax.add_collection3d(poly)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim): setter(-5,5)
    ax.set_box_aspect((1,1,1)); ax.view_init(25,-55); ax.set_axis_off()

# 1. Actual intermediate arrays, not a conceptual substitute for the renderer.
sim = sim_for(); depth,rgb = sim.render()
fig = plt.figure(figsize=(12,5.6), layout='constrained')
gs = fig.add_gridspec(2,3)
ax = fig.add_subplot(gs[0,0], projection='3d'); mesh_panel(ax,sim); ax.set_title('(a) Input mesh + pose',loc='left')
ax = fig.add_subplot(gs[0,1]); surf = np.where(np.isfinite(sim.surface),(sim.surface-sim.first_contact)*1000,np.nan)
im = panel(ax,surf,'(b) Nearest surface [mm]',cmap='viridis'); fig.colorbar(im,ax=ax,shrink=.7)
ax = fig.add_subplot(gs[0,2]); panel(ax,sim.raw_depth>0,'(c) Geometric contact mask',cmap='gray',vmin=0,vmax=1)
ax = fig.add_subplot(gs[1,0]); im=panel(ax,depth*1000,'(d) Approximate indentation [mm]',cmap='viridis',vmin=0,vmax=.8); fig.colorbar(im,ax=ax,shrink=.7)
ax = fig.add_subplot(gs[1,1]); panel(ax,(sim.sampler.normals(-depth)+1)/2,'(e) Encoded normals: (n + 1)/2')
ax = fig.add_subplot(gs[1,2]); panel(ax,rgb,'(f) Synthetic visuotactile image')
np.savez_compressed(OUT/'pipeline_arrays.npz',surface_m=sim.surface,raw_depth_m=sim.raw_depth,depth_m=depth,normals=sim.sampler.normals(-depth),rgb=rgb)
save(fig,'fig01_generation_pipeline','Methods. Production pipeline for a square-pyramid STL standardized to an 8 mm maximum extent, Euler XYZ pose (155,20,15) degrees and 0.8 mm cut depth. Nearest-surface rasterization gives d=max(z_plane-z_surface,0); Gaussian softness is a visual approximation, not elasticity. Normals encode the lowered surface. Image origin is upper left; rows increase sensor Y. RGB uses the saved real-reference preset. Surface and indentation color scales represent different quantities.')

# 2. Geometric depth sweep, all synthetic.
shapes=['sphere','cube','pyramid_square','cone_trunk']
fig,axes=plt.subplots(4,4,figsize=(11,8),layout='constrained')
for row,shape in enumerate(shapes):
    s=sim_for(shape)
    for col,mm in enumerate((.2,.5,.8,1.1)):
        s.cut_depth=mm/1000; d,r=s.render()
        panel(axes[row,col],r,f'{shape.replace("_"," ")} | {mm:.1f} mm')
save(fig,'fig02_shape_depth_sweep','Methods / qualitative simulator behavior. Four supplied mesh assets, each scaled to 8 mm maximum extent, at four geometric cut depths. Fixed pose per row: pyramid (155,20,15), others (25,30,10) degrees, Euler XYZ. These are synthetic demonstrations, not paired real measurements or force-controlled deformation. Cut depth is measured from the lowest oriented vertex.')

# 3. Call the same gather function as the GUI and retain all source outputs.
s=sim_for(); settings=GatherSettings(count=24,seed=42,random_xy=True,random_effect_seed=True)
state=gather(s,OUT/'generated_dataset',settings)
assert state['status']=='complete' and state['completed']==24, state
run=Path(state['directory']); records=[json.loads(line) for line in (run/'manifest.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(4,6,figsize=(12,6.8),layout='constrained')
for ax,record in zip(axes.flat,records):
    panel(ax,np.asarray(Image.open(run/record['files']['tactile.png'])),f'#{record["index"]:02d} | {record["cut_depth_m"]*1000:.2f} mm')
save(fig,'fig03_automatic_dataset','Methods. All 24 consecutive samples from one square-pyramid mesh using the app gather() function, seed 42. No sample selection or rejection. Independent uniform Euler angles in [-180,180] degrees, cut depth 0.1-2.0 mm, XY offsets -2 to 2 mm and randomized effect seeds. This sampler is not uniform on SO(3) and does not guarantee surface coverage. Full images, depth, masks and metadata are retained under generated_dataset.')
fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
cuts=np.array([r['cut_depth_m']*1000 for r in records]); fractions=np.array([r['contact_fraction']*100 for r in records])
axes[0].scatter(cuts,fractions,color='#237a91'); axes[0].set(xlabel='Sampled cut depth [mm]',ylabel='Geometric contact pixels [%]',title='(a) Dataset contact statistics')
axes[1].plot(np.arange(1,25),fractions,'o-',color='#237a91',markersize=4); axes[1].set(xlabel='Generation order',ylabel='Geometric contact pixels [%]',title='(b) All samples, without filtering')
save(fig,'fig04_dataset_statistics','Preliminary results. Raw contact-mask area for exactly the 24 samples in Figure 3. Contact fraction is the proportion of sensor pixels with positive raw geometric indentation; it is not a measure of object-surface coverage or dataset informativeness. Pose and offset vary, so this is not an isolated depth-response experiment.')

# 5. Paired fit reconstruction from frozen parameters, no reoptimization.
W,H=246,185; data=ROOT.parent/'Touchlab-VTS/dataset/GelSight'
def read(p): return np.asarray(Image.open(p).convert('RGB').resize((W,H)),dtype=float)/255
paths=[data/f'pyramid/video/sequence_{i:03}/sequence_{i:03}_frame_0000_raw.png' for i in range(220,229)]
target_path=data/'hemisphere/video/sequence_003/sequence_003_frame_0012_raw.png'
base=np.median([read(p) for p in paths],axis=0); target=read(target_path)
cfg=SensorConfig.from_dict(PRESET['sensor']); cfg.camera.width=W; cfg.camera.height=H
cfg.gel.size_x=.0186; cfg.gel.size_y=.0143; cfg.optics.noise_sigma=0
appearance=GelLighting.from_dict(PRESET['gel_lighting'])
sampler=SurfaceSampler(cfg,np.array([-.0186/2,-.0143/2,0]),np.array([.0186/2,.0143/2,0]))
yy,xx=np.mgrid[:H,:W]; cx,cy=480/984*W,350/739*H; r=175/984*W
dx=(xx-cx)*.0186/W; dy=(yy-cy)*.0143/H; radius=.0045; cr=r*.0186/W
d=gaussian_filter(np.maximum(np.sqrt(np.maximum(radius**2-dx*dx-dy*dy,0))-np.sqrt(radius**2-cr**2),0),.8)
renderer=GelSightRenderer(cfg); renderer.set_background(np.broadcast_to(appearance.background,(H,W,3)))
approx=appearance.overlay(renderer.render(sampler,-d,as_uint8=False))
roi=(xx-cx)**2+(yy-cy)**2<(r*1.12)**2
error=np.abs(approx-target).mean(axis=-1); empty=appearance.image((H,W))
metrics={'fit_roi_mae':float(error[roi].mean()),'fit_roi_rmse':float(np.sqrt(((approx-target)[roi]**2).mean())),
         'background_only_roi_mae':float(np.abs(empty-target)[roi].mean()),'roi_pixels':int(roi.sum()),'assumed_peak_depth_mm':float(d.max()*1000)}
np.savez_compressed(OUT/'real_comparison_arrays.npz',real=target,approximate=approx,roi=roi,absolute_rgb_error=error,estimated_background=base)
fig,axes=plt.subplots(1,4,figsize=(12,3),layout='constrained')
panel(axes[0],target,'(a) Real: fitting frame'); panel(axes[1],approx,'(b) Approximate: frozen fit')
im=panel(axes[2],error,'(c) Mean absolute RGB error',cmap='magma',vmin=0,vmax=.15)
axes[2].contour(roi,levels=[.5],colors='cyan',linewidths=.6); fig.colorbar(im,ax=axes[2],shrink=.65)
axes[3].bar(['Background\nonly','Fitted\ncontact'],[metrics['background_only_roi_mae'],metrics['fit_roi_mae']],color=['#a4aeb6','#237a91'])
axes[3].set(ylabel='ROI MAE [normalized RGB]',title='(d) In-sample comparison',ylim=(0,.1))
save(fig,'fig05_real_vs_approximate','Preliminary fitting result, NOT held-out validation. Real hemisphere sequence_003 frame_0012 versus the frozen optical fit evaluated on its assumed spherical-cap geometry (radius 4.5 mm; contact outline estimated from the same image). Display and evaluation resolution 246x185. Cyan contour shows the original fitting ROI; errors use RGB in [0,1]. Background-only is a diagnostic control, not another simulator. No post-alignment or refitting performed. This reproduces the analytic fitting path: it bypasses mesh rasterization and app camera effects, unlike Figures 1-4. Real reference files are unchanged.')

# 6. Production-renderer ablation at identical geometry.
s=sim_for(); fig,axes=plt.subplots(1,4,figsize=(12,3),layout='constrained')
for i,label in enumerate(('Full preset','No camera effects','No spatial glows','Directional sources')):
    a=deepcopy(s)
    if i==1: a.effects.enabled=False
    if i==2: a.gel_lighting.glows=[]; a.gel_lighting.vignette=0
    if i==3: a.cfg.optics.side_lighting=False
    _,rgb=a.render(); panel(axes[i],rgb,f'({chr(97+i)}) {label}')
save(fig,'fig06_appearance_ablation','Methods / qualitative ablation. Identical square-pyramid mesh, pose and 0.8 mm cut. Each variant changes only the named component relative to the full preset. No spatial glows removes both Gaussian glows and gel vignette; directional sources disables side-light arrays, retaining fitted light colors/intensities without refitting. This is a sensitivity illustration, not a quantitative comparison against Taxim, TACTO or Gomes.')

# 7. Actual CPU timing, warmup then five repeats, no export I/O.
timings=[]
for w,h in [(246,185),(492,370),(984,739)]:
    s=sim_for('pyramid_square',(w,h)); s.render()
    for mode in ('Cached surface','Recomputed surface'):
        for rep in range(5):
            if mode=='Recomputed surface': s._key=None
            start=time.perf_counter(); s.render(); elapsed=time.perf_counter()-start
            timings.append(dict(width=w,height=h,mode=mode,repeat=rep,seconds=elapsed,triangles=len(s.mesh.faces)))
fig,ax=plt.subplots(figsize=(6.5,3.4),layout='constrained')
for mode,color in [('Cached surface','#237a91'),('Recomputed surface','#bf713e')]:
    groups=[np.array([t['seconds']*1000 for t in timings if t['width']==w and t['mode']==mode]) for w in (246,492,984)]
    means=np.array([g.mean() for g in groups]); std=np.array([g.std(ddof=1) for g in groups])
    ax.errorbar(range(3),means,yerr=std,fmt='o-',label=mode,color=color,capsize=4)
ax.set(xticks=range(3),xticklabels=['246 x 185','492 x 370','984 x 739'],xlabel='Render resolution [pixels]',ylabel='Render time [ms]',title='CPU rendering: mean +/- sample SD (n = 5)'); ax.legend()
save(fig,'fig07_render_runtime','Preliminary computational benchmark on this host. Five timed render() calls per condition after a warmup; full appearance preset and one square-pyramid asset. Recomputed surface explicitly invalidates the rasterization cache. Timings exclude GUI, file I/O and mesh loading; sequential measurements are not independent hardware trials. Error bars are sample standard deviations. Results are not comparisons with competing simulators and must not be generalized to complex meshes.')

book.close()
sources=[ROOT/'mesh2tact/geometric.py',ROOT/'mesh2tact/gather.py',ROOT/'mesh2tact/render/optical.py',ROOT/'configs/sensors/Real GelSight reference.json',*paths,target_path]
provenance=dict(created=datetime.now().isoformat(),python=sys.version,platform=platform.platform(),processor=platform.processor(),
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},preset=PRESET,
                metrics=metrics,timings=timings,dataset=state,render_resolution=[328,246])
(OUT/'provenance_and_metrics.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
text='# Mesh2Tact figure set\n\nGenerated with the production app renderer and gather function; no UI screenshots or invented benchmark results. PNG (300 dpi), vector PDF and SVG versions are provided. Raster image panels retain their native source resolution; 300 dpi does not add sensor detail.\n\n'
for name,caption in captions: text+=f'## {name}\n\n{caption}\n\n'
text+='## Recommended paper selection\n\nUse Figures 1 and 3 for methods; Figure 5 for preliminary fitting results. Figure 2, 4, 6 and 7 can be supplementary until stronger evaluation is available. For final results still collect held-out real-versus-synthetic pairs, competing-simulator benchmarks and real-task recognition results. Do not report confusion matrices, generalization, surface coverage, force accuracy or data efficiency from this figure set.\n\nReproduce: `conda run -n mesh2tact python publication/scripts/build_paper_figures.py`. Each invocation creates a separate output folder. Exact source hashes, frozen settings, fit metrics and timing samples are in provenance_and_metrics.json.\n'
(OUT/'FIGURE_CAPTIONS.md').write_text(text,encoding='utf-8')
print('OUTPUT='+str(OUT),flush=True)
