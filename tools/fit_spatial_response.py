"""Fit a smooth position/normal RGB response to the Camera Roll collection."""
from pathlib import Path
import sys,json
from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,gaussian_filter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mesh2tact.calibration import load_session,make_sim,render_float
from mesh2tact.calibration_archive import save_run,sensor_config
from mesh2tact.render.optical import spatial_features
OUT=ROOT/'benchmarks/camera_roll_calibration'

def main():
    latest=json.loads((OUT/'latest_run.json').read_text())
    initial,refs,previous=load_session(Path(latest['run'])/'collection.npz',include_result=True)
    settings=deepcopy(previous['settings'])
    settings['sensor']['optics']['spatial_response']=None
    size=(328,246)
    photos=[np.asarray(Image.fromarray(r['photo']).resize(size,Image.Resampling.LANCZOS),float)/255 for r in refs]
    matrices=[];targets=[]; sims=[]; masks=[]
    for ref,photo in zip(refs,photos):
        sim=make_sim(ref,settings,size);depth=sim.depth()
        sims.append(sim);masks.append(binary_dilation(sim.raw_depth>0,iterations=3))
        if ref['blank'] or ref['validation']: continue
        features=spatial_features(sim.sampler.normals(-depth))
        mask=masks[-1]
        target=photo-photos[0]
        # Remove only each exposure's global RGB offset, outside the contact.
        target-=np.median(target[~mask],axis=0)
        matrix=features[mask][::max(1,int(mask.sum()/2500))]
        target=target[mask][::max(1,int(mask.sum()/2500))]
        weight=1/np.sqrt(len(matrix))
        matrices.append(matrix*weight);targets.append(target*weight)
    A=np.concatenate(matrices); B=np.concatenate(targets)
    # Ridge regularization limits poorly observed position/normal combinations.
    penalty=np.repeat([.003,.01,.01,.03,.03,.03],5)
    coef=np.linalg.solve(A.T@A+np.diag(penalty),A.T@B)
    settings['sensor']['optics']['spatial_response']=coef.tolist()
    settings['effects']['blur_px']=1.
    settings['gel_lighting']['preserve_contact']=True
    result=deepcopy(previous)
    def renders(data):
        return [render_float(make_sim(r,data,size)) for r in refs]
    before=renders(initial)
    spatial=renders(settings)
    def contact_error(images):
        return float(np.mean([np.abs(im-ph)[mask].mean() for r,im,ph,mask in zip(refs,images,photos,masks)
                             if not r['blank'] and not r['validation']]))
    old=renders(previous['settings'])
    assert contact_error(spatial)<contact_error(old), 'Spatial fit did not improve training contacts'
    # Select blur only with training contacts, never with the excluded balls.
    best=contact_error(spatial)
    for blur in (.5,1.5,2.,3.):
        candidate=deepcopy(settings);candidate['effects']['blur_px']=blur
        images=renders(candidate);cost=contact_error(images)
        if cost<best: settings,spatial,best=candidate,images,cost
    result.update(settings=settings,initial=before,final=spatial,photos=photos,size=size)
    result['stages']=[s for s in previous['stages'] if s['stage'] in ('Background','Lighting')]
    unblurred=deepcopy(settings);unblurred['effects']['blur_px']=0
    result['stages'] += [dict(stage='Spatial',settings=unblurred,accepted=True,
        algorithm='Ridge regression of five normal terms multiplied by six quadratic position terms',
        before_contact_mae=contact_error(old),after_contact_mae=best,regularization=penalty.tolist()),
        dict(stage='Blur',settings=deepcopy(settings),accepted=True),
        dict(stage='Production verification',accepted=True,before_contact_mae=contact_error(before),
             after_contact_mae=best,message='Training contact error improved; excluded balls did not select coefficients or blur.')]
    result['metrics']=[]
    for ref,photo,b,a,mask in zip(refs,photos,before,spatial,masks):
        row=dict(name=ref['name'],validation=ref['validation'])
        for prefix,rgb in [('before',b),('after',a)]:
            row[prefix+'_mae']=float(np.abs(rgb-photo).mean())
            row[prefix+'_rmse']=float(np.sqrt(np.mean((rgb-photo)**2)))
            row[prefix+'_contact_mae']=float(np.abs(rgb-photo)[mask].mean()) if mask.any() else None
        result['metrics'].append(row)
    result['options']['algorithm']='spatial_normal_ridge'
    result['metric_note']+=' Spatial response is an empirical RGB fit, not identified LED geometry; extrapolation beyond sampled positions/normals is unvalidated.'
    run=save_run(OUT/'runs',initial,refs,result)
    config=sensor_config(result);config['sensor']['name']='GelSight Camera Roll spatial lighting'
    config['calibration']['geometry_note']=result['metric_note']
    Path(latest['config']).write_text(json.dumps(config,indent=2)+'\n')
    latest['run']=str(run);(OUT/'latest_run.json').write_text(json.dumps(latest,indent=2))
    columns=[('Real capture','reference.png'),('Before','before.png'),('Background','stage_01_background.png'),
             ('Spatial response','stage_03_spatial.png'),('After blur','after.png')]
    selected=['blank','top_left3','top_right3','center_left3','center3','center_right3','low_left3','low_right3','ball2']
    sheet=Image.new('RGB',(246*5,207*len(selected)+30),'white');draw=ImageDraw.Draw(sheet)
    names=[r['name'] for r in refs]
    for c,(title,_) in enumerate(columns):draw.text((246*c+8,8),title,fill='black')
    for row,name in enumerate(selected):
        for c,(_,file) in enumerate(columns):
            y=30+row*207;draw.text((c*246+8,y+3),name+(' (excluded)' if name.startswith('ball') else ''),fill='black')
            with Image.open(run/f'reference_{names.index(name)+1:03d}'/file) as im:sheet.paste(im.resize((246,185)),(c*246,y+22))
    sheet.save(OUT/'stage_comparison.jpg',quality=95)
    print(json.dumps(result['metrics'],indent=2),flush=True)
    print('SAVED '+str(run),flush=True)

if __name__=='__main__':main()
