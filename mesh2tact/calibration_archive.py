"""Independent, portable records of completed calibration runs."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import csv
import json
import time

import numpy as np
from PIL import Image


def sensor_config(result):
    from .preview import link_capture_controls
    data = deepcopy(result['settings'])
    c, g = data['sensor']['camera'], data['sensor']['gel']
    controls = link_capture_controls(dict(softness=data['softness']*1000,
        width=g['size_x']*1000, height=g['size_y']*1000,
        depth_range=c['max_depth']*1000, output_w=c['width'], output_h=c['height'], preview_scale=33))
    c['width'], c['height'] = controls['output_w'], controls['output_h']
    return dict(format='mesh2tact-sensor-config', version=1, sensor=data['sensor'],
        gel_lighting=data['gel_lighting'], effects=data['effects'], controls=controls,
        calibration=dict(metrics=result['metrics'], stages=result['stages'],
            resolution=result['size'], options=result.get('options', {}),
            metric_note=result.get('metric_note', 'Deterministic pixel errors')))


def save_run(root, settings, references, result):
    from .calibration import save_session, make_sim, render_float
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = stamp + '-' + uuid4().hex[:8]
    pending = root / (name + '.incomplete')
    pending.mkdir()
    def write_json(name, data):
        (pending/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_json('sensor_config.json', sensor_config(result))
    report = {key: value for key, value in result.items() if key not in ('settings', 'initial', 'final', 'photos')}
    report['created_utc'] = stamp
    report['images'] = []
    for i, ref in enumerate(references):
        folder = pending / f'reference_{i+1:03d}'
        folder.mkdir()
        Image.fromarray(np.asarray(ref['photo'], dtype=np.uint8)).save(folder/'reference_original.png')
        for key, filename in [('photos', 'reference.png'), ('initial', 'before.png'), ('final', 'after.png')]:
            Image.fromarray((np.clip(result[key][i], 0, 1)*255).astype(np.uint8)).save(folder/filename)
        for index, stage in enumerate(result['stages']):
            if 'settings' not in stage:
                continue
            data = stage['settings']
            sim = make_sim(ref, data, result['size'])
            sim.effects.blur_px *= result['size'][0]/data['sensor']['camera']['width']
            rgb = render_float(sim)
            Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(
                folder/f"stage_{index+1:02d}_{stage['stage'].lower()}.png")
        report['images'].append(dict(name=ref['name'], folder=folder.name,
            validation=ref['validation'], blank=ref['blank']))
    write_json('metrics.json', report)
    with (pending/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['metrics'][0]))
        writer.writeheader()
        writer.writerows(result['metrics'])
    save_session(pending/'collection.npz', settings, references, result)
    (pending/'README.txt').write_text(
        'Completed calibration run. Nothing is applied automatically.\n'
        'Load sensor_config.json manually using Settings > Saved sensor configurations > Load config file.\n'
        'Before/after PNGs are deterministic renders at the metric resolution; reference_original.png preserves the input image.\n'
        'metrics.json includes fitting options and production acceptance. A rejected fit retains the original settings and images.\n'
        'collection.npz contains the references, meshes, starting settings and result for reopening in Calibration.\n', encoding='utf-8')
    destination = root/name
    # Windows indexers/antivirus can briefly hold newly written image files.
    # Retry only this verified run directory, never overwrite another run.
    if pending.resolve().parent != root.resolve() or destination.resolve().parent != root.resolve():
        raise ValueError('Calibration run paths must stay inside the run directory')
    for attempt in range(8):
        try:
            pending.rename(destination)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.15*(attempt+1))
    return destination
