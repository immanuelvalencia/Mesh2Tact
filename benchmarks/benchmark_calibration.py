"""Reproducible synthetic calibration profile; never reads/writes user datasets."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import cProfile
import hashlib
import io
import json
import platform
import pstats
import time

import numpy as np

from mesh2tact.calibration import appearance, fit_references, render_float
from mesh2tact.geometric import GeometricSim


def run(output):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 128, 96
    settings = appearance(sim)
    references = []
    for index, x in enumerate((-.002, .002)):
        sim.offset = (x, 0.)
        sim.cut_depth = .0008
        references.append(dict(name=f'sphere-{index}', shape='sphere',
            vertices=np.asarray(sim.mesh.vertices), faces=np.asarray(sim.mesh.faces),
            rotation=list(sim.rotation), offset=list(sim.offset), cut_depth=sim.cut_depth,
            blank=False, validation=False, locked=True,
            photo=(render_float(sim)*255).astype(np.uint8)))
    initial = deepcopy(settings)
    for led in initial['sensor']['optics']['leds']:
        led['elevation'] += 8
        led['intensity'] *= .7
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = fit_references(initial, references, size=(128, 96), fit_background=False,
                            max_nfev=8, search_starts=1, calibration_seed=42)
    profiler.disable()
    lighting_seconds = time.perf_counter()-start
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(22)
    lighting_profile = stream.getvalue()
    noise_refs = []
    sim.cut_depth = -.001
    sim.effects.read_noise = .02
    sim.effects.texture = .05
    sim.effects.texture_scale_mm = .3
    for index in range(2):
        sim.effects.seed = index+3
        ref = deepcopy(references[index])
        ref.update(blank=True, cut_depth=-.001, repeat_group='empty',
                   photo=(render_float(sim, stochastic=True)*255).astype(np.uint8))
        noise_refs.append(ref)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    noise = fit_references(settings, noise_refs, size=(64, 48), fit_background=False,
                          fit_lighting=False, fit_noise=True, fit_texture=True,
                          max_nfev=12, calibration_seed=42)
    profiler.disable()
    noise_seconds = time.perf_counter()-start
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(22)
    root = Path(__file__).resolve().parents[1]
    report = dict(python=platform.python_version(), platform=platform.platform(),
        source_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                       ('mesh2tact/calibration.py', 'mesh2tact/calibration_noise.py', 'mesh2tact/render/effects.py')},
        lighting=dict(seconds=lighting_seconds, profile=lighting_profile,
                      metrics=result['metrics'], stages=result['stages'], settings=result['settings']),
        noise_texture=dict(seconds=noise_seconds, profile=stream.getvalue(),
                           stages=noise['stages'], settings=noise['settings']))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value['seconds'] for key, value in report.items() if isinstance(value, dict) and 'seconds' in value}))
    print(lighting_profile)
    print(stream.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
