"""Sequential collection with one indexed, lowercase label per mesh."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from .gather import gather
from .outputs import model_name


def model_entries(paths):
    entries, seen_paths, labels = [], set(), set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path in seen_paths:
            continue
        if not path.is_file():
            raise ValueError(f'Model is missing: {path}')
        if path.suffix.lower() not in ('.stl', '.obj', '.ply', '.glb'):
            raise ValueError(f'Unsupported mesh format: {path.name}. Export CAD as STL, OBJ, PLY or GLB.')
        label = model_name(str(path)).lower()
        if label in labels:
            raise ValueError(f'Duplicate object label "{label}". Rename one model before adding it.')
        seen_paths.add(path)
        labels.add(label)
        entries.append((str(path), label))
    return entries


def gather_batch(sim, paths, directory, settings, units='mm', scale=1.,
                 cancelled=lambda: False, progress=lambda record: None):
    entries = model_entries(paths)
    if not entries:
        raise ValueError('Add at least one 3D model')
    settings.validate()
    results = []
    status = 'complete'
    for number, (path, label) in enumerate(entries, 1):
        if cancelled():
            status = 'cancelled'
            break
        context = dict(model_number=number, model_count=len(entries), model=path, label=label)
        progress(dict(context, loading=True))
        try:
            snapshot = deepcopy(sim)
            snapshot.load(path, units=units, scale=scale)
            snapshot.set_quality(sim.quality_level, sim.smoothing_iterations)
            snapshot.set_base_rotation(sim.base_rotation)
            result = gather(snapshot, directory, replace(settings, save_layout='object_label',
                            object_label=label, still_capture=False), cancelled,
                            lambda record: progress(dict(record, **context)))
        except Exception as exc:
            result = dict(status='failed', completed=0, error=str(exc))
        results.append(dict(result, **context))
        progress(dict(context, model_result=result))
        if result['status'] != 'complete':
            status = result['status']
            break
    return dict(status=status, results=results, requested_models=len(entries),
                completed_models=sum(r['status'] == 'complete' for r in results),
                completed=sum(r['completed'] for r in results))
