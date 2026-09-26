"""Reproducible geometric data collection, independent of Qt."""
from dataclasses import dataclass, asdict, field
import os
import secrets
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from .outputs import SaveOptions, create_run_directory, reserve_labelled_capture, reserve_labelled_still

import numpy as np


@dataclass
class GatherSettings:
    count: int = 100
    seed: int | None = None
    random_rotation: bool = True
    rotation_axes: tuple = (True, True, True)
    rotation_min: tuple = (-180., -180., -180.)
    rotation_max: tuple = (180., 180., 180.)
    random_cut: bool = True
    cut_min_mm: float = .1
    cut_max_mm: float = 2.
    random_xy: bool = False
    xy_min_mm: tuple = (-2., -2.)
    xy_max_mm: tuple = (2., 2.)
    random_effect_seed: bool = False
    random_scale: bool = False
    scale_min: float = 0.8
    scale_max: float = 1.2
    scale_step: float = 0.0  # zero retains continuous stratified sampling
    scale_steps: int | None = None  # None reads legacy continuous/increment settings
    balanced: bool = True
    geometry_backend: str = 'cpu'
    save_layout: str = "date_time"
    object_label: str = ""
    still_capture: bool = False
    require_contact: bool = False
    min_contact_pixels: int = 1
    attempts_per_sample: int = 20
    outputs: SaveOptions = field(default_factory=SaveOptions)

    def __post_init__(self):
        if isinstance(self.outputs, dict):
            self.outputs = SaveOptions(**self.outputs)

    def validate(self):
        self.outputs.validate()
        if self.scale_steps is not None and (type(self.scale_steps) is not int or not 1 <= self.scale_steps <= 10):
            raise ValueError('Scale steps must be an integer from 1 to 10')
        if not isinstance(self.random_rotation, bool):
            raise ValueError('Random rotations must be a boolean')
        if isinstance(self.scale_step, bool) or not isinstance(self.scale_step, (int, float)) or not np.isfinite(self.scale_step) or self.scale_step < 0:
            raise ValueError('Scale increment must be finite and nonnegative (0 means continuous)')
        if self.geometry_backend not in ('cpu', 'cuda', 'auto'):
            raise ValueError('Choose CPU, CUDA or Auto geometry processing')
        if not isinstance(self.random_scale, bool) or not isinstance(self.balanced, bool):
            raise ValueError('Scale variation and balanced sampling must be booleans')
        if not np.isfinite([self.scale_min, self.scale_max]).all() or not 0 < self.scale_min <= self.scale_max:
            raise ValueError('Scale bounds must be positive, finite, and ordered')
        if not isinstance(self.require_contact, bool):
            raise ValueError('Require contact must be a boolean')
        for value, label, upper in ((self.min_contact_pixels, 'Minimum contact pixels', 100000000),
                                    (self.attempts_per_sample, 'Attempts per sample', 1000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{label} must be an integer between 1 and {upper}')
        if len(self.rotation_axes) != 3 or any(not isinstance(value, bool) for value in self.rotation_axes):
            raise ValueError('Rotation axes must contain three booleans')
        if not isinstance(self.still_capture, bool) or (self.still_capture and self.count != 1):
            raise ValueError('A still capture must contain exactly one sample')
        if not isinstance(self.count, int) or not 1 <= self.count <= 100000:
            raise ValueError("Sample count must be between 1 and 100,000")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("Seed must be a nonnegative integer")
        if self.random_scale and (self.scale_steps is not None or self.scale_step):
            self.scale_levels()
        if self.save_layout not in ("date_time", "object_label"):
            raise ValueError("Choose a valid capture folder layout")
        if self.save_layout == "object_label" and (not isinstance(self.object_label, str) or not self.object_label.strip()):
            raise ValueError("Enter an object label for indexed capture folders")
        for low, high, length in [(self.rotation_min, self.rotation_max, 3),
                                  (self.xy_min_mm, self.xy_max_mm, 2),
                                  ((self.cut_min_mm,), (self.cut_max_mm,), 1)]:
            if len(low) != length or len(high) != length:
                raise ValueError("Range dimensions are invalid")
            if not np.isfinite([low, high]).all() or np.any(np.array(low) > high):
                raise ValueError("Every random minimum must be finite and at most its maximum")

    def scale_levels(self):
        if self.scale_steps is not None:
            levels = self.scale_steps if self.scale_min != self.scale_max else 1
            if self.count % levels:
                suggested = ((self.count+levels-1)//levels)*levels
                raise ValueError(f'Equal scale counts require a sample count divisible by {levels} '
                                 f'(for example {suggested}, maximum 100000)')
            return (np.array([(self.scale_min+self.scale_max)/2]) if levels == 1
                    else np.linspace(self.scale_min, self.scale_max, levels))
        low, high, step = (Decimal(str(v)) for v in (self.scale_min, self.scale_max, self.scale_step))
        intervals = (high-low)/step
        if intervals != intervals.to_integral_value():
            raise ValueError('Scale increment must divide the min-to-max range exactly; adjust the maximum or increment')
        levels = int(intervals)+1
        if levels > self.count or self.count % levels:
            suggested = ((self.count+levels-1)//levels)*levels
            raise ValueError(f'Equal scale counts require a sample count divisible by {levels} scale values '
                             f'(for example {suggested}, maximum 100000)')
        return np.array([float(low+i*step) for i in range(levels)])


def gather(sim, directory, settings, cancelled=lambda: False, progress=lambda record: None):
    """Run on a private simulator snapshot and save only selected payloads.

    Each sample is staged until all selected outputs have been written, then
    moved into its final flat output folder. No run JSON or manifest is made.
    Partial directories are retained if an output operation fails.
    """
    settings.validate()
    backend = settings.geometry_backend
    if backend in ('auto', 'cuda'):
        from .geometry.gpu_raster import cuda_available
        available = cuda_available()
        if backend == 'cuda' and not available:
            raise ValueError('CUDA is unavailable. Choose CPU or Auto geometry processing.')
        backend = 'cuda' if available else 'cpu'
    sim.geometry_backend = backend
    if sim.max_penetration is not None and settings.random_cut and settings.cut_max_mm > sim.max_penetration*1000+1e-9:
        raise ValueError('Requested indentation range exceeds the maximum indentation depth')
    indexed = settings.save_layout == 'object_label'
    labelled_still = indexed and settings.still_capture
    reservations = []
    # The tactile branch is the canonical location. RGB variants live beside it
    # under their own top-level data folders so datasets can be addressed by
    # image type without copying files.
    root = Path(directory).expanduser().resolve()
    if labelled_still:
        out, still_name, reservations = reserve_labelled_still(root / 'tactile', settings.object_label)
        first_index = 1
    elif indexed:
        out, first_index, reservations = reserve_labelled_capture(
            root / 'tactile', settings.object_label, settings.count,
            other_roots=[root / name for name in ('clean', 'default', 'mask')])
    else:
        out, first_index = create_run_directory(root / "tactile", sim.source), 1
    # Indexed captures have no run subfolder (``.../tactile/<shape>``), while
    # date/time captures do (``.../tactile/<shape>/<run>``).
    shape = out.parent.name if labelled_still else out.name if indexed else out.parent.name
    variant_leaf = Path('stills') if labelled_still else Path() if indexed else Path(out.name)
    variant_directories = {
        "tactile": out,
        "clean": root / "clean" / shape / variant_leaf,
        "default": root / "default" / shape / variant_leaf,
        "mask": root / "mask" / shape / variant_leaf,
    }
    mesh_name = (f'{still_name}_processed_mesh.ply' if labelled_still else
                 f"processed_mesh_{first_index:06d}.ply" if indexed else 'processed_mesh.ply')
    state = dict(status="running", completed=0, requested=settings.count,
                 geometry_backend=backend,
                 settings=asdict(settings), source=sim.source,
                 sampling=("Stratified marginal ranges; retries retain assigned intervals" if settings.balanced
                           else "Independent uniform Euler XYZ angles and uniform cut/XY/scale ranges"),
                 directory=str(out), first_sample_index=first_index,
                 output_directories={name: str(path) for name, path in variant_directories.items()})
    effective_seed = secrets.randbits(31) if settings.seed is None else settings.seed
    state['sampling_seed'] = effective_seed
    state['settings']['seed'] = effective_seed
    rng = np.random.default_rng(effective_seed)
    effect_seeds = None
    if settings.random_effect_seed:
        # Separate stream: enabling appearance variation must not change poses.
        # One unique seed per saved slot; contact retries retain that slot's seed.
        effect_rng = np.random.default_rng(np.random.SeedSequence([effective_seed, 1]))
        initial_effect_seed = sim.effects.seed
        exclude_initial = 0 <= initial_effect_seed <= 1_000_000
        effect_seeds = effect_rng.choice(1_000_000 if exclude_initial else 1_000_001,
                                         size=settings.count, replace=False)
        if exclude_initial:
            effect_seeds += effect_seeds >= initial_effect_seed
    # Independent permutations form a Latin hypercube. Advance only after a
    # successful save, so rejection cannot replace hard intervals with easy ones.
    strata = rng.permuted(np.tile(np.arange(settings.count)[:, None], (1, 7)), axis=0) if settings.balanced else None
    scale_schedule = None
    if settings.random_scale and (settings.scale_steps is not None or settings.scale_step):
        levels = settings.scale_levels()
        scale_rng = np.random.default_rng(np.random.SeedSequence([effective_seed, 2]))
        scale_schedule = scale_rng.permutation(np.tile(levels, settings.count//len(levels)))
        state['scale_levels'] = levels.tolist()
        state['samples_per_scale'] = settings.count//len(levels)

    def sample(low, high, columns):
        low, high = np.asarray(low), np.asarray(high)
        u = rng.random(low.shape)
        if strata is not None:
            u = (strata[state['completed'], columns] + u) / settings.count
        return low + (high - low) * u

    initial_z = sim.object_z
    initial_rotation = tuple(sim.rotation)
    state.update(attempted=0, rejected=0)
    attempt_limit = settings.count * (settings.attempts_per_sample if settings.require_contact else 1)
    if settings.still_capture:
        attempt_limit = 1
    try:
        if settings.outputs.mesh and not settings.random_scale:
            sim.mesh.export(out / mesh_name)
        for attempt in range(attempt_limit):
            if cancelled():
                state["status"] = "cancelled"
                break
            state['attempted'] += 1
            if settings.random_scale:
                sim.set_scale(float(scale_schedule[state['completed']] if scale_schedule is not None
                                    else sample(settings.scale_min, settings.scale_max, 6)))
            if settings.random_rotation:
                sampled = sample(settings.rotation_min, settings.rotation_max, slice(0, 3))
                sim.rotation = tuple(sampled[i] if settings.rotation_axes[i] else initial_rotation[i]
                                     for i in range(3))
            else:
                sim.rotation = initial_rotation
            if settings.random_cut:
                sim.cut_depth = float(sample(settings.cut_min_mm, settings.cut_max_mm, 3))/1000
            else:
                sim.object_z = initial_z
            if settings.random_xy:
                sim.offset = tuple(sample(settings.xy_min_mm, settings.xy_max_mm, slice(4, 6))/1000)
            if settings.random_effect_seed:
                sim.effects.seed = int(effect_seeds[state['completed']])
            if settings.require_contact:
                # Contact is geometric; shading/noise are unnecessary here.
                if hasattr(sim, 'depth'):
                    sim.depth()
                else:
                    sim.render()
                contact_pixels = int(np.count_nonzero(sim.raw_depth > 0))
                if contact_pixels < settings.min_contact_pixels:
                    state['rejected'] += 1
                    progress(dict(rejected=True, completed=state['completed'],
                                  attempted=state['attempted'], skipped=state['rejected'],
                                  contact_pixels=contact_pixels, directory=str(out)))
                    continue
            local_index = state['completed']
            index = first_index + local_index
            name = still_name if labelled_still else f"sample_{index:06d}"
            staging = out / (name + ".partial")
            sample_mesh = f'{name}_processed_mesh.ply' if settings.random_scale else mesh_name
            sim.export(staging, processed_mesh_ref=sample_mesh, outputs=settings.outputs)
            if settings.outputs.settings:
                import json
                metadata_path = staging / 'settings.json'
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                metadata['gather_settings'] = state['settings']
                metadata['sampling_seed'] = effective_seed
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            if settings.outputs.mesh and settings.random_scale:
                sim.mesh.export(staging / 'processed_mesh.ply')
            files = {}
            for item in staging.iterdir():
                variant = ("clean" if item.name == "tactile_clean.png" else
                           "default" if item.name == "tactile_default.png" else
                           "mask" if item.name == "contact.png" else "tactile")
                target_dir = variant_directories[variant]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{name}_{item.name}"
                item.rename(target)
                files[item.name] = os.path.relpath(target, out)
            staging.rmdir()
            record = dict(index=index, sample=name, files=files, directory=str(out),
                          rotation_xyz_deg=list(sim.rotation), offset_xy_m=list(sim.offset),
                          cut_depth_m=sim.cut_depth, effect_seed=sim.effects.seed,
                          scale=getattr(sim, 'scale', 1.0),
                          sampling_seed=effective_seed,
                          object_z_m=sim.object_z, plane_z_m=0.0, max_penetration_m=sim.max_penetration,
                          contact_fraction=float(np.count_nonzero(sim.raw_depth)/sim.raw_depth.size),
                          peak_depth_m=float(sim.raw_depth.max()))
            state["completed"] = local_index+1
            record.update(completed=state['completed'], skipped=state['rejected'])
            progress(record)
            if state['completed'] == settings.count:
                state['status'] = 'complete'
                break
        else:
            state["status"] = "incomplete"
            state['error'] = (f"Contact check: saved {state['completed']} of {settings.count} after "
                              f"{state['attempted']} attempts; skipped {state['rejected']} masks below "
                              f"{settings.min_contact_pixels} contact pixels. Adjust ranges or the threshold. "
                              "Incomplete runs are not guaranteed balanced.")
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
    finally:
        for reservation in reservations:
            reservation.unlink(missing_ok=True)
    return state
