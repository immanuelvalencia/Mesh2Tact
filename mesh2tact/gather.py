"""Reproducible geometric data collection, independent of Qt."""
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from .outputs import SaveOptions, create_run_directory, reserve_labelled_capture

import numpy as np


@dataclass
class GatherSettings:
    count: int = 100
    seed: int = 0
    random_rotation: bool = True
    rotation_min: tuple = (-180., -180., -180.)
    rotation_max: tuple = (180., 180., 180.)
    random_cut: bool = True
    cut_min_mm: float = .1
    cut_max_mm: float = 2.
    random_xy: bool = False
    xy_min_mm: tuple = (-2., -2.)
    xy_max_mm: tuple = (2., 2.)
    random_effect_seed: bool = False
    save_layout: str = "date_time"
    object_label: str = ""
    outputs: SaveOptions = field(default_factory=SaveOptions)

    def __post_init__(self):
        if isinstance(self.outputs, dict):
            self.outputs = SaveOptions(**self.outputs)

    def validate(self):
        self.outputs.validate()
        if not isinstance(self.count, int) or not 1 <= self.count <= 100000:
            raise ValueError("Sample count must be between 1 and 100,000")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
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


def gather(sim, directory, settings, cancelled=lambda: False, progress=lambda record: None):
    """Run on a private simulator snapshot. Keep complete samples on stop/error.

    A sample is only listed after all its files have been written and its
    staging directory renamed. A failed partial directory is retained for
    diagnosis but never included in the manifest.
    """
    settings.validate()
    if sim.max_penetration is not None and settings.random_cut and settings.cut_max_mm > sim.max_penetration*1000+1e-9:
        raise ValueError('Requested indentation range exceeds the maximum indentation depth')
    indexed = settings.save_layout == 'object_label'
    reservations = []
    if indexed:
        out, first_index, reservations = reserve_labelled_capture(directory, settings.object_label, settings.count)
    else:
        out, first_index = create_run_directory(directory, sim.source), 1
    run_name = f"run_{first_index:06d}.json" if indexed else 'run.json'
    manifest_name = f"manifest_{first_index:06d}.jsonl" if indexed else 'manifest.jsonl'
    mesh_name = f"processed_mesh_{first_index:06d}.ply" if indexed else 'processed_mesh.ply'
    state = dict(status="running", completed=0, requested=settings.count,
                 settings=asdict(settings), source=sim.source,
                 sampling="Independent uniform Euler XYZ angles and uniform cut/XY ranges",
                 directory=str(out), first_sample_index=first_index,
                 run_file=run_name, manifest_file=manifest_name)

    def save_state():
        temporary = out / (run_name + '.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(out / run_name)

    try:
        save_state()
    except Exception:
        for reservation in reservations:
            reservation.unlink(missing_ok=True)
        raise
    rng = np.random.default_rng(settings.seed)
    initial_z = sim.object_z
    try:
        if settings.outputs.mesh:
            sim.mesh.export(out / mesh_name)
        with (out / manifest_name).open("x", encoding="utf-8") as manifest:
            for local_index in range(settings.count):
                if cancelled():
                    state["status"] = "cancelled"
                    break
                if settings.random_rotation:
                    sim.rotation = tuple(rng.uniform(settings.rotation_min, settings.rotation_max))
                if settings.random_cut:
                    sim.cut_depth = float(rng.uniform(settings.cut_min_mm, settings.cut_max_mm))/1000
                else:
                    sim.object_z = initial_z
                if settings.random_xy:
                    sim.offset = tuple(rng.uniform(settings.xy_min_mm, settings.xy_max_mm)/1000)
                if settings.random_effect_seed:
                    sim.effects.seed = int(rng.integers(0, 1_000_001))
                index = first_index + local_index
                name = f"sample_{index:06d}"
                staging = out / (name + ".partial")
                sim.export(staging, processed_mesh_ref=mesh_name, outputs=settings.outputs)
                files = {}
                for item in staging.iterdir():
                    target = out / f"{name}_{item.name}"
                    item.rename(target)
                    files[item.name] = target.name
                staging.rmdir()
                record = dict(index=index, sample=name, files=files, directory=str(out),
                              rotation_xyz_deg=list(sim.rotation), offset_xy_m=list(sim.offset),
                              cut_depth_m=sim.cut_depth, effect_seed=sim.effects.seed,
                              object_z_m=sim.object_z, plane_z_m=0.0, max_penetration_m=sim.max_penetration,
                              contact_fraction=float(np.count_nonzero(sim.raw_depth)/sim.raw_depth.size),
                              peak_depth_m=float(sim.raw_depth.max()))
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                state["completed"] = local_index+1
                save_state()
                progress(record)
            else:
                state["status"] = "complete"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
    try:
        save_state()
    finally:
        for reservation in reservations:
            reservation.unlink(missing_ok=True)
    return state
