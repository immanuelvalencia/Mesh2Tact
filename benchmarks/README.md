# Calibration performance

## Full-resolution capture

Run `conda run -n vtsim python -m benchmarks.benchmark_capture` from the project
root. This exports a synthetic Ultra-quality sphere with GelSightV1 settings and
all outputs into temporary directories, then removes those benchmark outputs.
The input mesh has 20,480 faces and 0.8 mm cut depth. Timings include cProfile.

| Resolution | Before (s) | After (s) |
|---|---:|---:|
| 984 × 757 | 5.546 | 2.250 |
| 2048 × 1575 | 21.295 | 6.664 |

These single local measurements show a 68.7% reduction in 2K capture time.
Lighting is evaluated in blocks of at most 16,384 pixels using global image
coordinates and the same 12 samples per edge. Tests compare tiled and full-frame
shading exactly. A bounded one-entry flat-light cache benefits successive frames
in a gathering run and invalidates when optics, sensor dimensions, or resolution
change. A fresh single-capture snapshot still computes its first full-resolution
flat field. Single capture now uses the existing gathering worker, including
progress, duplicate-start prevention, and safe stopping after the current frame.

## Calibration

Run from the repository root in the existing Conda environment:

```powershell
conda run -n vtsim python -m benchmarks.benchmark_calibration --output benchmarks/calibration_current.json
```

The synthetic benchmark uses two sphere contacts for lighting and two empty-pad
images for noise/texture. Resolution, seeds, optimisation budgets, and the
12-sample edge-light model were unchanged between runs. Timings include cProfile
overhead and are single local measurements, not a real-sensor or general hardware
performance claim.

| Workload | Before (s) | After (s) | Time reduction |
|---|---:|---:|---:|
| Lighting fitting | 5.511 | 3.160 | 42.7% |
| Noise and texture fitting | 4.096 | 2.368 | 42.2% |

Both runs returned identical fitted settings; the lighting error metrics also
matched exactly. Detailed profiles and results are in `calibration_before.json`
and `calibration_after.json`.

Edge shading now shares coordinate grids across LEDs and performs component-wise
vector arithmetic, avoiding repeated three-channel temporary arrays. Texture
fitting and simulator rendering use a bounded per-workload cache of seeded
texture sources and sampled fields. Cache keys include seed, texture scale,
physical sensor dimensions, and output dimensions. Cached and uncached effects
are tested for identical pixels, including noise. Changing seeds during dataset
generation naturally reduces cache reuse. No fitting iterations were removed.
