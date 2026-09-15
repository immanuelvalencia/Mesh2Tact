# Preliminary results and discussion: implementation improvements

Working material for the results section. The values below are measured development results, not a completed journal evaluation. See the [engineering record](../review/implementation_improvements.md) for input settings, source files, and evidence limits.

## Capture performance

Full-resolution export was dominated by analytic edge-light evaluation. In the baseline 2048 × 1575 experiment, edge shading accounted for approximately 19.0 s of the 21.3 s export time. Following block-wise evaluation, the same workload required 6.664 s, with approximately 4.4 s spent in edge shading. Table 1 reports the observed end-to-end rendering/export times.

**Table 1. Preliminary capture timings for a synthetic sphere using GelSightV1, with all output types enabled. Each entry is one profiled local run.**

| Capture grid | Pixels | Baseline (s) | Improved (s) | Speedup | Time reduction |
|---|---:|---:|---:|---:|---:|
| 984 × 757 | 744,888 | 5.546 | 2.250 | 2.465× | 59.43% |
| 2048 × 1575 | 3,225,600 | 21.295 | 6.664 | 3.196× | 68.71% |

Speedup is \(T_{\mathrm{baseline}}/T_{\mathrm{improved}}\), while percentage time reduction is \(100(1-T_{\mathrm{improved}}/T_{\mathrm{baseline}})\). The improvement retained the requested capture grid and 12 illumination samples per edge. Each capture used a fresh simulator, so these values do not measure the additional benefit of reusing a warm flat-field cache across dataset frames.

The results indicate that light evaluation, rather than file writing, was the principal bottleneck in this workload. Smaller working arrays reduced elapsed time without reducing the specified lighting samples. This interpretation is consistent with the profile, but hardware-cache behaviour and memory bandwidth were not directly measured. The experiment does not establish real-time operation, universal speedup, or performance relative to another simulator.

## Earlier calibration optimisation

An earlier development-stage benchmark evaluated shared/component-wise shading and seeded-texture caching before the subsequent capture-blocking work. Lighting fitting time decreased from 5.511456 to 3.159575 s, while noise/texture fitting time decreased from 4.095877 to 2.367838 s (Table 2).

**Table 2. Preliminary synthetic calibration timings. Budgets, input procedure, and seeds were retained between each baseline/improved pair.**

| Fitting workload | Baseline (s) | Improved (s) | Speedup | Time reduction |
|---|---:|---:|---:|---:|
| Lighting | 5.511456 | 3.159575 | 1.744× | 42.67% |
| Noise and texture | 4.095877 | 2.367838 | 1.730× | 42.19% |

The retained JSON records contain identical fitted settings for the paired runs; the lighting error metrics also match exactly. For example, the post-fit full-image MAE for the two synthetic sphere references was approximately 0.0019190 and 0.0019193 on the RGB [0, 1] scale in both implementations. These are synthetic fitting-set residuals, not errors against independent physical sensor observations. The timing reduction therefore supports an implementation-efficiency claim for this benchmark, not improved calibration fidelity.

Tables 1 and 2 represent different experiments and code stages. Their speedup factors must not be multiplied. The calibration timing of the final capture-optimised implementation remains to be measured under the same frozen protocol.

## Numerical and functional verification

Tiled shading matched untiled shading exactly for the seeded numerical test field and edge assignments examined. Cache tests compared cached output with fresh rendering and exercised invalidation after changes to lighting, physical sensor width, and image resolution. Seeded texture-effect tests also matched cached and uncached pixels exactly in the tested cases. These checks support preservation of the tested computations; they do not establish exact equivalence for all possible settings or agreement with real sensor images.

The final recorded broad software run passed 58 tests with two exclusions caused by unavailable legacy artifacts. A subsequent targeted asynchronous-capture test passed with explicit worker and control-state assertions. Single capture now runs outside the UI thread, and built-in presets require a named copy before user edits. No quantitative interface-responsiveness, usability, or configuration-error reduction was measured, so these changes should be reported as implementation features rather than experimental performance outcomes.

## Limitations and next evaluation

The runtime observations contain profiler overhead and lack repeated-run statistics, a complete frozen baseline release, and hardware metadata recorded at measurement time. The capture benchmark uses one sphere and one saved appearance configuration. Accordingly, these numbers should be described as preliminary until repeated unprofiled timing covers representative meshes, contact conditions, resolutions, and effects. Separate cold-capture latency, warm batch throughput, memory consumption, and UI latency measurements are needed.

No new independent real-versus-synthetic image comparison, classification accuracy, or Sim2Real/Real2Real/Real2Sim result was produced in this optimisation work. Real-image calibration and downstream recognition must be evaluated independently, with acquisition-level separation to avoid leakage. The computational changes do not add force, elasticity, shear, or physical gel-deformation modelling to the geometric renderer.
