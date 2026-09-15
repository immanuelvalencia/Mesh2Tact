# Mesh2Tact publication

Working title: **Mesh2Tact: A Mesh-to-Tactile Framework for Automated Synthetic Dataset Generation for Visuotactile Sensing**

Target journal: IEEE Robotics and Automation Letters (RA-L).

## Implementation evidence for methodology and results

- [Problems, improvements, evidence, and remaining validation](review/implementation_improvements.md)
- [Methodology insertion notes](manuscript/04_methods_implementation_notes.md)
- [Preliminary results and discussion](manuscript/05_results_implementation_notes.md)
- [Recorded timing table for later figures](review/implementation_timings.csv)

These development records distinguish implemented features, single-run timing
observations, software tests, and experiments still required for publication.

## Organization

| Folder | Contents |
| --- | --- |
| `manuscript/` | Editable DOCX and PDF of the preliminary lighting-fitting manuscript |
| `figures/paper_figures_20260914_002838/` | Seven paper figures in PNG, PDF and SVG; combined figure book; captions; numerical arrays and provenance |
| `figures/paper_figures_20260914_002838/generated_dataset/` | The 24 exported samples used in the dataset figures, with metadata, depth and contact masks |
| `figures/reference_lighting/` | Original fitting comparisons and metrics used by the manuscript builder |
| `scripts/` | Reproducible fitting, figure-generation and manuscript-generation scripts |
| `references/` | 36 annotated published references, BibTeX, curated JSON and source metadata |
| `review/manuscript_render/` | Existing manuscript page renders and visual-review files |

Start with [the figure book](figures/paper_figures_20260914_002838/Mesh2Tact_figure_book.pdf) and [figure captions](figures/paper_figures_20260914_002838/FIGURE_CAPTIONS.md).

For writing, use the [annotated reference library](references/Mesh2Tact_related_references.md) and [BibTeX bibliography](references/Mesh2Tact_references.bib).

## Generation

Run from the repository root:

```powershell
conda run -n mesh2tact python publication/scripts/build_paper_figures.py
```

Each run creates a new timestamped directory inside `publication/figures/`.

To deliberately refit the lighting:

```powershell
conda run -n mesh2tact python publication/scripts/fit_reference_lighting.py
```

The fitting command updates the app's `configs/sensors/Real GelSight reference.json` and the reference figures/metrics. It requires the original sibling `Touchlab-VTS/dataset/GelSight` dataset.

`publication/scripts/build_lighting_manuscript.py` creates the DOCX in `publication/manuscript/` using a Python runtime with python-docx. PDF conversion and page rendering are separate steps; store review renders in `publication/review/`.

## Evidence and relocation notes

The real-versus-approximate comparison is an in-sample appearance fit using assumed sphere geometry, not held-out validation. Runtime results are local preliminary measurements. The synthetic gallery is not proof of surface coverage or recognition accuracy.

Existing files were moved here on 2026-09-14 without rerunning experiments or changing figure pixels. Absolute dataset-directory references in the JSON manifests were updated to their new location. Original source hashes and measured results were retained. The existing manuscript is an archived draft and may mention former script/output paths; the builder now uses the paths above.

App presets remain under `configs/`, meshes under `assets/`, and real source datasets in their original location.
