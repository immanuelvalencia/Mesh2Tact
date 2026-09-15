# Mesh2Tact — Geometric tactile image app

Project folder: `C:\Users\Jim\Development\mesh2tact`. Python imports and the
installed command use `mesh2tact`. Open this folder in your editor after the
rename. Existing `vtsim-*` preset formats remain readable; newly saved presets
use `mesh2tact-*`. Historical dataset metadata and benchmark records retain
their original provenance paths; generated images and meshes are preserved.

Load an STL, position it against a virtual sensor plane, and render its geometric
imprint as a depth map and a GelSight-style RGB image. This application uses
surface projection and image processing. It does not calculate contact forces,
elasticity, shear, or physical gel deformation.

## Install and launch

```powershell
conda activate torch_gpu
pip install -e .[gui,dev]
python main.py
# Or preload an object (STL coordinates default to millimetres):
python main.py assets/sphere.stl
```

Mesh2Tact is launched from `torch_gpu`. Install CUDA-compatible `torch` and
`torchvision` there using the command appropriate for your GPU and CUDA driver;
the Predict tab uses that same environment and does not invoke a separate Python
environment. New environments created from `environment.yml` use the name
`mesh2tact`; it remains a full historical export rather than the recommended ML
runtime.

`environment.yml` is based on a full export of the installed Windows Conda
environment, including its pip-installed packages; the machine-specific prefix
is omitted. `requirements.txt` is a `pip freeze` package snapshot. Both files
have been updated to install the renamed local Mesh2Tact package.
Some Conda-managed packages appear there as local build URLs, so use the Conda
environment file to recreate this setup rather than installing that pip snapshot
on another machine. These exports include all installed packages, not just the
simulator's minimum dependencies.
`python main.py` launches the geometric app. The installed `mesh2tact` command also launches it.
No Taichi, CUDA runtime, or physics solver is required. The GUI uses Qt and VTK.

## Main window

### Predict

The **Predict** tab uses the current tactile RGB image at the configured capture
resolution. Choose a model folder and it recursively lists every `.pth`
checkpoint, finding the closest `labels.txt`, `classes.txt`, `labels.json`, or
`classes.json` file. Check any number of labelled models and run them
together; the result table shows their top class, confidence, and top three
classes side by side. Supported torchvision checkpoints include ResNet,
DenseNet, EfficientNet, Swin, and ViT variants used by the training runs.

General-object workspace controls (including maximum penetration, capture
resolution, and preview scale) can be adjusted without preset-copy prompts.
Changing these controls does not overwrite the selected sensor config. Updating
or deleting a saved built-in still requires a new named copy. Movement buttons
use red for X, green for Y, and blue for Z; view fitting is teal, loading is blue,
and restore/reset actions are amber.

GelSightV1 is the startup sensor configuration. `Default` is an editable
uncalibrated profile and contains the user's chosen baseline appearance.
GelSightV1 and the legacy `Real GelSight reference`, if present, are read-only;
editing their controls, updating, or deleting prompts for a new named copy.
Cancelling preserves the current values and preset file.

The navigation bar's **Settings** button opens a separate, non-modal settings
window with a larger default size, maximize/minimize controls, and a section
sidebar. Changes update the image previews immediately. Closing Settings hides
it; reopening it keeps your values.

The main sidebar has **General object** and **Data gathering**. The 3D view sits
beside depth and RGB previews. Initially no object is loaded and gathering is
disabled. **Load STL…** is the first object control; OBJ, PLY and GLB are also
accepted. Units and import scale apply to the next imported mesh.

**Object quality** is directly below Load STL and defaults to **Ultra**:

| Setting | Processing of imported meshes |
| --- | --- |
| Very low | Topology-preserving decimation, target 85% fewer faces |
| Low | Target 60% fewer faces |
| Medium | Target 25% fewer faces |
| Original | Original triangle mesh |
| High | Up to one subdivision pass |
| Ultra | Up to two subdivision passes |
| Ultra high | Up to three subdivision passes |

Quality selection applies automatically. Topology preservation can prevent the
exact reduction target. Refinement is capped at one million faces; input meshes
above that size are retained without further subdivision. Subdivision improves
sampling density but does not recover detail missing from an STL. Optional shape
smoothing rounds the mesh and may change dimensions. Quality and smoothing apply
automatically; Restore original beside these controls removes refinement/smoothing.
Every operation starts from the loaded original and never overwrites its file.

## Object Z and penetration limit

Translation arrows and rotation rings are always enabled for loaded objects.
Drag an arrow or ring; RGB/depth update on release. Numeric XYZ rotation and XY
position controls, object Z and move-step buttons are also available. Dragging the empty
scene controls the camera. Fit 3D view resets framing.

The sensor plane is fixed at Z=0 in both rendering and exported metadata. The
numeric Object Z field and vertical slider translate the object origin in
millimetres: upward lifts the object; downward increases penetration. First
contact is generally not object Z=0, because meshes have their own local origin.

Maximum penetration limits the lowest point of the rotated object below the
sensor plane, including points outside the sensor crop. This limit is independent
of the depth colour scale. The slider's lower bound follows this limit; its upper
bound allows clearance of twice the rotated object's height (a 0.1 mm minimum
handles flat meshes). The slider midpoint is zero penetration / first contact,
not necessarily object-origin Z=0. Its lower half spans the entire penetration
limit, giving fine control near contact; its upper half spans the clearance range.
Rotation preserves
object Z unless the new geometry requires lifting it to satisfy the limit. Dragging
and dataset generation obey the same limit. This is a geometric constraint, not
a model of gel force or elasticity. The renderer projects the nearest surface
seen from below and does not wrap hidden undercuts.

Sensor configs store the penetration limit. Older configs initialise it from
their depth display maximum. Complete workspace presets also store preview
selections. Legacy `cut_depth_m` and preset `cut` fields remain supported as
penetration relative to first contact; new exports include `object_z_m`,
`plane_z_m=0`, and `max_penetration_m`.

## Selectable right-hand previews

Use the **Preview panels** checkboxes above the right panel, then click **Apply**.
**Tactile image** is first, followed by geometric depth, clean tactile image before image effects, raw depth,
contact mask, and RGB-encoded surface normals. Enabled panels reflow into a grid
as the panel width changes; a scrollbar accommodates more views. All panels may
be hidden. These options control inspection views, independently of saved outputs.
**Apply** saves selections to `configs/preview_preferences.json` and updates the visible
panels. Choices restore on startup, including hiding every panel. Unapplied checkbox
changes do not affect the visible panels or saved file. Loading a workspace preset can
show its own panels without overwriting your saved defaults; click **Apply** to keep them.

## Settings window

All sensor, lighting, and image-effect controls live here. Use the left sidebar
to show one section at a time; each section scrolls independently:

- Sensor size and edge softness: physical sensor dimensions, Gaussian smoothing
  sigma in mm, and the fixed depth display maximum.
- Gel background and gradients: a protected GelSight preset, user presets,
  a custom color picker, zero to eight soft
  elliptical color layers, strengths, positions, spreads and rotation, and corner
  darkening. The two starting gradients are blue and pink, approximating the
  supplied GelSight reference. Master brightness (0–300%) scales the background,
  all gradients, and contact lighting together before image effects. It is saved
  with sensor configs.
  Contrast, hue shift, saturation, and gamma also adjust the gel illumination
  before image effects. Create preset saves the current complete gel appearance;
  select a user preset, change controls, then Update preset to edit it. Delete
  preset removes it from `configs/gel_backgrounds.json`. GelSight cannot be
  overwritten or deleted. Restore to default restores the built-in gel appearance
  without deleting user presets. All gradients can be removed for a plain background.
- Directional lighting: LED colors, azimuth/elevation, intensity, ambient,
  exposure, diffuse/specular response, and reset to defaults.
  Full-edge LED arrays are enabled by default. Each light spans its selected side
  (Top, Bottom, Left, Right). The renderer integrates 12 emitters along the entire
  edge, with normalized total intensity. Defaults are red top, green left, blue right.
  Older azimuths select the nearest edge (0° right, 90° bottom, 180° left, 270° top).
  Elevation sets its height relative to the center. Edge distance and distance
  falloff control its spatial influence. Reflection uses the deformed surface's
  normals and depth; disable full-edge arrays for uniform directional light.
  The deformed-minus-flat lighting response modifies the base background color;
  gel gradients then overlay the shaded contact image. Gradient sliders retain
  numeric inputs for precise adjustment. This approximation does not include cast shadows, internal gel
  scattering, or refraction. Saved sensor configs include these lighting controls.
- Image effects: Gaussian read noise, photon noise, speckle, fixed gel texture,
  blur, vignette, contrast, gamma, and reproducible seeds.

Directional lighting and image effects each have a preset selector with Create,
Update, and Delete actions. Select a user preset, edit the controls, then Update
to save changes. The built-in Default is protected. Libraries persist in
`configs/directional_presets.json` and `configs/effect_presets.json`; deleting a
preset keeps the current appearance. Effects presets include the enable flag and seed.

Select **Real GelSight reference** in Saved sensor configs to apply the appearance
fit; this is also the desktop startup default when no explicit sensor file is supplied.
The detailed methodology, in-sample results, and limitations are in
`publication/manuscript/GelSight_reference_fitting_manuscript.docx`.
The fit uses the supplied sphere and pyramid images. The Directional lighting
page also has **Apply real GelSight reference lighting** to apply only its optics.
This config uses effective red/bottom, green/left and blue/top array directions,
with fitted channel mixing and reduced distance falloff. These are image-coordinate
parameters, not measurements of the actual LED hardware. Local surface normals
and distance to each array sample continue to change with position and indentation.
`python publication/scripts/fit_reference_lighting.py` reproduces the fit using the supplied
Touchlab-VTS files read-only, writing comparison images under
`publication/figures/reference_lighting`. Sphere geometry was approximated from its outline;
no depth labels were supplied, so this is not a metric or force calibration.

Effects modify RGB only. Texture scale is in mm; blur sigma is in output pixels.
Positions of color gradients are image fractions (0 = left/top). Reference colors
are a visual approximation, not a calibrated model of a real sensor.

The sensor-config selector scans `configs/sensors/*.json` every two seconds.
Choose a file to apply it. **Add current** creates a named config; edit controls
then **Update selected** to save changes. **Delete selected** removes that file
while retaining live settings. Refresh list scans immediately. External edits
apply when the file is selected again. Invalid configs leave the current setup
unchanged. Configs include resolution, sensor dimensions, gradients, lights and
effects; switching them preserves the object, pose, and gathering configuration.

### Calibrate from reference photos

**Calibration** in the main window fits a new sensor preset from manually aligned reference
contacts. Load the object through **General object → Load STL…**, add reference
photos in the **Calibration** tab, then choose **Align reference**. Position the object with the main controls or 3D
handles while comparing the uploaded reference and overlay on the right. **Save
reference** stores the image, shape name, mesh and pose in a collection. Repeat
for each STL, or choose **No shape** for an empty sensor image. **Calibrate all
saved references** fits one shared sensor configuration against the collection. Fitting can include light directions and edge
geometry, RGB light weights, background, blur/vignette/contrast/gamma, read/shot/
speckle noise, and texture strength/size. Repeated unchanged-contact photos can
share a Repeat group for more reliable noise estimation. Validation references
stay outside fitting. Standard/Thorough searches use a reproducible search seed.

Use the large zoomable previews to review the fitted appearance, including
seeded noise/texture. Save a fitted sensor config and select it in Saved sensor
configs to apply it to previews and dataset generation. Calibration sessions
retain photos, meshes, alignments and results. See [the calibration guide](docs/calibration.md)
for algorithms, controls and estimation limits.

## Resolution

General object includes capture dimensions (64–2048 pixels per axis), with
presets through 2048 × 1536, and a **Preview scale** from 10% to 100%. The preview
size is calculated from capture resolution and displayed below the scale.
For example, 50% of 1280 × 960 gives 640 × 480; 100% gives full resolution.
Capture width and height follow the physical sensor width/height ratio. Editing
either pixel dimension recalculates the other; changing sensor size updates the
capture grid and resolution preset labels. The readout shows sensor dimensions,
capture dimensions, and micrometres per pixel. Pixel rounding is unavoidable.
Older presets are adjusted when loaded, without rewriting their source files.
The default is approximately one third (328 × 252 for a 984 × 757 capture
over the 18.6 × 14.3 mm sensor). Both pixel axes stay within 64–2048 pixels;
sensor aspect ratios that cannot fit those limits are rejected.
Changing capture resolution preserves the preview scale and aspect ratio,
subject to rounding to whole pixels. Captures always use capture resolution.

Data-gathering actions use blue for capturing the current frame, green for
starting automatic gathering, and red for stopping. Unavailable actions are grey.
Single capture also runs in the background, so the interface stays responsive
while full-resolution rendering and file saving finish. Stop finishes the current
sample safely. Edge lighting uses small processing blocks without reducing capture
resolution or light samples; gathering reuses unchanged flat-background lighting.
Sensor configs save the scale; older width/height presets are converted to the
largest fitting scale between 10% and 100% when loaded. Physical sensor dimensions
still define the crop.

## Data gathering

Capture current or start an automatic batch. All output goes to:

```text
data/<model-name>/<date-time>/
    run.json
    manifest.jsonl
    processed_mesh.ply                 # when selected, shared by all samples
    sample_000001_tactile.png
    sample_000001_tactile_clean.png
    sample_000001_depth.png
    sample_000001_depth_m.npy
    sample_000001_raw_depth_m.npy
    sample_000001_contact.png
    sample_000001_settings.json
    ...
```

The **Save layout** selector in Data gathering also offers **Object label with
index**. Enter a label such as `sphere_red_20mm` to save this instead:

```text
data/sphere_red_20mm/sample_000001_tactile.png
data/sphere_red_20mm/sample_000001_settings.json
data/sphere_red_20mm/sample_000002_tactile.png
data/sphere_red_20mm/manifest_000001.jsonl
data/sphere_red_20mm/run_000001.json
```

There is no date-time or numbered run folder in this mode. Before each capture
the program finds the highest saved sample index and continues from the next
number. Sample indices are reserved exclusively before saving, so an existing
capture is never replaced, even if another capture begins at the same time.
Labels are made filesystem-safe automatically.

Checkboxes choose the payloads to save. At least one must be selected. Run status
and the manifest are always written. Depth arrays are floating-point metres.
The processed mesh is in metres before pose transformation. The manifest records
each completed sample's filenames, pose, depth, seed, and contact fraction.

For auto gathering choose sample count and random XYZ rotation, cut-depth and
optional XY ranges. Angles are independently uniform in the specified ranges,
not uniformly distributed over all orientations. Unchecked random options keep
the current value. A separate option varies image-effect seeds. All samples,
including empty contacts, are retained.

The worker freezes a copy of the object and settings at Start. Later edits do
not alter that batch. Stop finishes the current sample and preserves completed
files; incomplete writes are excluded from the manifest and kept in a `.partial`
directory for diagnosis. Closing the app during a run requests a stop and waits
for the current write. Existing datasets are never overwritten.

## Headless API and tests

```python
from mesh2tact.geometric import GeometricSim
from mesh2tact.gather import GatherSettings, gather

sim = GeometricSim()
sim.load("assets/sphere.stl", units="mm")
sim.set_quality(2)  # Ultra; -3..3 correspond to the quality levels above
sim.cut_depth = 0.001
depth, rgb = sim.render()
result = gather(sim, "data", GatherSettings(count=10, seed=42))
```

`gather` mutates the supplied simulator; pass a private copy when needed.

```powershell
conda run -n mesh2tact python -m pytest tests -q
```

## Project layout and physics archive

- `mesh2tact/`: geometric projection, optical rendering, Qt GUI and collection code.
- `configs/sensors/`: editable sensor configs.
- `assets/`: STL inputs. Existing datasets/captures are preserved separately.
- `tests/`: geometric app tests.
- `physics_simulator/`: standalone physics application and its own dependencies,
  entry point, configs, assets, examples and tests. Move this entire folder to
  another location and launch its `main.py` there. It does not depend on this
  geometric project. `separation_manifest.json` records verified copied files;
  `original_sources/` retains the physics-only originals from separation.

The geometric app does not import the physics archive. Legacy config dataclasses
remain readable for compatibility, but no physics engine runs in this app.
