# Methodology revision plan

Status: Agreed structure for the next methodology rewrite. This plan does not replace `04_methods.md` or report completed evaluation experiments.

Target manuscript: Mesh2Tact, IEEE Robotics and Automation Letters.

## Organisation and flow

Describe the computational process rather than the interface layout:

**Mesh preparation → geometric contact construction → tactile image rendering → dataset generation → classification evaluation.**

Use eight subsections. Keep appearance fitting separate because it estimates parameters before dataset generation rather than running for every generated image. Ground the eventual prose and equations in the active geometric implementation.

## A. Framework Overview

Briefly introduce Mesh2Tact and the workflow. Explain that saved sensor configurations determine rendering appearance, while reference-image fitting establishes the GelSight Mini appearance parameters.

Figure: the three-column framework diagram.

## B. Mesh Preparation and Contact Geometry

Explain how a 3D model becomes a contact-depth map:

- Supported application imports: STL, OBJ, PLY, and GLB.
- Unit conversion, scaling, centring, and mesh cleanup.
- Mesh quality controls: subdivision, simplification, and optional mesh smoothing; effects on surface detail and computational cost.
- Three-axis rotation and lateral offsets relative to the sensing area.
- Cut depth relative to the initial geometric contact position.
- Projection onto the sensor grid, depth-map construction, contact-mask extraction, depth smoothing, and surface-normal calculation.

Equations: vertex transformation, pixel-to-sensor coordinates, contact depth, depth smoothing, and surface normals.

Figure: a sphere at different offsets or cut depths, accompanied by depth maps and contact masks. Use an asymmetric object if demonstrating rotation effects, since sphere rotation alone does not change its ideal geometry.

Distinguish mesh smoothing, which modifies the object surface, from depth smoothing, which modifies the reconstructed contact profile.

## C. Visuotactile Sensor Configuration

Introduce GelSight Mini as the reference sensor for the appearance benchmark. Describe the configurable sensing geometry and image resolution.

Cover sensing width and height, output resolution, preview resolution, contact softness, and depth-visualisation range. Explain that the depth-visualisation range changes displayed depth colours, whereas cut depth changes the generated contact.

Describe the two configuration mechanisms:

- Sensor configurations preserve reusable sensing, background, lighting, and image-effect settings.
- Complete workbench presets additionally preserve object settings, pose, and dataset-sampling settings.

Table: principal parameters, units, meanings, and effects. Explain lighting and processing parameters in their respective subsections without repeating the full description here.

Only describe parameters that affect the active geometric pipeline; legacy solver or material fields must not be presented as operative physics.

## D. Gel Background and Optical Rendering

Explain how contact geometry becomes a coloured tactile image.

Start with base gel colour, spatial colour gradients, and vignetting. Explain that these approximate the nonuniform background appearance observed in the real reference images. Distinguish spatial colour gradients from the surface derivatives used to compute normals.

Then describe:

- Light colour, intensity, azimuth, and elevation.
- Directional and edge-distributed lighting.
- Source distance and spatial falloff.
- Diffuse and optional specular response, identifying which terms the fitted preset uses.
- Combination of the contact-dependent response with the gel background.

Equations: spatial background fields, illumination response, and background–contact composition.

Figure: background alone, lighting response, and combined tactile image.

This subsection explains what the model computes. The next explains how its parameters are selected.

## E. Reference-Based Appearance Fitting

Describe the separate fitting procedure reproducibly:

- Reference-image selection and preprocessing.
- Background estimation from multiple real frames.
- Background-parameter fitting.
- Sphere-based contact approximation for illumination fitting.
- Fitted parameters versus fixed parameters.
- Objective functions, bounds, and the resulting saved appearance configuration.

Equations: background and illumination fitting objectives.

Figure: real image, fitted approximation, and error map.

Call this appearance fitting. Distinguish fitting images from independent evaluation images. Explain the sphere approximation and other assumptions faithfully. Define error calculations here; report measured fitting errors in Results. Do not present fitting residuals as independent validation.

## F. Image Processing and Appearance Variation

Describe processing after optical rendering in implemented order:

- Spatial texture.
- Additional vignetting and Gaussian image blur.
- Shot noise, multiplicative noise, and read noise.
- Contrast and gamma adjustment.
- Random-seed control.

Explain the intended effect of each operation without claiming exact reproduction of physical sensor noise.

Figure: before processing, after processing, and a difference map with identical contact geometry and lighting. Keep actual images unenhanced; label any difference-map scaling.

Briefly reference depth smoothing from B and distinguish it from RGB image blur.

## G. Automated Dataset Generation

Explain how single-contact rendering is repeated automatically:

- Sample count and sampling seed.
- Rotation, lateral-offset, and cut-depth ranges.
- Fixed versus randomised controls.
- Optional variation of the image-effect seed.
- Rendering and export for each sampled contact.
- Recorded parameters and metadata for reproducibility.

Describe selectable outputs: processed and clean RGB images, depth visualisations, numerical depth maps, contact masks, configuration metadata, and the processed mesh. Distinguish numerical raw and filtered depth outputs.

Equation: sampling rules for pose, offset, and cut depth.

State that Euler angles are sampled independently; do not claim uniform sampling of 3D orientations or guaranteed surface coverage.

Figure: generated examples across objects, poses, and depths.

## H. ResNet-Based Transfer Evaluation

Present two separate classification tasks with separately trained models:

1. Contact-shape classification.
2. Object classification.

| Experiment | Training domain | Test domain | Purpose |
|---|---|---|---|
| Real2Real | Real | Real | Reference baseline |
| Sim2Real | Synthetic | Real | Evaluate transfer to real tactile images |
| Real2Sim | Real | Synthetic | Evaluate recognition of synthetic images by a real-trained model |

Specify the ResNet variant, initialisation, preprocessing, loss, optimiser, training budget, model-selection procedure, and repeated seeds once fixed. Keep architecture and training comparisons controlled within each task. Real2Sim can evaluate the same real-trained checkpoint used for Real2Real.

Keep complete acquisition sequences and related synthetic variants within the same split. Exclude fitting images from held-out evaluation. Sim2Real and Real2Real should use the same held-out real test set. Define shared class labels across domains and distinguish local contact-shape labels from object labels.

Report accuracy, macro-F1, per-class performance, confusion matrices, and variability across runs. Include a controlled image-processing ablation to assess whether filters improve transfer.

Keep planned experiments in future tense until performed. Do not invent model settings, sample counts, or results. State whether the split tests unseen instances or only held-out contacts of known objects.

## Writing and figure requirements

- Keep the framework overview brief and ensure each subsection leads into the next.
- Explain geometric controls once in B, then their sampling ranges in G.
- Use a parameter table rather than turning the methodology into a control-by-control user manual.
- Use the supplied reference library only unless the user explicitly requests external research; place APA citations beside supported claims.
- Keep implementation, fitted assumptions, planned evaluation, and measured findings distinct.
- Prepare figures for the paper's two-column layout; retain readable labels at their intended printed size.
- Rewrite the Markdown methodology first, then refresh the Word version with native equations and verified figure layout when requested.

## Next step

Pause manuscript rewriting and return to app coding. The specific coding change remains to be supplied by the user.
