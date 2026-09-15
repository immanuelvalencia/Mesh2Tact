<div align="center">

# Mesh2Tact

### A Mesh-to-Tactile Generation of Synthetic Datasets for Visuo-Tactile Sensing

Generate synthetic tactile images, geometric depth maps, and contact masks from 3D meshes.

**[Quick start](#quick-start) · [Usage](#usage) · [Dataset outputs](#dataset-outputs) · [Paper and authors](#paper-and-authors)**

</div>

---

## Overview

Mesh2Tact is a Python desktop application and programmatic generator for creating synthetic visuo-tactile datasets from 3D meshes. It projects mesh geometry onto a virtual sensor plane, derives a contact depth map, and renders a GelSight-style RGB image using configurable lighting, background appearance, and image effects.

The desktop interface supports mesh positioning, sensor configuration, reference-image fitting, and automated data collection. A Python API provides rendering and dataset generation without opening the interface.

### Features

- **Mesh-based generation:** load STL, OBJ, or PLY geometry and adjust its scale, position, orientation, and penetration.
- **Interactive inspection:** view the object alongside tactile RGB, geometric depth, raw depth, contact masks, and encoded surface normals.
- **Configurable sensor appearance:** control sensor dimensions, background colors and gradients, directional lighting, blur, vignette, noise, and texture.
- **Reference-image fitting:** align real contact photographs and fit a shared sensor configuration across a reference collection.
- **Automated collection:** sample rotation, penetration, position, and image-effect seeds within configurable ranges.
- **Traceable exports:** save selected image and array outputs with per-sample settings, run metadata, and a manifest.
- **Optional classification:** compare predictions from supported, locally supplied torchvision checkpoints.

## Paper and authors

**Paper title**

*Mesh2Tact: A Mesh-to-Tactile Generation of Synthetic Datasets for Visuo-Tactile Sensing*

**Project and paper authors, in author order**

Immanuel Jose Valencia, Ryan Rhay Vicerra, Aaron Raymond See, Renann Baldovino, Robert Kerwin Billones, Elmer Dadios, Argel Bandala, and Raouf Naguib

Manuscript source is maintained in [`publication/manuscript/`](publication/manuscript/), with separate Markdown files for the available sections and a consolidated reference list. Publication venue, year, and DOI will be added when confirmed.

## Quick start

### Requirements

- Python **3.10 or newer**; the example below uses Python 3.11.
- Conda for environment management.
- A desktop environment for the Qt/VTK interface.

The geometric renderer does not require CUDA or a physics engine. The optional **Predict** feature requires PyTorch, torchvision, and compatible model checkpoints.

### Installation

```powershell
git clone https://github.com/immanuelvalencia/Mesh2Tact.git
cd Mesh2Tact

conda create -n mesh2tact python=3.11 -y
conda activate mesh2tact
python -m pip install -e ".[gui,dev]"
```

If you already use the project's `torch_gpu` Conda environment, activate it and run the editable installation there instead.

Dependencies are declared in [`pyproject.toml`](pyproject.toml). The included `environment.yml` and `requirements.txt` are historical environment snapshots; the installation above uses the package's declared dependencies and avoids machine-specific paths in those snapshots.

### Launch

```powershell
python main.py
```

Open an example mesh on launch:

```powershell
python main.py assets/sphere.stl
```

The installed command provides the same entry point:

```powershell
mesh2tact
mesh2tact --list-sensors
```

## Usage

### 1. Load and position a mesh

Choose **Load STL…** in **General object**, then select a mesh. Check its import units and scale; STL inputs default to millimetres. Use the translation handles, rotation rings, or numeric controls to position it over the sensor.

The sensor plane is fixed at **Z = 0**. Object Z describes the mesh origin, so first contact depends on the mesh geometry and orientation. **Maximum penetration** bounds how far the object's lowest point can extend below the sensor plane.

Mesh refinement increases sampling density; it does not recover missing geometric detail. Optional smoothing changes the mesh shape.

### 2. Configure the sensor

Select a saved configuration from [`configs/sensors/`](configs/sensors/), or open **Settings** to adjust:

| Control group | Configuration |
| --- | --- |
| Sensor geometry | Physical dimensions, edge softness, and depth display range |
| Background | Base color, gradients, brightness, and color response |
| Lighting | LED colors, edge positions, intensity, and diffuse/specular response |
| Image effects | Noise, texture, blur, vignette, contrast, gamma, and seeds |

Capture resolution controls saved output dimensions. **Preview scale** changes interactive preview resolution. Image effects modify RGB outputs without changing geometric depth or contact masks.

Save a named sensor configuration to reuse the same appearance across captures and collection runs.

### 3. Fit reference images, if available

In **Calibration**, add reference photographs, align each mesh and pose to its contact image, and save the references into a collection. **Calibrate all saved references** fits a shared sensor configuration. References marked for validation stay outside fitting.

See the [calibration guide](docs/calibration.md) for the workflow, fitting controls, and estimation limits. Reference photographs and calibration session archives are local inputs and are not bundled with this repository.

### 4. Capture a dataset

Use **Data gathering** to capture the current frame or collect an automatic batch. Choose the output payloads, sample count, parameter ranges, and random seed before starting.

Automatic collection can vary XYZ rotation, penetration, XY position, and image-effect seeds. Rotation angles are sampled independently within their ranges; this does not produce a uniform distribution over all 3D orientations. Empty contacts are retained.

A batch uses a snapshot of the object and settings taken at its start. Stopping a run finishes the current sample and preserves completed captures.

### Optional: classify a tactile image

The **Predict** tab evaluates the current tactile RGB image using supported local `.pth` checkpoints. Select a model folder containing checkpoints and corresponding label files (`labels.txt`, `classes.txt`, `labels.json`, or `classes.json`). Supported model families include ResNet, DenseNet, EfficientNet, Swin, and ViT.

PyTorch and torchvision must be installed in the environment running Mesh2Tact. Model weights and training datasets are not included.

## Dataset outputs

The default layout groups captures by object and run time:

```text
data/<model-name>/<date-time>/
├── run.json
├── manifest.jsonl
├── processed_mesh.ply
├── sample_000001_tactile.png
├── sample_000001_tactile_clean.png
├── sample_000001_depth.png
├── sample_000001_depth_m.npy
├── sample_000001_raw_depth_m.npy
├── sample_000001_contact.png
└── sample_000001_settings.json
```

Payload files depend on the selected save options. Run metadata and the manifest are always written.

| Output | Contents |
| --- | --- |
| Tactile RGB | Rendered sensor appearance with selected image effects |
| Clean tactile RGB | Rendered image before image effects |
| Depth PNG | Depth visualization |
| Depth arrays | Floating-point geometric depth in **metres** |
| Contact mask | Projected contact region |
| Processed mesh | Mesh in metres, before the sample pose transformation |
| Settings and manifest | Sample configuration, filenames, pose, penetration, seed, and contact fraction |

The **Object label with index** layout saves sequentially numbered samples directly under `data/<object-label>/`, with indexed run and manifest files. Sample numbering continues from existing captures.

Generated datasets, trained weights, reference-library PDFs, and rendered publication outputs are excluded from Git. Some publication scripts require these separately maintained local inputs.

## Python API

Render a single contact and generate a small dataset without opening the GUI:

```python
from mesh2tact.geometric import GeometricSim
from mesh2tact.gather import GatherSettings, gather

sim = GeometricSim()
sim.load("assets/sphere.stl", units="mm")
sim.cut_depth = 0.001  # 1 mm penetration relative to first contact

depth, rgb = sim.render()

settings = GatherSettings(count=10, seed=42)
result = gather(sim, "data", settings)
```

`GatherSettings` enables random rotation and penetration by default. Set `random_rotation=False` and `random_cut=False` to preserve the current pose and penetration. The `gather` function updates the supplied simulator during collection; use a separate instance when its state must be preserved.

## Repository structure

```text
Mesh2Tact/
├── mesh2tact/             # Geometry, rendering, calibration, collection, and GUI
├── assets/               # Example meshes
├── configs/              # Sensor and geometric configurations
├── benchmarks/           # Benchmark scripts and selected records
├── docs/                 # Detailed documentation
├── publication/
│   ├── manuscript/       # Markdown manuscript sections and references
│   └── scripts/          # Manuscript and figure preparation tools
├── tests/                # Automated tests
├── tools/                # Reference-image calibration utilities
├── main.py               # Desktop entry point
└── pyproject.toml        # Package metadata and dependencies
```

## Development

Activate the environment used for installation, then run:

```powershell
python -m pytest tests -q
```

The repository uses `main` for the shared baseline and `develop` for ongoing development. Create feature or fix branches from `develop` and describe relevant validation when proposing changes.

## Scope and limitations

Mesh2Tact uses geometric projection and image-based appearance rendering. It does not solve contact forces, elasticity, shear, or physical gel deformation. The depth representation follows the surface visible from below the sensor plane and cannot represent hidden undercuts or gel wrapping.

Fitting rendered images to reference photographs estimates appearance parameters. It does not by itself establish metric depth accuracy, force calibration, or independent agreement with a physical sensor. Those claims require separate measurements and evaluation.
