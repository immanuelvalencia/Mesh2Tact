# Reference-guided appearance calibration

Open the **Calibration** tab in the main window. This is the single calibration entry point. Load the object with **Load STL…** in **General object**, using the source-unit and import-scale controls there. Calibration uses a snapshot and does not change saved sensor presets automatically.

1. Load the first STL in **General object** and select the sensor settings you want to start from.
2. Open **Calibration**, click **Upload reference…**, and enter the name of the **Shape shown in the reference**. The current STL and pose are copied for this reference.
3. Click **Align reference**. Use the normal main-window movement and rotation controls to match the uploaded photo. **Load STL…** remains available to replace the draft mesh.
4. Click **Save reference** beside the overlay. This saves the image, processed mesh, shape name and pose into a collection archive. The list marks it **Saved**.
5. Load the next STL in **General object**, return to Calibration, upload its image, align and save it. Earlier reference meshes and poses remain independent.
6. For an image of the empty sensor, select **No shape**, then **Save reference**. No alignment is required.
7. Once every entry is saved, click **Calibrate all saved references**. One shared appearance configuration is fitted against the collection, with a before/after error for each image. Background, lighting, directions, filters, noise and texture fitting are enabled by default. Saved geometry and poses remain fixed. **Validation only** references are evaluated but excluded from optimisation.
8. Review the overlays and per-reference errors, then **Save fitted sensor config…**. Select that config in sensor settings to apply it to previews and captures.

Each saved reference updates an automatically named archive in `configs/calibration_sessions/`. Fitting results are also stored in that archive. **Open collection…** resumes it; **Save collection as…** creates a named copy; **New collection** starts another without deleting previous archives. Draft entries are marked in the list and must be saved before fitting.

Every completed fitting run also creates a separate timestamped folder in `configs/calibration_sessions/runs/`. It contains `metrics.json`, `metrics.csv`, an importable `sensor_config.json`, a self-contained `collection.npz`, and each reference's original image, resized comparison image, and before/after simulation PNGs. These deterministic PNGs use the same resolution as the reported metrics. Earlier runs are never overwritten. Rejected fits are recorded with their acceptance status and retained original settings; cancelled or failed fits have no completed result. Disk-write failures are reported and partially written folders end in `.incomplete`.

Run configs are not applied automatically. To apply one later, use **Settings → Saved sensor configurations → Load config file…** and choose its `sensor_config.json`.

The overlay supports opacity blending, contact outline, absolute error and simulation only. Alignment previews use the capture width and height selected in General object, including while moving. Zoom with the mouse wheel and drag an image to pan. **Cancel alignment** discards the live draft mesh and pose.

For repeated exposures of an unchanged contact, assign the same optional **Repeat group**. Standard/Thorough search controls the fitting budget. Fitting runs in the background and can be cancelled.

**Fitting algorithm** offers two choices:

- **Current — robust least squares** (default): the existing bounded image-error fitter.
- **AI-assisted — learned surrogate (experimental)**: a local Gaussian-process model learns error from trial renders and proposes settings using uncertainty-aware search, followed by robust least-squares refinement. It trains during each fit, needs no downloaded model or cloud service, and may take longer. It is not a pretrained image model, and improved accuracy over the current fitter has not been established on real references.

Both algorithms fit the same shared settings, including brightness when image-filter fitting is enabled, exclude validation references from training, and use the same final production-render acceptance checks. Noise and texture retain their statistical estimators. The chosen algorithm and seed are recorded with the fitted result and restored when the collection is reopened.

To revise starting settings, edit them in the app and click **Use current sensor settings for this session**. References and poses are retained, but previous fit results are cleared. Changing the sensing area's dimensions unlocks alignments for rechecking.

## What is fitted

- Background: base RGB, vignette, and the colours, strengths, positions, widths, heights and angles of existing glow fields. The number of fields remains fixed; add a narrow edge glow before fitting if the reference has a separate illuminated strip.
- Lighting: effective RGB weights represented as normalised colour plus intensity. **Fit light directions / edge geometry** also fits elevation and azimuth. In edge-lighting mode, azimuth selects one of four full edges; the fitter searches edge assignments, elevation, shared side distance and falloff. In directional mode, azimuth and elevation are continuous. The light count, lighting mode, exposure, ambient, diffuse and specular settings remain fixed to avoid redundant intensity parameters.
- Image filters: shared tactile-image brightness, blur, additional vignetting, contrast and gamma. Brightness is matched jointly across the saved contact and no-shape references and saved with the calibrated settings. Alternatively select only **Refine blur**. Contact softness and all reference poses remain fixed.
- Noise: read-noise amplitude, intensity-dependent shot noise and shared-channel multiplicative speckle. The fitted read-noise value contains the total additive camera noise; optical `noise_sigma` is set to zero to avoid adding it twice.
- Texture: spatial texture strength and physical texture size. The noise seed is never optimised to copy pixels from the reference.

### Optimisation strategy

Background, lighting and filters use bounded trust-region least squares with a soft-L1 robust loss. Parameters are scaled to [0, 1], with absolute finite-difference steps that work at zero weights and zero angles. Background fitting uses pixels outside the aligned contact region. Lighting and filter fitting balance contact/background regions and references. Sampling retains all three channels of each selected pixel. A deterministic stage is accepted only when its training objective improves. A final uncached production render checks full-image training MAE and MSE; a degraded candidate is rejected and the original settings are retained. Validation images never determine acceptance.

Lighting uses two coarse coordinate-search sweeps over edge assignments or azimuth quadrants, followed by seeded multistart continuous refinement. Each direction also gets a conditional RGB-weight proposal, so an initially disabled light can become active and an incorrect starting colour does not prevent direction selection. The full nonlinear objective decides whether to accept that proposal. Per-reference light-response bases are cached through the production shader, so changing RGB weights does not recompute all source geometry. This is a practical mixed discrete/continuous search, not a guarantee of a global optimum.

Noise uses native-resolution, smooth, unsaturated patches. Repeated exposures use pair differences divided by the square root of two to remove static appearance. Single images use a normalised second-difference filter that suppresses planes. Robust variance estimates and RGB cross-channel covariance constrain nonnegative read, shot and speckle variance components. The variance model includes the small shot–speckle product because speckle follows Poisson sampling in the renderer. Contrast/gamma are inverted before estimating these components. The report flags single-frame estimates and narrow intensity ranges, where components are less identifiable.

Texture uses seeded differential evolution on strength and logarithmic physical size. It compares multiscale residual powers with two fixed random realisations of the actual image-effects pipeline at native resolution. It matches spatial statistics rather than the location of individual texture specks. Broad illumination mismatch is suppressed, but the report notes that remaining geometry or material detail can still inflate estimated texture.

Validation photographs never enter fitting or noise/texture statistics. Empty images constrain background and noise but cannot determine contact lighting. There is no universally best optimiser for all reference sets; the hybrid is chosen for this renderer's discrete edges, bounded continuous settings and stochastic effects.

The fitter uses the production optical and gel rendering functions with cached contact depth. Alignment, review, background, lighting, and filter/blur fitting all use the exact capture width and height selected in General object. Calibration does not reduce or cap that resolution. The selected dimensions are retained in comparison images and the saved sensor config. Full-resolution rendering can take longer; bounded residual sampling limits optimization memory without resizing the rendered image. Noise and texture use the original photo pixels, without downsampling. Before/after pixel-error metrics use the same final deterministic fitting resolution, recorded in the report; they are not necessarily full-resolution validation measurements.

Older fits may contain a nearly green background because the previous sampler could skip two RGB channels. Loading a saved config whose recorded training error increased displays a refitting notice. Reopen the original calibration session and fit again, or reimport and align the reference photos. Existing config files are not rewritten automatically.

**Show fitted noise and texture in preview** renders the saved effects with a fixed seed; turn it off to inspect the deterministic contact. Statistical objectives and parameter changes are reported separately from deterministic pixel MAE/RMSE. Independent random noise is not expected to reduce pixelwise error against a particular photo. Selecting filters, noise or texture enables image effects in the fitted preset. No force, elasticity, unique physical sensor identification or automatic pose refinement is performed. Disable calibration lookup-table mode to fit analytic lighting.

For useful calibration, first match the object's dimensions and contact geometry. Then use multiple contact orientations, positions and depths, preferably with an empty-sensor image and independent validation contacts. A low training residual alone does not establish generalisation to other contacts.

### Stage images and Camera Roll workflow

**Preserve contact colours under gradients**, in the gel-background controls, treats glow fields as empty-pad illumination. It adds the shader's signed contact response to that background instead of painting translucent glows over the contact. This prevents a strong fitted centre glow from erasing the imprint. The setting is saved as `gel_lighting.preserve_contact`; older presets retain their previous compositing behavior by default.

Completed deterministic stages retain their own settings in the run report. Each reference folder also contains `stage_01_background.png`, `stage_02_lighting.png`, and a filter/blur image when that stage was selected. Stage numbering follows the selected stages. These are production renders at the final comparison resolution. If production verification rejects a candidate, `after.png` retains the original configuration; intermediate stage images still show the attempted fit.

`conda run -n vtsim python tools/calibrate_camera_roll.py` fits the supplied `benchmarks/Camera Roll` collection and saves a new timestamped run under `benchmarks/camera_roll_calibration/runs/`. It fits the blank background first, then shared edge-source lighting across 21 location/depth images, followed by blur. The two small-ball images are excluded from fitting. It saves `configs/sensors/GelSight Camera Roll spatial.json`, a resumable collection, estimated alignments, per-image errors and a stage comparison sheet. Original JPG files are never modified.

This workflow estimates contact outlines by subtracting the empty frame and uses approximate sphere geometry. The nominal bed size and sphere radii are scale assumptions, not measured dimensions. Consequently, the saved configuration is an appearance fit; the estimated cut depths are not pressure, force or calibrated physical depth. The numbered images represent different presses, not repeated exposures. Small-ball comparisons are excluded-image appearance checks with image-estimated geometry, not independent geometric validation.

The Camera Roll command then runs `tools/fit_spatial_response.py`: it fits five normal terms `(nx, ny, nz−1, nx·ny, nx²−ny²)` multiplied by six position terms `(1, x, y, x², x·y, y²)`, with ridge regularization and equal per-reference weighting. Targets are contact-minus-blank RGB differences after removing an exposure offset outside contact. The resulting finite 30×3 coefficient matrix is stored as `sensor.optics.spatial_response` and replaces analytic LED shading. Flat normals give zero contact response. This allows a central red response and different edge colours without attenuating the contact under the background glows. LED controls do not change this empirical response; exposure and image appearance controls still apply. The library fit button supports analytic LEDs; rerun the spatial script to refit this response, or remove the spatial response field to return to analytic mode. Extrapolation outside sampled positions/normals is unvalidated.

`tools/review_camera_roll.py` verifies GUI loading and exports the same sphere/depth at nine positions, a compact preview, and a numerical summary. The complete command runs these steps automatically. Stage archives retain the analytic attempt as well as the final spatial-response stage.

