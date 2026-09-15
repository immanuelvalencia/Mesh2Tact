# Methodology insertion notes: configuration, calibration, and efficient capture

Working material for integration into `04_methods.md`. These notes describe implementation verified from the current source; they do not replace the agreed methodology structure or introduce literature claims. See the [engineering record](../review/implementation_improvements.md) for evidence and limitations.

## Sensor dimensions and image sampling

The physical sensor area and image resolution are configured separately. For sensor width \(L_x\), height \(L_y\), and capture width \(W\), the application computes

\[
H=\operatorname{round}\!\left(W\frac{L_y}{L_x}\right),\qquad
\Delta x=\frac{L_x}{W},\quad \Delta y=\frac{L_y}{H}.
\]

When the user edits height, width is recomputed using the reciprocal relation. Both dimensions are constrained jointly to 64–2048 pixels. Ratios that cannot satisfy these bounds are rejected. The physical field of view remains unchanged; integer rounding introduces a small difference between horizontal and vertical pixel spacing. For an 18.6 × 14.3 mm area, a capture width of 984 pixels produces 757 rows. Preview resolution is a scaled display grid; exported captures use the configured capture grid.

Built-in configurations are protected within the application. The user creates a named copy before editing or replacing a built-in configuration. Calibration exports also require a new name. Configurations store sensor geometry, optical parameters, background/gradient settings, image effects, and capture controls. This workflow supports traceability of the settings used to generate each dataset; it does not substitute for recording the physical sensor and acquisition conditions.

## Reference-guided appearance fitting

Each reference combines a photograph, a known mesh, and a manually aligned pose defined by XYZ rotations, XY offsets, and cut depth. The pose is locked before fitting. Several reference contacts can share one fitted appearance configuration; optional empty frames constrain background appearance. Validation-only references are withheld from optimisation, including noise and texture statistics. Manual alignment and fixed contact geometry should be stated explicitly, since the procedure is not joint automatic pose and appearance estimation.

Background, lighting, and deterministic image filters are fitted in stages using bounded trust-region least squares with a soft-L1 loss. Parameters are normalised to [0, 1] before finite-difference optimisation. Background fitting uses non-contact regions, while lighting and filter stages balance references and contact/background regions. Sampling limits the number of residual values used for numerical optimisation. A deterministic stage retains the previous settings if its candidate does not improve the training objective.

The light search combines coarse edge assignments or azimuth-quadrant candidates with conditional RGB-weight proposals and seeded multistart continuous refinement. In edge-lighting mode, the parameters include light elevation and shared distance/falloff; edge selection is discrete. The light count, lighting mode, exposure, ambient term, and diffuse/specular parameters remain fixed during this fitting stage. Thus, the method estimates a selected appearance parameterisation rather than uniquely identifying the physical illumination system.

Noise is estimated from native-resolution smooth, unsaturated regions. Where aligned repeated exposures are available, the difference of two frames divided by \(\sqrt{2}\) suppresses static image structure under the equal-variance, independent-noise assumption. Single-image estimation instead uses a normalised second-difference filter. Robust variance and cross-channel statistics constrain nonnegative read, shot, and shared-channel multiplicative noise components. These assumptions and single-frame identifiability limitations must accompany the method.

Texture strength and logarithmic physical scale are fitted with seeded differential evolution using multiscale residual statistics and two fixed stochastic realisations. The random seed is not fitted to reproduce the reference’s individual noise pixels. Deterministic pixel errors and stochastic-statistics objectives are reported separately. Review images can include the fitted stochastic effects, but independent random noise is not expected to reduce pixelwise error against a particular measured frame.

The application normally fits background/lighting at a width of 192 pixels, preserving the configured aspect ratio. Filter refinement increases width up to the capture width, capped at 768 pixels; blur is converted between output and fitting pixel units. Noise/texture use native reference pixels. Review previews use a width of 768 pixels. Report the actual fitting resolution from the result record rather than treating every displayed image as a full-resolution evaluation.

## Computational implementation

Edge illumination is evaluated using 12 midpoint samples along each selected edge. The optimisation preserves these samples and the original analytic response. Coordinate grids are shared, and vector components are evaluated directly to reduce temporary three-channel arrays.

For full-resolution rendering, the image is processed in row blocks. With a target block size \(B=16{,}384\) pixels, the block height is

\[
b=\max\!\left(1,\left\lfloor B/W\right\rfloor\right).
\]

For a pixel at global row \(i\) and column \(j\), coordinates remain

\[
x_j=\left(\frac{j+1/2}{W}-\frac12\right)L_x,\qquad
y_i=\left(\frac{i+1/2}{H}-\frac12\right)L_y.
\]

Using global coordinates prevents each block from being treated as a separate sensor area. Blocking changes the working array sizes, not image resolution, light quadrature, or contact geometry. The implementation remains CPU-based.

With adjustable background enabled, the renderer subtracts the flat-surface light response from the contact response before adding the background. A one-entry cache stores the analytic flat response, keyed by image shape, physical sensor width/height, and optical settings. Changing these values invalidates the cached entry; changing only the contact pose can reuse it. Lookup-table calibration does not use this analytic cache. The cache is local to a simulator instance, so a fresh single-capture snapshot computes its first full-resolution flat field.

Texture computation uses separate bounded caches for seeded 512 × 512 random source fields and resampled texture fields, with up to four entries in each. Field keys include seed, physical texture scale, sensor dimensions, and output dimensions. Cache reuse preserves the effect-generation sequence and does not change the random seed. During calibration, per-reference geometry and light-response bases also avoid repeated work when compatible parameters remain unchanged.

## Dataset capture and timing protocol

Single and automatic capture operate on a snapshot of the selected simulator settings in a worker thread. Export uses the capture grid and preserves the selected pose for single capture. The interface displays progress and prevents overlapping capture starts. A stop request completes the current sample before ending collection, preserving completed outputs. This separates interface interaction from rendering/export work; it is not a real-time rendering guarantee.

For the preliminary runtime check, all output types were enabled and a synthetic sphere with 20,480 faces was captured at 0.8 mm depth using GelSightV1. The timed region covered rendering and export at 984 × 757 and 2048 × 1575, using fresh simulator instances. Measurements included profiler overhead and excluded GUI snapshot creation and interaction. This protocol must be expanded to repeated, unprofiled runs and multiple workloads for the final evaluation.

The planned ResNet-based contact-shape and object-classification evaluations remain separate experiments. Real2Real denotes real training and independent real testing; Sim2Real denotes synthetic training and real testing; Real2Sim denotes real training and synthetic testing. Acquisition groups and their derived frames/augmentations must remain within one split. No transfer accuracy is established by rendering-speed or software-correctness tests.
