# 4. Methodology

## A. Mesh-to-Tactile Generation Process

Mesh2Tact is a mesh-to-tactile generator for static synthetic visuo-tactile data. It takes a 3D mesh, a sensor configuration, and contact-sampling settings. After import and optional mesh refinement, the generator poses the mesh relative to a fixed sensor plane, reconstructs its contact geometry, and renders the resulting surface as a visuo-tactile image. The same process also produces depth and contact representations required for inspection, supervision, and reproducible data generation.

Figure 1 summarizes the generation process. Sensor settings specify the sensing area, image resolution, pad colour, background gradients, lighting, and image effects. Appearance parameters can be configured directly or estimated from reference visuo-tactile images paired with locked contact geometry and poses. The resulting configuration is reused during generation. For each specified or sampled contact, mesh rasterization produces raw depth and a contact mask; optional depth smoothing precedes normal estimation. Optical rendering produces clean RGB, and image processing produces the final visuo-tactile image. Batch generation repeats contact sampling and rendering, exporting the selected images, geometric representations, and settings with a dataset manifest. Mesh2Tact represents static geometry and appearance; it does not model force, elastomer dynamics, friction, or marker motion.

![Generic Mesh2Tact generation workflow with reusable sensor configuration, contact sampling, geometric reconstruction, optical rendering, image processing, and export.](C:/Users/Jim/Development/mesh2tact/publication/figures/framework/fig_framework.png)

**Fig. 1. Mesh2Tact generation process.** Sensor settings are configured directly or refined through optional calibration using reference visuo-tactile images and locked contact geometry. The reusable configuration supplies the geometric sampling grid, optical appearance, and image-effect parameters. Each contact passes through mesh positioning, depth and mask reconstruction, normal estimation, optical rendering, and image processing. Selected outputs include clean and final visuo-tactile images, raw and smoothed depth, contact masks, mesh data, and generation metadata. The two-row grid follows the numbered arrows from steps 1–3 across the top row and steps 4–6 from right to left across the bottom row. Batch generation repeats steps 2–6 while reusing the prepared mesh and sensor configuration. Optional reference calibration supplies the configuration before generation and is not repeated for each sample.

## B. Mesh Preparation and Contact Geometry

### 1) Mesh representation and pose

Mesh2Tact accepts STL, OBJ, PLY, and GLB meshes. The selected source unit and import scale are converted to metres before rendering. A valid input must contain finite vertices and triangular faces. The application retains the imported mesh and can create a processed copy through simplification, subdivision, or Taubin smoothing. Simplification reduces rasterization cost, subdivision increases tessellation, and Taubin smoothing changes the surface itself. These mesh operations are distinct from the depth smoothing described below.

For a mesh vertex $\mathbf{v}_k$, an XYZ Euler rotation and lateral offset define the posed vertex

$$
\widetilde{\mathbf{v}}_k = R_{xyz}(\theta_x,\theta_y,\theta_z)\mathbf{v}_k + (t_x,t_y,0)^\mathsf{T}.
\tag{1}
$$

~~~latex
\widetilde{\mathbf{v}}_k = R_{xyz}(\theta_x,\theta_y,\theta_z)\mathbf{v}_k + (t_x,t_y,0)^\mathsf{T}.
~~~

The sensor plane remains at $z=0$. Rather than moving a cutting plane, the application translates the object origin along $z$. If $z_o$ is this object position, the full posed vertex is $\mathbf{v}_k'=\widetilde{\mathbf{v}}_k+(0,0,z_o)^\mathsf{T}$. This convention makes a negative displacement move the object into the gel and a positive displacement release it. Let $z_{\min}=\min_k\widetilde v_{k,z}$ be the lowest rotated mesh vertex. The geometric indentation setting is

$$
d=-(z_{\min}+z_o), \qquad 0\leq d\leq d_{\max},
\tag{2}
$$

~~~latex
d=-(z_{\min}+z_o), \qquad 0\leq d\leq d_{\max}.
~~~

where $d_{\max}$ is the user-defined maximum indentation depth. The bound is applied after changes in pose or object position. It limits the simulated contact range; it is not a measured force or material limit. The user interface derives its vertical travel range from the rotated mesh height so that first contact is centred and the available downward travel corresponds to $d_{\max}$.

For a sensing area $L_x\times L_y$ sampled at $W\times H$ pixels, pixel centres correspond to

$$
x_j=\left(\frac{j+1/2}{W}-\frac12\right)L_x,\qquad
y_i=\left(\frac{i+1/2}{H}-\frac12\right)L_y.
\tag{3}
$$

~~~latex
x_j=\left(\frac{j+1/2}{W}-\frac12\right)L_x,\qquad
y_i=\left(\frac{i+1/2}{H}-\frac12\right)L_y.
~~~

The renderer orthographically rasterizes each triangle over this grid. At each covered pixel, it retains the minimum interpolated surface height $s(i,j)$; uncovered pixels are assigned $+\infty$. The raw indentation map and binary contact mask are

$$
d_0(i,j)=\max\{-[s(i,j)+z_o],0\}, \qquad
M(i,j)=\mathbb{1}[d_0(i,j)>0].
\tag{4}
$$

~~~latex
d_0(i,j)=\max\{-[s(i,j)+z_o],0\}, \qquad
M(i,j)=\mathbb{1}[d_0(i,j)>0].
~~~

The minimum-height rule represents the surface first encountered by the sensor-facing view. It is a geometric construction, so it does not model deformation of an elastic body. An optional Gaussian filter creates a smoothed depth field $d=G_\sigma*d_0$, where $\sigma$ is specified in physical units. The associated height and normal field are $h=-d$ and $\mathbf n=(-h_x,-h_y,1)^\mathsf{T}/\sqrt{h_x^2+h_y^2+1}$. The raw contact mask remains tied to $d_0$; smoothing may extend nonzero depth slightly beyond its boundary.

## C. Sensor Configuration and Optical Rendering

Mesh2Tact uses a configurable GelSight Mini-style appearance benchmark. A sensor configuration stores the physical sensing area, capture resolution, depth-display range, background, illumination, and image-effect parameters. The capture height follows the physical aspect ratio:

$$
H=\operatorname{round}\!\left(W\frac{L_y}{L_x}\right), \qquad
\Delta x=\frac{L_x}{W},\quad \Delta y=\frac{L_y}{H}.
\tag{5}
$$

~~~latex
H=\operatorname{round}\!\left(W\frac{L_y}{L_x}\right), \qquad
\Delta x=\frac{L_x}{W},\quad \Delta y=\frac{L_y}{H}.
~~~

Editing the capture width or height updates the other dimension to preserve this relation. The preview grid is a scaled display of the capture grid and does not change exported resolution. A named sensor configuration preserves appearance and sensing settings; a complete workbench preset additionally records mesh pose and dataset-sampling controls. Built-in configurations are protected from replacement so that generated data can be traced to a named copy.

### 1) Pad colour and background gradients

The apparent colour of the pad is specified independently of the contact geometry. The base RGB triplet sets the uniform background before spatial illumination layers are applied. A dark, neutral, or coloured base therefore changes the simulated pad appearance without changing its indentation or surface normals. These RGB values describe effective image colour, rather than measured pigment composition, spectral reflectance, or material stiffness. In optical tactile sensing, the observed colour depends jointly on illumination and coating reflectance (Gomes et al., 2021); the simulator does not identify those physical contributions separately.

Up to eight elliptical Gaussian glow layers can be superimposed on the base field; the default configuration contains two. Each layer has an RGB colour, strength, horizontal and vertical centre, width, height, and in-plane rotation. Centres and widths are expressed relative to the image dimensions. The layer blends its colour with the underlying image using a spatially varying opacity: strength determines the peak contribution, widths determine its spread, and rotation sets the ellipse orientation. Layers are applied in their saved order, so overlapping gradients are not interchangeable. A radial vignette subsequently darkens the field towards the corners.

These background gradients describe broad colour variation over the pad, including regions outside contact. They are distinct from the surface gradients $h_x$ and $h_y$, which determine the normal field and generate the contact-dependent colour bands. This separation permits background adjustment without altering the reconstructed contact shape. The base RGB field, glow layers, and vignette are applied to the image containing the contact response; consequently, a strong glow can also attenuate local contact contrast.

### 2) Illumination arrangement and contact response

Each light group has an independently specified RGB colour, intensity, azimuth, and elevation. Pure red, green, and blue lights are one configuration; mixed colours and white illumination are also permitted. Changing colour or intensity changes the contribution to each output channel, whereas changing direction changes which surface orientations are bright. Diffuse gain controls the broad orientation-dependent response. Specular gain and the shininess exponent control highlight amplitude and concentration, and exposure scales the resulting optical image.

The analytic renderer supports two lighting arrangements. In directional mode, each group supplies one spatially constant direction defined by azimuth and elevation. In edge mode, azimuth selects the nearest of four sensor edges, and each group is represented by 12 equally spaced midpoint samples along that edge. A shared distance multiplier places the sources relative to the sensor half-width or half-height; elevation sets their height above the reference plane. The distance-falloff exponent controls spatial attenuation, with each sample weighted by one twelfth of the group contribution. Thus, edge lighting varies across the sensing area, whereas directional lighting neglects source-distance variation.

For channel $c$, the contact-dependent shading response is

$$
S_c = a_c + \sum_{\ell,q}w_{\ell c}A_{\ell q}
\left[k_d\max(\mathbf n\!\cdot\!\mathbf l_{\ell q},0)
+k_s\max(\mathbf n\!\cdot\!\mathbf b_{\ell q},0)^p\right],
\tag{6}
$$

~~~latex
S_c = a_c + \sum_{\ell,q}w_{\ell c}A_{\ell q}
\left[k_d\max(\mathbf n\!\cdot\!\mathbf l_{\ell q},0)
+k_s\max(\mathbf n\!\cdot\!\mathbf b_{\ell q},0)^p\right].
~~~

where $a_c$ is ambient intensity, $w_{\ell c}$ is light colour multiplied by intensity, $A_{\ell q}$ includes attenuation and sample weighting, $\mathbf l_{\ell q}$ is the unit source direction, and $\mathbf b_{\ell q}$ is the normalized half-vector between the source and viewing directions. The viewing direction is $(0,0,1)^\mathsf{T}$. The implementation uses diffuse shading and a half-vector specular term; this is an adaptation of empirical local illumination, rather than the original reflected-vector formulation of Phong (1975). With the configurable gel background enabled, the renderer subtracts the corresponding flat-surface shading before adding the base RGB field. A spatially constant ambient term therefore cancels in this branch and cannot be identified independently from the background.

### 3) Global colour controls and configuration scope

After optical shading and background compositing, hue rotates the image colours, saturation scales their chromatic strength, contrast rescales intensity about 0.5, gamma applies an inverse-power transform, and brightness multiplies the result. These global controls act before the image effects in Section D. Background vignetting and global contrast/gamma are separate parameters from the additional image-effect vignette and contrast/gamma; recording both sets is necessary to reproduce a render. Overlapping colour controls also make multiple parameter combinations capable of producing similar images.

The configurable-background switch selects whether the base field and glow layers are used. Without this field, the optical renderer can use its fixed reference-inspired background correction. A supplied gradient-to-RGB lookup table can alternatively replace analytic shading; the reference-guided procedure in Section E fits the analytic branch and requires this lookup-table mode to be disabled. Neither branch models multilayer refraction, cast shadows, or physical changes in coating properties. Although the configuration schema retains material, solver, and marker fields, the static geometric path described here does not use them to compute elastic deformation or marker displacement.

## D. Image Processing and Appearance Variation

Image processing is part of the forward model used during appearance calibration. It is described before calibration because its deterministic parameters and stochastic statistics are among the quantities estimated from reference images. Geometric Gaussian smoothing remains a separate operation before normal estimation (Section B); its physical width is fixed during appearance fitting.

The RGB effects follow a fixed order: spatial texture, additional vignetting, Gaussian blur, shot noise, multiplicative noise, additive read noise, contrast, and gamma. Texture multiplies all channels by a common spatial field whose amplitude and correlation scale are configurable; its scale is specified in millimetres and converted using the sensing area. A seeded Gaussian field is smoothed on a fixed reference grid and resampled to the output grid. It represents reflectance variation in the image, without perturbing the depth map. The additional vignette multiplies intensity by a radially decreasing field. Gaussian RGB blur then smooths each colour channel using a standard deviation specified in output pixels.

Shot noise uses Poisson sampling with an effective photon count of $1/\sigma_s^2$, where $\sigma_s$ is its configured standard deviation at unit intensity. Multiplicative noise, labelled speckle in the interface, applies one zero-mean Gaussian multiplier per pixel shared across RGB channels. Read noise adds independent zero-mean Gaussian samples to the channels. The optical-noise setting and image-effect read-noise setting are combined in quadrature and applied at this stage. These are statistical image models; the speckle control does not simulate coherent optical interference.

Finally, contrast scales intensity around 0.5, the result is clipped to $[0,1]$, and gamma applies the exponent $1/\gamma$. The final RGB image is quantized to eight bits per channel. A stored seed controls stochastic realizations and may remain fixed or vary between dataset samples. Disabling the effects stage preserves the clean optical RGB image. None of these RGB operations changes numerical depth, surface normals, or the raw contact mask.

![Individual depth and RGB processing effects for a fixed sphere contact.](C:/Users/Jim/Development/mesh2tact/publication/figures/methodology_revision/fig02_individual_filters.png)

**Fig. 2. Individual processing effects on a sphere contact.** (a, b) Raw depth and depth smoothed with a 30 µm Gaussian standard deviation, displayed on the same 0–1.5 mm scale. (c) Clean RGB obtained from the smoothed geometry. (d–k) Each RGB effect applied separately to the same clean rendering; all other RGB effects are neutral. (l) All eight illustrated RGB effects applied in their production order. Texture strength is 0.12 with a 0.15 mm correlation-scale setting; the additional vignette is 0.60; RGB blur is 5 output pixels; shot, multiplicative, and read-noise settings are 0.06, 0.12, and 0.04; contrast and gamma are 1.50. These deliberately visible settings illustrate operator behaviour and are not fitted values. The sphere radius is 4.5 mm, and the image grid is 984 × 739. A fixed seed of 17 makes each rendering reproducible; no display enhancement is applied to the RGB panels.

## E. Reference-Guided Appearance Calibration

Reference-guided calibration estimates an appearance configuration from real photographs without changing the live configuration automatically. Each calibration reference contains a photograph, a known primitive or imported mesh, and a manually adjusted pose. The user aligns XYZ rotation, XY offset, and contact depth against the image, then locks the alignment before fitting. An empty-sensor photograph may be included for background and noise estimation. References marked validation only are retained for review but excluded from optimization and from noise and texture statistics.

Figure 3 summarizes the calibration procedure. The fitting process first estimates the background using non-contact regions. It then estimates effective illumination and deterministic image filters while the reference geometry and alignment remain fixed. Background fitting adjusts base RGB, vignette, and existing glow colours and strengths; glow count, positions, widths, and rotations remain fixed. Filter fitting adjusts RGB blur, additional vignette, contrast, and gamma, or optionally refines blur alone. The global gel colour controls remain fixed. For a set of training references $\mathcal R$ and render parameters $\boldsymbol\theta$, the deterministic objective is

$$
\boldsymbol\theta^*=\arg\min_{\boldsymbol\theta\in\mathcal B}
\sum_{r\in\mathcal R}\sum_{p\in\Omega_r}
\rho\!\left(I_r(p)-\widehat I_r(p;\boldsymbol\theta)\right),
\tag{7}
$$

~~~latex
\boldsymbol\theta^*=\arg\min_{\boldsymbol\theta\in\mathcal B}
\sum_{r\in\mathcal R}\sum_{p\in\Omega_r}
\rho\!\left(I_r(p)-\widehat I_r(p;\boldsymbol\theta)\right).
~~~

where $I_r$ and $\widehat I_r$ are real and simulated RGB values, $\Omega_r$ is the selected image region, $\mathcal B$ contains parameter bounds, and $\rho$ is a soft-$L_1$ robust loss. Background fitting emphasizes non-contact pixels; lighting and filter fitting balance contact and background regions. Parameters are normalized for bounded trust-region least squares, and a candidate stage is retained only when it improves the training objective.

The illumination search combines a coarse search over edge assignments or azimuth quadrants with seeded multistart continuous refinement. Conditional RGB-weight proposals allow a disabled or incorrectly coloured source to become active. In edge-lighting mode, the discrete edge assignment is searched together with elevation, shared source distance, and falloff. In directional mode, azimuth and elevation are continuous. Light count, rendering mode, exposure, and diffuse/specular settings are fixed during a fit. The result is therefore an effective image-formation configuration, not an identification of the physical LED locations.

Noise and texture are estimated separately from deterministic pixel fitting. Repeated aligned exposures use pair differences divided by $\sqrt{2}$ to suppress static image structure under an equal-variance, independent-noise assumption. Single references use a normalized second-difference estimate in smooth, unsaturated regions. These measurements constrain read noise, intensity-dependent shot noise, and shared-channel multiplicative noise. Texture strength and physical scale are fitted with seeded differential evolution against multiscale residual statistics. The seed is never optimized to reproduce individual pixels in a reference photograph.

The fitting path normally uses a width of 192 pixels for background and illumination searches. Filter refinement increases the width to the requested output width, capped at 768 pixels, with blur converted from output pixels to fitting pixels. Noise and texture estimation use native reference pixels. Before/after deterministic errors use the same final fitting resolution, which is recorded separately from the display resolution. The saved record retains fitting settings, locked poses, reference membership, objectives, and before/after error measures. A fitted configuration can then be saved under a new name and used as a normal sensor preset.

![Reference-guided calibration from locked contact references to a saved sensor configuration.](C:/Users/Jim/Development/mesh2tact/publication/figures/methodology_revision/fig03_calibration_process.png)

**Fig. 3. Reference-guided appearance calibration.** The two-row grid follows the numbered arrows: reference preparation, background fitting, illumination fitting, filter fitting, noise and texture estimation, and review. Reference preparation includes geometry alignment, pose locking, and selection of reserved validation images. Geometry and pose remain fixed during fitting. Deterministic stages use the training objective; noise and texture use separate statistical estimates. Reserved validation references enter review only and are excluded from every fitting stage. Noise seeds are not optimized. The resulting configuration describes effective appearance, not uniquely identified material properties or physical LED positions.

### Sphere illustration of the rendering stages

Figure 4 shows default appearance, reference-fitted appearance before image effects, the same fitted appearance after image effects, and the actual GelSight reference. The three synthetic panels use one triangulated sphere, one pose, and one depth field, so changes between them arise from appearance settings. This order describes image generation: the calibrated parameters are loaded first and filtering is applied during rendering, even though filter parameters may themselves be estimated within calibration.

This illustration reuses the archived preliminary reference fit, whose sphere radius and contact outline were assumed from the photograph. It does not document a new run of every stage in Fig. 3. The archived image-effect settings were selected for appearance generation, rather than estimated by the subsequently implemented noise/texture fitter. The example therefore establishes neither metric geometric calibration nor generalization to unseen contacts. Independent contacts at different positions, orientations, and depths are required to evaluate those properties.

![Default, reference-fitted and filtered sphere renderings beside the actual GelSight reference.](C:/Users/Jim/Development/mesh2tact/publication/figures/methodology_revision/fig04_sphere_stages.png)

**Fig. 4. Sphere rendering from default appearance to reference comparison.** (a) Uncalibrated appearance defaults, with the same geometry and sampling as (b, c). (b) Archived GelSight reference-fitted background and lighting, without RGB effects. (c) The same rendering with the archived image-effect settings: 0.5-pixel blur, texture strength 0.008 and scale 0.14 mm, shot noise 0.002, and seed 17; other effects are neutral. (d) Actual GelSight hemisphere image, sequence 003, frame 0012, used in the original appearance fit. The 4.5 mm sphere radius and contact outline are assumed, not independently measured. The historical 984 × 739 registration is retained across panels; it differs from the 984 × 757 grid produced by the current aspect-ratio rule for an 18.6 × 14.3 mm area. The photograph is shown without colour adjustment. This is an in-sample qualitative illustration, not a held-out comparison.

## F. Automated Dataset Generation and Export

Automatic generation repeats the same rendering pipeline from a reproducible random stream. A run may independently sample XYZ Euler angles, XY offsets, indentation depth, and an image-effect seed. For each enabled scalar parameter $q$, the sampler draws

$$
q \sim \mathcal U(q_{\min},q_{\max}).
\tag{8}
$$

~~~latex
q \sim \mathcal U(q_{\min},q_{\max}).
~~~

The orientation components are sampled independently; this is not uniform sampling on $SO(3)$ and does not guarantee equal coverage of object surfaces. The requested indentation range is checked against $d_{\max}$ before the run starts. Fixed controls remain unchanged. The seed, requested ranges, and completed-sample count are written to the run record so that a run can be reproduced with the same mesh, configuration, and software version.

For each completed sample, the user may export final RGB, clean RGB before image effects, raw and smoothed depth arrays in metres, a depth visualization, a binary contact mask, the processed mesh, and settings metadata. The per-sample record stores rotation, XY offset, object position, indentation depth, maximum indentation depth, contact fraction, peak raw depth, and output filenames. A sample is added to the manifest only after all selected files are written. This preserves partial-run diagnostics without treating incomplete samples as dataset members.

## G. Transfer-Evaluation Protocol

The planned learning evaluation has two separate single-image classification tasks: contact-shape classification and object classification. Contact-shape labels describe the local geometry visible in an imprint; object labels describe the source-object category. The tasks require separate classifiers because a single object can yield different local imprints and similar local imprints can arise from different objects.

A ResNet-based classifier will be trained under the three domain conditions in Table I. Architecture variant, initialization, image preprocessing, optimizer, augmentation, training budget, and model-selection rule will be fixed within each task before training. Dataset sizes and class distributions will be reported once the real dataset is finalized; they are not specified here because the experiments have not yet been run.

| Condition | Training domain | Test domain | Role |
|---|---|---|---|
| Real2Real | Real | held-out real | Real-data reference baseline |
| Sim2Real | Synthetic | the same held-out real set | Transfer from Mesh2Tact data to real tactile images |
| Real2Sim | Real | held-out synthetic | Reverse-transfer diagnostic |

Splits will be defined before augmentation. Frames from the same physical contact sequence and their augmentations will remain in one split. Synthetic renders derived from the same base contact, including noise or filter variants, will also remain together. Calibration photographs will be excluded from the held-out real test set, and validation-only calibration references will never enter fitting. Real2Real and Sim2Real will use the same held-out real groups. Results will report accuracy, macro-F1, per-class recall, and confusion matrices, together with variability over repeated training seeds. Any claim about unseen-object generalization will require physical object instances and their corresponding mesh instances to be separated across splits.

The transfer experiments are planned evaluation, not current results. The reference-guided fit will be treated as a fixed dataset-generation step, and an ablation will compare geometric smoothing, RGB blur, and the complete fitted appearance configuration using matched samples and split groups. This distinguishes an improvement in image similarity from an improvement in real-data recognition utility, which is the relevant Sim2Real outcome (Gomes et al., 2021).
