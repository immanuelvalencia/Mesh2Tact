"""Measured background and robust position/normal response calibration.

Geometry is locked by the collection. This estimates appearance, not LED
positions, indentation depths, or membrane mechanics.
"""
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt

from .render.optical import spatial_features
from .render.boundary import boundary_height


def fit_spatial(settings, references, size=None, cancelled=lambda: False,
                progress=lambda message: None, **options):
    from .calibration import Cancelled, make_sim, render_float, fit_references
    start = perf_counter()
    indentation = options.pop('indentation_response', False)
    training = [r for r in references if not r['validation']]
    blanks = [r for r in training if r['blank']]
    if not training or any(not r['locked'] for r in references):
        raise ValueError('Save and lock every reference; include training references')
    if not any(options[k] for k in ('fit_background', 'fit_lighting', 'fit_filters', 'fit_blur', 'fit_noise', 'fit_texture')):
        raise ValueError('Select at least one fitting stage')
    if options['fit_background'] and not blanks:
        raise ValueError('Measured background fitting needs a training No shape reference')
    if options['fit_lighting'] and not any(not r['blank'] for r in training):
        raise ValueError('Contact response fitting needs a training contact reference')
    current = deepcopy(settings)
    from .config import SensorConfig
    current['sensor'] = asdict(SensorConfig.from_dict(current['sensor']))
    camera = settings['sensor']['camera']
    size = tuple(size or (camera['width'], camera['height']))
    scale = size[0]/camera['width']
    stages, sims, depths, masks, photos, boundaries = [], [], [], [], [], []

    def check():
        if cancelled():
            raise Cancelled()

    for ref in references:
        check()
        sim = make_sim(ref, settings, size)
        depth = sim.depth()
        mask = binary_dilation(sim.raw_depth > 0, iterations=3)
        contact = sim.raw_depth > 1e-8
        spacing = (sim.cfg.gel.size_y/size[1], sim.cfg.gel.size_x/size[0])
        field_options = current['sensor']['optics']
        radius = 3*(max(.35, field_options['boundary_width_mm'])+
                    max(.8, field_options['boundary_growth'])*float(sim.raw_depth.max())*1000)/1000
        boundary = np.zeros_like(contact)
        if contact.any() and not contact.all():
            outside = distance_transform_edt(~contact, sampling=spacing)
            inside = distance_transform_edt(contact, sampling=spacing)
            boundary = np.where(contact, inside, outside) <= radius
        boundaries.append(boundary)
        if not ref['blank'] and not mask.any():
            raise ValueError(ref['name']+': aligned shape has no visible contact')
        sims.append(sim); depths.append(depth); masks.append(mask)
        photos.append(np.asarray(Image.fromarray(ref['photo']).resize(size, Image.Resampling.LANCZOS), dtype=np.float32)/255)

    def renders(data, fresh=False):
        from .config import SensorConfig
        from .render.gel import GelLighting
        from .render.effects import ImageEffects
        output = []
        for ref, sim, depth in zip(references, sims, depths):
            check()
            if fresh:
                sim = make_sim(ref, data, size)
                depth = sim.depth()
            else:
                sim.cfg.optics = SensorConfig.from_dict(data['sensor']).optics
                sim.gel_lighting = GelLighting.from_dict(data['gel_lighting'])
                sim.effects = ImageEffects(**data['effects'])
            sim.effects.blur_px = data['effects']['blur_px']*scale
            output.append(render_float(sim, depth))
        return output

    initial = renders(settings)
    optics = current['sensor']['optics']
    if options['fit_background']:
        progress('Background: measuring the fixed empty-pad colour field')
        # Native photos are reduced before combining, bounding memory and
        # removing texture rather than storing it as part of illumination.
        grid_size = (min(96, size[0]), min(72, size[1]))
        grids = [np.asarray(Image.fromarray(r['photo']).resize(grid_size, Image.Resampling.BOX), dtype=np.float32)/255 for r in blanks]
        optics['calibrated_background'] = np.median(grids, axis=0).tolist()
        optics['calibrated_background_enabled'] = True
        # The grid already contains measured exposure and colour processing.
        # Reset these once to avoid applying the previous tone curve twice.
        optics.update(exposure=1., reference_background=False, calibration_path=None,
                      depth_shading=0., depth_relief=0., shadow_strength=0.)
        current['gel_lighting'].update(brightness=1., contrast=1., saturation=1., hue=0., gamma=1., preserve_contact=True)
        current['effects'].update(contrast=1., gamma=1., vignette=0., blur_px=0.)
        stages.append(dict(stage='Background', accepted=True, settings=deepcopy(current),
                           algorithm='Median of reduced training empty-pad images'))

    if options['fit_lighting']:
        if optics.get('calibrated_background') is None:
            raise ValueError('Fit the measured background first, or load a measured-background preset')
        # Start the response at zero and fit signed RGB changes around the base.
        optics.update(spatial_response=np.zeros((35, 3)).tolist(), calibration_path=None,
                      spatial_response_enabled=True,
                      contact_depth_response=None, spatial_slope_damping=0.,
                      depth_shading=0., depth_relief=0., shadow_strength=0.)
        if indentation:
            # Prevent an old brightness/shadow overlay from being absorbed into
            # a new learned response and then applied a second time at runtime.
            optics.update(boundary_enabled=True, boundary_gain=1.,
                          boundary_response=np.zeros((35, 3)).tolist(),
                          depth_overlay_enabled=False, perimeter_shadow_enabled=False,
                          exposure=1.)
            current['gel_lighting'].update(brightness=1., contrast=1., saturation=1., hue=0., gamma=1.)
            current['effects'].update(contrast=1., gamma=1., vignette=0., blur_px=0.)
        else:
            optics.update(boundary_enabled=False, boundary_response=None)
        baseline = renders(current)
        samples, targets = [], []
        for ref, sim, depth, mask, boundary, photo, base in zip(references, sims, depths, masks, boundaries, photos, baseline):
            check()
            if ref['validation'] or ref['blank']:
                continue
            # All three channels use the same deterministic pixel sample.
            # Balance the contact and surrounding band, independently of area.
            regions = [mask, boundary & ~mask] if indentation else [mask]
            selected, weights = [], []
            for region in regions:
                indices = np.flatnonzero(region)[::max(1, int(np.ceil(region.sum()/2000)))]
                if len(indices):
                    selected.append(indices)
                    weights.append(np.full(len(indices), 1/np.sqrt(len(indices))))
            indices, weight = np.concatenate(selected), np.concatenate(weights)
            features = spatial_features(sim.sampler.normals(-depth), indices, terms=35)
            target = (photo-base).reshape(-1, 3)[indices]
            samples.append((sim, indices, weight, features))
            targets.append(target*weight[:, None])
        B = np.concatenate(targets)
        penalty = np.repeat([.0001, .001, .001, .01, .01, .01, .0001], 5)
        candidates = [(0., 0.)]
        if indentation:
            candidates = [(width, growth) for width in (.08, .2, .35) for growth in (.2, .5, .8)]
            candidates.append((optics['boundary_width_mm'], optics['boundary_growth']))
            penalty = np.r_[penalty, penalty*2]
        best, search = None, []
        for width, growth in dict.fromkeys(candidates):
            check()
            progress(f'Contact and boundary response: width {width:g} mm, growth {growth:g}')
            matrices = []
            for sim, indices, weight, features in samples:
                if indentation:
                    spread = boundary_height(sim.raw_depth, sim.cfg.gel.size_x,
                                             sim.cfg.gel.size_y, width, growth)
                    features = np.c_[features, spatial_features(sim.sampler.normals(spread), indices, terms=35)]
                matrices.append(features*weight[:, None])
            A = np.concatenate(matrices)
            weights = np.ones(len(A))
            for iteration in range(5):
                check()
                lhs = A.T @ (weights[:, None]*A) + np.diag(penalty)
                coef = np.linalg.solve(lhs, A.T @ (weights[:, None]*B))
                error = np.linalg.norm(A@coef-B, axis=1)
                robust_scale = max(float(np.median(error))*1.4826, 1e-6)
                weights = np.minimum(1., 1.5*robust_scale/np.maximum(error, 1e-12))
            score = float(np.mean(np.abs(A@coef-B)))
            search.append(dict(width_mm=width, growth=growth, sampled_error=score))
            if best is None or score < best[0]:
                best = (score, coef.copy(), width, growth, int(np.linalg.matrix_rank(A)))
        _, coef, width, growth, rank = best
        optics['spatial_response'] = coef[:35].tolist()
        if indentation:
            optics.update(boundary_response=coef[35:].tolist(), boundary_width_mm=width, boundary_growth=growth)
        warnings = []
        contact_refs = [r for r in training if not r['blank']]
        if indentation and len({round(r['cut_depth'], 7) for r in contact_refs}) < 3:
            warnings.append('Fewer than three training indentations: boundary growth is weakly constrained. Add a depth sweep.')
        if len({tuple(r['offset']) for r in contact_refs}) < 3:
            warnings.append('Fewer than three training positions: spatial response away from these contacts is unvalidated.')
        if rank < len(coef):
            warnings.append(f'Design rank {rank}/{len(coef)}: regularization supplies unconstrained coefficient directions.')
        stages.append(dict(stage='Contact response', accepted=True, settings=deepcopy(current),
            algorithm='Signed contact and indentation boundary response' if indentation else 'Position-conditioned normal polynomial',
            sampled_pixels=len(A), regularization=penalty.tolist(),
            design_rank=rank, boundary_search=search, warnings=warnings))

    if any(options[k] for k in ('fit_filters', 'fit_blur', 'fit_noise', 'fit_texture')):
        progress('Refining camera effects after background and contact response')
        refined = fit_references(current, references, size=size, fit_background=False,
            fit_lighting=False, fit_directions=False, fit_filters=options['fit_filters'],
            fit_blur=options['fit_blur'], fit_noise=options['fit_noise'],
            fit_texture=options['fit_texture'], max_nfev=options['max_nfev'],
            calibration_seed=options['calibration_seed'], cancelled=cancelled, progress=progress)
        current = refined['settings']
        for report in refined['stages']:
            report = deepcopy(report)
            if report['stage'] == 'Production verification':
                report['stage'] = 'Filter verification'
            stages.append(report)

    progress('Verifying fresh production renders and training contact errors')
    final = renders(current, fresh=True)
    def errors(images):
        whole, contact, edge = [], [], []
        for ref, rgb, photo, mask, boundary in zip(references, images, photos, masks, boundaries):
            if ref['validation']:
                continue
            e = rgb-photo
            whole.append((np.abs(e).mean(), (e*e).mean()))
            if indentation and not ref['blank'] and boundary.any():
                be = e[boundary]
                edge.append((np.abs(be).mean(), (be*be).mean()))
            if not ref['blank'] and mask.any():
                e = e[mask]
                contact.append((np.abs(e).mean(), (e*e).mean()))
        return np.concatenate([np.mean(whole, axis=0)] + ([np.mean(contact, axis=0)] if contact else []) + ([np.mean(edge, axis=0)] if edge else []))
    before, after = errors(initial), errors(final)
    accepted = bool(np.isfinite(after).all() and np.all(after <= before+1e-7))
    if not accepted:
        current, final = deepcopy(settings), initial
    stages.append(dict(stage='Production verification', accepted=accepted,
        before_errors=before.tolist(), candidate_errors=after.tolist(),
        message='Training whole-image, contact and (in indentation mode) boundary MAE/MSE checked; original settings retained on regression.'))
    metrics = []
    for ref, photo, b, a, mask, boundary in zip(references, photos, initial, final, masks, boundaries):
        record = dict(name=ref['name'], validation=ref['validation'])
        for label, rgb in [('before', b), ('after', a)]:
            e = rgb-photo
            record[label+'_mae'] = float(np.abs(e).mean())
            record[label+'_rmse'] = float(np.sqrt((e*e).mean()))
            record[label+'_contact_mae'] = float(np.abs(e[mask]).mean()) if mask.any() else None
            record[label+'_boundary_mae'] = float(np.abs(e[boundary]).mean()) if boundary.any() else None
        metrics.append(record)
    return dict(settings=current, initial=initial, final=final, photos=photos,
        metrics=metrics, stages=stages, size=size,
        options=dict(options, algorithm='indentation_response' if indentation else 'spatial_response'), seconds=perf_counter()-start,
        metric_note='Empirical appearance fit using locked supplied geometry. Validation references never fit the background, response or filters. Position/slope extrapolation is unvalidated; image-aligned geometry is not independent geometric validation.')
