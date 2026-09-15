"""Reference-guided appearance fitting through the production renderer (no Qt)."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares
import trimesh

from .geometric import GeometricSim
from .config import SensorConfig, LEDConfig
from .render.gel import GelLighting
from .render.effects import ImageEffects
from .render.optical import GelSightRenderer


class Cancelled(Exception):
    pass


def appearance(sim):
    return dict(sensor=asdict(sim.cfg), gel_lighting=asdict(sim.gel_lighting),
                effects=asdict(sim.effects), softness=sim.softness)


def apply_appearance(sim, data):
    sim.cfg = SensorConfig.from_dict(deepcopy(data['sensor']))
    sim.gel_lighting = GelLighting.from_dict(data['gel_lighting'])
    sim.effects = ImageEffects(**data['effects'])
    sim.softness = data['softness']
    sim._key = None


def read_photo(path, size):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize(size, Image.Resampling.LANCZOS)).copy()


def make_sim(reference, settings, size):
    sim = GeometricSim()
    apply_appearance(sim, settings)
    sim.cfg.camera.width, sim.cfg.camera.height = size
    sim.set_mesh(trimesh.Trimesh(reference['vertices'], reference['faces'], process=False), reference['name'])
    sim.rotation = tuple(reference['rotation'])
    sim.offset = tuple(reference['offset'])
    sim.cut_depth = -1 if reference['blank'] else reference['cut_depth']
    return sim


def render_float(sim, depth=None, blur=True, stochastic=False, apply_effects=True, shade_cache=None):
    """Production optical pipeline; optionally include seeded camera/gel effects."""
    depth = sim.depth() if depth is None else depth
    cfg = deepcopy(sim.cfg)
    cfg.optics.noise_sigma = 0
    renderer = GelSightRenderer(cfg)
    renderer._flat_cache = sim._flat_lighting_cache
    if shade_cache is not None and not cfg.optics.calibration_path and cfg.optics.spatial_response is None:
        original_shade = renderer.shade
        def cached_shade(normals, optics=None, height=None):
            # Valid only for this reference's fixed depth/normals. RGB weights
            # are linear before clipping, so reuse each light's scalar basis
            # across colour Jacobian columns and unchanged geometric parameters.
            optics = optics or renderer.optics
            rgb = np.zeros(normals.shape[:2]+(3,), dtype=np.float32)
            rgb += np.asarray(optics.ambient, dtype=np.float32)
            for led in optics.leds:
                key = (normals.shape, height is None, bool(height is not None and np.any(height)),
                       led.azimuth, led.elevation, optics.side_lighting, optics.side_distance,
                       optics.side_falloff, optics.diffuse_gain, optics.specular_gain, optics.shininess)
                basis = shade_cache.get(key)
                if basis is None:
                    unit = deepcopy(optics)
                    unit.ambient = (0., 0., 0.)
                    unit.leds = [LEDConfig(color=(1., 1., 1.), intensity=1., azimuth=led.azimuth, elevation=led.elevation)]
                    basis = original_shade(normals, unit, height)[..., 0].copy()
                    if len(shade_cache) >= 64:
                        shade_cache.pop(next(iter(shade_cache)))
                    shade_cache[key] = basis
                rgb += basis[..., None]*(np.asarray(led.color, dtype=np.float32)*led.intensity)
            return rgb
        renderer.shade = cached_shade
    if sim.gel_lighting.enabled:
        renderer.set_background(np.broadcast_to(sim.gel_lighting.background, (*depth.shape, 3)))
    rgb = renderer.render(sim.sampler, -depth, use_gpu=False, as_uint8=False)
    if sim.gel_lighting.enabled:
        rgb = sim.gel_lighting.overlay(rgb)
    rgb = sim.gel_lighting.adjust(rgb)
    if not apply_effects:
        return rgb
    effects = deepcopy(sim.effects)
    if stochastic:
        effects.read_noise = float(np.hypot(effects.read_noise, sim.cfg.optics.noise_sigma))
    else:
        effects.read_noise = effects.shot_noise = effects.speckle = effects.texture = 0
    if not blur:
        effects.blur_px = 0
    return effects.apply(rgb, cfg.gel.size_x, cfg.gel.size_y)


def save_session(path, settings, references, result=None):
    """Self-contained archive: images and processed meshes; never pickle."""
    arrays = {}
    records = []
    for i, ref in enumerate(references):
        records.append({k: v for k, v in ref.items() if k not in ('photo', 'vertices', 'faces')})
        for key in ('photo', 'vertices', 'faces'):
            arrays[f'{i}_{key}'] = ref[key]
    result_meta = None
    if result is not None:
        result_meta = {key: value for key, value in result.items() if key not in ('initial', 'final', 'photos')}
        for key in ('initial', 'final', 'photos'):
            arrays['result_'+key] = np.asarray(result[key])
    arrays['metadata'] = np.array(json.dumps(dict(version=1, settings=settings, references=records, result=result_meta), allow_nan=False))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_session(path, include_result=False):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['version'] != 1:
            raise ValueError('Unsupported calibration session version')
        refs = metadata['references']
        for i, ref in enumerate(refs):
            for key in ('photo', 'vertices', 'faces'):
                ref[key] = archive[f'{i}_{key}'].copy()
            if ref['photo'].ndim != 3 or ref['photo'].shape[2] != 3:
                raise ValueError('Reference must be an RGB image')
            make_sim(ref, metadata['settings'], (32, 24)).depth()
        result = metadata.get('result')
        if result is not None:
            for key in ('initial', 'final', 'photos'):
                result[key] = list(archive['result_'+key].copy())
    return (metadata['settings'], refs, result) if include_result else (metadata['settings'], refs)


def fit_references(settings, references, size=None, fit_background=True,
                   fit_lighting=True, fit_blur=False, max_nfev=35,
                   cancelled=lambda: False, progress=lambda message: None,
                   fit_directions=True, fit_filters=False, fit_noise=False,
                   fit_texture=False, search_starts=2, calibration_seed=0,
                   algorithm='least_squares'):
    if algorithm not in ('least_squares', 'ai_surrogate'):
        raise ValueError('Unknown calibration fitting algorithm: '+str(algorithm))
    if not references or not any(not r['validation'] for r in references):
        raise ValueError('Add at least one fitting reference')
    if any(not r['locked'] for r in references):
        raise ValueError('Lock every reference alignment before fitting')
    if settings['sensor']['optics'].get('calibration_path'):
        raise ValueError('Select analytic lighting (disable the calibration lookup table) before fitting')
    if fit_lighting and settings['sensor']['optics'].get('spatial_response') is not None:
        raise ValueError('This preset uses an empirical spatial response. Refit with tools/fit_spatial_response.py, or clear spatial_response to fit analytic LEDs.')
    if not (fit_background or fit_lighting or fit_blur or fit_filters or fit_noise or fit_texture):
        raise ValueError('Select at least one fitting stage')
    current = deepcopy(settings)
    if size is None:
        camera = current['sensor']['camera']
        size = (camera['width'], camera['height'])
    if fit_filters or fit_noise or fit_texture or fit_blur:
        current['effects']['enabled'] = True
    sims, depths, masks, photos = [], [], [], []
    for ref in references:
        if cancelled():
            raise Cancelled()
        sim = make_sim(ref, current, size)
        depth = sim.depth()
        mask = binary_dilation(sim.raw_depth > 0, iterations=3)
        if not ref['blank'] and not mask.any():
            raise ValueError(f"{ref['name']}: aligned shape has no visible contact")
        sims.append(sim)
        depths.append(depth)
        masks.append(mask)
        photos.append(np.asarray(Image.fromarray(ref['photo']).resize(size, Image.Resampling.LANCZOS), dtype=float)/255)
    # Blur is specified in output pixels, not optimization-preview pixels.
    output_width = current['sensor']['camera']['width']
    scale = size[0] / output_width

    def images(data, training_only=False):
        output = []
        for ref, sim, depth in zip(references, sims, depths):
            if cancelled():
                raise Cancelled()
            if training_only and ref['validation']:
                output.append(None)
                continue
            sim.cfg.optics = SensorConfig.from_dict(data['sensor']).optics
            sim.gel_lighting = GelLighting.from_dict(data['gel_lighting'])
            sim.effects = ImageEffects(**data['effects'])
            sim.effects.blur_px *= scale
            if not hasattr(sim, '_calibration_shade_cache'):
                sim._calibration_shade_cache = {}
            output.append(render_float(sim, depth, shade_cache=sim._calibration_shade_cache))
        return output

    initial = images(settings)
    stages = []
    if fit_background:
        if not current['gel_lighting']['enabled']:
            raise ValueError('Enable the gel background before fitting it')
        stages.append('Background')
    if fit_lighting:
        if not any(not r['blank'] and not r['validation'] for r in references):
            raise ValueError('Lighting fitting needs a training contact reference')
        stages.append('Lighting')
    if fit_filters:
        stages.append('Filters')
    elif fit_blur:
        stages.append('Blur')
    reports = []

    def image_residual(data, stage):
        parts = []
        for ref, predicted, photo, mask in zip(references, images(data, training_only=True), photos, masks):
            if ref['validation']:
                continue
            regions = [~mask] if stage == 'Background' else [mask, ~mask]
            regions = [region for region in regions if region.any()]
            for region in regions:
                error = (predicted-photo)[region]
                # Bound Jacobian memory at high resolution. Deterministic sample
                # locations remain fixed throughout each stage and each restart.
                # Sample pixels before flattening: a stride divisible by three
                # in flattened RGB silently drops two entire colour channels.
                error = error[::max(1, int(np.ceil(len(error)/2000)))].ravel()
                parts.append(error*np.sqrt(1000/(len(error)*len(regions))))
        if not parts:
            raise ValueError('No visible background pixels; add an empty-sensor reference')
        return np.concatenate(parts)

    def robust_cost(residual):
        return float(np.mean(2*.03**2*(np.sqrt(1+(residual/.03)**2)-1)))

    for stage in stages:
        base = deepcopy(current)
        initial_stage_cost = robust_cost(image_residual(base, stage))
        edge_trials = 0
        if stage == 'Lighting' and fit_directions:
            # Azimuth is a categorical edge in this renderer. Continuous
            # derivatives are identically zero between edge boundaries.
            best_cost = initial_stage_cost
            for sweep in range(2):
                improved = False
                for led_index in range(len(base['sensor']['optics']['leds'])):
                    side_mode = base['sensor']['optics']['side_lighting']
                    progress(f'Lighting: {"edge" if side_mode else "azimuth"} search {sweep+1}/2, light {led_index+1}')
                    selected = base
                    off = deepcopy(base)
                    off['sensor']['optics']['leds'][led_index]['intensity'] = 0.
                    without_light = images(off, training_only=True)
                    azimuth = base['sensor']['optics']['leds'][led_index]['azimuth']
                    candidates = (0., 90., 180., 270.) if side_mode else tuple((azimuth+d+180)%360-180 for d in (-180, -90, 0, 90))
                    for edge in candidates:
                        candidate = deepcopy(base)
                        candidate['sensor']['optics']['leds'][led_index]['azimuth'] = edge
                        cost = robust_cost(image_residual(candidate, stage))
                        edge_trials += 1
                        if cost < best_cost:
                            selected, best_cost, improved = candidate, cost, True
                        # Conditional RGB projection gives each direction a fair
                        # trial even if this source starts switched off or with
                        # the wrong colour. The nonlinear objective still decides
                        # acceptance; clipping/tone curves make this a proposal,
                        # not an exact linear solve of the complete renderer.
                        unit = deepcopy(candidate)
                        unit['sensor']['optics']['leds'][led_index].update(color=[1., 1., 1.], intensity=1.)
                        numerator, denominator = np.zeros(3), np.zeros(3)
                        for ref, lit, unlit, photo, mask in zip(references, images(unit, training_only=True), without_light, photos, masks):
                            if ref['validation'] or not mask.any():
                                continue
                            basis = (lit-unlit)[mask]
                            target = (photo-unlit)[mask]
                            numerator += np.mean(basis*target, axis=0)
                            denominator += np.mean(basis*basis, axis=0)
                        weights = np.clip(numerator/np.maximum(denominator, 1e-10), 0., 5.)
                        intensity = max(float(weights.max()), 1e-8)
                        unit['sensor']['optics']['leds'][led_index].update(color=list(weights/intensity), intensity=intensity)
                        projected_cost = robust_cost(image_residual(unit, stage))
                        if projected_cost < best_cost:
                            selected, best_cost, improved = unit, projected_cost, True
                    base = selected
                if not improved:
                    break
        if stage == 'Background':
            g = base['gel_lighting']
            x0 = list(g['background']) + [g['vignette']]
            low, high = [0.]*4, [1.]*4
            for glow in g['glows']:
                x0 += list(glow['color']) + [glow[k] for k in ('strength', 'x', 'y', 'width', 'height', 'angle')]
                low += [0.]*6 + [.01, .01, -180.]
                high += [1.]*8 + [180.]
            bounds = (low, high)
        elif stage == 'Lighting':
            optics = base['sensor']['optics']
            light_count = len(optics['leds'])
            x0 = list(np.array([np.array(led['color'])*led['intensity'] for led in optics['leds']]).ravel())
            if not len(x0):
                raise ValueError('Add at least one light source')
            low, high = [0.]*len(x0), [5.]*len(x0)
            if fit_directions:
                for led in optics['leds']:
                    x0.append(led['elevation'])
                    low.append(0.)
                    high.append(85.)
                    if not optics['side_lighting']:
                        x0.append(led['azimuth'])
                        low.append(-360.)
                        high.append(360.)
                if optics['side_lighting']:
                    x0 += [optics['side_distance'], optics['side_falloff']]
                    low += [1., 0.]
                    high += [3., 2.]
            bounds = (low, high)
        elif stage == 'Filters':
            x0 = [max(1., base['effects']['blur_px']), base['effects']['vignette'],
                  base['effects']['contrast'], base['effects']['gamma'],
                  base['gel_lighting']['brightness']]
            bounds = ([0., 0., .1, .1, 0.], [5., 1., 3., 3., 3.])
        else:
            x0, bounds = [max(2., base['effects']['blur_px'])], ([0.], [5.])

        def unpack(x):
            data = deepcopy(base)
            if stage == 'Background':
                data['gel_lighting']['background'] = list(x[:3])
                data['gel_lighting']['vignette'] = float(x[3])
                for i, glow in enumerate(data['gel_lighting']['glows']):
                    start = 4+i*9
                    glow['color'] = list(x[start:start+3])
                    for key, value in zip(('strength', 'x', 'y', 'width', 'height', 'angle'), x[start+3:start+9]):
                        glow[key] = float(value)
            elif stage == 'Lighting':
                for led, weight in zip(data['sensor']['optics']['leds'], np.asarray(x[:3*light_count]).reshape(-1, 3)):
                    intensity = max(float(max(weight)), 1e-8)
                    led['intensity'], led['color'] = intensity, list(weight/intensity)
                if fit_directions:
                    position = 3*light_count
                    for led in data['sensor']['optics']['leds']:
                        led['elevation'] = float(x[position])
                        position += 1
                        if not optics['side_lighting']:
                            led['azimuth'] = float(x[position])
                            position += 1
                    if optics['side_lighting']:
                        data['sensor']['optics']['side_distance'] = float(x[position])
                        data['sensor']['optics']['side_falloff'] = float(x[position+1])
            elif stage == 'Filters':
                for key, value in zip(('blur_px', 'vignette', 'contrast', 'gamma'), x):
                    data['effects'][key] = float(value)
                data['gel_lighting']['brightness'] = float(x[4])
            else:
                data['effects']['blur_px'] = float(x[0])
            return data

        calls = 0
        def residual(x):
            nonlocal calls
            calls += 1
            if calls % 5 == 1:
                progress(f'{stage}: evaluation {calls}')
            return image_residual(unpack(x), stage)

        # Scale all parameters to [0,1]. Use absolute finite differences so zero
        # weights, zero blur and zero-degree directions have usable derivatives.
        lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
        def scaled_residual(u):
            return residual(lo+u*(hi-lo))
        def jacobian(u):
            baseline = scaled_residual(u)
            columns = []
            for j in range(len(u)):
                shifted = u.copy()
                step = .001 if u[j] <= .999 else -.001
                shifted[j] += step
                columns.append((scaled_residual(shifted)-baseline)/step)
            return np.column_stack(columns)

        rng = np.random.default_rng(calibration_seed)
        start = np.clip((np.asarray(x0)-lo)/(hi-lo), 1e-6, 1-1e-6)
        base_cost = robust_cost(image_residual(base, stage))
        best_data, best_cost = deepcopy(base), base_cost
        surrogate_trials = 0
        if algorithm == 'ai_surrogate':
            from .calibration_search import surrogate_search
            def check_cancel():
                if cancelled():
                    raise Cancelled()
            start, cost, surrogate_trials = surrogate_search(
                lambda u: robust_cost(scaled_residual(u)), start,
                max(8, max_nfev), rng, check_cancel,
                lambda message: progress(f'{stage}: {message}'))
            if cost < best_cost:
                best_data, best_cost = unpack(lo+start*(hi-lo)), cost
        attempts = max(1, search_starts) if stage == 'Lighting' and fit_directions else 1
        for attempt in range(attempts):
            initial_u = start.copy()
            if attempt:
                initial_u[3*light_count:] = np.clip(initial_u[3*light_count:]+rng.normal(0, .15, len(initial_u)-3*light_count), .001, .999)
            progress(f'{stage}: bounded robust refinement {attempt+1}/{attempts}')
            result = least_squares(scaled_residual, initial_u, jac=jacobian,
                                   bounds=(np.zeros(len(start)), np.ones(len(start))), loss='soft_l1',
                                   f_scale=.03, max_nfev=max_nfev, ftol=1e-5, x_scale='jac')
            candidate = unpack(lo+result.x*(hi-lo))
            cost = robust_cost(image_residual(candidate, stage))
            if cost < best_cost:
                best_data, best_cost = candidate, cost
        current = best_data
        reports.append(dict(stage=stage, evaluations=calls, message=result.message, resolution=size,
                            settings=deepcopy(current),
                            algorithm=('Gaussian-process surrogate search + bounded robust least squares' if algorithm == 'ai_surrogate'
                                       else 'Bounded robust trust-region least squares; scaled absolute-difference Jacobian'),
                            surrogate_trials=surrogate_trials,
                            starts=attempts, edge_trials=edge_trials, initial_cost=initial_stage_cost,
                            final_cost=best_cost, accepted=bool(best_cost < initial_stage_cost), seed=calibration_seed))

    if fit_noise or fit_texture:
        from .calibration_noise import flat_patches, estimate_noise, fit_texture as estimate_texture
        records = []
        for i, ref in enumerate(references):
            if ref['validation']:
                continue
            if cancelled():
                raise Cancelled()
            progress(f'Noise / texture: preparing native reference {i+1}/{len(references)}')
            native = (ref['photo'].shape[1], ref['photo'].shape[0])
            sim = make_sim(ref, current, native)
            sim.effects.blur_px *= native[0]/output_width
            clean = render_float(sim)
            photo = np.asarray(ref['photo'], dtype=float)/255
            records.append(dict(reference=ref, photo=photo, clean=clean,
                                base=render_float(sim, apply_effects=False), regions=flat_patches(photo, clean)))
        if fit_noise:
            noise_effects = ImageEffects(**current['effects'])
            noise_effects.read_noise = float(np.hypot(noise_effects.read_noise, current['sensor']['optics']['noise_sigma']))
            estimate = estimate_noise(records, noise_effects, cancelled, progress)
            if estimate is None:
                raise Cancelled()
            parameters, report = estimate
            current['effects'].update(parameters)
            # The app combines optical/read noise in quadrature; store only one.
            current['sensor']['optics']['noise_sigma'] = 0.
            reports.append(report)
        if fit_texture:
            try:
                parameters, report = estimate_texture(records, current, cancelled, progress,
                    seed=calibration_seed, iterations=max(4, min(25, max_nfev//3)))
            except InterruptedError as exc:
                raise Cancelled() from exc
            current['effects'].update(parameters)
            reports.append(report)
    # Verify through fresh, uncached production renders. Only training photos
    # decide acceptance; held-out photos remain evaluation-only.
    def verified_images(data):
        output = []
        for ref in references:
            if cancelled():
                raise Cancelled()
            sim = make_sim(ref, data, size)
            sim.effects.blur_px *= scale
            output.append(render_float(sim))
        return output

    initial, final = verified_images(settings), verified_images(current)
    def training_errors(images):
        errors = [rgb-photo for ref, rgb, photo in zip(references, images, photos)
                  if not ref['validation']]
        return (float(np.mean([np.abs(e).mean() for e in errors])),
                float(np.mean([(e*e).mean() for e in errors])))
    before_error, after_error = training_errors(initial), training_errors(final)
    accepted = all(np.isfinite(b) and b <= a + 1e-7
                   for a, b in zip(before_error, after_error))
    if not accepted:
        current, final = deepcopy(settings), initial
    reports.append(dict(stage='Production verification', accepted=accepted,
        before_mae=before_error[0], candidate_mae=after_error[0],
        message=('Training MAE and MSE did not increase.' if accepted else
                 'Rejected degraded fit; original settings retained. Recheck reference alignment and fitting options.')))
    metrics = []
    for ref, photo, before, after, mask in zip(references, photos, initial, final, masks):
        record = dict(name=ref['name'], validation=ref['validation'])
        for label, rgb in [('before', before), ('after', after)]:
            record[label+'_mae'] = float(np.abs(rgb-photo).mean())
            record[label+'_rmse'] = float(np.sqrt(np.mean((rgb-photo)**2)))
            record[label+'_contact_mae'] = float(np.abs(rgb-photo)[mask].mean()) if mask.any() else None
        metrics.append(record)
    return dict(settings=current, initial=initial, final=final, photos=photos,
                metrics=metrics, stages=reports, size=size,
                options=dict(algorithm=algorithm, fit_background=fit_background, fit_lighting=fit_lighting,
                    fit_directions=fit_directions, fit_filters=fit_filters, fit_blur=fit_blur,
                    fit_noise=fit_noise, fit_texture=fit_texture, search_starts=search_starts,
                    calibration_seed=calibration_seed, max_nfev=max_nfev),
                metric_note='Pixel MAE/RMSE compare deterministic images. Noise/texture are fitted statistically at native photo resolution.')
