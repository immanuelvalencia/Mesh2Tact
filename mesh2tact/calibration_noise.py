"""Native-pixel noise estimation and seeded spatial texture matching.

Estimates appearance parameters, not a unique physical camera noise model.
Repeated exposures cancel fixed texture; single frames use a plane-annihilating
high-pass filter and conservative flat-patch selection.
"""
from copy import deepcopy

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.optimize import least_squares, differential_evolution

from .render.effects import ImageEffects, TextureCache


HIGH_PASS = np.outer([1., -2., 1.], [1., -2., 1.])/6


def inverse_tone(rgb, effects):
    if effects.contrast < 1e-4:
        raise ValueError('Noise estimation needs nonzero image contrast')
    return (np.asarray(rgb, dtype=float)**effects.gamma-.5)/effects.contrast+.5


def flat_patches(photo, clean, count=24):
    """Select unsaturated, smooth patches by deterministic image gradient."""
    h, w = photo.shape[:2]
    width = min(32, h//2, w//2)
    if width < 8:
        raise ValueError('Noise estimation needs reference photos at least 16 × 16 pixels')
    smooth = gaussian_filter(clean, (2, 2, 0))
    gy, gx = np.gradient(smooth, axis=(0, 1))
    gradient = np.sqrt(np.mean(gx*gx+gy*gy, axis=2))
    choices = []
    for y in range(0, h-width+1, width):
        for x in range(0, w-width+1, width):
            region = (slice(y, y+width), slice(x, x+width))
            patch = photo[region]
            if np.mean((patch > .02) & (patch < .98)) < .98:
                continue
            score = float(np.mean(gradient[region]))
            if score < .012:
                choices.append((score, y, x, region))
    # Keep a spread of intensities amongst smooth areas, rather than one corner.
    choices.sort(key=lambda item: item[0])
    pool = choices[:max(count, len(choices)//2)]
    if len(pool) > count:
        pool = [pool[i] for i in np.linspace(0, len(pool)-1, count).astype(int)]
    if len(pool) < 2:
        raise ValueError('Too few smooth, unsaturated patches. Add empty-sensor photos or gentler contacts.')
    return [item[3] for item in pool]


def variance_rows(signal, residual, regions):
    means, variances, cross = [], [], []
    for region in regions:
        # Filter borders are omitted; median absolute deviation resists edges.
        error = residual[region][2:-2, 2:-2].reshape(-1, 3)
        error = error-np.median(error, axis=0)
        sigma = np.maximum(np.median(np.abs(error), axis=0)/.67448975, 1e-5)
        winsor = np.clip(error, -4*sigma, 4*sigma)
        means.append(np.mean(signal[region], axis=(0, 1)))
        variances.append(sigma*sigma)
        cross.append([np.mean(winsor[:, a]*winsor[:, b]) for a, b in ((0, 1), (0, 2), (1, 2))])
    return np.asarray(means), np.asarray(variances), np.asarray(cross)


def estimate_noise(records, effects, cancelled, progress):
    """Fit read/shot/speckle variance and shared-channel covariance."""
    rows = []
    groups = {}
    for record in records:
        group = record['reference'].get('repeat_group', '').strip()
        if group:
            groups.setdefault(group, []).append(record)
    used = set()
    paired = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        anchor = group[0]
        for other in group[1:]:
            if cancelled():
                return None
            if anchor['photo'].shape != other['photo'].shape:
                raise ValueError('Photos in a repeat group must have identical pixel dimensions')
            if np.mean(np.abs(anchor['clean']-other['clean'])) > .005:
                raise ValueError('Repeat groups must use the same shape, pose and cut depth')
            difference = gaussian_filter(anchor['photo']-other['photo'], (3, 3, 0))
            if np.mean(np.abs(difference)) > .015:
                raise ValueError('Repeat photos differ in position or exposure. Use unchanged-contact exposures, or leave their repeat group empty.')
            a, b = inverse_tone(anchor['photo'], effects), inverse_tone(other['photo'], effects)
            signal = (a+b)/2
            rows.append(variance_rows(signal, (a-b)/np.sqrt(2), anchor['regions']))
            used.update((id(anchor), id(other)))
            paired += 1
    single = 0
    for record in records:
        if id(record) in used:
            continue
        if cancelled():
            return None
        photo = inverse_tone(record['photo'], effects)
        residual = convolve(photo, HIGH_PASS[..., None], mode='reflect')
        rows.append(variance_rows(gaussian_filter(photo, (2, 2, 0)), residual, record['regions']))
        single += 1
    means, variances, cross = [np.concatenate([row[i] for row in rows]) for i in range(3)]
    # Keep the variance components nonnegative. Robust relative residuals stop
    # bright regions dominating, while RGB covariance helps separate speckle.
    def residual(coefficients):
        read, shot, speckle = coefficients
        # Speckle follows Poisson sampling in ImageEffects, so it also scales
        # shot variance (the small product term matters at strong noise).
        predicted = read + shot*means*(1+speckle) + speckle*means**2
        diagonal = (predicted-variances)/np.maximum(variances, 2e-6)
        shared = np.stack([speckle*means[:, a]*means[:, b] for a, b in ((0, 1), (0, 2), (1, 2))], axis=1)
        off_diagonal = (shared-cross)/np.maximum(np.mean(variances, axis=1, keepdims=True), 2e-6)
        return np.concatenate((diagonal.ravel(), off_diagonal.ravel()))
    progress('Noise: robust intensity-dependent variance and RGB covariance fit')
    upper = np.array([.2, .2, .5])**2
    best = None
    for seed in ([1e-4, 1e-5, 1e-5], [1e-6, 1e-3, 1e-6], [1e-6, 1e-6, 1e-3]):
        if cancelled():
            return None
        result = least_squares(residual, seed, bounds=(np.zeros(3), upper), x_scale='jac',
                               loss='soft_l1', f_scale=.5, max_nfev=150)
        if best is None or result.cost < best.cost:
            best = result
    values = np.sqrt(best.x)
    # Sub-quantisation estimates are not useful; avoid adding noise to clean data.
    values[values < 1/(255*np.sqrt(12))] = 0
    warnings = []
    if single:
        warnings.append('Single-frame noise estimates may include fine surface detail or model mismatch; repeated unchanged contacts are more reliable.')
    if np.ptp(means) < .15:
        warnings.append('Limited intensity range: read and shot noise are weakly distinguishable.')
    def cost(coefficients):
        errors = residual(coefficients)/.5
        return float(.5**2*np.sum(np.sqrt(1+errors*errors)-1))
    initial_cost = cost(np.array([effects.read_noise, effects.shot_noise, effects.speckle])**2)
    return dict(read_noise=float(values[0]), shot_noise=float(values[1]), speckle=float(values[2])), dict(
        stage='Noise', algorithm='Robust bounded variance-component regression with RGB covariance',
        repeat_pairs=paired, single_frames=single, patches=len(means), warnings=warnings,
        initial_cost=initial_cost, final_cost=cost(values**2), native_pixel_statistics=True)


def spatial_statistics(rgb, clean, regions):
    """Band-pass residual powers: phase independent, shared-channel texture."""
    relative = np.mean((rgb-clean)/np.maximum(clean, .1), axis=2)
    # Remove broad illumination mismatch. Do not match a random texture pattern.
    relative -= gaussian_filter(relative, 12)
    summaries = []
    for sigma in (1., 2., 4.):
        band = gaussian_filter(relative, sigma)-gaussian_filter(relative, sigma*2)
        patches = []
        for region in regions:
            p = band[region][2:-2, 2:-2]
            patches.append(float(np.mean(np.minimum(p*p, np.quantile(p*p, .98)))))
        summaries.append(float(np.median(patches)))
    return np.asarray(summaries)


def fit_texture(records, settings, cancelled, progress, seed=0, iterations=12):
    """Seeded differential evolution on texture amplitude and physical scale.

Uses the actual ImageEffects pipeline at native resolution with two fixed
realizations. The objective compares residual band powers, not random pixels.
"""
    effects = ImageEffects(**settings['effects'])
    effects.read_noise = float(np.hypot(effects.read_noise, settings['sensor']['optics']['noise_sigma']))
    sx, sy = settings['sensor']['gel']['size_x'], settings['sensor']['gel']['size_y']
    targets = [spatial_statistics(r['photo'], r['clean'], r['regions']) for r in records]
    calls = 0
    texture_cache = TextureCache()

    def objective(parameters):
        nonlocal calls
        if cancelled():
            # Caller supplies the application's cancellation exception.
            raise InterruptedError('Texture fitting cancelled')
        calls += 1
        if calls % 5 == 1:
            progress(f'Texture: statistical search evaluation {calls}')
        effects.texture, effects.texture_scale_mm = float(parameters[0]), float(np.exp(parameters[1]))
        errors = []
        for record, target in zip(records, targets):
            if cancelled():
                raise InterruptedError('Texture fitting cancelled')
            simulated = []
            for fixed_seed in (seed+137, seed+911):
                effects.seed = fixed_seed
                effects.blur_px = settings['effects']['blur_px']*record['photo'].shape[1]/settings['sensor']['camera']['width']
                rgb = effects.apply(record['base'], sx, sy, texture_cache=texture_cache)
                simulated.append(spatial_statistics(rgb, record['clean'], record['regions']))
            prediction = np.mean(simulated, axis=0)
            log_error = np.log((prediction+1e-7)/(target+1e-7))
            errors.append(np.mean(2*(np.sqrt(1+log_error*log_error)-1)))
        return float(np.mean(errors))

    initial = [effects.texture, np.log(effects.texture_scale_mm)]
    initial_cost = objective(initial)
    result = differential_evolution(objective, [(0., .5), (np.log(.01), np.log(2.))],
        seed=seed, popsize=5, maxiter=iterations, polish=False, tol=.02, x0=initial)
    accepted = result.fun < initial_cost
    chosen = result.x if accepted else initial
    return dict(texture=float(chosen[0]), texture_scale_mm=float(np.exp(chosen[1]))), dict(
        stage='Texture', algorithm='Seeded differential evolution of multiscale residual powers',
        evaluations=calls, initial_cost=initial_cost, final_cost=float(min(result.fun, initial_cost)),
        accepted=bool(accepted), seed=seed, warnings=[
            'Texture is an effective spatial appearance estimate; residual geometry/lighting error can inflate it. Statistics use native photo pixels.'])
