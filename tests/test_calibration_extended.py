from copy import deepcopy

import numpy as np
import pytest

from mesh2tact.calibration import (appearance, render_float, fit_references, Cancelled,
    make_sim, save_session, load_session)
from mesh2tact.calibration_noise import flat_patches, estimate_noise, fit_texture
from mesh2tact.geometric import GeometricSim
from mesh2tact.config import LEDConfig
from mesh2tact.render.effects import ImageEffects


def contact(sim, blank=False, validation=False):
    sim.cut_depth = -.001 if blank else .0008
    return dict(name='reference', shape='sphere', vertices=np.asarray(sim.mesh.vertices),
        faces=np.asarray(sim.mesh.faces), rotation=list(sim.rotation), offset=list(sim.offset),
        cut_depth=sim.cut_depth, blank=blank, validation=validation, locked=True,
        photo=(render_float(sim, stochastic=True)*255).astype(np.uint8))


@pytest.mark.parametrize('side', [True, False])
def test_wrong_light_direction_is_recovered(side):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 48, 36
    sim.cfg.optics.leds = [LEDConfig(color=(1., .2, .1), azimuth=90 if side else 60, elevation=30, intensity=.8)]
    sim.cfg.optics.side_lighting = side
    sim.cfg.optics.specular_gain = 0
    ref = contact(sim)
    settings = appearance(sim)
    settings['sensor']['optics']['leds'][0]['azimuth'] = 270 if side else -120
    settings['sensor']['optics']['leds'][0]['elevation'] = 60
    result = fit_references(settings, [ref], size=(48, 36), fit_background=False,
        fit_directions=True, search_starts=2, max_nfev=20)
    assert result['metrics'][0]['after_contact_mae'] < result['metrics'][0]['before_contact_mae']*.5
    fitted = result['settings']['sensor']['optics']['leds'][0]
    assert abs((fitted['azimuth']-(90 if side else 60)+180)%360-180) < 20
    assert abs(fitted['elevation']-60) > 5
    assert result['stages'][0]['edge_trials'] >= 4


def test_direction_search_can_activate_an_initially_off_light():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 32, 24
    sim.cfg.optics.leds = [LEDConfig(color=(1., .1, .1), azimuth=90, elevation=30, intensity=.8)]
    ref = contact(sim)
    settings = appearance(sim)
    settings['sensor']['optics']['leds'][0].update(azimuth=270, intensity=0.)
    result = fit_references(settings, [ref], size=(32, 24), fit_background=False, max_nfev=15, search_starts=1)
    assert result['settings']['sensor']['optics']['leds'][0]['azimuth'] == 90
    assert result['metrics'][0]['after_contact_mae'] < result['metrics'][0]['before_contact_mae']*.5


@pytest.mark.parametrize('brightness', [.6, 1.4])
@pytest.mark.parametrize('algorithm', ['least_squares', 'ai_surrogate'])
def test_shared_brightness_matches_contact_and_empty_references(tmp_path, brightness, algorithm):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 48, 36
    sim.gel_lighting.brightness = brightness
    refs = [contact(sim), contact(sim, blank=True)]
    settings = appearance(sim)
    settings['gel_lighting']['brightness'] = 1.
    result = fit_references(settings, refs, size=(48, 36), fit_background=False,
        fit_lighting=False, fit_filters=True, max_nfev=40, algorithm=algorithm)
    assert abs(result['settings']['gel_lighting']['brightness'] - 1.) > .05
    path = tmp_path / 'brightness.npz'
    save_session(path, result['settings'], refs, result)
    restored, saved_refs = load_session(path)
    for ref, metric in zip(saved_refs, result['metrics']):
        fitted = make_sim(ref, restored, (48, 36))
        _, production = fitted.render()
        assert abs(production.mean() - ref['photo'].mean()) < 2.
        assert metric['after_mae'] < metric['before_mae'] * .15
        np.testing.assert_array_equal(production, (render_float(fitted)*255).astype(np.uint8))


def test_filter_fit_and_noisy_render_match_production():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 64, 64
    sim.effects = ImageEffects(contrast=1.2, gamma=.8, vignette=.2, blur_px=1.2)
    ref = contact(sim)
    settings = appearance(sim)
    settings['effects'].update(contrast=1., gamma=1., vignette=0., blur_px=0.)
    result = fit_references(settings, [ref], size=(64, 64), fit_background=False,
        fit_lighting=False, fit_filters=True, max_nfev=20)
    assert result['metrics'][0]['after_mae'] < result['metrics'][0]['before_mae']*.3
    sim.effects = ImageEffects(read_noise=.02, shot_noise=.03, speckle=.04, texture=.05, texture_scale_mm=.2, seed=15)
    sim.cfg.optics.noise_sigma = .01
    _, production = sim.render()
    np.testing.assert_array_equal(production, (render_float(sim, stochastic=True)*255).astype(np.uint8))


@pytest.mark.parametrize('side', [True, False])
def test_cached_lighting_basis_matches_production(side):
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 64, 48
    sim.cfg.optics.side_lighting = side
    sim.cfg.optics.specular_gain = .15
    cache = {}
    for intensity, elevation in ((.8, 30), (1.2, 30), (1.2, 50)):
        sim.cfg.optics.leds[0].intensity = intensity
        sim.cfg.optics.leds[0].elevation = elevation
        np.testing.assert_allclose(render_float(sim, shade_cache=cache), render_float(sim), atol=3e-7)


def test_repeated_noise_components():
    yy, xx = np.mgrid[:192, :256]
    clean = np.stack([.15+.6*xx/256, .2+.5*xx/256, .25+.4*xx/256], axis=-1)
    effects = ImageEffects(read_noise=.025, shot_noise=.045, speckle=.055)
    records = []
    for seed in range(4):
        effects.seed = seed
        photo = effects.apply(clean, .0186, .0143)
        records.append(dict(reference={'repeat_group': 'unchanged'}, photo=photo, clean=clean,
                            regions=flat_patches(photo, clean)))
    params, report = estimate_noise(records, effects, lambda: False, lambda text: None)
    assert params['read_noise'] == pytest.approx(.025, abs=.007)
    assert params['shot_noise'] == pytest.approx(.045, abs=.012)
    assert params['speckle'] == pytest.approx(.055, abs=.012)
    assert report['repeat_pairs'] == 3
    assert report['single_frames'] == 0


def test_texture_statistics_improve_and_are_reproducible():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 96, 72
    sim.effects = ImageEffects(texture=.08, texture_scale_mm=.4, seed=12)
    base = np.broadcast_to([.3, .45, .55], (72, 96, 3)).copy()
    photo = sim.effects.apply(base, .0186, .0143)
    records = [dict(reference={}, photo=photo, clean=base, base=base, regions=flat_patches(photo, base))]
    settings = appearance(sim)
    settings['effects'].update(texture=0., texture_scale_mm=.1)
    params, report = fit_texture(records, settings, lambda: False, lambda text: None, seed=5, iterations=4)
    assert params['texture'] > .01
    assert report['final_cost'] < report['initial_cost']*.5
    again, _ = fit_texture(records, settings, lambda: False, lambda text: None, seed=5, iterations=4)
    assert again == params


def test_noise_native_resolution_validation_exclusion_and_cancellation():
    sim = GeometricSim()
    sim.cfg.camera.width, sim.cfg.camera.height = 96, 72
    sim.effects.read_noise = .03
    ref = contact(sim, blank=True)
    heldout = deepcopy(ref)
    heldout.update(validation=True, photo=np.zeros_like(ref['photo']))
    settings = appearance(sim)
    settings['effects']['read_noise'] = 0.
    settings['sensor']['optics']['noise_sigma'] = .01
    result = fit_references(settings, [ref, heldout], size=(32, 24), fit_background=False,
        fit_lighting=False, fit_noise=True)
    assert result['settings']['effects']['read_noise'] == pytest.approx(.03, abs=.012)
    assert result['settings']['sensor']['optics']['noise_sigma'] == 0
    assert result['stages'][0]['single_frames'] == 1
    assert result['stages'][0]['native_pixel_statistics']
    assert settings['effects']['read_noise'] == 0
    with pytest.raises(Cancelled):
        fit_references(settings, [ref], fit_background=False, fit_lighting=False,
                       fit_noise=True, cancelled=lambda: True)
