import cProfile, pstats, io, json, tempfile, time
from pathlib import Path
from mesh2tact.geometric import GeometricSim
from mesh2tact.config import SensorConfig
from mesh2tact.render.gel import GelLighting
from mesh2tact.render.effects import ImageEffects

raw = json.loads(Path('configs/sensors/GelSightV1.json').read_text())
for width in (984, 2048):
    sim = GeometricSim()
    sim.cfg = SensorConfig.from_dict(raw['sensor'])
    sim.gel_lighting = GelLighting.from_dict(raw['gel_lighting'])
    sim.effects = ImageEffects(**raw['effects'])
    sim.softness = raw['controls']['softness']/1000
    sim.cfg.camera.width = width
    sim.cfg.camera.height = round(width*sim.cfg.gel.size_y/sim.cfg.gel.size_x)
    sim.set_quality(2)
    sim.cut_depth = .0008
    with tempfile.TemporaryDirectory(prefix='mesh2tact_capture_profile_') as folder:
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        sim.export(Path(folder)/'capture')
        profiler.disable()
        print(f'{width}x{sim.cfg.camera.height}: {time.perf_counter()-start:.3f}s, faces={len(sim.mesh.faces)}')
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(14)
        print(stream.getvalue())
