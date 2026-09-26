"""Rigid geometric imprint: mesh -> nearest surface -> indentation -> optics.

All distances are metres. The sensor looks along +Z; the minimum Z intersection
at each pixel is the accessible surface. Cut depth is measured from the lowest
vertex of the oriented object, including when it lies outside the sensor crop.
No elasticity, forces, SDF, ray-tree dependency, or Taichi initialization.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import copy, deepcopy
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from .config import SensorConfig
from .geometry.mesh import load_mesh, primitive
from .render.optical import GelSightRenderer
from .render.effects import ImageEffects, TextureCache
from .render.gel import GelLighting
from .sensor.heightmap import SurfaceSampler


DEFAULT_SENSOR_PROFILE_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'sensors' / 'Default.json'


@lru_cache(maxsize=1)
def _default_sensor_profile():
    """Load the built-in, uncalibrated profile once for comparison exports."""
    data = json.loads(DEFAULT_SENSOR_PROFILE_PATH.read_text(encoding='utf-8'))
    if data.get('format') not in ('mesh2tact-sensor-config', 'vtsim-sensor-config') or data.get('version') != 1:
        raise ValueError('Default sensor profile is invalid')
    return (SensorConfig.from_dict(data['sensor']), GelLighting.from_dict(data['gel_lighting']),
            ImageEffects(**data['effects']), float(data.get('max_penetration_mm', data['sensor']['camera']['max_depth']*1000)))


def invalidate_default_sensor_profile():
    """Use the most recently saved Default profile for later comparison exports."""
    _default_sensor_profile.cache_clear()


def rasterize_surface(vertices, faces, width, height, size_x, size_y):
    """Orthographic minimum-Z buffer at pixel centres, inf outside the mesh.

    Triangle interpolation preserves planar facets. Vertical triangles have
    zero projected area and do not cover pixels. Overlapping parts resolve to
    the first surface visible from below, rather than an internal/back face.
    """
    if width < 2 or height < 2 or min(size_x, size_y) <= 0:
        raise ValueError("Sensor dimensions must be positive and resolution at least 2x2")
    surface = np.full((height, width), np.inf, dtype=np.float64)
    projected = np.asarray(vertices, dtype=float).copy()
    projected[:, 0] = (projected[:, 0] / size_x + .5) * width - .5
    projected[:, 1] = (projected[:, 1] / size_y + .5) * height - .5
    triangles = projected[np.asarray(faces)]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominators = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    lower = np.maximum(np.ceil(triangles[:, :, :2].min(axis=1)).astype(int), 0)
    upper = np.minimum(np.floor(triangles[:, :, :2].max(axis=1)).astype(int), [width-1, height-1])
    visible = (np.abs(denominators) >= 1e-12) & np.all(lower <= upper, axis=1)
    small = visible & np.all(upper-lower < 16, axis=1)
    indices = np.flatnonzero(small)
    flat = surface.ravel()
    # Dense meshes contain thousands of tiny projected triangles. Evaluate
    # their pixel boxes in bounded batches rather than a Python loop per face.
    for start in range(0, len(indices), 512):
        group = indices[start:start+512]
        widths, heights = (upper[group]-lower[group]+1).T
        counts = widths*heights
        local = np.repeat(np.arange(len(group)), counts)
        offset = np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts, counts)
        xx = lower[group[local], 0]+offset % widths[local]
        yy = lower[group[local], 1]+offset // widths[local]
        a, b, c = triangles[group[local], 0], triangles[group[local], 1], triangles[group[local], 2]
        denominator = denominators[group[local]]
        u = ((b[:, 1]-c[:, 1])*(xx-c[:, 0]) + (c[:, 0]-b[:, 0])*(yy-c[:, 1]))/denominator
        v = ((c[:, 1]-a[:, 1])*(xx-c[:, 0]) + (a[:, 0]-c[:, 0])*(yy-c[:, 1]))/denominator
        w = 1-u-v
        inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
        z = u*a[:, 2]+v*b[:, 2]+w*c[:, 2]
        np.minimum.at(flat, (yy*width+xx)[inside], z[inside])
    # Cull subpixel, vertical and off-sensor triangles in one vectorized pass.
    # Keep the same pixel-centre rasterization and full input mesh.
    for triangle, denominator, (xmin, ymin), (xmax, ymax) in zip(
            triangles[visible & ~small], denominators[visible & ~small],
            lower[visible & ~small], upper[visible & ~small]):
        a, b, c = triangle
        yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
        u = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / denominator
        v = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / denominator
        w = 1-u-v
        inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
        z = u*a[2] + v*b[2] + w*c[2]
        tile = surface[ymin:ymax+1, xmin:xmax+1]
        np.minimum(tile, np.where(inside, z, np.inf), out=tile)
    return surface


class GeometricSim:
    def __init__(self, cfg=None):
        self.cfg = deepcopy(cfg) if cfg is not None else SensorConfig()
        if cfg is None:
            self.cfg.optics.noise_sigma = 0.0
        self.mesh = primitive("sphere", .004)
        self.original_mesh = self.mesh.copy()
        self.quality_level = 0
        self.smoothing_iterations = 0
        self.effects = ImageEffects()
        self._texture_cache = TextureCache()
        self._flat_lighting_cache = {}
        self.max_penetration = None  # metres; None preserves legacy API behaviour
        self.gel_lighting = GelLighting()
        self.source = "primitive:sphere"
        self.units = "m"
        self.scale = 1.0
        self.base_rotation = (0., 0., 0.)
        self.rotation = (0., 0., 0.)
        self.offset = (0., 0.)
        self.cut_depth = .001
        self.softness = 0.0  # Gaussian sigma in metres; visual approximation
        self._key = None
        self.surface = None
        self.geometry_backend = 'cpu'

    def load(self, path, units="mm", scale=1.0):
        if units not in ("mm", "cm", "m", "in") or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Choose valid mesh units and a positive scale")
        mesh = load_mesh(path, units=units, scale=scale)
        self.set_mesh(mesh, str(Path(path).resolve()))
        self.units, self.scale = units, scale

    def set_mesh(self, mesh, source="mesh"):
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
            raise ValueError("A finite, nonempty triangle mesh is required")
        self.mesh = mesh.copy()
        self.original_mesh = mesh.copy()
        self.quality_level = 0
        self.smoothing_iterations = 0
        self.source = source
        self.units, self.scale = 'm', 1.0
        self.base_rotation = (0., 0., 0.)
        self._key = None

    def set_base_rotation(self, angles):
        """Set an absolute mesh orientation underneath sampled pose rotations."""
        angles = tuple(angles)
        if len(angles) != 3 or not np.isfinite(angles).all():
            raise ValueError('Base rotation must contain three finite angles')
        transform = np.eye(4)
        transform[:3, :3] = (Rotation.from_euler('xyz', angles, degrees=True) *
                             Rotation.from_euler('xyz', self.base_rotation, degrees=True).inv()).as_matrix()
        self.mesh.apply_transform(transform)
        self.original_mesh.apply_transform(transform)
        self.base_rotation = angles
        self._key = None

    def set_scale(self, scale):
        """Resize the loaded geometry without reimporting or losing mesh quality."""
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Choose a positive finite object scale')
        if scale == self.scale:
            return
        ratio = scale / self.scale
        mesh, original = self.mesh.copy(), self.original_mesh.copy()
        mesh.apply_scale(ratio)
        original.apply_scale(ratio)
        self.mesh, self.original_mesh = mesh, original
        self.scale = float(scale)
        self._key = None

    def set_quality(self, level=0, smoothing_iterations=0):
        """Rebuild from the original, never cumulatively modify or overwrite it.

        Demo curves are regenerated at higher tessellation. Imported meshes are
        subdivided linearly; only the explicit Taubin smoothing changes shape.
        """
        if level not in (-3, -2, -1, 0, 1, 2, 3) or not 0 <= smoothing_iterations <= 50:
            raise ValueError("Quality level must be -3–3 and smoothing iterations 0–50")
        if (level, smoothing_iterations) == (self.quality_level, self.smoothing_iterations):
            return
        kind = self.source.removeprefix("primitive:") if self.source.startswith("primitive:") else None
        if kind == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=3+level, radius=.004)
        elif kind in ("cylinder", "cone"):
            factory = getattr(trimesh.creation, kind)
            mesh = factory(radius=.004, height=.008, sections=max(8, int(32*2**level)))
        else:
            mesh = self.original_mesh.copy()
            if level < 0:
                import pyvista as pv
                faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
                poly = pv.PolyData(mesh.vertices, faces)
                reduced = poly.decimate_pro({-1: .25, -2: .6, -3: .85}[level], preserve_topology=True)
                mesh = trimesh.Trimesh(reduced.points, reduced.faces.reshape(-1, 4)[:, 1:], process=False)
            for _ in range(max(0, level)):
                if len(mesh.faces)*4 > 1_000_000:
                    break
                mesh = mesh.subdivide()
        if smoothing_iterations:
            trimesh.smoothing.filter_taubin(mesh, iterations=int(smoothing_iterations))
        if kind in ('sphere', 'cylinder', 'cone'):
            mesh.apply_scale(self.scale)
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', self.base_rotation, degrees=True).as_matrix()
            mesh.apply_transform(transform)
        self.mesh = mesh
        self.quality_level, self.smoothing_iterations = level, smoothing_iterations
        self._key = None

    def prepare(self):
        c, g = self.cfg.camera, self.cfg.gel
        key = (*self.rotation, *self.offset, c.width, c.height, g.size_x, g.size_y, self.geometry_backend)
        if key != self._key:
            vertices = Rotation.from_euler("xyz", self.rotation, degrees=True).apply(self.mesh.vertices)
            vertices[:, :2] += self.offset
            self.vertices = vertices
            self.first_contact = float(vertices[:, 2].min())
            if self.geometry_backend == 'cuda':
                from .geometry.gpu_raster import rasterize_surface_cuda
                self.surface = rasterize_surface_cuda(vertices, self.mesh.faces, c.width, c.height, g.size_x, g.size_y)
            else:
                self.surface = rasterize_surface(vertices, self.mesh.faces, c.width, c.height, g.size_x, g.size_y)
            self.sampler = SurfaceSampler(self.cfg, np.array([-g.size_x/2, -g.size_y/2, 0]),
                                          np.array([g.size_x/2, g.size_y/2, g.thickness]))
            self._key = key

    @property
    def plane_z(self):
        return 0.0

    @property
    def object_z(self):
        # Object placement must not rasterize an image just to move a 3D actor.
        bottom = Rotation.from_euler('xyz', self.rotation, degrees=True).apply(self.mesh.vertices)[:, 2].min()
        cut = self.cut_depth
        if self.max_penetration is not None:
            if not np.isfinite(self.max_penetration) or self.max_penetration < 0:
                raise ValueError('Maximum indentation depth must be finite and nonnegative')
            cut = min(cut, self.max_penetration)
        return -float(bottom)-cut

    @object_z.setter
    def object_z(self, value):
        if not np.isfinite(value):
            raise ValueError('Object Z must be finite')
        bottom = Rotation.from_euler('xyz', self.rotation, degrees=True).apply(self.mesh.vertices)[:, 2].min()
        self.cut_depth = -float(bottom)-float(value)
        if self.max_penetration is not None:
            self.cut_depth = min(self.cut_depth, self.max_penetration)

    def depth(self):
        if not np.isfinite(self.cut_depth) or not np.isfinite(self.softness) or self.softness < 0:
            raise ValueError("Cut depth must be finite and softness finite and nonnegative")
        self.prepare()
        if self.max_penetration is not None:
            _ = self.object_z  # Validate the configured physical limit.
            self.cut_depth = min(self.cut_depth, self.max_penetration)
        raw = np.maximum(-(self.surface + self.object_z), 0).astype(np.float32)
        self.raw_depth = raw
        if self.softness:
            dx, dy = self.sampler.pixel_size
            return gaussian_filter(raw, (self.softness/dy, self.softness/dx), mode="constant")
        return raw.copy()

    def render(self):
        depth = self.depth()
        optical_cfg = deepcopy(self.cfg)
        # Apply camera noise after optical blur and before final quantization.
        optical_cfg.optics.noise_sigma = 0
        renderer = GelSightRenderer(optical_cfg)
        renderer._flat_cache = self._flat_lighting_cache
        use_manual_background = self.gel_lighting.enabled and renderer._background is None
        if use_manual_background:
            renderer.set_background(np.broadcast_to(self.gel_lighting.background, (*depth.shape, 3)))
        # Indentation lowers the gel surface; a positive depth has negative height.
        rgb = renderer.render(self.sampler, -depth, use_gpu=False, as_uint8=False,
                              contact_depth=self.raw_depth)
        if use_manual_background:
            rgb = self.gel_lighting.overlay(rgb)
        rgb = self.gel_lighting.adjust(rgb)
        rgb = renderer.depth_overlay(rgb, depth)
        if self.cfg.optics.perimeter_shadow_enabled:
            rgb *= renderer.perimeter_shadow(depth, self.raw_depth)[..., None]
        self.clean_rgb = (rgb*255).astype(np.uint8)
        effects = deepcopy(self.effects)
        effects.read_noise = float(np.hypot(effects.read_noise, self.cfg.optics.noise_sigma))
        rgb = effects.apply(rgb, self.cfg.gel.size_x, self.cfg.gel.size_y, texture_cache=self._texture_cache)
        return depth, (rgb*255).astype(np.uint8)

    def render_default_profile(self, depth):
        """Render the object with the complete built-in Default sensor profile.

        Object pose stays fixed. The capture resolution stays with the active
        General object settings, while the Default profile supplies the pad,
        optics, gel lighting, effects, and indentation limit.
        """
        profile_cfg, profile_lighting, profile_effects, maximum_mm = _default_sensor_profile()
        # Rendering never edits the mesh or cached arrays. Avoid copying the
        # full geometry and image caches for every Default-profile sample.
        comparison = copy(self)
        comparison._flat_lighting_cache = dict(self._flat_lighting_cache)
        comparison._texture_cache = copy(self._texture_cache)
        comparison._texture_cache.sources = self._texture_cache.sources.copy()
        comparison._texture_cache.fields = self._texture_cache.fields.copy()
        comparison.cfg = deepcopy(profile_cfg)
        comparison.cfg.camera.width, comparison.cfg.camera.height = self.cfg.camera.width, self.cfg.camera.height
        comparison.gel_lighting = deepcopy(profile_lighting)
        comparison.effects = deepcopy(profile_effects)
        comparison.max_penetration = maximum_mm/1000
        # A different pad must rasterize again. Identical pad dimensions may
        # reuse the prepared surface while applying the Default depth limit.
        if (comparison.cfg.gel.size_x, comparison.cfg.gel.size_y) != (self.cfg.gel.size_x, self.cfg.gel.size_y):
            comparison._key = None
        elif self._key is not None:
            g = comparison.cfg.gel
            comparison.sampler = SurfaceSampler(comparison.cfg, np.array([-g.size_x/2, -g.size_y/2, 0]),
                                                np.array([g.size_x/2, g.size_y/2, g.thickness]))
        _, rgb = comparison.render()
        return rgb

    def export(self, directory, processed_mesh_ref=None, outputs=None):
        """Export one reproducible frame; refuse to overwrite a previous capture."""
        import cv2
        from .outputs import SaveOptions
        outputs = outputs or SaveOptions()
        outputs.validate()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=False)
        needs_rgb = outputs.tactile or outputs.clean
        if needs_rgb:
            depth, rgb = self.render()
        else:
            depth, rgb = self.depth(), None
        default_rgb = self.render_default_profile(depth) if outputs.default_tactile else None
        if outputs.depth_array:
            np.save(out / "depth_m.npy", depth)
        if outputs.raw_depth:
            np.save(out / "raw_depth_m.npy", self.raw_depth)
        image_outputs = [(outputs.tactile, "tactile.png", rgb[:, :, ::-1] if outputs.tactile else None),
                         (outputs.clean, "tactile_clean.png", self.clean_rgb[:, :, ::-1] if outputs.clean else None),
                           (outputs.depth_image, "depth.png", GelSightRenderer.depth_colormap(depth, self.cfg.camera.max_depth)[:, :, ::-1] if outputs.depth_image else None),
                           (outputs.contact, "contact.png", (self.raw_depth > 0).astype(np.uint8)*255)]
        if outputs.default_tactile:
            image_outputs.insert(1, (True, "tactile_default.png", default_rgb[:, :, ::-1]))
        for enabled, name, data in image_outputs:
            if not enabled:
                continue
            if not cv2.imwrite(str(out / name), data):
                raise OSError(f"Could not write {out / name}")
        metadata = dict(mode="geometric", source=self.source, source_units=self.units, scale=self.scale,
                        geometry_backend=self.geometry_backend,
                        base_rotation_xyz_deg=list(self.base_rotation),
                        image_effects=asdict(self.effects), quality_level=self.quality_level,
                        gel_lighting=asdict(self.gel_lighting),
                        smoothing_iterations=self.smoothing_iterations, processed_mesh=(processed_mesh_ref or "processed_mesh.ply") if outputs.mesh else None,
                        rotation_xyz_deg=list(self.rotation), offset_xy_m=list(self.offset),
                        cut_depth_m=self.cut_depth, plane_z_m=0.0, object_z_m=self.object_z,
                        max_penetration_m=self.max_penetration, softness_sigma_m=self.softness,
                        depth_units="metres", image_rows="increasing sensor Y", sensor=asdict(self.cfg),
                        default_tactile_profile=str(DEFAULT_SENSOR_PROFILE_PATH) if outputs.default_tactile else None,
                        note="Nearest surface geometric imprint; Gaussian softness is not elasticity.")
        if outputs.settings:
            (out / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if outputs.mesh and processed_mesh_ref is None:
            self.mesh.export(out / "processed_mesh.ply")
        return out
