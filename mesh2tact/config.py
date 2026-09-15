"""Configuration dataclasses for the visuo-tactile simulator.

All physical quantities are SI (metres, seconds, kilograms, pascals) unless a
field name says otherwise.  A whole sensor is described by :class:`SensorConfig`,
which is normally loaded from a YAML file in ``configs/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import yaml


# --------------------------------------------------------------------------- #
# gel / elastomer
# --------------------------------------------------------------------------- #
def _coerce(obj) -> None:
    """Force numeric dataclass fields to float/int.

    YAML 1.1 (PyYAML) reads ``3.0e5`` as a *string* unless the exponent carries
    a sign, which is an easy trap when hand-editing sensor configs.  Coercing
    here means a config file never silently poisons the solver.
    """
    for name, f in obj.__dataclass_fields__.items():
        if f.type in ("float", "float | None") or f.type is float:
            value = getattr(obj, name)
            if value is not None and not isinstance(value, float):
                setattr(obj, name, float(value))
        elif f.type == "int" or f.type is int:
            setattr(obj, name, int(getattr(obj, name)))


@dataclass
class GelConfig:
    """Geometry and material of the elastomer pad.

    The gel is a rectangular slab.  Its bottom face (``z = 0``) is glued to the
    rigid acrylic window of the sensor; its top face (``z = thickness``) is the
    free surface that objects press into.
    """

    size_x: float = 0.0186          # m  (GelSight Mini sensing area ~18.6 mm)
    size_y: float = 0.0143          # m
    thickness: float = 0.0030       # m  elastomer depth

    youngs_modulus: float = 3.0e5   # Pa  (soft silicone, ~0.1-1 MPa typical)
    poissons_ratio: float = 0.45    # nearly incompressible
    density: float = 1100.0         # kg/m^3

    # Numerical: MPM particles seeded per grid cell along each axis
    # (2.0 -> 8 particles per cell, the usual choice for elastic solids).
    particles_per_cell_axis: float = 2.0
    # Rayleigh-style velocity damping applied on the grid each substep.
    damping: float = 8.0            # 1/s

    def __post_init__(self) -> None:
        _coerce(self)

    @property
    def lame(self) -> tuple[float, float]:
        """(mu, lambda) Lame parameters."""
        e, nu = self.youngs_modulus, self.poissons_ratio
        mu = e / (2.0 * (1.0 + nu))
        lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return mu, lam


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    """MPM solver settings.

    The simulation domain is a box that contains the gel plus head-room above it
    for the indenter.  ``grid_res`` is the number of MPM grid cells along the
    longest domain axis.
    """

    grid_res: int = 96
    headroom: float = 0.006         # m of empty space modelled above the gel
    margin: float = 0.002           # m of padding around the gel footprint

    dt: float = 1.0e-5              # s   (CFL-limited by sqrt(E/rho))
    substeps: int = 120             # substeps advanced per :meth:`step` call
    gravity: float = 0.0            # m/s^2  (gravity is negligible vs contact)

    friction: float = 0.9           # Coulomb coefficient, gel <-> object
    arch: str = "cuda"              # "cuda" | "vulkan" | "cpu"  (auto-fallback)
    device_memory_gb: float = 2.0

    def __post_init__(self) -> None:
        _coerce(self)

    @property
    def cfl_dt(self) -> float:
        """Informational: caller can compare ``dt`` against this."""
        return float("inf")


# --------------------------------------------------------------------------- #
# camera behind the gel
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    """The camera looks up through the transparent backing at the gel surface.

    The model is orthographic: the imaged region is exactly the gel footprint,
    which is a good approximation for GelSight-class sensors with a flat pad.
    """

    width: int = 984
    height: int = 739
    # Depth range used when normalising the height map for visualisation (m).
    max_depth: float = 0.0015

    def __post_init__(self) -> None:
        _coerce(self)


# --------------------------------------------------------------------------- #
# optics
# --------------------------------------------------------------------------- #
@dataclass
class LEDConfig:
    """One coloured light source inside the sensor.

    ``azimuth`` is measured in the image plane (degrees, CCW from +x) and
    ``elevation`` above that plane, matching how GelSight-style illumination is
    usually specified.
    """

    color: Sequence[float] = (1.0, 0.0, 0.0)
    azimuth: float = 90.0
    elevation: float = 25.0
    intensity: float = 1.0

    def __post_init__(self) -> None:
        self.color = tuple(float(c) for c in self.color)
        _coerce(self)


@dataclass
class OpticsConfig:
    # Optional empirical RGB response: 6 spatial terms x 5 normal terms x RGB.
    # Stored inline so calibrated presets have no external lookup-file dependency.
    spatial_response: list | None = None
    leds: list[LEDConfig] = field(
        default_factory=lambda: [
            LEDConfig(color=(1.0, 0.0, 0.0), azimuth=270.0, elevation=25.0),
            LEDConfig(color=(0.0, 1.0, 0.0), azimuth=180.0, elevation=25.0),
            LEDConfig(color=(0.0, 0.0, 1.0), azimuth=0.0, elevation=25.0),
        ]
    )
    ambient: Sequence[float] = (0.10, 0.10, 0.12)
    diffuse_gain: float = 0.85
    specular_gain: float = 0.25
    shininess: float = 24.0
    # Optional Taxim-style calibrated lookup table (.npz).  When present it
    # replaces the analytic shading model.
    calibration_path: str | None = None
    # Sensor noise, applied after shading (std-dev in 0-1 units).
    noise_sigma: float = 0.006
    exposure: float = 1.0
    reference_background: bool = True
    side_lighting: bool = True
    side_distance: float = 1.1
    side_falloff: float = 1.0

    def __post_init__(self) -> None:
        self.ambient = tuple(float(c) for c in self.ambient)
        _coerce(self)


# --------------------------------------------------------------------------- #
# markers painted on the gel
# --------------------------------------------------------------------------- #
@dataclass
class MarkerConfig:
    enabled: bool = True
    spacing: float = 0.0012         # m between marker centres
    radius_px: float = 3.0
    darkness: float = 0.75          # 1.0 = fully black dot

    def __post_init__(self) -> None:
        _coerce(self)


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
@dataclass
class SensorConfig:
    name: str = "gelsight_mini_like"
    gel: GelConfig = field(default_factory=GelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    markers: MarkerConfig = field(default_factory=MarkerConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "SensorConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SensorConfig":
        optics_raw = dict(raw.get("optics", {}))
        leds = [LEDConfig(**led) for led in optics_raw.pop("leds", [])]
        optics = OpticsConfig(**optics_raw)
        if leds:
            optics.leds = leds
        return cls(
            name=raw.get("name", "sensor"),
            gel=GelConfig(**raw.get("gel", {})),
            solver=SolverConfig(**raw.get("solver", {})),
            camera=CameraConfig(**raw.get("camera", {})),
            optics=optics,
            markers=MarkerConfig(**raw.get("markers", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)


def load_config(path: str | Path | None = None) -> SensorConfig:
    """Load a sensor config, falling back to the packaged default."""
    if path is None:
        default = Path(__file__).resolve().parent.parent / "configs" / "gelsight_mini.yaml"
        if default.exists():
            return SensorConfig.from_yaml(default)
        return SensorConfig()
    return SensorConfig.from_yaml(path)
