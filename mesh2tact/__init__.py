"""Geometric visuo-tactile image renderer."""
from .config import SensorConfig
__version__ = "1.0.0"
__all__ = ["GeometricSim", "SensorConfig"]

def __getattr__(name):
    if name == "GeometricSim":
        from .geometric import GeometricSim
        return GeometricSim
    raise AttributeError(name)
