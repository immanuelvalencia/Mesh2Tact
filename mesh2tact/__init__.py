"""Geometric visuo-tactile image renderer."""
from .config import SensorConfig
__version__ = "0.2.0"
__all__ = ["GeometricSim", "SensorConfig"]

def __getattr__(name):
    if name == "GeometricSim":
        from .geometric import GeometricSim
        return GeometricSim
    raise AttributeError(name)
