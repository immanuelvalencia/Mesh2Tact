"""Geometric desktop application."""
def run(sensor=None, mesh=None):
    from .geometric import run as launch
    return launch(sensor=sensor, mesh=mesh)
