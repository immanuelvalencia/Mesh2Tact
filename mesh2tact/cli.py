"""Launch Mesh2Tact, the mesh-to-tactile desktop application."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", nargs="?", help="STL/OBJ/PLY file to load")
    parser.add_argument("--sensor", help="Optional legacy sensor YAML")
    parser.add_argument("--geometry", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list-sensors", action="store_true")
    args = parser.parse_args(argv)
    if args.list_sensors:
        folder = Path(__file__).resolve().parents[1] / "configs/sensors"
        for path in sorted(folder.glob("*.json")):
            print(path.name)
        return 0
    if args.mesh and not Path(args.mesh).is_file():
        parser.error(f"No such mesh: {args.mesh}")
    from .gui import run
    return run(sensor=args.sensor, mesh=args.mesh)
