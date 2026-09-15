"""Selectable capture payloads and safe automatic output paths."""
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class SaveOptions:
    tactile: bool = True
    default_tactile: bool = False
    clean: bool = True
    depth_image: bool = True
    depth_array: bool = True
    raw_depth: bool = True
    contact: bool = True
    settings: bool = True
    mesh: bool = True

    def validate(self):
        values = list(asdict(self).values())
        if not all(isinstance(value, bool) for value in values) or not any(values):
            raise ValueError("Select at least one output to save")


def model_name(source):
    name = source.split(":", 1)[1] if source.startswith("primitive:") else Path(source).stem
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "model"
    if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1,10)), *(f"LPT{i}" for i in range(1,10))}:
        name = "model_" + name
    return name[:100]


def create_run_directory(root, source):
    """Create one new capture directory without replacing an earlier run.

    This preserves the historic ``data/model/date-time`` layout.
    """
    parent = Path(root).expanduser().resolve() / model_name(source)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    for suffix in range(1000):
        out = parent / (stamp if suffix == 0 else f"{stamp}_{suffix}")
        try:
            out.mkdir(exist_ok=False)
            return out
        except FileExistsError:
            continue
    raise OSError("Could not create a unique capture directory")


def reserve_labelled_capture(root, label, count):
    """Reserve contiguous sample indices directly in ``data/<label>``.

    Reservation files are temporary and use exclusive creation. This makes
    repeated or simultaneous saves advance safely without creating a run folder
    inside the label directory.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Enter an object label for indexed capture folders")
    if not isinstance(count, int) or count < 1:
        raise ValueError("Capture count must be a positive integer")
    parent = Path(root).expanduser().resolve() / model_name(label.strip())
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'^(?:sample|run|manifest|processed_mesh|\.sample)_(\d{6})')
    used = [int(match.group(1)) for path in parent.iterdir() if (match := pattern.match(path.name))]
    start = max(used, default=0) + 1
    while start <= 99_000_000:
        reservations = [parent / f".sample_{index:06d}.lock" for index in range(start, start + count)]
        claimed = []
        try:
            for reservation in reservations:
                with reservation.open("x", encoding="utf-8"):
                    pass
                claimed.append(reservation)
            return parent, start, claimed
        except FileExistsError:
            for reservation in claimed:
                reservation.unlink(missing_ok=True)
            start += 1
    raise OSError("Could not reserve unique capture indices")
