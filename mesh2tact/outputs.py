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


def next_capture_index(directories, excluded=()):
    """Find the next index across every payload branch, including partial saves."""
    pattern = re.compile(r'^(?:sample|run|manifest|processed_mesh|\.sample)_(\d+)(?=[_.]|$)')
    excluded = set(excluded)
    maximum = 0
    for directory in directories:
        directory = Path(directory)
        if directory.is_dir():
            for path in directory.iterdir():
                if path not in excluded and (match := pattern.match(path.name)):
                    maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def reserve_labelled_capture(root, label, count, other_roots=()):
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
    directories = [parent] + [Path(root).expanduser().resolve() / parent.name for root in other_roots]
    start = next_capture_index(directories)
    while start <= 99_000_000:
        reservations = [parent / f".sample_{index:06d}.lock" for index in range(start, start + count)]
        claimed = []
        try:
            for reservation in reservations:
                with reservation.open("x", encoding="utf-8"):
                    pass
                claimed.append(reservation)
            # Another collector may have finished and removed its locks between
            # our initial scan and reservation. Recheck payloads under our locks.
            next_index = next_capture_index(directories, excluded=claimed)
            if next_index > start:
                for reservation in claimed:
                    reservation.unlink(missing_ok=True)
                start = next_index
                continue
            return parent, start, claimed
        except FileExistsError:
            for reservation in claimed:
                reservation.unlink(missing_ok=True)
            start = max(start + 1, next_capture_index(directories))
        except Exception:
            for reservation in claimed:
                reservation.unlink(missing_ok=True)
            raise
    raise OSError("Could not reserve unique capture indices")


def reserve_labelled_still(root, label):
    """Reserve a timestamp prefix inside the label's flat stills directory."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError('Enter an object label for still captures')
    parent = Path(root).expanduser().resolve() / model_name(label.strip()) / 'stills'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    for suffix in range(1000):
        name = stamp if suffix == 0 else f'{stamp}_{suffix}'
        reservation = parent / f'.{name}.lock'
        try:
            with reservation.open('x', encoding='utf-8'):
                pass
        except FileExistsError:
            continue
        if any(parent.glob(name+'_*')) or (parent/(name+'.partial')).exists():
            reservation.unlink()
            continue
        return parent, name, [reservation]
    raise OSError('Could not reserve a unique still timestamp')
