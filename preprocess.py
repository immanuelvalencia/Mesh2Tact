"""Prepare aligned ImageFolder datasets from Mesh2Tact image branches.

Run ``python preprocess.py`` for the desktop UI. The backend has no Qt dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SAMPLE_ID = re.compile(r"^(sample_\d+)(?:_|$)", re.IGNORECASE)
SPLITS = ("train", "val", "test")
VARIANTS = ("clean", "tactile", "default")
# Both the current short names and older Mesh2Tact export names identify the
# same capture.  Only the terminal image-type suffix is removed for pairing.
VARIANT_SUFFIXES = {"clean": ("_tactile_clean", "_clean"),
                    "tactile": ("_tactile",),
                    "default": ("_tactile_default", "_default")}


@dataclass(frozen=True)
class ImageRecord:
    source: Path
    label: str
    group: str
    digest: str


@dataclass(frozen=True)
class PairedRecord:
    label: str
    key: str
    group: str
    sources: dict[str, Path]
    digests: dict[str, str]


@dataclass
class VariantScan:
    labels: list[str]
    pairs: list[PairedRecord]
    summary: dict


def _classes(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"Dataset folder does not exist: {source}")
    names = {path.name.lower() for path in source.iterdir() if path.is_dir()}
    if len({"tactile", "clean", "default"} & names) >= 2:
        raise ValueError("Select one image branch, for example data/tactile, rather than data itself.")
    if len({"train", "val", "test"} & names) >= 2:
        raise ValueError("Select the unsplit class folder, not an already split dataset.")
    classes = sorted((path for path in source.iterdir() if path.is_dir() and not path.name.startswith(".")),
                     key=lambda path: path.name)
    if len(classes) < 2:
        raise ValueError("At least two class folders are required.")
    if len({path.name.casefold() for path in classes}) != len(classes):
        raise ValueError("Class names differ only by letter case; rename them before export.")
    return classes


def _is_empty_contact(path: Path) -> bool:
    match = SAMPLE_ID.match(path.stem)
    if not match:
        return False
    settings = path.with_name(f"{match.group(1)}_settings.json")
    if not settings.is_file():
        return False
    try:
        fraction = json.loads(settings.read_text(encoding="utf-8")).get("contact_fraction")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read contact metadata: {settings}") from exc
    return fraction is not None and float(fraction) <= 0.0


def scan_dataset(source: str | Path, group_by_subfolder: bool = True) -> tuple[list[str], list[ImageRecord], dict]:
    """Validate images and remove identical images within each class."""
    source = Path(source).expanduser().resolve()
    labels = [folder.name for folder in _classes(source)]
    records: list[ImageRecord] = []
    digest_owner: dict[str, str] = {}
    skipped = defaultdict(int)
    capture_suffixes = VARIANT_SUFFIXES.get(source.name.lower())
    for label in labels:
        folder = source / label
        images = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                         and (capture_suffixes is None or p.stem.lower().endswith(capture_suffixes))),
                        key=lambda p: p.as_posix())
        if not images:
            raise ValueError(f"No images in class {label!r}.")
        for path in images:
            if _is_empty_contact(path):
                skipped["empty_contact"] += 1
                continue
            try:
                with Image.open(path) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    if min(image.size) < 16:
                        raise ValueError("image is smaller than 16 pixels")
                    digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable image {path}: {exc}") from exc
            owner = digest_owner.get(digest)
            if owner is not None:
                if owner != label:
                    raise ValueError(f"Identical image appears in classes {owner!r} and {label!r}: {path}")
                skipped["duplicate_image"] += 1
                continue
            digest_owner[digest] = label
            relative = path.relative_to(folder)
            sample = SAMPLE_ID.match(path.stem)
            if group_by_subfolder and len(relative.parts) > 1:
                group = relative.parts[0]
            else:
                group = str(relative.parent / (sample.group(1) if sample else path.stem))
            records.append(ImageRecord(path, label, group, digest))
        if not any(record.label == label for record in records):
            raise ValueError(f"Class {label!r} has no usable images after filtering.")
    return labels, records, dict(skipped)


def _group_counts(count: int, ratios: tuple[int, int, int]) -> tuple[int, int, int]:
    if count < 3:
        raise ValueError("Each class needs at least three independent groups for train/validation/test.")
    counts = [1, 1, 1]
    remaining = count - 3
    targets = [remaining * ratio / 100 for ratio in ratios]
    for index, value in enumerate(targets):
        counts[index] += int(value)
    for index in sorted(range(3), key=lambda item: (targets[item] - int(targets[item]), -item), reverse=True)[:remaining - sum(int(x) for x in targets)]:
        counts[index] += 1
    return tuple(counts)


def split_records(records: list[ImageRecord], ratios: tuple[int, int, int], seed: int) -> dict[str, list[ImageRecord]]:
    if len(ratios) != 3 or any(not isinstance(value, int) or value <= 0 for value in ratios) or sum(ratios) != 100:
        raise ValueError("Train, validation, and test percentages must be positive integers totalling 100.")
    if seed < 0:
        raise ValueError("Seed must be nonnegative.")
    result: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}
    by_class: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record.label][record.group].append(record)
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        random.Random(f"{seed}:{label}").shuffle(groups)
        counts = _group_counts(len(groups), ratios)
        offset = 0
        for split, count in zip(SPLITS, counts):
            for group in groups[offset:offset + count]:
                result[split].extend(by_class[label][group])
            offset += count
    return result


def prepare_dataset(source: str | Path, output: str | Path, ratios: tuple[int, int, int] = (70, 15, 15),
                    seed: int = 42, group_by_subfolder: bool = True, progress=None) -> dict:
    """Write a fresh RGB ImageFolder export, labels, and an auditable manifest."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose an output folder outside the source dataset.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new folder to preserve existing data.")
    labels, records, skipped = scan_dataset(source, group_by_subfolder)
    assignments = split_records(records, ratios, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.incomplete-{uuid4().hex[:8]}")
    staging.mkdir(exist_ok=False)
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    rows = []
    try:
        (staging / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        total = len(records)
        completed = 0
        for split in SPLITS:
            for label in labels:
                folder = staging / split / label
                folder.mkdir(parents=True)
                items = sorted((r for r in assignments[split] if r.label == label), key=lambda r: r.source.as_posix())
                counts[split][label] = len(items)
                for index, record in enumerate(items, 1):
                    target = folder / f"image_{index:06d}.png"
                    with Image.open(record.source) as raw:
                        ImageOps.exif_transpose(raw).convert("RGB").save(target, format="PNG")
                    rows.append((split, label, str(record.source), target.relative_to(staging).as_posix(),
                                 record.group, record.digest))
                    completed += 1
                    if progress:
                        progress(completed, total)
        with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("split", "label", "source", "output", "group", "rgb_sha256"))
            writer.writerows(rows)
        summary = {"source": str(source), "labels": labels, "ratios_requested": dict(zip(SPLITS, ratios)),
                   "seed": seed, "group_by_subfolder": group_by_subfolder, "counts": counts,
                   "skipped": skipped, "total_images": total,
                   "note": "Splits evaluate new images of the listed classes, not unseen object instances."}
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
        return summary
    except Exception:
        # Retain this run's partial export for inspection. The requested output is untouched.
        raise


def scan_variants(root: str | Path, group_by_subfolder: bool = True, progress=None) -> VariantScan:
    """Compare class/sample identity across the three image branches, read only.

    Matching means the same class and sample IDs, not pixel equality: the three
    render types intentionally have different RGB appearances.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Source folder does not exist: {root}")
    missing_branches = [name for name in VARIANTS if not (root / name).is_dir()]
    if missing_branches:
        raise ValueError("Missing image branches: " + ", ".join(missing_branches))

    class_sets = {
        name: {folder.name for folder in (root / name).iterdir()
               if folder.is_dir() and not folder.name.startswith(".")}
        for name in VARIANTS
    }
    labels = sorted(set.union(*class_sets.values()))
    if len(labels) < 2:
        raise ValueError("At least two class folders are required in every image branch.")
    issue_counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)

    def issue(kind: str, detail: str):
        issue_counts[kind] += 1
        if len(examples[kind]) < 8:
            examples[kind].append(detail)

    for name in VARIANTS:
        for label in labels:
            if label not in class_sets[name]:
                issue("missing_class", f"{name}: missing class {label}")
    if len({label.casefold() for label in labels}) != len(labels):
        issue("class_case_collision", "Class names differ only by letter case.")
    for label in labels:
        if label != label.strip() or "\n" in label or "\r" in label:
            issue("invalid_label", f"Class name has surrounding whitespace or a newline: {label!r}")

    indexed: dict[str, dict[tuple[str, str], tuple[Path, str, tuple[int, int]]]] = {}
    source_paths: dict[str, dict[tuple[str, str], list[Path]]] = {}
    invalid: set[tuple[str, str]] = set()
    branch_counts: dict[str, dict[str, int]] = {}
    scanned_classes = 0
    total_classes = len(labels) * len(VARIANTS)
    for name in VARIANTS:
        indexed[name] = {}
        source_paths[name] = defaultdict(list)
        branch_counts[name] = {}
        seen_digest: dict[str, tuple[str, str]] = {}
        for label in labels:
            folder = root / name / label
            if not folder.is_dir():
                branch_counts[name][label] = 0
                scanned_classes += 1
                if progress:
                    progress(scanned_classes, total_classes)
                continue
            image_files = sorted((path for path in folder.rglob("*") if path.is_file()
                                  and path.suffix.lower() in IMAGE_EXTENSIONS),
                                 key=lambda path: path.as_posix())
            images = []
            for path in image_files:
                stem = path.stem.lower()
                suffix = next((value for value in VARIANT_SUFFIXES[name] if stem.endswith(value)), None)
                if suffix is not None:
                    images.append(path)
                elif name != "tactile" or not stem.endswith(("_depth", "_contact", "_normals")):
                    issue("unexpected_image", f"{name}: unrecognized image file {path}")
            branch_counts[name][label] = len(images)
            for path in images:
                relative = path.relative_to(folder)
                suffix = next(value for value in VARIANT_SUFFIXES[name]
                              if path.stem.lower().endswith(value))
                base = path.stem[:-len(suffix)]
                key = (label, (relative.parent / base).as_posix())
                source_paths[name][key].append(path)
                if key in indexed[name]:
                    invalid.add(key)
                    issue("duplicate_sample_id", f"{name}: multiple files for {label}/{key[1]}")
                    continue
                try:
                    with Image.open(path) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                        if min(image.size) < 16:
                            raise ValueError("image is smaller than 16 pixels")
                        size = image.size
                        digest = hashlib.sha256(image.tobytes() + str(size).encode()).hexdigest()
                except (OSError, ValueError) as exc:
                    invalid.add(key)
                    issue("unreadable_image", f"{name}: {path}: {exc}")
                    continue
                indexed[name][key] = (path, digest, size)
                previous = seen_digest.get(digest)
                if previous is not None:
                    invalid.add(key)
                    if previous[0] != label:
                        invalid.add(previous)
                        issue("cross_class_duplicate", f"{name}: identical RGB in {previous[0]}/{previous[1]} and {label}/{key[1]}")
                    else:
                        issue("duplicate_rgb", f"{name}: duplicate RGB in {label}/{previous[1]} and {key[1]}")
                else:
                    seen_digest[digest] = key
            scanned_classes += 1
            if progress:
                progress(scanned_classes, total_classes)

    all_keys = sorted(set.union(*(set(indexed[name]) for name in VARIANTS)) | invalid)
    unmatched_files = []
    for key in sorted(set.union(*(set(source_paths[name]) for name in VARIANTS))):
        present = [name for name in VARIANTS if key in source_paths[name]]
        if len(present) != len(VARIANTS):
            for name in present:
                for path in source_paths[name][key]:
                    unmatched_files.append({"variant": name, "label": key[0], "key": key[1],
                                            "path": str(path)})
    pairs = []
    excluded = []
    empty_contacts = 0
    for key in all_keys:
        if key in invalid:
            excluded.append({"label": key[0], "key": key[1], "reason": "duplicate_or_invalid_image"})
            continue
        present = [name for name in VARIANTS if key in indexed[name]]
        if len(present) != len(VARIANTS):
            issue("missing_sample", f"{key[0]}/{key[1]}: missing from {', '.join(name for name in VARIANTS if name not in present)}")
            excluded.append({"label": key[0], "key": key[1], "reason": "missing_sample"})
            continue
        sizes = {indexed[name][key][2] for name in VARIANTS}
        if len(sizes) != 1:
            issue("size_mismatch", f"{key[0]}/{key[1]}: image dimensions differ across branches")
            excluded.append({"label": key[0], "key": key[1], "reason": "size_mismatch"})
            continue
        tactile_path = indexed["tactile"][key][0]
        try:
            if _is_empty_contact(tactile_path):
                empty_contacts += 1
                excluded.append({"label": key[0], "key": key[1], "reason": "empty_contact"})
                continue
        except ValueError as exc:
            issue("invalid_metadata", str(exc))
            excluded.append({"label": key[0], "key": key[1], "reason": "invalid_metadata"})
            continue
        relative = Path(key[1])
        sample = SAMPLE_ID.match(relative.name)
        if group_by_subfolder and len(relative.parts) > 1:
            group = relative.parts[0]
        else:
            group = str(relative.parent / (sample.group(1) if sample else relative.name))
        pairs.append(PairedRecord(
            label=key[0], key=key[1], group=group,
            sources={name: indexed[name][key][0] for name in VARIANTS},
            digests={name: indexed[name][key][1] for name in VARIANTS},
        ))
    usable = {label: sum(pair.label == label for pair in pairs) for label in labels}
    groups = {label: len({pair.group for pair in pairs if pair.label == label}) for label in labels}
    for label, count in groups.items():
        if count < len(SPLITS):
            issue("insufficient_groups", f"{label}: {count} independent groups; at least three are required")
    summary = {"source": str(root), "labels": labels, "branch_counts": branch_counts,
               "matched_usable_samples": usable, "matched_total": len(pairs),
               "independent_groups": groups, "group_by_subfolder": group_by_subfolder,
               "excluded_empty_contacts": empty_contacts, "excluded_samples": excluded,
               "unmatched_files": unmatched_files,
               "issues": dict(issue_counts),
               "issue_examples": dict(examples),
               "meaning": "Matched sample IDs and sizes; RGB pixels may differ between render types."}
    return VariantScan(labels, pairs, summary)


def prepare_variants(root: str | Path, output: str | Path, ratios: tuple[int, int, int] = (70, 15, 15),
                     seed: int = 42, matched_only: bool = False, group_by_subfolder: bool = True,
                     balance_classes: bool = False,
                     progress=None, scan: VariantScan | None = None) -> dict:
    """Export three aligned ImageFolder datasets using one shared sample split."""
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Choose an output folder outside the source dataset.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new folder to preserve existing data.")
    if scan is None:
        scan = scan_variants(root, group_by_subfolder, progress=progress)
    elif scan.summary["source"] != str(root) or scan.summary["group_by_subfolder"] != group_by_subfolder:
        raise ValueError("Source or grouping changed since the scan; scan again.")
    if any(scan.summary["issues"].get(kind) for kind in
           ("missing_class", "class_case_collision", "invalid_label", "insufficient_groups")):
        raise ValueError("Class folders or independent group counts are unsuitable for three splits. Review the scan.")
    if scan.summary["issues"] and not matched_only:
        raise ValueError("The branches have missing, invalid, or duplicate samples. Review the scan or enable matched-only export.")
    if any(scan.summary["matched_usable_samples"][label] == 0 for label in scan.labels):
        raise ValueError("Every class needs usable matched samples in all three branches.")
    assignments = split_records(scan.pairs, ratios, seed)
    balance_removed = {split: {label: 0 for label in scan.labels} for split in SPLITS}
    if balance_classes:
        for split in SPLITS:
            target = min(sum(pair.label == label for pair in assignments[split]) for label in scan.labels)
            selected = []
            for label in scan.labels:
                candidates = [pair for pair in assignments[split] if pair.label == label]
                kept = random.Random(f"{seed}:{split}:{label}:balance").sample(candidates, target)
                selected.extend(kept)
                balance_removed[split][label] = len(candidates) - target
            assignments[split] = selected
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.incomplete-{uuid4().hex[:8]}")
    staging.mkdir(exist_ok=False)
    labels_text = "\n".join(scan.labels) + "\n"
    (staging / "labels.txt").write_text(labels_text, encoding="utf-8")
    per_variant = {name: [] for name in VARIANTS}
    paired_rows = []
    counts = {split: {label: 0 for label in scan.labels} for split in SPLITS}
    total = sum(map(len, assignments.values())) * len(VARIANTS)
    completed = 0
    for name in VARIANTS:
        (staging / name).mkdir()
        (staging / name / "labels.txt").write_text(labels_text, encoding="utf-8")
    for split in SPLITS:
        for label in scan.labels:
            items = sorted((pair for pair in assignments[split] if pair.label == label), key=lambda pair: pair.key)
            counts[split][label] = len(items)
            for name in VARIANTS:
                (staging / name / split / label).mkdir(parents=True)
            for index, pair in enumerate(items, 1):
                relative = Path(split) / label / f"image_{index:06d}.png"
                paired_rows.append((split, label, pair.key, pair.group, relative.as_posix()))
                for name in VARIANTS:
                    target = staging / name / relative
                    with Image.open(pair.sources[name]) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
                        if digest != pair.digests[name]:
                            raise ValueError(f"Source image changed since the scan: {pair.sources[name]}")
                        image.save(target, format="PNG")
                    per_variant[name].append((split, label, pair.key, str(pair.sources[name]),
                                              relative.as_posix(), pair.group, pair.digests[name]))
                    completed += 1
                    if progress:
                        progress(completed, total)
    for name in VARIANTS:
        with (staging / name / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("split", "label", "key", "source", "output", "group", "rgb_sha256"))
            writer.writerows(per_variant[name])
    with (staging / "paired_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "label", "key", "group", "output"))
        writer.writerows(paired_rows)
    summary = {**scan.summary, "ratios_requested": dict(zip(SPLITS, ratios)), "seed": seed,
               "matched_only": matched_only, "source_modified": False,
               "excluded_from_each_output": len(scan.summary["excluded_samples"]),
               "balance_classes": balance_classes,
               "balance_removed": balance_removed, "exported_total": total // len(VARIANTS),
               "counts": counts,
               "output_layout": "<image_type>/<train|val|test>/<class>/image_######.png",
               "note": "The same source sample IDs and split assignments are used in all image types."}
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        raise FileExistsError(f"Output appeared during export: {output}. Partial data remains at {staging}.")
    staging.rename(output)
    return summary


def main() -> None:
    from PyQt5 import QtCore, QtWidgets

    class Worker(QtCore.QThread):
        progress = QtCore.pyqtSignal(int, int)
        scanned = QtCore.pyqtSignal(object)
        finished_ok = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, mode, source, output=None, ratios=None, seed=42,
                     matched_only=False, group_subfolders=True, balance_classes=False,
                     scan_result=None):
            super().__init__()
            self.mode = mode
            self.options = (source, output, ratios, seed, matched_only, group_subfolders, balance_classes)
            self.scan_result = scan_result

        def run(self):
            try:
                if self.mode == "scan":
                    source, _, _, _, _, group_subfolders, _ = self.options
                    self.scanned.emit(scan_variants(source, group_subfolders, progress=self.progress.emit))
                else:
                    self.finished_ok.emit(prepare_variants(*self.options, progress=self.progress.emit,
                                                           scan=self.scan_result))
            except Exception as exc:
                self.failed.emit(str(exc))

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Mesh2Tact dataset preparation")
            self.resize(900, 650)
            layout = QtWidgets.QVBoxLayout(self)
            form = QtWidgets.QFormLayout()
            self.source = QtWidgets.QLineEdit()
            self.output = QtWidgets.QLineEdit()
            for title, field, picker in (("Folder with clean / tactile / default", self.source, self.pick_source),
                                         ("New output folder", self.output, self.pick_output)):
                row = QtWidgets.QHBoxLayout()
                row.addWidget(field)
                button = QtWidgets.QPushButton("Browse…")
                button.clicked.connect(picker)
                row.addWidget(button)
                form.addRow(title, row)
            layout.addLayout(form)
            ratios_row = QtWidgets.QHBoxLayout()
            self.spins = []
            for title, value in (("Train %", 70), ("Validation %", 15), ("Test %", 15)):
                ratios_row.addWidget(QtWidgets.QLabel(title))
                spin = QtWidgets.QSpinBox()
                spin.setRange(1, 98)
                spin.setValue(value)
                ratios_row.addWidget(spin)
                self.spins.append(spin)
            layout.addLayout(ratios_row)
            self.seed = QtWidgets.QSpinBox()
            self.seed.setRange(0, 2_147_483_647)
            self.seed.setValue(42)
            form.addRow("Random seed", self.seed)
            self.group_subfolders = QtWidgets.QCheckBox("Keep each class subfolder together (trial/run grouping)")
            self.group_subfolders.setChecked(True)
            self.group_subfolders.toggled.connect(self.invalidate_scan)
            layout.addWidget(self.group_subfolders)
            self.matched_only = QtWidgets.QCheckBox(
                "Exclude missing, duplicate, or invalid captures from all three OUTPUT datasets")
            self.matched_only.setChecked(True)
            self.matched_only.toggled.connect(self.update_prepare_enabled)
            layout.addWidget(self.matched_only)
            self.balance_classes = QtWidgets.QCheckBox("Balance classes within train, validation, and test (matched triplets)")
            layout.addWidget(self.balance_classes)
            matching_note = QtWidgets.QLabel("Matching means the same class and sample IDs in all three branches. "
                                             "Their RGB appearance is expected to differ.")
            matching_note.setWordWrap(True)
            layout.addWidget(matching_note)
            self.scan_button = QtWidgets.QPushButton("Scan and compare branches")
            self.scan_button.clicked.connect(self.scan)
            layout.addWidget(self.scan_button)
            self.start = QtWidgets.QPushButton("Prepare dataset")
            self.start.clicked.connect(self.prepare)
            self.start.setEnabled(False)
            layout.addWidget(self.start)
            self.bar = QtWidgets.QProgressBar()
            layout.addWidget(self.bar)
            self.status = QtWidgets.QPlainTextEdit()
            self.status.setReadOnly(True)
            layout.addWidget(self.status)
            self.worker = None
            self.scanned_source = None
            self.scan_result = None

        def invalidate_scan(self):
            self.scanned_source = None
            self.scan_result = None
            self.start.setEnabled(False)

        def update_prepare_enabled(self):
            if self.scan_result is None:
                self.start.setEnabled(False)
                return
            summary = self.scan_result.summary
            fatal = sum(summary["issues"].get(kind, 0) for kind in
                        ("missing_class", "class_case_collision", "invalid_label", "insufficient_groups"))
            issues = sum(summary["issues"].values())
            self.start.setEnabled(not fatal and (not issues or self.matched_only.isChecked()))

        def pick_source(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder containing clean, tactile, and default")
            if folder:
                self.source.setText(folder)
                self.output.setText(str(Path(folder).parent / "ml_dataset"))
                self.invalidate_scan()

        def pick_output(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output parent folder")
            if folder:
                self.output.setText(str(Path(folder) / "ml_dataset"))

        def scan(self):
            if not self.source.text().strip():
                QtWidgets.QMessageBox.warning(self, "Missing folder", "Choose the source folder first.")
                return
            self.start.setEnabled(False)
            self.scan_button.setEnabled(False)
            self.status.setPlainText("Scanning class folders, sample IDs, and image contents…")
            self.worker = Worker("scan", self.source.text(), group_subfolders=self.group_subfolders.isChecked())
            self.worker.progress.connect(self.on_progress)
            self.worker.scanned.connect(self.on_scan)
            self.worker.failed.connect(self.on_error)
            self.worker.finished.connect(lambda: self.scan_button.setEnabled(True))
            self.worker.start()

        def on_scan(self, scan_result):
            self.scanned_source = str(Path(self.source.text()).expanduser().resolve())
            self.scan_result = scan_result
            excluded = len(scan_result.summary["excluded_samples"])
            self.status.setPlainText(
                f"{excluded} capture index(es) will be omitted from EACH output dataset when exclusion is enabled.\n"
                "Source photos will not be changed.\n\n" + json.dumps(scan_result.summary, indent=2))
            self.update_prepare_enabled()

        def prepare(self):
            ratios = tuple(spin.value() for spin in self.spins)
            if sum(ratios) != 100:
                QtWidgets.QMessageBox.warning(self, "Invalid split", "Percentages must add up to 100.")
                return
            if not self.source.text().strip() or not self.output.text().strip():
                QtWidgets.QMessageBox.warning(self, "Missing folder", "Choose source and output folders.")
                return
            if self.scanned_source != str(Path(self.source.text()).expanduser().resolve()):
                QtWidgets.QMessageBox.warning(self, "Scan first", "Scan the selected source folder before preparing.")
                return
            self.start.setEnabled(False)
            self.scan_button.setEnabled(False)
            self.status.setPlainText("Checking pairs again and exporting aligned datasets…")
            self.worker = Worker("prepare", self.source.text(), self.output.text(), ratios,
                                 self.seed.value(), self.matched_only.isChecked(),
                                 self.group_subfolders.isChecked(), self.balance_classes.isChecked(),
                                 scan_result=self.scan_result)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished_ok.connect(self.on_done)
            self.worker.failed.connect(self.on_error)
            self.worker.finished.connect(lambda: self.scan_button.setEnabled(True))
            self.worker.start()

        def on_progress(self, completed, total):
            self.bar.setMaximum(total)
            self.bar.setValue(completed)

        def on_done(self, summary):
            self.status.setPlainText(json.dumps(summary, indent=2))
            QtWidgets.QMessageBox.information(
                self, "Datasets ready",
                f"Aligned datasets were written. {summary['excluded_from_each_output']} problematic "
                "capture index(es) were omitted from clean, tactile, and default. Source photos are unchanged.")

        def on_error(self, message):
            self.invalidate_scan()
            self.status.setPlainText(message)
            QtWidgets.QMessageBox.critical(self, "Preparation failed", message)

    app = QtWidgets.QApplication([])
    window = Window()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
