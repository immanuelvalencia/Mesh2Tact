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
VARIANTS = ("clean", "default", "tactile", "mask")
# Both the current short names and older Mesh2Tact export names identify the
# same capture.  Only the terminal image-type suffix is removed for pairing.
VARIANT_SUFFIXES = {"clean": ("_tactile_clean", "_clean"),
                    "tactile": ("_tactile",),
                    "default": ("_tactile_default", "_default"),
                    "mask": ("_contact", "_mask")}


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


def selected_variants(root, variants=None):
    names = tuple(name for name in VARIANTS if (Path(root) / name).is_dir()) if variants is None else tuple(variants)
    if not names or len(set(names)) != len(names) or any(name not in VARIANTS for name in names):
        raise ValueError('Select at least one supported image type')
    return names


def sample_inventory(root, variants=None, labels=None):
    """Index all recognized captures, including incomplete/unreadable samples."""
    root = Path(root).resolve()
    result = defaultdict(lambda: defaultdict(list))
    for name in (VARIANTS if variants is None else variants):
        branch = root / name
        if not branch.is_dir():
            continue
        for folder in sorted(branch.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or (labels is not None and folder.name not in labels):
                continue
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                suffix = next((s for s in VARIANT_SUFFIXES[name] if path.stem.lower().endswith(s)), None)
                if suffix:
                    key = (path.relative_to(folder).parent / path.stem[:-len(suffix)]).as_posix()
                    result[(folder.name, key)][name].append(path)
    return {key: dict(value) for key, value in sorted(result.items())}


def raw_inventory(root, labels=None):
    root = Path(root).resolve()
    return {(folder.name, path.relative_to(folder).as_posix()): {'gelsight': [path]}
            for folder in sorted(root.iterdir())
            if folder.is_dir() and not folder.name.startswith('.')
            and (labels is None or folder.name in labels)
            for path in sorted(folder.rglob('*'))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def mask_contact_pixels(path):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
        colors = image.getcolors(maxcolors=2)
        if colors is None or any(color not in ((0, 0, 0), (255, 255, 255)) for _, color in colors):
            raise ValueError('Mask must be binary (0 or 255)')
        return sum(count for count, color in colors if color == (255, 255, 255))


def quarantine_samples(root, keys, raw=False):
    """Remove whole sample identities across every available branch; retain undo data."""
    root = Path(root).resolve()
    inventory = raw_inventory(root) if raw else sample_inventory(root)
    paths = [path for key in set(keys) for paths in inventory.get(tuple(key), {}).values() for path in paths]
    if not paths:
        raise ValueError('No matching source images remain; rescan')
    for path in paths:
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Refusing to move images outside the source root')
    trash = root / '.inspection-trash' / uuid4().hex
    trash.mkdir(parents=True)
    manifest = trash / 'removed.json'
    relative_paths = [str(path.relative_to(root)) for path in paths]
    manifest.write_text(json.dumps({'root': str(root), 'files': relative_paths}, indent=2), encoding='utf-8')
    moved = []
    try:
        for path in paths:
            target = trash / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            moved.append((path, target))
    except Exception:
        for path, target in reversed(moved):
            target.rename(path)
        raise
    return manifest


def restore_quarantine(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    root = Path(data['root']).resolve()
    if not manifest.is_relative_to(root / '.inspection-trash'):
        raise ValueError('Invalid recovery location')
    moves = []
    for relative in data['files']:
        source, target = (manifest.parent / relative).resolve(), (root / relative).resolve()
        if not source.is_relative_to(manifest.parent) or not target.is_relative_to(root):
            raise ValueError('Invalid recovery path')
        if target.exists() or not source.is_file():
            raise ValueError(f'Cannot restore without overwriting or missing files: {target}')
        moves.append((source, target))
    restored = []
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            restored.append((source, target))
    except Exception:
        for source, target in reversed(restored):
            target.rename(source)
        raise


def _classes(source: Path, gelsight=False) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"Dataset folder does not exist: {source}")
    names = {path.name.lower() for path in source.iterdir() if path.is_dir()}
    if not gelsight and len(set(VARIANTS) & names) >= 2:
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


def scan_dataset(source: str | Path, group_by_subfolder: bool = True, selected_labels=None, gelsight=False) -> tuple[list[str], list[ImageRecord], dict]:
    """Validate images and remove identical images within each class."""
    source = Path(source).expanduser().resolve()
    labels = [folder.name for folder in _classes(source, gelsight)
              if selected_labels is None or folder.name in selected_labels]
    if not labels:
        raise ValueError('Select at least one class.')
    records: list[ImageRecord] = []
    digest_owner: dict[str, str] = {}
    skipped = defaultdict(int)
    capture_suffixes = None if gelsight else VARIANT_SUFFIXES.get(source.name.lower())
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
                    seed: int = 42, group_by_subfolder: bool = True, progress=None,
                    selected_labels=None, gelsight=False, balance_classes=False) -> dict:
    """Write a fresh RGB ImageFolder export, labels, and an auditable manifest."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose an output folder outside the source dataset.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new folder to preserve existing data.")
    labels, records, skipped = scan_dataset(source, group_by_subfolder, selected_labels, gelsight)
    if len(labels) < 2:
        raise ValueError('At least two classes are required for export.')
    assignments = split_records(records, ratios, seed)
    if balance_classes:
        for split, items in assignments.items():
            count = min(sum(r.label == label for r in items) for label in labels)
            balanced = []
            for label in labels:
                candidates = [r for r in items if r.label == label]
                random.Random(f'{seed}:{split}:{label}').shuffle(candidates)
                balanced.extend(candidates[:count])
            assignments[split] = balanced
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.incomplete-{uuid4().hex[:8]}")
    staging.mkdir(exist_ok=False)
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    rows = []
    try:
        (staging / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        total = sum(len(items) for items in assignments.values())
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
                   "dataset_kind": "gelsight" if gelsight else "single",
                   "note": "Splits evaluate new images of the listed classes, not unseen object instances."}
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
        return summary
    except Exception:
        # Retain this run's partial export for inspection. The requested output is untouched.
        raise


def scan_variants(root: str | Path, group_by_subfolder: bool = True, progress=None,
                  variants=None, selected_labels=None, require_contact=False, min_contact_pixels=1) -> VariantScan:
    """Compare class/sample identity across selected image branches, read only.

    Matching means the same class and sample IDs, not pixel equality: the
    render types intentionally have different RGB appearances.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Source folder does not exist: {root}")
    variants = selected_variants(root, variants)
    if type(min_contact_pixels) is not int or min_contact_pixels < 1:
        raise ValueError("Minimum contact pixels must be a positive integer")
    mask_index = sample_inventory(root, ("mask",)) if require_contact else {}
    missing_branches = [name for name in variants if not (root / name).is_dir()]
    if missing_branches:
        raise ValueError("Missing image branches: " + ", ".join(missing_branches))

    class_sets = {
        name: {folder.name for folder in (root / name).iterdir()
               if folder.is_dir() and not folder.name.startswith(".")}
        for name in variants
    }
    labels = sorted(set.union(*class_sets.values()))
    if selected_labels is not None:
        if not selected_labels or not set(selected_labels).issubset(labels):
            raise ValueError("Choose existing class folders")
        labels = sorted(set(selected_labels))
    if not labels:
        raise ValueError("No class folders found")
    issue_counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)

    def issue(kind: str, detail: str):
        issue_counts[kind] += 1
        if len(examples[kind]) < 8:
            examples[kind].append(detail)

    for name in variants:
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
    total_classes = len(labels) * len(variants)
    for name in variants:
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
                    if name == 'mask':
                        mask_contact_pixels(path)
                except (OSError, ValueError) as exc:
                    invalid.add(key)
                    issue("unreadable_image", f"{name}: {path}: {exc}")
                    continue
                indexed[name][key] = (path, digest, size)
                # Repeated binary silhouettes are legitimate, including across classes.
                previous = seen_digest.get(digest) if name != "mask" else None
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

    all_keys = sorted(set.union(*(set(indexed[name]) for name in variants)) | invalid)
    unmatched_files = []
    for key in sorted(set.union(*(set(source_paths[name]) for name in variants))):
        present = [name for name in variants if key in source_paths[name]]
        if len(present) != len(variants):
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
        present = [name for name in variants if key in indexed[name]]
        if len(present) != len(variants):
            issue("missing_sample", f"{key[0]}/{key[1]}: missing from {', '.join(name for name in variants if name not in present)}")
            excluded.append({"label": key[0], "key": key[1], "reason": "missing_sample"})
            continue
        sizes = {indexed[name][key][2] for name in variants}
        if len(sizes) != 1:
            issue("size_mismatch", f"{key[0]}/{key[1]}: image dimensions differ across branches")
            excluded.append({"label": key[0], "key": key[1], "reason": "size_mismatch"})
            continue
        if require_contact:
            paths = mask_index.get(key, {}).get("mask", [])
            try:
                if len(paths) != 1:
                    raise ValueError("missing or ambiguous mask")
                pixels = mask_contact_pixels(paths[0])
                if pixels < min_contact_pixels:
                    empty_contacts += 1
                    excluded.append({"label": key[0], "key": key[1], "reason": "empty_contact"})
                    continue
            except (OSError, ValueError) as exc:
                issue("invalid_mask", f"{key}: {exc}")
                excluded.append({"label": key[0], "key": key[1], "reason": "invalid_mask"})
                continue
        tactile_path = indexed['tactile' if 'tactile' in variants else variants[0]][key][0]
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
            sources={name: indexed[name][key][0] for name in variants},
            digests={name: indexed[name][key][1] for name in variants},
        ))
    usable = {label: sum(pair.label == label for pair in pairs) for label in labels}
    groups = {label: len({pair.group for pair in pairs if pair.label == label}) for label in labels}
    for label, count in groups.items():
        if count < len(SPLITS):
            issue("insufficient_groups", f"{label}: {count} independent groups; at least three are required")
    summary = {"variants": list(variants), "selected_labels": list(selected_labels) if selected_labels is not None else None,
               "require_contact": require_contact, "min_contact_pixels": min_contact_pixels,
               "source": str(root), "labels": labels, "branch_counts": branch_counts,
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
                     progress=None, scan: VariantScan | None = None, variants=None, selected_labels=None,
                     require_contact=False, min_contact_pixels=1) -> dict:
    """Export selected aligned ImageFolder datasets using one shared sample split."""
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Choose an output folder outside the source dataset.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new folder to preserve existing data.")
    variants = selected_variants(root, variants)
    if scan is not None and (scan.summary.get("variants") != list(variants)
            or scan.summary.get("selected_labels") != (list(selected_labels) if selected_labels is not None else None)
            or scan.summary.get("require_contact", False) != require_contact
            or scan.summary.get("min_contact_pixels", 1) != min_contact_pixels):
        raise ValueError("Selection or contact criteria changed since the scan; scan again")
    if scan is None:
        scan = scan_variants(root, group_by_subfolder, progress=progress, variants=variants,
                             selected_labels=selected_labels, require_contact=require_contact, min_contact_pixels=min_contact_pixels)
    elif scan.summary["source"] != str(root) or scan.summary["group_by_subfolder"] != group_by_subfolder:
        raise ValueError("Source or grouping changed since the scan; scan again.")
    if len(scan.labels) < 2:
        raise ValueError("At least two classes are required for training export")
    if any(scan.summary["issues"].get(kind) for kind in
           ("missing_class", "class_case_collision", "invalid_label", "insufficient_groups")):
        raise ValueError("Class folders or independent group counts are unsuitable for three splits. Review the scan.")
    if scan.summary["issues"] and not matched_only:
        raise ValueError("The branches have missing, invalid, or duplicate samples. Review the scan or enable matched-only export.")
    if any(scan.summary["matched_usable_samples"][label] == 0 for label in scan.labels):
        raise ValueError("Every class needs usable matched samples in all selected branches.")
    assignments = split_records(scan.pairs, ratios, seed)
    if require_contact:
        masks = sample_inventory(root, ('mask',))
        for pair in scan.pairs:
            paths = masks.get((pair.label, pair.key), {}).get('mask', [])
            if len(paths) != 1 or mask_contact_pixels(paths[0]) < min_contact_pixels:
                raise ValueError(f'Mask contact changed since the scan: {pair.label}/{pair.key}; scan again')
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
    per_variant = {name: [] for name in variants}
    paired_rows = []
    counts = {split: {label: 0 for label in scan.labels} for split in SPLITS}
    total = sum(map(len, assignments.values())) * len(variants)
    completed = 0
    for name in variants:
        (staging / name).mkdir()
        (staging / name / "labels.txt").write_text(labels_text, encoding="utf-8")
    for split in SPLITS:
        for label in scan.labels:
            items = sorted((pair for pair in assignments[split] if pair.label == label), key=lambda pair: pair.key)
            counts[split][label] = len(items)
            for name in variants:
                (staging / name / split / label).mkdir(parents=True)
            for index, pair in enumerate(items, 1):
                relative = Path(split) / label / f"image_{index:06d}.png"
                paired_rows.append((split, label, pair.key, pair.group, relative.as_posix()))
                for name in variants:
                    target = staging / name / relative
                    with Image.open(pair.sources[name]) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
                        if digest != pair.digests[name]:
                            raise ValueError(f"Source image changed since the scan: {pair.sources[name]}")
                        image.save(target, format="PNG")
                    per_variant[name].append((split, label, pair.key, str(pair.sources[name]),
                                              relative.as_posix(), pair.group, pair.digests[name], name))
                    completed += 1
                    if progress:
                        progress(completed, total)
    for name in variants:
        with (staging / name / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("split", "label", "key", "source", "output", "group", "rgb_sha256", "image_type"))
            writer.writerows(per_variant[name])
    with (staging / "paired_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "label", "key", "group", "output"))
        writer.writerows(paired_rows)
    summary = {**scan.summary, "ratios_requested": dict(zip(SPLITS, ratios)), "seed": seed,
               "matched_only": matched_only, "source_modified": False,
               "excluded_from_each_output": len(scan.summary["excluded_samples"]),
               "balance_classes": balance_classes,
               "balance_removed": balance_removed, "exported_total": total // len(variants),
               "counts": counts,
               "output_layout": "<image_type>/<train|val|test>/<class>/image_######.png",
               "note": "The same source sample IDs and split assignments are used in all image types."}
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        raise FileExistsError(f"Output appeared during export: {output}. Partial data remains at {staging}.")
    staging.rename(output)
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gelsight', action='store_true', help='Raw RGB class-folder dataset')
    parser.add_argument('--source', type=Path, help='Folder containing class folders')
    parser.add_argument('--output', type=Path, help='New prepared dataset folder')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.source or args.output:
        if not args.gelsight or not args.source or not args.output:
            parser.error('CLI export requires --gelsight --source and --output')
        print(json.dumps(prepare_dataset(args.source, args.output, seed=args.seed, gelsight=True), indent=2))
        return
    from mesh2tact.gui.preprocess_app import run
    run(gelsight=args.gelsight)


if __name__ == "__main__":
    main()
