"""Train torchvision classifiers on a dataset exported by preprocess.py.

The architecture suites, classifier replacement, validation selection, and test
reports are integrated with Mesh2Tact's Predict tab.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mesh2tact.architectures import ALL_MODELS, SUITES

IMAGE_TYPES = ("tactile", "default", "clean", "mask")
PROGRESS_PREFIX = "@@M2T_PROGRESS@@"


class TrainingStopped(Exception):
    """The UI requested that the sequential training queue stop."""


def _check_stop(stop_file: Path | None) -> None:
    if stop_file is not None and stop_file.is_file():
        raise TrainingStopped("Stop requested; no further models will be started.")


def _progress_event(enabled: bool, **values) -> None:
    if enabled:
        print(PROGRESS_PREFIX + json.dumps(values, separators=(",", ":")), flush=True)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", "--dataset_dir", required=True, type=Path)
    dataset_mode = parser.add_mutually_exclusive_group()
    dataset_mode.add_argument("--image-type", choices=(*IMAGE_TYPES, "gelsight"),
                              help="Select an aligned branch, or gelsight for a single prepared raw RGB dataset")
    dataset_mode.add_argument("--all-image-types", action="store_true",
                              help="Train each aligned branch independently")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("train"))
    parser.add_argument("--models", nargs="+", default=[], help="Architecture names, e.g. resnet18 swin_t")
    parser.add_argument("--all", action="store_true", help="Train the full architecture suite")
    for family in SUITES:
        parser.add_argument(f"--{family}", action="store_true", help=f"Train all {family} models")
    parser.add_argument("--weights", choices=("default", "none", "all"), default="default",
                        help="Pretrained weight selection; all runs each available weight variant")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--test-grid-per-class", type=int, default=4,
                        help="Held-out test photos saved per label for the reusable visual test grid")
    parser.add_argument("--ui-progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1
            or args.seed < 0 or args.workers < 0 or args.test_grid_per_class < 1):
        parser.error("Epochs, batch size, learning rate, and patience must be positive; seed/workers nonnegative.")
    selected = []
    if args.all:
        selected.extend(ALL_MODELS)
    for family, names in SUITES.items():
        if getattr(args, family):
            selected.extend(names)
    selected.extend(args.models)
    args.selected_models = list(dict.fromkeys(selected))
    if not args.selected_models:
        parser.error("Select --models, a family flag, or --all.")
    unknown = sorted(set(args.selected_models) - set(ALL_MODELS))
    if unknown:
        parser.error("Unsupported architectures: " + ", ".join(unknown))
    return args


def resolve_datasets(root: Path, image_type: str | None, all_image_types: bool) -> list[tuple[str | None, Path]]:
    root = root.expanduser().resolve()
    if (root / "train").is_dir() and (root / "val").is_dir() and (root / "test").is_dir():
        if image_type == "gelsight" and not all_image_types:
            return [("gelsight", root)]
        if image_type or all_image_types:
            raise ValueError("dataset-dir already points to a single split dataset; omit image-type flags.")
        return [(None, root)]
    if image_type == "gelsight":
        raise ValueError("GelSight training needs a prepared train/val/test dataset. Select GelSight in preprocess.py and export your class folders first.")
    available = [name for name in IMAGE_TYPES if (root / name / "train").is_dir()]
    if not available:
        raise ValueError(f"No train/val/test dataset found in {root}. Run preprocess.py first.")
    _validate_aligned_manifests(root, available)
    if image_type:
        if image_type not in available:
            raise ValueError(f"{image_type} dataset is missing from {root}.")
        return [(image_type, root / image_type)]
    if all_image_types:
        return [(name, root / name) for name in available]
    raise ValueError("Choose --image-type clean|tactile|default|mask or --all-image-types for this dataset root.")


def _validate_aligned_manifests(root: Path, variants: list[str]) -> None:
    paired = root / "paired_manifest.csv"
    labels = root / "labels.txt"
    if not paired.is_file() or not labels.is_file():
        raise ValueError(f"Missing paired_manifest.csv or labels.txt in {root}; prepare the branches together.")

    def entries(path: Path) -> list[tuple[str, str, str, str]]:
        if not path.is_file():
            raise ValueError(f"Missing manifest: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"split", "label", "key", "output"} <= set(reader.fieldnames or ()):
                raise ValueError(f"Missing pairing columns in {path}")
            return sorted((row["split"], row["label"], row["key"], row["output"]) for row in reader)

    expected = entries(paired)
    if not expected:
        raise ValueError("The paired manifest is empty.")
    label_text = labels.read_text(encoding="utf-8")
    for variant in variants:
        branch = root / variant
        branch_labels = branch / "labels.txt"
        if not branch_labels.is_file() or branch_labels.read_text(encoding="utf-8") != label_text:
            raise ValueError(f"{variant} labels.txt differs from the shared labels.txt")
        if entries(branch / "manifest.csv") != expected:
            raise ValueError(f"{variant} sample IDs or split assignments differ from paired_manifest.csv")


def replace_classifier(model, name: str, classes: int, nn):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, classes)
        elif isinstance(classifier, nn.Sequential):
            for index in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[index], nn.Linear):
                    classifier[index] = nn.Linear(classifier[index].in_features, classes)
                    break
            else:
                raise ValueError(f"No linear classifier in {name}")
        else:
            raise ValueError(f"Unsupported classifier in {name}")
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, classes)
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = nn.Linear(model.heads.head.in_features, classes)
    else:
        raise ValueError(f"Cannot replace classifier in {name}")


def _dataset(root: Path, transforms, datasets, labels):
    image_sets = {}
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f"Missing required split: {folder}")
        image_sets[split] = datasets.ImageFolder(folder, transforms)
        if image_sets[split].classes != labels:
            raise ValueError(f"{split} class order does not match labels.txt: {image_sets[split].classes}")
        counts = Counter(index for _, index in image_sets[split].samples)
        missing = [labels[index] for index in range(len(labels)) if counts[index] == 0]
        if missing:
            raise ValueError(f"{split} has no images for: {', '.join(missing)}")
    manifest = root / "manifest.csv"
    if manifest.is_file():
        seen: dict[str, str] = {}
        groups = {}
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                digest, split = row["rgb_sha256"], row["split"]
                group = (row.get('label'), row.get('group'))
                if row.get('group') and group in groups and groups[group] != split:
                    raise ValueError('An acquisition group appears in more than one split.')
                groups[group] = split
                # Identical binary silhouettes do not imply identical captures.
                is_mask = row.get('image_type', root.name) == 'mask'
                if not is_mask and digest in seen and seen[digest] != split:
                    raise ValueError("Identical image content appears in more than one split.")
                seen[digest] = split
    return image_sets


def _weights_for(name: str, option: str, models):
    if option == "none":
        return [("scratch", None)]
    choices = models.get_model_weights(name)
    if option == "default":
        return [(choices.DEFAULT.name, choices.DEFAULT)]
    return [(weight.name, weight) for weight in choices]


def _epoch(model, loader, criterion, optimizer, device, train_mode, torch, progress=None,
           stop_file=None):
    model.train(train_mode)
    loss_sum = 0.0
    correct = 0
    count = 0
    total_batches = len(loader)
    report_interval = max(1, math.ceil(total_batches / 10))
    for batch_index, (images, targets) in enumerate(loader, 1):
        _check_stop(stop_file)
        images = images.to(device)
        targets = targets.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, targets)
            if train_mode:
                loss.backward()
                optimizer.step()
        batch = len(targets)
        loss_sum += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        count += batch
        if progress is not None and (batch_index % report_interval == 0 or batch_index == total_batches):
            progress(batch_index, total_batches, loss_sum / count, correct / count)
        _check_stop(stop_file)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def _test(model, loader, device, torch, progress=None, stop_file=None):
    truth, predicted = [], []
    model.eval()
    total_batches = len(loader)
    report_interval = max(1, math.ceil(total_batches / 10))
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader, 1):
            _check_stop(stop_file)
            logits = model(images.to(device))
            truth.extend(int(value) for value in targets.tolist())
            predicted.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
            if progress is not None and (batch_index % report_interval == 0 or batch_index == total_batches):
                correct = sum(left == right for left, right in zip(truth, predicted))
                progress(batch_index, total_batches, correct / len(truth))
            _check_stop(stop_file)
    return truth, predicted


def save_test_grid(dataset_root, image_set, labels, folder, per_class):
    """Save copied held-out photos, a labelled contact sheet, and their index map."""
    from PIL import Image, ImageDraw, ImageOps

    image_dir = folder / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    source_keys = {}
    manifest = dataset_root / "manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "test":
                    source_keys[row["output"]] = row.get("key", "")
    selections = []
    for label_index, label in enumerate(labels):
        candidates = [(index, path) for index, (path, target) in enumerate(image_set.samples)
                      if target == label_index]
        count = min(per_class, len(candidates))
        positions = ([0] if count == 1 else
                     [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)])
        for position in positions:
            index, source = candidates[position]
            relative_source = Path(source).relative_to(dataset_root).as_posix()
            target = image_dir / f"{len(selections) + 1:04d}.png"
            with Image.open(source) as raw:
                ImageOps.exif_transpose(raw).convert("RGB").save(target)
            selections.append({"dataset_index": index, "label": label,
                               "key": source_keys.get(relative_source, Path(source).stem),
                               "source": str(Path(source).resolve()),
                               "image": target.relative_to(folder).as_posix()})
    if not selections:
        raise ValueError("The held-out test split has no images for a test grid.")
    cell_w, cell_h, thumb = 184, 214, 168
    canvas = Image.new("RGB", (cell_w * per_class, cell_h * len(labels)), "white")
    draw = ImageDraw.Draw(canvas)
    for label_index, label in enumerate(labels):
        label_rows = [entry for entry in selections if entry["label"] == label]
        for column, entry in enumerate(label_rows):
            x, y = column * cell_w, label_index * cell_h
            with Image.open(folder / entry["image"]) as raw:
                tile = ImageOps.contain(raw.convert("RGB"), (thumb, thumb))
            canvas.paste(tile, (x + (cell_w - tile.width) // 2, y + 3))
            draw.text((x + 6, y + 175), label[:24], fill="black")
            draw.text((x + 6, y + 190), entry["key"][-26:], fill="black")
    canvas.save(folder / "test_grid.png")
    with (folder / "test_grid_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset_index", "label", "key", "source", "image"))
        writer.writeheader()
        writer.writerows(selections)
    return folder, selections


def _save_test_predictions(run_dir, image_set, labels, truth, predicted, grid_folder, grid_selections):
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source", "true_label", "predicted_label", "correct"))
        for (source, _), true, result in zip(image_set.samples, truth, predicted):
            writer.writerow((source, labels[true], labels[result], true == result))
    with (run_dir / "test_grid_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("grid_image", "capture_key", "true_label", "predicted_label", "correct"))
        for entry in grid_selections:
            index = entry["dataset_index"]
            image = (grid_folder / entry["image"]).relative_to(run_dir).as_posix()
            writer.writerow((image, entry["key"],
                             labels[truth[index]], labels[predicted[index]], truth[index] == predicted[index]))


def _save_reports(run_dir, labels, history, truth, predicted):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    indices = list(range(len(labels)))
    report = classification_report(truth, predicted, labels=indices, target_names=labels,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(truth, predicted, labels=indices, target_names=labels, zero_division=0)
    (run_dir / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"))
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, axis in (("loss", axes[0]), ("accuracy", axes[1])):
        for split in ("train", "val"):
            axis.plot([row[f"{split}_{key}"] for row in history], label=split)
        axis.set_title(key.title())
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_history.png", dpi=150)
    plt.close(fig)
    matrix = confusion_matrix(truth, predicted, labels=indices)
    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("actual\\predicted", *labels))
        for label, row in zip(labels, matrix):
            writer.writerow((label, *row.tolist()))
    fig, axis = plt.subplots(figsize=(max(6, len(labels) * 0.7), max(5, len(labels) * 0.7)))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(indices, labels, rotation=90)
    axis.set_yticks(indices, labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return float(report["accuracy"])


def train_one(name, weight_name, weight, args, labels, loaders, device, torch, models, nn,
              session_dir, run_index=1, run_total=1):
    _check_stop(args.stop_file)
    image_type = args.image_type or args.dataset_dir.name
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{image_type}_{name}_{weight_name}")
    run_dir = session_dir / "runs" / f"{run_index:03d}_{identifier}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.dataset_dir / "labels.txt", run_dir / "labels.txt")
    import torchvision
    config = {"architecture": name, "weights": weight_name, "image_type": image_type,
              "dataset": str(args.dataset_dir.resolve()), "labels": labels,
              "epochs": args.epochs, "batch_size": args.batch_size,
              "learning_rate": args.lr, "patience": args.patience, "seed": args.seed,
              "workers": args.workers, "device": str(device), "resize": [224, 224],
              "normalization_mean": [0.485, 0.456, 0.406],
              "normalization_std": [0.229, 0.224, 0.225],
              "torch_version": str(torch.__version__),
              "torchvision_version": str(torchvision.__version__)}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    def log(message):
        print(message, flush=True)
        with (run_dir / "training_log.txt").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"\nRun {run_index}/{run_total}: {image_type} / {name} / {weight_name} -> {run_dir}")
    _progress_event(args.ui_progress, event="run_start", run=run_index, runs=run_total,
                    image_type=image_type, architecture=name, weights=weight_name,
                    epochs=args.epochs, overall_percent=100 * (run_index - 1) / run_total)
    status_path = run_dir / "run_status.json"
    status_path.write_text(json.dumps({"status": "running"}, indent=2) + "\n", encoding="utf-8")
    model = None
    history = []
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    checkpoint = run_dir / f"best_{name}_{weight_name}_model.pth"
    last_checkpoint = run_dir / f"last_{name}_{weight_name}_model.pth"
    started = time.monotonic()
    try:
        grid_folder, grid_selections = save_test_grid(args.dataset_dir, loaders["test"].dataset,
                                                      labels, run_dir / "sample_images",
                                                      args.test_grid_per_class)
        log(f"  Held-out test grid: {grid_folder / 'test_grid.png'}")
        model = getattr(models, name)(weights=weight)
        replace_classifier(model, name, len(labels), nn)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        for epoch in range(1, args.epochs + 1):
            _check_stop(args.stop_file)
            def report(phase, batch_index, total_batches, loss, accuracy):
                fraction = batch_index / total_batches
                epoch_fraction = 0.8 * fraction if phase == "train" else 0.8 + 0.2 * fraction
                model_percent = 90 * ((epoch - 1) + epoch_fraction) / args.epochs
                overall_percent = 100 * ((run_index - 1) + model_percent / 100) / run_total
                phase_percent = 100 * fraction
                log(f"  [{run_index}/{run_total}] epoch {epoch}/{args.epochs} {phase} "
                    f"{phase_percent:5.1f}% ({batch_index}/{total_batches} batches) | "
                    f"loss {loss:.4f} | accuracy {accuracy * 100:.1f}% | "
                    f"model {model_percent:.1f}% | overall {overall_percent:.1f}%")
                _progress_event(args.ui_progress, event="batch", run=run_index, runs=run_total,
                                image_type=image_type, architecture=name, weights=weight_name,
                                epoch=epoch, epochs=args.epochs, phase=phase, batch=batch_index,
                                batches=total_batches, phase_percent=phase_percent,
                                model_percent=model_percent, overall_percent=overall_percent,
                                loss=loss, accuracy=accuracy)

            train_metrics = _epoch(model, loaders["train"], criterion, optimizer, device, True, torch,
                                   lambda batch, total, loss, accuracy: report(
                                       "train", batch, total, loss, accuracy), args.stop_file)
            val_metrics = _epoch(model, loaders["val"], criterion, optimizer, device, False, torch,
                                 lambda batch, total, loss, accuracy: report(
                                     "validation", batch, total, loss, accuracy), args.stop_file)
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                            "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
                            "val_accuracy": val_metrics["accuracy"]})
            log(f"  {epoch:03d}: train loss {train_metrics['loss']:.4f}, acc {train_metrics['accuracy']:.3f} | "
                f"val loss {val_metrics['loss']:.4f}, acc {val_metrics['accuracy']:.3f}")
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                no_improvement = 0
                torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, checkpoint)
            else:
                no_improvement += 1
            torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                       last_checkpoint)
            if no_improvement >= args.patience:
                break
        log(f"  [{run_index}/{run_total}] testing best checkpoint on held-out images…")
        _progress_event(args.ui_progress, event="test_start", run=run_index, runs=run_total,
                        image_type=image_type, architecture=name, weights=weight_name,
                        model_percent=90, overall_percent=100 * ((run_index - 1) + 0.90) / run_total)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        def report_test(batch, total, accuracy):
            phase_percent = 100 * batch / total
            model_percent = 90 + 8 * batch / total
            overall_percent = 100 * ((run_index - 1) + model_percent / 100) / run_total
            log(f"  [{run_index}/{run_total}] test {phase_percent:5.1f}% "
                f"({batch}/{total} batches) | accuracy {accuracy * 100:.1f}% | "
                f"model {model_percent:.1f}% | overall {overall_percent:.1f}%")
            _progress_event(args.ui_progress, event="batch", run=run_index, runs=run_total,
                            image_type=image_type, architecture=name, weights=weight_name,
                            phase="test", batch=batch, batches=total, phase_percent=phase_percent,
                            model_percent=model_percent, overall_percent=overall_percent,
                            accuracy=accuracy)

        truth, predicted = _test(model, loaders["test"], device, torch, report_test, args.stop_file)
        log(f"  [{run_index}/{run_total}] exporting reports and model artifacts…")
        _progress_event(args.ui_progress, event="export", run=run_index, runs=run_total,
                        image_type=image_type, architecture=name, weights=weight_name,
                        model_percent=98, overall_percent=100 * ((run_index - 1) + 0.98) / run_total)
        test_accuracy = _save_reports(run_dir, labels, history, truth, predicted)
        _save_test_predictions(run_dir, loaders["test"].dataset, labels, truth, predicted,
                               grid_folder, grid_selections)
        metrics = {"architecture": name, "weights": weight_name, "dataset": str(args.dataset_dir.resolve()),
                   "run_dir": str(run_dir.resolve()),
                   "checkpoint": str(checkpoint.resolve()),
                   "checkpoint_file": checkpoint.name,
                   "last_checkpoint": str(last_checkpoint.resolve()),
                   "last_checkpoint_file": last_checkpoint.name,
                   "session": str(session_dir.resolve()),
                   "run_relative": run_dir.relative_to(session_dir).as_posix(),
                   "image_type": image_type,
                   "test_grid": str((grid_folder / "test_grid.png").resolve()),
                   "test_grid_file": "sample_images/test_grid.png",
                   "classes": labels, "best_epoch": best_epoch, "best_val_loss": best_loss,
                   "test_accuracy": test_accuracy, "test_count": len(truth), "seed": args.seed,
                   "epochs_requested": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
                   "elapsed_seconds": round(time.monotonic() - started, 2), "device": str(device)}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        status_path.write_text(json.dumps({"status": "complete", "best_epoch": best_epoch,
                                           "epochs_completed": len(history)}, indent=2) + "\n",
                               encoding="utf-8")
        log(f"  best epoch {best_epoch}; held-out test accuracy {test_accuracy:.3f}")
        _progress_event(args.ui_progress, event="run_complete", run=run_index, runs=run_total,
                        image_type=image_type, architecture=name, weights=weight_name,
                        model_percent=100, overall_percent=100 * run_index / run_total,
                        test_accuracy=test_accuracy)
        return metrics
    except TrainingStopped:
        log("  STOPPED: training was interrupted before this model completed.")
        status_path.write_text(json.dumps({"status": "stopped", "epochs_completed": len(history)},
                                          indent=2) + "\n", encoding="utf-8")
        raise
    except Exception as exc:
        log(f"  FAILED: {exc}")
        status_path.write_text(json.dumps({"status": "failed", "error": str(exc),
                                           "epochs_completed": len(history)}, indent=2) + "\n",
                               encoding="utf-8")
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def save_comparison(output_dir: Path, completed: list[dict]) -> Path:
    """Save one row per architecture/weight with each image type's test results."""
    grouped: dict[tuple[str, str, int], dict[str, dict]] = {}
    for metrics in completed:
        key = (metrics["architecture"], metrics["weights"], metrics["seed"])
        grouped.setdefault(key, {})[Path(metrics["dataset"]).name] = metrics
    fields = ["architecture", "weights", "seed", "complete_triplet", "complete_quartet"]
    for variant in IMAGE_TYPES:
        fields.extend((f"{variant}_test_accuracy", f"{variant}_best_val_loss",
                       f"{variant}_test_count", f"{variant}_run_dir"))
    fields.extend(("tactile_minus_clean", "default_minus_clean"))
    rows = []
    for (architecture, weights, seed), variants in sorted(grouped.items()):
        row = {"architecture": architecture, "weights": weights, "seed": seed,
               "complete_triplet": all(name in variants for name in ("tactile", "default", "clean")),
               "complete_quartet": all(name in variants for name in IMAGE_TYPES)}
        for variant in IMAGE_TYPES:
            if variant in variants:
                metrics = variants[variant]
                row.update({f"{variant}_test_accuracy": metrics["test_accuracy"],
                            f"{variant}_best_val_loss": metrics["best_val_loss"],
                            f"{variant}_test_count": metrics["test_count"],
                            f"{variant}_run_dir": metrics["run_dir"]})
        if row["complete_triplet"]:
            row["tactile_minus_clean"] = (variants["tactile"]["test_accuracy"]
                                          - variants["clean"]["test_accuracy"])
            row["default_minus_clean"] = (variants["default"]["test_accuracy"]
                                          - variants["clean"]["test_accuracy"])
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_run_index(output_dir: Path, completed: list[dict]) -> Path:
    """Summarize every completed model run, including single-branch training."""
    fields = ("architecture", "weights", "image_type", "dataset", "seed", "best_epoch",
              "best_val_loss", "test_accuracy", "test_count", "elapsed_seconds",
              "checkpoint", "last_checkpoint", "test_grid", "run_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runs.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: metrics.get(field) for field in fields} for metrics in completed)
    return path


def main(argv=None):
    args = parse_args(argv)
    _check_stop(args.stop_file)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    selected_datasets = resolve_datasets(args.dataset_dir, args.image_type, args.all_image_types)
    model_variants = {name: _weights_for(name, args.weights, models) for name in args.selected_models}
    run_plan = [{"architecture": name, "weights": weight_name,
                 "image_type": image_type or dataset_root.name}
                for name in args.selected_models
                for weight_name, _ in model_variants[name]
                for image_type, dataset_root in selected_datasets]
    run_total = len(run_plan)
    _progress_event(args.ui_progress, event="plan", runs=run_total,
                    queue=run_plan, overall_percent=0)
    transform = transforms.Compose((transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))))
    device = torch.device("cuda" if (args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this PyTorch installation.")
    export_root = args.output_dir.expanduser().resolve()
    session_dir = export_root / f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    (session_dir / "runs").mkdir(parents=True, exist_ok=False)
    session_config = {"dataset": str(args.dataset_dir.expanduser().resolve()),
                      "image_types": [name or folder.name for name, folder in selected_datasets],
                      "architectures": args.selected_models, "weights": args.weights,
                      "epochs": args.epochs, "batch_size": args.batch_size,
                      "learning_rate": args.lr, "patience": args.patience,
                      "seed": args.seed, "workers": args.workers, "device": str(device),
                      "test_grid_per_class": args.test_grid_per_class,
                      "expected_runs": run_total, "run_queue": run_plan}
    (session_dir / "session_config.json").write_text(
        json.dumps(session_config, indent=2) + "\n", encoding="utf-8")
    print(f"Session export: {session_dir}", flush=True)
    _progress_event(args.ui_progress, event="session", path=str(session_dir))
    completed = []
    failures = []
    stopped = False
    run_index = 0
    prepared_datasets = []
    for image_type, dataset_root in selected_datasets:
        if args.stop_file is not None and args.stop_file.is_file():
            stopped = True
            break
        labels_path = dataset_root / "labels.txt"
        if not labels_path.is_file():
            raise ValueError(f"Missing labels.txt: {labels_path}. Run preprocess.py first.")
        labels = labels_path.read_text(encoding="utf-8-sig").splitlines()
        if len(labels) < 2 or any(not value for value in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"{labels_path} must contain at least two distinct nonempty classes.")
        image_sets = _dataset(dataset_root, transform, datasets, labels)
        loaders = {split: DataLoader(image_sets[split], batch_size=args.batch_size,
                                     shuffle=(split == "train"), num_workers=args.workers,
                                     pin_memory=(device.type == "cuda")) for split in ("train", "val", "test")}
        print(f"Dataset: {dataset_root}; classes: {labels}; split sizes: "
              + ", ".join(f"{s}={len(d)}" for s, d in image_sets.items()))
        prepared_datasets.append((image_type, dataset_root, labels, loaders))
    print(f"Device: {device}; models: {', '.join(args.selected_models)}", flush=True)
    for name in args.selected_models:
        if stopped or args.stop_file is not None and args.stop_file.is_file():
            stopped = True
            break
        for weight_name, weight in model_variants[name]:
            if stopped or args.stop_file is not None and args.stop_file.is_file():
                stopped = True
                break
            for image_type, dataset_root, labels, loaders in prepared_datasets:
                if args.stop_file is not None and args.stop_file.is_file():
                    stopped = True
                    break
                run_args = copy.copy(args)
                run_args.dataset_dir = dataset_root
                run_index += 1
                try:
                    random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(args.seed)
                    completed.append(train_one(name, weight_name, weight, run_args, labels, loaders,
                                               device, torch, models, nn, session_dir, run_index, run_total))
                except TrainingStopped:
                    stopped = True
                    break
                except Exception as exc:
                    failures.append({"image_type": image_type, "architecture": name,
                                     "weights": weight_name, "error": str(exc)})
                    print(f"FAILED {image_type or 'dataset'}/{name}/{weight_name}: {exc}", flush=True)
            if stopped:
                break
        if stopped:
            break
    print(f"Completed {len(completed)} training runs.")
    if completed:
        index = save_run_index(session_dir, completed)
        print(f"Run index: {index}")
    if args.all_image_types and completed:
        comparison = save_comparison(session_dir, completed)
        print(f"Side-by-side held-out test results: {comparison}")
    summary = {"status": "stopped" if stopped else "failed" if failures or not completed else "complete",
               "session": str(session_dir), "expected_runs": run_total,
               "completed_runs": len(completed), "failed_runs": len(failures),
               "run_directories": [str(Path(item["run_dir"]).relative_to(session_dir)) for item in completed],
               "attempted_run_directories": [path.relative_to(session_dir).as_posix()
                                             for path in sorted((session_dir / "runs").iterdir())
                                             if path.is_dir()]}
    (session_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if stopped:
        print("Training stopped. Completed runs and their reports were retained; no further model was started.",
              flush=True)
        _progress_event(args.ui_progress, event="stopped", run=run_index, runs=run_total,
                        completed=len(completed), overall_percent=100 * len(completed) / run_total)
        raise TrainingStopped("Training stopped by user.")
    if failures:
        failure_path = session_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} training run(s) failed; details: {failure_path}")
    if not completed:
        raise RuntimeError("No training run completed successfully.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        from mesh2tact.gui.training import launch
        launch()
    else:
        try:
            main()
        except TrainingStopped as exc:
            print(str(exc), flush=True)
            sys.exit(130)
