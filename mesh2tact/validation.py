"""Reproducible split evaluation and automatically exported validation runs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .prediction import IMAGE_TYPES, LoadedClassifier, infer_image_type


VALIDATION_ROOT = Path(__file__).resolve().parents[1] / "validation"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def discover_split(root, split):
    """Return class folders for a processed dataset, accepting val or valid."""
    root = Path(root).expanduser().resolve()
    if not (root / "train").is_dir() or not (root / "test").is_dir():
        raise ValueError("Choose a processed dataset containing train, test, and val (or valid). "
                         "For paired exports, choose the tactile, clean, default, or mask folder inside it.")
    validation = [name for name in ("val", "valid") if (root / name).is_dir()]
    if not validation:
        raise ValueError("The processed folder has no val or valid split.")
    if split not in ("test", "val", "valid"):
        raise ValueError("Choose test or valid; training data cannot be evaluated here.")
    if split != "test":
        if len(validation) > 1:
            raise ValueError("Both val and valid exist. Keep one validation split to avoid ambiguity.")
        split = validation[0]
    folder = root / split
    classes = {
        directory.name: sorted(path for path in directory.rglob("*")
                               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        for directory in sorted(folder.iterdir()) if directory.is_dir()
    }
    if not classes or not any(classes.values()):
        raise ValueError(f"No class-organized photos found in {folder}")
    return folder, classes


def sample_split(classes, percentage=100, seed=42):
    """Deterministically select ceil(N * percentage / 100) images per class."""
    if not 0 < percentage <= 100:
        raise ValueError("Split percentage must be greater than zero and at most 100.")
    rng = random.Random(seed)
    selected = []
    for label, paths in sorted(classes.items()):
        count = math.ceil(len(paths) * percentage / 100)
        selected.extend((label, path) for path in sorted(rng.sample(sorted(paths), count)))
    return selected


def classification_metrics(labels, truth, predicted, probabilities):
    """Single-label metrics; undefined per-class rates use zero and are documented."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        matrix[index[actual], index[guess]] += 1
    support = matrix.sum(axis=1)
    total = int(matrix.sum())
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = support - tp
    tn = total - tp - fp - fn

    def divide(a, b):
        return np.divide(a, b, out=np.zeros(len(labels), dtype=float), where=b != 0)

    precision = divide(tp, tp + fp)
    recall = divide(tp, support)
    f1 = divide(2 * precision * recall, precision + recall)
    specificity = divide(tn, tn + fp)
    per_class = [dict(label=label, precision=float(precision[i]), recall=float(recall[i]),
                      f1=float(f1[i]), specificity=float(specificity[i]), support=int(support[i]),
                      predicted_count=int(matrix[:, i].sum()), tp=int(tp[i]), fp=int(fp[i]),
                      fn=int(fn[i]), tn=int(tn[i])) for i, label in enumerate(labels)]
    metrics = {"evaluated": total, "accuracy": float(tp.sum() / total) if total else None,
               "balanced_accuracy": float(recall[support > 0].mean()) if total else None,
               "macro_precision": float(precision.mean()) if total else None,
               "macro_recall": float(recall.mean()) if total else None,
               "macro_f1": float(f1.mean()) if total else None,
               "weighted_precision": float(precision @ support / total) if total else None,
               "weighted_recall": float(recall @ support / total) if total else None,
               "weighted_f1": float(f1 @ support / total) if total else None,
               "top_k": min(3, len(labels)), "top_k_accuracy": None, "log_loss": None,
               "zero_division": 0, "labels": list(labels), "per_class": per_class,
               "confusion_matrix": matrix.tolist(),
               "metric_scope": "Successfully evaluated images only; errors and cancellation excluded. "
                               "Macro averages include all dataset classes; balanced accuracy includes supported classes."}
    if total:
        probs = np.asarray(probabilities)
        actual = np.array([index[label] for label in truth])
        top = np.argsort(probs, axis=1)[:, -metrics["top_k"]:]
        metrics["top_k_accuracy"] = float(np.any(top == actual[:, None], axis=1).mean())
        metrics["log_loss"] = float(-np.log(np.clip(probs[np.arange(total), actual], 1e-15, 1)).mean())
    return metrics


def write_json(path, payload):
    # Replace atomically so the app never reads half a run index.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_confusion(directory, metrics):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    labels = metrics["labels"]
    matrix = np.array(metrics["confusion_matrix"])
    sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, sums, out=np.zeros_like(matrix, dtype=float), where=sums != 0)
    for name, values in (("confusion_counts", matrix), ("confusion_normalized", normalized)):
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Actual / Predicted", *labels])
            writer.writerows([label, *values[i].tolist()] for i, label in enumerate(labels))
        side = min(24, max(6, len(labels) * .48 + 2))
        figure = Figure(figsize=(side + 1, side), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        normalized_plot = name.endswith("normalized")
        plot = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized_plot else max(1, int(matrix.max())))
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                 xlabel="Predicted class", ylabel="Actual class",
                 title="Confusion matrix — " + ("fraction within each actual class" if normalized_plot else "counts"))
        axis.tick_params(axis="x", labelrotation=60, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        if len(labels) <= 30:
            threshold = float(values.max()) / 2
            for row in range(len(labels)):
                for column in range(len(labels)):
                    value = values[row, column]
                    axis.text(column, row, f"{value:.2f}" if normalized_plot else str(value),
                              ha="center", va="center", fontsize=7,
                              color="white" if value > threshold else "black")
        figure.colorbar(plot, ax=axis, shrink=.8)
        figure.savefig(directory / f"{name}.png", dpi=160)
        figure.savefig(directory / f"{name}.pdf")
        figure.clear()


SUMMARY_FIELDS = ("checkpoint", "image_type", "status", "selected", "evaluated", "errors", "not_evaluated", "coverage",
                  "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                  "weighted_precision", "weighted_recall", "weighted_f1", "top_k_accuracy", "log_loss",
                  "elapsed_seconds")


def update_run_image_type(directory, model_directory, image_type):
    """Correct a saved run annotation, keeping its JSON and CSV summaries consistent."""
    if image_type not in IMAGE_TYPES:
        raise ValueError("Choose a supported image type")
    directory = Path(directory).resolve()
    model_directory = Path(model_directory).resolve()
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run["status"] == "running":
        raise ValueError("Wait for this run to finish before editing its image types.")
    target = next((entry for entry in run["models"]
                   if (directory / entry["directory"]).resolve() == model_directory), None)
    if target is None or not model_directory.is_relative_to(directory):
        raise ValueError("Model does not belong to this saved run")
    reports = []
    for entry in run["models"]:
        folder = (directory / entry["directory"]).resolve()
        if not folder.is_relative_to(directory):
            raise ValueError("Saved model directory is outside the run")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        metrics.setdefault("image_type", infer_image_type(metrics["checkpoint"]))
        if folder == model_directory:
            metrics["image_type"] = image_type
            metrics["image_type_source"] = "user annotation"
            updated = metrics
        reports.append({key: metrics.get(key) for key in SUMMARY_FIELDS})
    target["image_type"] = image_type
    run.setdefault("model_image_types", {})[updated["checkpoint"]] = image_type
    write_json(model_directory / "metrics.json", updated)
    temporary = directory / "summary.csv.tmp"
    write_csv(temporary, SUMMARY_FIELDS, reports)
    temporary.replace(directory / "summary.csv")
    write_json(directory / "run.json", run)


def run_validation(root, split, classes, samples, candidates, percentage, seed, output_root=VALIDATION_ROOT,
                   progress=None, cancelled=None, model_ready=None, classifier_factory=LoadedClassifier):
    """Evaluate models sequentially; preserve completed and partial results on cancellation."""
    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)
    model_ready = model_ready or (lambda *_: None)
    if not samples or not candidates:
        raise ValueError("Select a nonempty split and at least one model.")
    directory = Path(output_root) / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    labels = sorted(classes)
    run = {"version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
           "dataset": str(Path(root).resolve()), "split": split, "percentage": percentage, "seed": seed,
           "sampling": "Sorted classes, seeded random sampling without replacement; ceil per class.",
           "class_counts": {label: len(paths) for label, paths in classes.items()},
           "selected": len(samples), "labels": labels, "models": [],
           "requested_checkpoints": [str(candidate.checkpoint) for candidate in candidates],
           "model_image_types": {str(candidate.checkpoint): candidate.resolved_image_type for candidate in candidates},
           "preprocessing": "EXIF orientation, RGB, resize 224x224, ImageNet mean/std; no augmentation."}
    write_json(directory / "run.json", run)
    write_csv(directory / "samples.csv", ("label", "path"),
              ({"label": label, "path": str(path)} for label, path in samples))
    total = len(samples) * len(candidates)
    done = 0
    summaries = []
    try:
        for number, candidate in enumerate(candidates, 1):
            if cancelled():
                break
            started = time.perf_counter()
            destination = directory / f"model_{number:03d}"
            destination.mkdir()
            rows, truth, predicted, probabilities = [], [], [], []
            error = ""
            classifier = None
            progress(done, total, f"Loading {candidate.checkpoint.name}")
            try:
                if set(candidate.labels) != set(labels):
                    raise ValueError("Model labels do not match dataset classes. "
                                     f"Missing: {sorted(set(labels) - set(candidate.labels))}; "
                                     f"extra: {sorted(set(candidate.labels) - set(labels))}")
                classifier = classifier_factory(candidate)
                order = [candidate.labels.index(label) for label in labels]
            except Exception as exc:
                error = str(exc)
            probability_fields = [f"probability:{label}" for label in labels]
            fields = ["path", "actual", "predicted", "confidence", "correct", "error", *probability_fields]
            # Stream predictions to disk; long runs retain rows even after an unexpected exit.
            with (destination / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for label, path in samples:
                    if cancelled():
                        break
                    row = dict(path=str(path), actual=label, predicted="", confidence="", correct="", error=error)
                    if not error:
                        try:
                            with Image.open(path) as photo:
                                pixels = np.array(ImageOps.exif_transpose(photo).convert("RGB"))
                            probs = np.asarray(classifier.probabilities(pixels), dtype=float)[order]
                            if (probs.shape != (len(labels),) or not np.all(np.isfinite(probs))
                                    or np.any(probs < 0) or not np.isclose(probs.sum(), 1, atol=1e-4)):
                                raise ValueError("Model returned invalid class probabilities")
                            guess = labels[int(probs.argmax())]
                            row.update(predicted=guess, confidence=float(probs.max()), correct=int(guess == label))
                            row.update(zip(probability_fields, probs.tolist()))
                            truth.append(label)
                            predicted.append(guess)
                            probabilities.append(probs)
                        except Exception as exc:
                            row["error"] = str(exc)
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    done += 1
                    progress(done, total, f"{number}/{len(candidates)} models • {candidate.checkpoint.name} • {path.name}")
            metrics = classification_metrics(labels, truth, predicted, probabilities)
            errors = sum(bool(row["error"]) for row in rows)
            status = "cancelled" if cancelled() else "failed" if not truth else "partial" if errors else "complete"
            metrics.update(checkpoint=str(candidate.checkpoint), image_type=candidate.resolved_image_type,
                           labels_path=str(candidate.labels_path),
                           model_labels=list(candidate.labels), status=status, selected=len(samples), errors=errors,
                           not_evaluated=len(samples) - len(rows), coverage=len(truth) / len(samples),
                           elapsed_seconds=time.perf_counter() - started, model_error=error,
                           device=str(getattr(classifier, "device", "")))
            # Record checkpoint identity without retaining another model in memory.
            digest = hashlib.sha256()
            try:
                with candidate.checkpoint.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                metrics["checkpoint_sha256"] = digest.hexdigest()
            except OSError as exc:
                metrics["checkpoint_sha256"] = None
                metrics["checkpoint_identity_error"] = str(exc)
            write_json(destination / "metrics.json", metrics)
            write_csv(destination / "per_class.csv", list(metrics["per_class"][0]), metrics["per_class"])
            write_csv(destination / "errors.csv", ("path", "actual", "error"),
                      ({key: row[key] for key in ("path", "actual", "error")} for row in rows if row["error"]))
            del classifier
            export_confusion(destination, metrics)
            summaries.append({key: metrics[key] for key in SUMMARY_FIELDS})
            write_csv(directory / "summary.csv", SUMMARY_FIELDS, summaries)
            run["models"].append({"directory": destination.name, "checkpoint": str(candidate.checkpoint),
                                  "image_type": candidate.resolved_image_type, "status": status})
            write_json(directory / "run.json", run)
            model_ready(str(destination))
        run["status"] = ("cancelled" if cancelled() else "complete" if all(m["status"] == "complete" for m in run["models"])
                         else "partial" if any(m["status"] in ("complete", "partial") for m in run["models"]) else "failed")
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)
        raise RuntimeError(f"Validation stopped: {exc}. Saved output: {directory}") from exc
    finally:
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "run.json", run)
    return directory
