import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess import prepare_dataset, prepare_variants, scan_dataset, scan_variants
from mesh2tact.prediction import ModelCandidate, architecture_name, _classifier_weights


def _image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((24, 30, 3), dtype=np.uint8)
    array[:, :, 0] = value
    array[:, :, 1] = np.arange(30, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_preparation_writes_all_classes_and_disjoint_group_splits(tmp_path):
    source = tmp_path / "source"
    for label_index, label in enumerate(("cube", "sphere")):
        for run in range(3):
            for sample in range(2):
                _image(source / label / f"run_{run}" / f"sample_{sample:06d}_tactile.png",
                       label_index * 100 + run * 10 + sample)
    output = tmp_path / "prepared"
    summary = prepare_dataset(source, output, (70, 15, 15), seed=7)
    assert (output / "labels.txt").read_text().splitlines() == ["cube", "sphere"]
    assert summary["total_images"] == 12
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    for label in ("cube", "sphere"):
        assigned = {(row["label"], row["group"]): row["split"] for row in rows if row["label"] == label}
        assert len(assigned) == 3
        assert set(assigned.values()) == {"train", "val", "test"}
        assert all((output / split / label).is_dir() for split in ("train", "val", "test"))
    assert len({row["rgb_sha256"] for row in rows}) == len(rows)
    assert json.loads((output / "summary.json").read_text())["seed"] == 7
    with pytest.raises(FileExistsError):
        prepare_dataset(source, output)


def test_empty_contacts_are_excluded_and_cross_class_duplicates_rejected(tmp_path):
    source = tmp_path / "source"
    for label in ("cube", "sphere"):
        for index in range(4):
            _image(source / label / f"sample_{index:06d}_tactile.png", index + (100 if label == "sphere" else 0))
    settings = source / "cube" / "sample_000000_settings.json"
    settings.write_text('{"contact_fraction": 0.0}', encoding="utf-8")
    _, records, skipped = scan_dataset(source)
    assert len(records) == 7
    assert skipped["empty_contact"] == 1
    _image(source / "sphere" / "sample_000005_tactile.png", 1)
    with pytest.raises(ValueError, match="Identical image appears"):
        scan_dataset(source)


def test_tactile_branch_does_not_include_depth_and_mask_pngs(tmp_path):
    source = tmp_path / "tactile"
    for label_index, label in enumerate(("cube", "sphere")):
        for index in range(3):
            folder = source / label
            _image(folder / f"sample_{index:06d}_tactile.png", label_index * 50 + index)
            _image(folder / f"sample_{index:06d}_depth.png", label_index * 50 + index + 10)
            _image(folder / f"sample_{index:06d}_contact.png", label_index * 50 + index + 20)
    _, records, _ = scan_dataset(source)
    assert len(records) == 6
    assert all(record.source.stem.endswith("_tactile") for record in records)


def _variant_source(root, samples=3):
    for variant_index, (variant, suffix) in enumerate((
        ("clean", "_tactile_clean"), ("tactile", "_tactile"), ("default", "_tactile_default")
    )):
        for label_index, label in enumerate(("cube", "sphere")):
            for index in range(samples):
                _image(root / variant / label / f"sample_{index:06d}{suffix}.png",
                       variant_index * 60 + label_index * 20 + index)


def test_short_variant_filenames_pair_on_capture_index(tmp_path):
    source = tmp_path / "data"
    for variant_index, variant in enumerate(("clean", "tactile", "default")):
        for label_index, label in enumerate(("cube", "sphere")):
            for index in (310, 311, 312):
                _image(source / variant / label / f"sample_{index:06d}_{variant}.png",
                       variant_index * 60 + label_index * 20 + index - 310)
    scan = scan_variants(source)
    assert scan.summary["issues"] == {}
    assert scan.summary["matched_total"] == 6
    assert {pair.key for pair in scan.pairs} == {f"sample_{index:06d}" for index in (310, 311, 312)}
    output = tmp_path / "prepared"
    prepare_variants(source, output)
    with (output / "paired_manifest.csv").open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 6
    for row in paired:
        for variant in ("clean", "tactile", "default"):
            assert (output / variant / row["output"]).is_file()
    for variant in ("clean", "tactile", "default"):
        with (output / variant / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {(row["split"], row["label"], row["key"], row["output"]) for row in rows} == {
            (row["split"], row["label"], row["key"], row["output"]) for row in paired
        }


def test_comparison_reports_three_independent_models(tmp_path):
    from train import save_comparison

    completed = [
        {"architecture": "resnet18", "weights": "scratch", "seed": 42,
         "dataset": str(tmp_path / variant), "run_dir": str(tmp_path / "train" / variant),
         "test_accuracy": accuracy, "best_val_loss": 0.5, "test_count": 10}
        for variant, accuracy in (("clean", 0.8), ("tactile", 0.6), ("default", 0.7))
    ]
    comparison = save_comparison(tmp_path / "train", completed)
    with comparison.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["complete_triplet"] == "True"
    assert float(rows[0]["tactile_minus_clean"]) == pytest.approx(-0.2)
    assert float(rows[0]["default_minus_clean"]) == pytest.approx(-0.1)


def test_three_variants_share_labels_samples_and_split_assignments(tmp_path):
    from train import _dataset, resolve_datasets
    from torchvision import datasets, transforms

    source = tmp_path / "data"
    _variant_source(source)
    _image(source / "tactile" / "cube" / "sample_000000_depth.png", 250)
    scan = scan_variants(source)
    assert scan.summary["matched_total"] == 6
    assert scan.summary["issues"] == {}
    output = tmp_path / "ml_dataset"
    summary = prepare_variants(source, output, (70, 15, 15), seed=3)
    assert summary["matched_total"] == 6
    assert len(resolve_datasets(output, None, True)) == 3
    assert resolve_datasets(output, "tactile", False)[0][1] == output / "tactile"
    manifests = []
    for variant in ("clean", "tactile", "default"):
        branch = output / variant
        assert (branch / "labels.txt").read_text().splitlines() == ["cube", "sphere"]
        with (branch / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            manifests.append(list(csv.DictReader(handle)))
        assert len(manifests[-1]) == 6
        for split in ("train", "val", "test"):
            assert sorted(path.name for path in (branch / split).iterdir()) == ["cube", "sphere"]
        image_sets = _dataset(branch, transforms.ToTensor(), datasets, ["cube", "sphere"])
        assert all(len(image_sets[split]) == 2 for split in ("train", "val", "test"))
    keys = lambda rows: {(row["split"], row["label"], row["key"], row["output"]) for row in rows}
    assert keys(manifests[0]) == keys(manifests[1]) == keys(manifests[2])
    example = manifests[0][0]["output"]
    with Image.open(output / "clean" / example) as clean, Image.open(output / "tactile" / example) as tactile:
        assert clean.size == tactile.size
        assert clean.tobytes() != tactile.tobytes()
    assert (output / "paired_manifest.csv").is_file()
    assert (source / "clean" / "cube" / "sample_000000_tactile_clean.png").is_file()
    clean_manifest = output / "clean" / "manifest.csv"
    clean_manifest.write_text(clean_manifest.read_text(encoding="utf-8").replace("sample_000000", "wrong_id", 1),
                              encoding="utf-8")
    with pytest.raises(ValueError, match="differ from paired_manifest"):
        resolve_datasets(output, None, True)


def test_scan_reports_missing_pairs_and_matched_only_export(tmp_path):
    source = tmp_path / "data"
    _variant_source(source, samples=4)
    (source / "default" / "cube" / "sample_000003_tactile_default.png").unlink()
    scan = scan_variants(source)
    assert scan.summary["issues"]["missing_sample"] == 1
    assert scan.summary["matched_usable_samples"]["cube"] == 3
    assert scan.summary["excluded_samples"] == [{"label": "cube", "key": "sample_000003", "reason": "missing_sample"}]
    with pytest.raises(ValueError, match="matched-only"):
        prepare_variants(source, tmp_path / "strict")
    summary = prepare_variants(source, tmp_path / "matched", matched_only=True)
    assert summary["counts"]["train"]["cube"] == 1
    assert sum(summary["counts"][split]["cube"] for split in ("train", "val", "test")) == 3
    assert summary["excluded_from_each_output"] == 1
    for variant in ("clean", "tactile", "default"):
        with (tmp_path / "matched" / variant / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            assert all(row["key"] != "sample_000003" or row["label"] != "cube"
                       for row in csv.DictReader(handle))
    assert (source / "clean" / "cube" / "sample_000003_tactile_clean.png").is_file()


def test_duplicate_capture_is_excluded_from_all_outputs_not_source(tmp_path):
    source = tmp_path / "data"
    _variant_source(source, samples=4)
    for variant_index, (variant, suffix) in enumerate((("clean", "_tactile_clean"),
                                                       ("tactile", "_tactile"),
                                                       ("default", "_tactile_default"))):
        _image(source / variant / "cube" / f"sample_000003{suffix}.png", variant_index * 60 + 2)
    scan = scan_variants(source)
    assert scan.summary["unmatched_files"] == []
    assert scan.summary["issues"]["duplicate_rgb"] == 3
    output = tmp_path / "prepared"
    summary = prepare_variants(source, output, matched_only=True, scan=scan)
    assert summary["excluded_from_each_output"] == 1
    for variant, suffix in (("clean", "_tactile_clean"), ("tactile", "_tactile"),
                            ("default", "_tactile_default")):
        with (output / variant / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            keys = {(row["label"], row["key"]) for row in csv.DictReader(handle)}
        assert ("cube", "sample_000003") not in keys
        assert (source / variant / "cube" / f"sample_000003{suffix}.png").is_file()


def test_balancing_equalizes_each_split_and_preserves_triplet_indexes(tmp_path):
    source = tmp_path / "data"
    for variant_index, (variant, suffix) in enumerate((("clean", "_clean"),
                                                       ("tactile", "_tactile"),
                                                       ("default", "_default"))):
        for label_index, (label, count) in enumerate((("cube", 7), ("sphere", 4))):
            for index in range(count):
                _image(source / variant / label / f"sample_{index:06d}{suffix}.png",
                       variant_index * 60 + label_index * 20 + index)
    output = tmp_path / "balanced"
    summary = prepare_variants(source, output, balance_classes=True)
    assert summary["balance_classes"] is True
    assert summary["exported_total"] < summary["matched_total"]
    for split in ("train", "val", "test"):
        assert summary["counts"][split]["cube"] == summary["counts"][split]["sphere"]
    branch_rows = []
    for variant in ("clean", "tactile", "default"):
        with (output / variant / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            branch_rows.append({(row["split"], row["label"], row["key"], row["output"])
                                for row in csv.DictReader(handle)})
    assert branch_rows[0] == branch_rows[1] == branch_rows[2]


def test_scan_detects_bad_images_and_cached_export_detects_source_change(tmp_path):
    source = tmp_path / "data"
    _variant_source(source, samples=4)
    invalid = source / "default" / "cube" / "sample_000003_tactile_default.png"
    invalid.write_bytes(b"not an image")
    report = scan_variants(source)
    assert report.summary["issues"]["unreadable_image"] == 1
    invalid.unlink()
    _image(invalid, 3)
    scan = scan_variants(source)
    changed = source / "clean" / "cube" / "sample_000001_tactile_clean.png"
    _image(changed, 240)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="changed since the scan"):
        prepare_variants(source, output, scan=scan)
    assert not output.exists()


def test_trial_grouping_blocks_a_single_run_per_class(tmp_path):
    source = tmp_path / "data"
    _variant_source(source)
    for variant, suffix in (("clean", "_tactile_clean"), ("tactile", "_tactile"),
                            ("default", "_tactile_default")):
        for label in ("cube", "sphere"):
            folder = source / variant / label
            run = folder / "run_1"
            run.mkdir()
            for path in folder.glob(f"*{suffix}.png"):
                path.rename(run / path.name)
    scan = scan_variants(source)
    assert scan.summary["issues"]["insufficient_groups"] == 2
    with pytest.raises(ValueError, match="independent group"):
        prepare_variants(source, tmp_path / "output", matched_only=True, scan=scan)


def test_prediction_accepts_architectures_from_train_suite(tmp_path):
    torch = pytest.importorskip("torch")
    from torchvision import models
    for name in ("mobilenet_v3_small", "convnext_tiny", "regnet_y_400mf"):
        model = getattr(models, name)(weights=None, num_classes=2)
        state = model.state_dict()
        candidate = ModelCandidate(tmp_path / f"best_{name}_model.pth", None, ("cube", "sphere"))
        assert architecture_name(candidate, state) == name
        assert _classifier_weights(state).shape[0] == 2
        del model


def test_training_ui_lists_all_families_and_confirms_labels(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtWidgets
    from mesh2tact.architectures import ALL_MODELS, SUITES
    from train import PROGRESS_PREFIX
    from train_ui import TrainingWindow, dataset_modes

    source = tmp_path / "source"
    _variant_source(source)
    dataset = tmp_path / "dataset"
    prepare_variants(source, dataset)
    assert [mode for _, mode in dataset_modes(dataset)] == ["all", "tactile", "default", "clean"]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TrainingWindow()
    window.dataset_edit.setText(str(dataset))
    window.dataset_mode.addItem("All three", "all")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.Yes)
    window.confirm_labels()
    assert window.confirmed
    assert "cube" in window.labels_view.text()
    assert window.model_tree.topLevelItemCount() == len(SUITES)
    window.select_all.setChecked(True)
    assert set(window.selected_models()) == set(ALL_MODELS)
    window.select_all.setChecked(False)
    assert window.selected_models() == []
    family = window.model_tree.topLevelItem(1)
    family.setCheckState(0, QtCore.Qt.Checked)
    assert len(window.selected_models()) == family.childCount()
    assert all(name.startswith("efficientnet_") for name in window.selected_models())
    from train import parse_args
    assert parse_args(window.command_arguments()[1:]).selected_models == window.selected_models()
    family.setCheckState(0, QtCore.Qt.Unchecked)
    assert window.selected_models() == []
    window.model_tree.topLevelItem(0).child(0).setCheckState(0, QtCore.Qt.Checked)
    assert window.selected_models() == ["resnet18"]
    arguments = window.command_arguments()
    assert "--all-image-types" in arguments and "--models" in arguments
    assert "--ui-progress" in arguments
    assert parse_args(arguments[1:]).selected_models == ["resnet18"]
    event = {"event": "batch", "run": 1, "runs": 3, "image_type": "clean",
             "architecture": "resnet18", "weights": "scratch", "epoch": 2, "epochs": 20,
             "phase": "train", "phase_percent": 50.0, "batch": 5, "batches": 10,
             "model_percent": 7.2, "overall_percent": 2.4,
             "loss": 0.5, "accuracy": 0.75}
    marker = PROGRESS_PREFIX + json.dumps(event) + "\n"
    window.consume_output("visible terminal line\n" + marker[:20])
    window.consume_output(marker[20:])
    assert "visible terminal line" in window.log.toPlainText()
    assert PROGRESS_PREFIX not in window.log.toPlainText()
    assert window.current_progress.value() == 72
    assert window.overall_progress.value() == 24
    assert "accuracy 75.0%" in window.status.text()
    window.close()
    assert app is not None


def test_training_ui_stop_sends_request_and_force_stops(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtWidgets
    from train_ui import TrainingWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TrainingWindow()
    window.stop_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    window.stop_path = Path(window.stop_directory.name) / "stop.request"
    window.process = QtCore.QProcess(window)
    try:
        window.process.start(sys.executable, ["-u", "-c", "import time; time.sleep(30)"])
        assert window.process.waitForStarted(5000)
        window.stop_training()
        assert window.stop_requested
        assert window.stop_path.is_file()
        assert window.force_stop_timer.isActive()
        window.force_stop_if_running()
        assert window.process.waitForFinished(5000)
        window.finished_training(window.process.exitCode(), window.process.exitStatus())
        assert "stopped" in window.status.text().lower()
        assert not window.force_stop_timer.isActive()
    finally:
        if window.process.state() != QtCore.QProcess.NotRunning:
            window.process.kill()
            window.process.waitForFinished(5000)
        window.cleanup_stop_request()
        window.close()
    assert app is not None


@pytest.mark.slow
def test_training_checkpoint_loads_in_predict(tmp_path, capsys):
    pytest.importorskip("torch")
    from train import main as train_main
    from mesh2tact.prediction import discover_models, predict_classifier

    source = tmp_path / "source"
    _variant_source(source)
    dataset = tmp_path / "dataset"
    prepare_variants(source, dataset, (70, 15, 15))
    output = tmp_path / "models"
    train_main(["--dataset-dir", str(dataset), "--all-image-types",
                "--output-dir", str(output), "--models", "resnet18", "resnet34",
                "--weights", "none", "--epochs", "1", "--batch-size", "2",
                "--ui-progress"])
    output_text = capsys.readouterr().out
    progress_events = [json.loads(line.split("@@M2T_PROGRESS@@", 1)[1])
                       for line in output_text.splitlines() if line.startswith("@@M2T_PROGRESS@@")]
    assert progress_events[0]["event"] == "plan" and progress_events[0]["runs"] == 6
    expected_order = [(name, variant) for name in ("resnet18", "resnet34")
                      for variant in ("tactile", "default", "clean")]
    assert [(item["architecture"], item["image_type"])
            for item in progress_events[0]["queue"]] == expected_order
    assert len([event for event in progress_events if event["event"] == "run_complete"]) == 6
    starts = [(index, event["run"], event["architecture"], event["image_type"])
              for index, event in enumerate(progress_events)
              if event["event"] == "run_start"]
    completions = [(index, event["run"]) for index, event in enumerate(progress_events)
                   if event["event"] == "run_complete"]
    assert [(name, variant) for _, _, name, variant in starts] == expected_order
    assert [run for _, run, _, _ in starts] == [1, 2, 3, 4, 5, 6]
    assert [run for _, run in completions] == [1, 2, 3, 4, 5, 6]
    assert all(completions[index][0] < starts[index + 1][0] for index in range(5))
    assert any(event["event"] == "batch" and event["phase"] == "train"
               and event["phase_percent"] == 100.0 for event in progress_events)
    sessions = list(output.glob("session_*"))
    assert len(sessions) == 1
    session = sessions[0]
    assert (session / "session_config.json").is_file()
    assert json.loads((session / "session_summary.json").read_text())["completed_runs"] == 6
    assert [(item["architecture"], item["image_type"])
            for item in json.loads((session / "session_config.json").read_text())["run_queue"]] == expected_order
    with (session / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with (session / "runs.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 6
    grids = list((session / "runs").glob("*/sample_images/test_grid.png"))
    assert len(grids) == 6
    grid_keys = {}
    for grid in grids:
        manifest = grid.parent / "test_grid_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["label"] for row in rows} == {"cube", "sphere"}
        assert all("/test/" in row["source"].replace("\\", "/") for row in rows)
        assert all((grid.parent / row["image"]).is_file() for row in rows)
        variant = grid.parent.parent.name.split("_", 2)[1]
        grid_keys[variant] = {(row["label"], row["key"]) for row in rows}
    assert grid_keys["clean"] == grid_keys["tactile"] == grid_keys["default"]
    runs = list((session / "runs").glob("*/metrics.json"))
    assert len(runs) == 6
    for metrics_file in runs:
        architecture = json.loads(metrics_file.read_text())["architecture"]
        for artifact in ("confusion_matrix.csv", "test_predictions.csv", "test_grid_predictions.csv",
                         "run_config.json", "run_status.json", "training_log.txt",
                         "classification_report.json", "training_history.png",
                         "confusion_matrix.png", "labels.txt", "sample_images/test_grid.png",
                         "sample_images/test_grid_manifest.csv", f"best_{architecture}_scratch_model.pth",
                         f"last_{architecture}_scratch_model.pth"):
            assert (metrics_file.parent / artifact).is_file()
        metrics = json.loads(metrics_file.read_text())
        assert json.loads((metrics_file.parent / "run_status.json").read_text())["status"] == "complete"
        assert Path(metrics["checkpoint"]).parent == metrics_file.parent
        assert Path(metrics["last_checkpoint"]).parent == metrics_file.parent
        assert (metrics_file.parent / metrics["checkpoint_file"]).is_file()
        assert (metrics_file.parent / metrics["last_checkpoint_file"]).is_file()
        assert (metrics_file.parent / metrics["test_grid_file"]).is_file()
        assert (session / metrics["run_relative"]) == metrics_file.parent
        assert "best epoch" in (metrics_file.parent / "training_log.txt").read_text()
    candidates = discover_models(output)
    assert len(candidates) == 12 and all(candidate.ready for candidate in candidates)
    tactile = next(candidate for candidate in candidates
                   if candidate.checkpoint.name.startswith("best_")
                   and any("tactile" in part for part in candidate.checkpoint.parts))
    prediction = predict_classifier(np.zeros((24, 30, 3), dtype=np.uint8), tactile)
    assert prediction.label in {"cube", "sphere"}
    last_tactile = next(candidate for candidate in candidates
                        if candidate.checkpoint.name.startswith("last_")
                        and any("tactile" in part for part in candidate.checkpoint.parts))
    assert predict_classifier(np.zeros((24, 30, 3), dtype=np.uint8), last_tactile).label in {"cube", "sphere"}
    train_main(["--dataset-dir", str(dataset / "clean"), "--output-dir", str(output),
                "--models", "resnet18", "--weights", "none", "--epochs", "1", "--batch-size", "2"])
    assert len(list(output.glob("session_*"))) == 2
    assert len(list((session / "runs").iterdir())) == 6


@pytest.mark.slow
def test_stop_request_prevents_next_selected_model(tmp_path, monkeypatch):
    from train import TrainingStopped, main as train_main
    import train

    source = tmp_path / "source"
    _variant_source(source)
    dataset = tmp_path / "dataset"
    prepare_variants(source, dataset)
    stop_file = tmp_path / "stop.request"
    original = train._progress_event

    def request_stop(enabled, **event):
        original(enabled, **event)
        if event.get("event") == "batch" and event.get("phase") == "train":
            stop_file.touch()

    monkeypatch.setattr(train, "_progress_event", request_stop)
    output = tmp_path / "models"
    with pytest.raises(TrainingStopped, match="stopped"):
        train_main(["--dataset-dir", str(dataset / "clean"), "--output-dir", str(output),
                    "--models", "resnet18", "resnet34", "--weights", "none",
                    "--epochs", "2", "--batch-size", "2", "--stop-file", str(stop_file),
                    "--ui-progress"])
    sessions = list(output.glob("session_*"))
    assert len(sessions) == 1
    assert not list((sessions[0] / "runs").glob("*resnet34*"))
    assert not (sessions[0] / "runs.csv").exists()
    assert json.loads((sessions[0] / "session_summary.json").read_text())["status"] == "stopped"
    partial_runs = list((sessions[0] / "runs").iterdir())
    assert len(partial_runs) == 1
    assert json.loads((partial_runs[0] / "run_status.json").read_text())["status"] == "stopped"
