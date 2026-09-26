"""Discovery and inference helpers for saved image-classification checkpoints.

PyTorch is deliberately imported only when inference is requested.  That keeps
the geometric renderer usable in an environment which has not installed ML
dependencies yet, while allowing the desktop Predict tab to use the active
Python environment's CUDA installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json
import re

import numpy as np

from .architectures import ALL_MODELS


LABEL_FILENAMES = ("labels.txt", "classes.txt", "labels.json", "classes.json")
IMAGE_TYPES = ("Tactile", "Default", "Clean", "Actual", "Mask", "Unknown")


def normalize_image_type(value):
    """Use one display vocabulary for simulated branches and physical captures."""
    aliases = {"tactile": "Tactile", "default": "Default", "default_tactile": "Default",
               "clean": "Clean", "actual": "Actual", "gelsight": "Actual", "real": "Actual",
               "mask": "Mask", "unknown": "Unknown"}
    return aliases.get(str(value).strip().lower(), "Unknown")


def infer_image_type(checkpoint):
    """Prefer training metadata, then unambiguous filename/nearest-folder tokens."""
    checkpoint = Path(checkpoint)
    for filename in ("run_config.json", "metrics.json"):
        try:
            payload = json.loads((checkpoint.parent / filename).read_text(encoding="utf-8"))
            value = normalize_image_type(payload.get("image_type"))
            if value != "Unknown":
                return value
        except (OSError, ValueError, AttributeError):
            pass
    for part in (checkpoint.stem, *(parent.name for parent in checkpoint.parents)):
        tokens = re.split(r"[^a-z0-9]+", part.lower())
        values = {normalize_image_type(token) for token in tokens} - {"Unknown"}
        # default_tactile is the existing SaveOptions name for the Default branch.
        if "default_tactile" in part.lower():
            values.discard("Tactile")
        if values:
            return next(iter(values)) if len(values) == 1 else "Unknown"
    return "Unknown"


@dataclass(frozen=True)
class ModelCandidate:
    """A checkpoint and the label file that describes its output indices."""

    checkpoint: Path
    labels_path: Path | None
    labels: tuple[str, ...]
    image_type: str = ""

    @property
    def resolved_image_type(self) -> str:
        return normalize_image_type(self.image_type) if self.image_type else infer_image_type(self.checkpoint)

    @property
    def ready(self) -> bool:
        return bool(self.labels)

    @property
    def relative_label_name(self) -> str:
        return self.labels_path.name if self.labels_path else "No labels found"


@dataclass(frozen=True)
class Prediction:
    checkpoint: Path
    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...]
    device: str


def read_labels(path: str | Path) -> tuple[str, ...]:
    """Read one-label-per-line text files or simple JSON label lists/maps."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, Mapping):
            values = [payload[str(index)] if str(index) in payload else payload[index]
                      for index in range(len(payload))]
        else:
            raise ValueError(f"{path.name} must contain a JSON list or index-to-label object")
        labels = tuple(str(value).strip() for value in values)
    else:
        labels = tuple(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                       if line.strip() and not line.lstrip().startswith("#"))
    if not labels:
        raise ValueError(f"{path.name} has no labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path.name} contains duplicate labels")
    return labels


def _find_labels(checkpoint: Path, root: Path) -> tuple[Path | None, tuple[str, ...]]:
    """Prefer sibling labels, then look in ancestor folders up to *root*."""
    folder = checkpoint.parent
    while True:
        for filename in LABEL_FILENAMES:
            candidate = folder / filename
            if candidate.is_file():
                try:
                    return candidate, read_labels(candidate)
                except (OSError, ValueError):
                    # Keep the checkpoint visible; the UI reports that it cannot be selected.
                    return candidate, ()
        if folder == root or folder.parent == folder:
            break
        folder = folder.parent
    return None, ()


def discover_models(folder: str | Path) -> list[ModelCandidate]:
    """Recursively find ``.pth`` checkpoints and their nearest label files."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Model folder does not exist: {root}")
    models: list[ModelCandidate] = []
    for checkpoint in sorted(path for path in root.rglob("*.pth") if path.is_file()):
        labels_path, labels = _find_labels(checkpoint, root)
        models.append(ModelCandidate(checkpoint, labels_path, labels, infer_image_type(checkpoint)))
    return models


def _checkpoint_state(checkpoint: Path, torch):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Checkpoint must be a PyTorch state dictionary")
    state = {str(key).removeprefix("module."): value for key, value in payload.items()}
    return state


SUPPORTED_ARCHITECTURES = tuple(sorted(ALL_MODELS, key=len, reverse=True))


def _resnet_name(state: Mapping[str, object]) -> str:
    """Infer a ResNet family when its filename has no architecture name."""
    if "conv1.weight" not in state or "fc.weight" not in state:
        raise ValueError("Checkpoint does not match a torchvision ResNet state dictionary")
    counts = tuple(sum(key.startswith(f"layer{layer}.") and key.endswith(".conv1.weight")
                       for key in state) for layer in range(1, 5))
    bottleneck = any(key.startswith("layer1.") and key.endswith(".conv3.weight") for key in state)
    table = {
        (False, (2, 2, 2, 2)): "resnet18",
        (False, (3, 4, 6, 3)): "resnet34",
        (True, (3, 4, 6, 3)): "resnet50",
        (True, (3, 4, 23, 3)): "resnet101",
        (True, (3, 8, 36, 3)): "resnet152",
    }
    try:
        return table[(bottleneck, counts)]
    except KeyError as exc:
        raise ValueError("Unsupported ResNet checkpoint layout") from exc


def architecture_name(candidate: ModelCandidate, state: Mapping[str, object]) -> str:
    """Identify supported torchvision architectures from the saved checkpoint path.

    Training saves the selected torchvision name in the run/checkpoint name.
    This is more reliable for model families with similar layer layouts, such as
    EfficientNet variants.  ResNet retains a state-dictionary fallback for old
    checkpoints whose filename only says ``model.pth``.
    """
    source = candidate.checkpoint.as_posix().lower()
    for name in SUPPORTED_ARCHITECTURES:
        if name in source:
            return name
    if "fc.weight" in state:
        return _resnet_name(state)
    raise ValueError(
        "Cannot identify the torchvision architecture from this checkpoint path. "
        "Supported families: ResNet, DenseNet, EfficientNet, MobileNet, ConvNeXt, Swin, ViT, and RegNet."
    )


def _classifier_weights(state: Mapping[str, object]):
    for key in ("fc.weight", "classifier.weight", "head.weight", "heads.head.weight"):
        weights = state.get(key)
        if hasattr(weights, "shape") and len(weights.shape) == 2:
            return weights
    for key in sorted(state, reverse=True):
        if key.startswith("classifier.") and key.endswith(".weight"):
            weights = state[key]
            if hasattr(weights, "shape") and len(weights.shape) == 2:
                return weights
    raise ValueError("Checkpoint has no supported torchvision classifier weights")


class LoadedClassifier:
    """Load one checkpoint once and reuse it for a sequence of images."""

    def __init__(self, candidate: ModelCandidate):
        if not candidate.ready:
            raise ValueError(f"{candidate.checkpoint.name} has no valid labels file")
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and TorchVision must be installed in the environment that launches Mesh2Tact."
            ) from exc
        state = _checkpoint_state(candidate.checkpoint, torch)
        weights = _classifier_weights(state)
        output_count = int(weights.shape[0])
        if output_count != len(candidate.labels):
            raise ValueError(
                f"Checkpoint has {output_count} outputs but its labels file has {len(candidate.labels)} labels"
            )
        self.architecture = architecture_name(candidate, state)
        self.model = getattr(models, self.architecture)(weights=None, num_classes=output_count)
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.candidate = candidate
        self.transform = transforms.Compose((transforms.Resize((224, 224)), transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))))

    def probabilities(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Prediction expects an H x W x 3 RGB tactile image")
        tensor = self.transform(Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return torch.softmax(self.model(tensor)[0], dim=0).detach().cpu().numpy()

    def predict(self, image_rgb: np.ndarray, top_k: int = 3) -> Prediction:
        probabilities = self.probabilities(image_rgb)
        count = min(max(1, int(top_k)), len(self.candidate.labels))
        indices = np.argsort(probabilities)[::-1][:count]
        ranked = tuple((self.candidate.labels[int(index)], float(probabilities[int(index)])) for index in indices)
        return Prediction(self.candidate.checkpoint, ranked[0][0], ranked[0][1], ranked, str(self.device))


def predict_classifier(image_rgb: np.ndarray, candidate: ModelCandidate, top_k: int = 3) -> Prediction:
    """Classify one image; bulk callers should reuse a LoadedClassifier."""
    return LoadedClassifier(candidate).predict(image_rgb, top_k)


# Kept for scripts written while the first Predict tab only accepted ResNet.
predict_resnet = predict_classifier
