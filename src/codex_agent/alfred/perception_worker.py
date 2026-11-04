"""Standalone Python 3.8 bridge to FILM's published perception architectures.

Launch by absolute file path in the configured FILM interpreter. The optional
torch/FILM dependencies are deliberately confined to this process. Dynamic model
values use Any at this external framework boundary, never in the actor API.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def prepare_model_images(
    rgb: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return detector RGB and depth BGR without modifying the source image.

    FILM's Mask R-CNN helper uses simulator RGB directly, whereas its released
    depth inference explicitly converts RGB to BGR. Both arrays are contiguous
    because torch.from_numpy cannot consume the reversed channel view's strides.
    """
    import numpy as np

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("model image input must be uint8 HxWx3 RGB")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(rgb[:, :, ::-1])


def normalize_horizon_degrees(horizon_degrees: float) -> float:
    """Clamp only Unity's sub-millidegree roundoff at the camera pitch limits."""
    tolerance = 1e-3
    if (
        not math.isfinite(horizon_degrees)
        or not -30 - tolerance <= horizon_degrees <= 60 + tolerance
    ):
        raise ValueError(f"horizon_degrees={horizon_degrees!r} must be in [-30, 60]")
    return min(60.0, max(-30.0, horizon_degrees))


def choose_depth_view(horizon_degrees: float) -> Literal["level", "downward"]:
    """Select the nearest released 0-degree or 45-degree checkpoint."""
    horizon_degrees = normalize_horizon_degrees(horizon_degrees)
    return "level" if abs(horizon_degrees) <= abs(horizon_degrees - 45) else "downward"


def normalize_depth_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the released DataParallel prefix without truncating arbitrary keys."""
    if not state or any(not key.startswith("module.") for key in state):
        raise ValueError("FILM depth checkpoint keys must have the 'module.' prefix")
    return {key[len("module.") :]: value for key, value in state.items()}


class FilmModels:
    """Loaded ALFRED-trained depth and two instance segmentation networks."""

    def __init__(self, args: argparse.Namespace) -> None:
        started = time.monotonic()
        sys.path.insert(0, str(Path(args.film_root).resolve()))
        self.torch = importlib.import_module("torch")
        self.numpy = importlib.import_module("numpy")
        self.image = importlib.import_module("PIL.Image")
        torchvision = importlib.import_module("torchvision")
        depth_module = importlib.import_module("models.depth.alfred_perception_models")
        segmentation = importlib.import_module("models.segmentation.alfworld_mrcnn")
        labels = importlib.import_module("models.segmentation.alfworld_constants")
        self.device = self.torch.device(args.device)
        self.torch.set_num_threads(1)
        self.torch.set_grad_enabled(False)
        self.score_threshold = args.score_threshold
        self.mask_threshold = args.mask_threshold
        root = Path(args.weights_root)
        self.detectors: list[tuple[Any, list[str]]] = []
        constructor = torchvision.models.detection.maskrcnn_resnet50_fpn

        def offline_constructor(*positional: Any, **keywords: Any) -> Any:
            # The released helper requests COCO initialization before replacing
            # every weight. Suppress those unnecessary downloads explicitly.
            keywords.pop("pretrained", None)
            keywords.pop("pretrained_backbone", None)
            keywords.update(weights=None, weights_backbone=None)
            return constructor(*positional, **keywords)

        with patch.object(
            torchvision.models.detection, "maskrcnn_resnet50_fpn", offline_constructor
        ):
            for kind, filename, categories in (
                ("obj", "objects_lr5e-3_005.pth", labels.OBJECTS_DETECTOR),
                ("recep", "receps_lr5e-3_003.pth", labels.STATIC_RECEPTACLES),
            ):
                model = (
                    segmentation.load_pretrained_model(
                        str(root / "maskrcnn_alfworld" / filename), self.device, kind
                    )
                    .eval()
                    .to(self.device)
                )
                self.detectors.append((model, list(categories)))
        self.depth_models: dict[str, Any] = {}
        for view, filename in (
            ("level", "valts0/model-102500-best_silog_17.00430"),
            ("downward", "valts/model-2000-best_silog_10.13741"),
        ):
            checkpoint = self.torch.load(
                str(root / "depth_models" / filename), map_location="cpu"
            )
            model = depth_module.AlfredSegmentationAndDepthModel()
            model.load_state_dict(
                normalize_depth_state(checkpoint["model"]), strict=True
            )
            self.depth_models[view] = model.eval().to(self.device)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        self.load_seconds = time.monotonic() - started
        self.inference_seconds = 0.0
        self.frames = 0

    def predict(self, request: dict[str, object]) -> dict[str, object]:
        """Write independent depth/instance predictions from RGB to the output folder."""
        import re

        started = time.monotonic()
        frame_id = request["frame_id"]
        if not isinstance(frame_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", frame_id
        ):
            raise ValueError("frame_id must be a safe filename component")
        horizon = request["horizon_degrees"]
        if not isinstance(horizon, (int, float)) or isinstance(horizon, bool):
            raise ValueError("horizon_degrees must be numeric")
        view = choose_depth_view(horizon)
        rgb_path, output = request["rgb_path"], request["output_dir"]
        if not isinstance(rgb_path, str) or not isinstance(output, str):
            raise ValueError("rgb_path and output_dir must be strings")
        with self.image.open(rgb_path) as image:
            if image.size != (300, 300) or image.mode != "RGB":
                raise ValueError("FILM perception requires a 300x300 RGB image")
            rgb = self.numpy.array(image)
        detector_rgb, depth_bgr = prepare_model_images(rgb)
        detector_tensor = (
            self.torch.from_numpy(detector_rgb).permute(2, 0, 1).float().to(self.device)
            / 255
        )
        depth_tensor = (
            self.torch.from_numpy(depth_bgr).permute(2, 0, 1).float().to(self.device)
            / 255
        )
        output_dir = Path(output).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = output_dir / (frame_id + "_depth.npy")
        if depth_path.exists():
            raise FileExistsError("prediction already exists: " + str(depth_path))
        with self.torch.inference_mode():
            _, estimate = self.depth_models[view].predict(depth_tensor.unsqueeze(0))
            depth = estimate.mle()[0, 0].cpu().numpy().astype(self.numpy.float32)
            if not self.numpy.isfinite(depth).all():
                raise ValueError("depth network returned nonfinite metric depth")
            self.numpy.save(str(depth_path), depth, allow_pickle=False)
            detections: list[dict[str, object]] = []
            for detector, categories in self.detectors:
                prediction = detector([detector_tensor])[0]
                detections.extend(
                    self._save_detections(
                        prediction, categories, output_dir, frame_id, len(detections)
                    )
                )
        self.inference_seconds += time.monotonic() - started
        self.frames += 1
        return {
            "frame_id": frame_id,
            "depth_path": str(depth_path),
            "depth_model": view,
            "detections": detections,
        }

    def statistics(self) -> dict[str, object]:
        """Return timing and peak device allocation without exposing model state."""
        return {
            "load_seconds": self.load_seconds,
            "inference_seconds": self.inference_seconds,
            "frames": self.frames,
            "peak_cuda_memory_bytes": (
                int(self.torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
            "device": str(self.device),
        }

    def _save_detections(
        self,
        prediction: Any,
        categories: list[str],
        output_dir: Path,
        frame_id: str,
        first_index: int,
    ) -> list[dict[str, object]]:
        detections: list[dict[str, object]] = []
        for index, score in enumerate(prediction["scores"]):
            confidence = float(score.item())
            if not math.isfinite(confidence):
                raise ValueError("instance network returned a nonfinite score")
            if confidence < self.score_threshold:
                continue
            label = int(prediction["labels"][index].item())
            if not 0 <= label < len(categories):
                raise ValueError(f"predicted class index={label} has no FILM category")
            probabilities = prediction["masks"][index, 0].cpu().numpy()
            if not self.numpy.isfinite(probabilities).all():
                raise ValueError("instance network returned a nonfinite mask")
            mask = (probabilities >= self.mask_threshold).astype(self.numpy.uint8)
            rows, columns = self.numpy.nonzero(mask)
            if rows.size == 0:
                continue
            bbox = [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ]
            mask_path = (
                output_dir / f"{frame_id}_mask_{first_index + len(detections):03d}.npy"
            )
            self.numpy.save(str(mask_path), mask, allow_pickle=False)
            detections.append(
                {
                    "category": categories[label],
                    "mask_path": str(mask_path),
                    "bbox_xyxy": bbox,
                    "confidence": confidence,
                }
            )
        return detections


def main() -> None:
    """Serve JSONL requests; unexpected failures are reported and terminate worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--mask-threshold", type=float, required=True)
    args = parser.parse_args()
    models: FilmModels | None = None
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                command = request.get("command")
                if command == "close":
                    break
                if command == "initialize":
                    if models is not None:
                        raise ValueError("perception models are already initialized")
                    models = FilmModels(args)
                    result: dict[str, object] = {"backend": "film_predicted_rgb_v1"}
                elif command == "predict" and models is not None:
                    result = models.predict(request)
                elif command == "statistics" and models is not None:
                    result = models.statistics()
                else:
                    raise ValueError(f"unexpected perception command={command!r}")
            response = {"ok": True, "result": result}
        except Exception as exc:
            # Process boundary: report failure and exit, never substitute models.
            traceback.print_exc(file=sys.stderr)
            protocol_stdout.write(
                json.dumps(
                    {"ok": False, "error_type": type(exc).__name__, "message": str(exc)}
                )
                + "\n"
            )
            protocol_stdout.flush()
            raise SystemExit(1) from exc
        protocol_stdout.write(json.dumps(response, allow_nan=False) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
