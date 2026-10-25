"""Real parking-inspector classifier backed by a TensorFlow Lite model.

Workflow: train an image classifier in TensorFlow/Keras, export it to
``.tflite``, then point the config at it::

    detectors:
      inspector:
        backend: tflite
        options:
          model_path: models/inspector.tflite
          labels: [not_inspector, inspector]
          input_size: [224, 224]   # optional; overridden by the model if it
                                   # declares a fixed input shape
          normalize: "0-1"         # "0-1" | "-1-1" | "none"

The TFLite interpreter is imported lazily, so importing this module is safe even
when neither ``tflite-runtime`` nor ``tensorflow`` is installed. Only actually
*using* the tflite backend requires one of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .base import ClassificationResult, InspectorClassifier


def _load_interpreter_class():
    """Return a TFLite Interpreter class, preferring the Pi-friendly runtime."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf  # type: ignore

        return tf.lite.Interpreter
    except ImportError:
        pass
    raise ImportError(
        "The tflite inspector backend needs 'tflite-runtime' (recommended on a "
        "Raspberry Pi) or 'tensorflow' (fine on a dev machine). Install one, or "
        "switch detectors.inspector.backend back to 'mock'."
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    total = exp.sum()
    return exp / total if total else exp


class TFLiteInspectorClassifier(InspectorClassifier):
    name = "tflite-inspector"

    def __init__(
        self,
        model_path: str | Path,
        labels: Optional[Sequence[str]] = None,
        input_size: Optional[Sequence[int]] = None,
        normalize: str = "0-1",
        positive_label: str = "inspector",
        input_index: int = 0,
        output_index: int = 0,
    ) -> None:
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        Interpreter = _load_interpreter_class()
        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        self._input_detail = self._interpreter.get_input_details()[input_index]
        self._output_detail = self._interpreter.get_output_details()[output_index]

        shape = self._input_detail["shape"]  # NHWC
        model_h, model_w = int(shape[1]), int(shape[2])
        if model_h > 0 and model_w > 0:
            self.input_size: Tuple[int, int] = (model_w, model_h)
        elif input_size is not None:
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            self.input_size = (224, 224)

        self.labels: Optional[List[str]] = list(labels) if labels else None
        self.normalize = normalize
        self.positive_label = positive_label

        self._in_dtype = self._input_detail["dtype"]
        self._out_dtype = self._output_detail["dtype"]
        in_q = self._input_detail.get("quantization", (0.0, 0)) or (0.0, 0)
        out_q = self._output_detail.get("quantization", (0.0, 0)) or (0.0, 0)
        self._in_scale, self._in_zero = float(in_q[0]), int(in_q[1])
        self._out_scale, self._out_zero = float(out_q[0]), int(out_q[1])

    # -- public API --------------------------------------------------------- #
    def classify(self, image: np.ndarray) -> ClassificationResult:
        tensor = self._preprocess(image)
        self._interpreter.set_tensor(self._input_detail["index"], tensor)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_detail["index"])
        return self._postprocess(raw)

    # -- internals ---------------------------------------------------------- #
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.normalize == "0-1":
            rgb /= 255.0
        elif self.normalize == "-1-1":
            rgb = rgb / 127.5 - 1.0
        elif self.normalize not in ("none", None):
            raise ValueError(f"unsupported normalize mode: {self.normalize!r}")

        tensor = rgb[None, ...]
        if self._in_dtype in (np.uint8, np.int8) and self._in_scale:
            tensor = tensor / self._in_scale + self._in_zero
            info = np.iinfo(self._in_dtype)
            tensor = np.clip(tensor, info.min, info.max)
        return tensor.astype(self._in_dtype)

    def _postprocess(self, raw: Any) -> ClassificationResult:
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        if self._out_dtype in (np.uint8, np.int8) and self._out_scale:
            scores = (scores - self._out_zero) * self._out_scale

        if scores.size and (scores.min() < 0.0 or abs(float(scores.sum()) - 1.0) > 1e-2):
            scores = _softmax(scores)

        index = int(np.argmax(scores)) if scores.size else 0
        label = self._label_for(index)
        named = {
            self._label_for(i): float(score) for i, score in enumerate(scores)
        }
        return ClassificationResult(label=label, confidence=float(scores[index]) if scores.size else 0.0, scores=named)

    def _label_for(self, index: int) -> str:
        if self.labels and index < len(self.labels):
            return self.labels[index]
        return str(index)
