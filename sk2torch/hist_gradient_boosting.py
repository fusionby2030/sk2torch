from typing import List, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from .tree import _packed_leaf_indices


class TorchHistGradientBoostingClassifier(nn.Module):
    """Inference on dense numerical features in sklearn's original column order.

    Keep the floating-point buffers in float64 to preserve split thresholds and
    raw scores. The input need not be float64; NaNs follow the fitted split.
    """

    def __init__(self, obj: HistGradientBoostingClassifier):
        super().__init__()
        check_is_fitted(obj, ["_predictors", "_baseline_prediction", "classes_"])
        if obj.loss != "log_loss":
            raise ValueError("only HistGradientBoostingClassifier log_loss is supported")
        if np.any(getattr(obj, "is_categorical_", False)) or getattr(
            obj, "_preprocessor", None
        ) is not None:
            raise ValueError("categorical features and fitted preprocessors are unsupported")

        classes = obj.classes_
        if classes.ndim != 1 or len(classes) < 2:
            raise ValueError("expected at least two classes for a single target")
        if classes.dtype.kind not in "biuf":
            raise ValueError("class labels must be numeric or boolean tensor values")
        try:
            labels = torch.from_numpy(classes.copy())
        except TypeError as exc:
            raise ValueError(f"unsupported class label dtype: {classes.dtype}") from exc
        self.register_buffer("classes", labels)

        self.n_features = int(obj.n_features_in_)
        self.n_outputs = 1 if len(classes) == 2 else len(classes)
        if (
            obj.n_trees_per_iteration_ != self.n_outputs
            or obj._baseline_prediction.shape != (1, self.n_outputs)
            or not obj._predictors
            or any(len(stage) != self.n_outputs for stage in obj._predictors)
        ):
            raise ValueError("unexpected histogram boosting baseline or predictor layout")
        self.register_buffer("baseline", torch.from_numpy(obj._baseline_prediction.copy()))

        required = {
            "feature_idx", "num_threshold", "missing_go_to_left", "left", "right",
            "depth", "is_leaf", "is_categorical", "value",
        }
        trees = []
        for stage in obj._predictors:
            for predictor in stage:
                nodes = getattr(predictor, "nodes", None)
                if (
                    nodes is None or nodes.ndim != 1 or not len(nodes)
                    or not required.issubset(nodes.dtype.names or ())
                ):
                    raise ValueError("unsupported sklearn TreePredictor node layout")
                if nodes["is_categorical"].any():
                    raise ValueError("categorical histogram tree splits are unsupported")
                trees.append(nodes)

        # Pack trees in iteration-major, then class order. Child indices are
        # tree-local in sklearn; make them absolute, with self-loops on leaves.
        sizes = [len(tree) for tree in trees]
        roots = np.cumsum([0] + sizes[:-1], dtype=np.int64)
        offsets = np.repeat(roots, sizes)
        nodes = np.concatenate(trees)
        leaf = nodes["is_leaf"].astype(bool)
        ids = np.arange(len(nodes), dtype=np.int64)
        self.max_depth = int(nodes["depth"].max())
        self.register_buffer("roots", torch.from_numpy(roots))
        for side in ("left", "right"):
            children = np.where(leaf, ids, offsets + nodes[side])
            self.register_buffer(side, torch.from_numpy(children))
        features = np.where(leaf, 0, nodes["feature_idx"]).astype(np.int64)
        self.register_buffer("feature", torch.from_numpy(features))
        self.register_buffer("threshold", torch.from_numpy(nodes["num_threshold"].astype(np.float64)))
        self.register_buffer("missing_left", torch.from_numpy(nodes["missing_go_to_left"].astype(bool)))
        self.register_buffer("value", torch.from_numpy(nodes["value"].astype(np.float64)))

        # Working tensors have one entry per (sample, tree). Bound their size
        # for large batches without serializing individual samples or trees.
        self.chunk_rows = max(1, 16384 // len(trees))

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [HistGradientBoostingClassifier]

    @classmethod
    def wrap(cls, obj: HistGradientBoostingClassifier) -> "TorchHistGradientBoostingClassifier":
        return cls(obj)

    def _raw_chunk(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        node = _packed_leaf_indices(
            x, self.roots, self.max_depth, self.feature, self.threshold,
            self.missing_left, self.left, self.right,
        )

        # Leaf values already include shrinkage. Include the baseline as the
        # first term (not after a tree reduction) to retain sklearn's sum order.
        leaves = self.value.index_select(0, node).view(n, -1, self.n_outputs)
        terms = torch.cat([self.baseline.expand(n, 1, self.n_outputs), leaves], dim=1)
        return terms.cumsum(1)[:, -1, :]

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_features:
            raise ValueError("expected a 2-D (samples, features) tensor with the fitted feature count")
        if x.layout != torch.strided or x.is_complex() or x.is_quantized:
            raise ValueError("expected a dense real numerical tensor")
        scores = self.baseline.new_empty((x.size(0), self.n_outputs))
        for i in range(0, x.size(0), self.chunk_rows):
            batch = x[i : i + self.chunk_rows].to(torch.float64)
            scores[i : i + self.chunk_rows] = self._raw_chunk(batch)
        return scores.squeeze(1) if self.n_outputs == 1 else scores

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.decision_function(x)
        if self.n_outputs == 1:
            p = scores.sigmoid()
            return torch.stack([1 - p, p], dim=1)
        return scores.softmax(dim=1)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.decision_function(x)
        index = (scores > 0).long() if self.n_outputs == 1 else scores.argmax(dim=1)
        return self.classes.index_select(0, index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)
