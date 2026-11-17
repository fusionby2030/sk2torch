from typing import List, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from .tree import _packed_leaf_indices


class TorchExtraTreesRegressor(nn.Module):
    """Predict with a fitted Extra Trees forest on a dense (samples, features) tensor."""

    def __init__(self, obj: ExtraTreesRegressor):
        super().__init__()
        check_is_fitted(obj, ["estimators_", "n_features_in_", "n_outputs_"])
        trees = [e.tree_ for e in obj.estimators_]
        self.n_features = int(obj.n_features_in_)
        self.n_outputs = int(obj.n_outputs_)
        self.max_depth = max(t.max_depth for t in trees)
        self.chunk_rows = max(1, 16384 // (len(trees) * self.n_outputs))

        # Offset each tree's local child indices into one flat forest. Leaves
        # point to themselves so shallow and deep trees can be walked together.
        sizes = [t.node_count for t in trees]
        roots = np.cumsum([0] + sizes[:-1], dtype=np.int64)
        offsets = np.repeat(roots, sizes)
        indices = np.arange(sum(sizes), dtype=np.int64)
        left = np.concatenate([t.children_left for t in trees])
        right = np.concatenate([t.children_right for t in trees])
        leaf = left == -1
        self.register_buffer("roots", torch.from_numpy(roots))
        self.register_buffer("left", torch.from_numpy(np.where(leaf, indices, offsets + left)))
        self.register_buffer("right", torch.from_numpy(np.where(leaf, indices, offsets + right)))
        features = np.concatenate([t.feature for t in trees]).astype(np.int64, copy=False)
        features[leaf] = 0  # Leaves have a negative feature sentinel in sklearn.
        self.register_buffer("feature", torch.from_numpy(features))
        self.register_buffer("threshold", torch.from_numpy(np.concatenate([t.threshold for t in trees])))
        # Older sklearn trees predate missing-value support and have no routing flags.
        supports_missing = getattr(obj.estimators_[0], "_support_missing_values", None)
        self.allow_nan = supports_missing is not None and bool(supports_missing(np.empty((0, self.n_features))))
        missing_left = (
            np.concatenate([t.missing_go_to_left for t in trees]).astype(bool)
            if self.allow_nan else np.zeros(sum(sizes), dtype=bool)
        )
        self.register_buffer("missing_left", torch.from_numpy(missing_left))
        values = np.concatenate([t.value[:, :, 0] for t in trees])
        self.register_buffer("value", torch.from_numpy(values))

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [ExtraTreesRegressor]

    @classmethod
    def wrap(cls, obj: ExtraTreesRegressor) -> "TorchExtraTreesRegressor":
        return cls(obj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_features or x.size(0) == 0:
            raise ValueError("expected a nonempty (samples, features) tensor with the fitted feature count")
        if x.layout != torch.strided or x.is_complex() or x.is_quantized:
            raise ValueError("expected a dense real numerical tensor")
        # sklearn rounds input features to float32, *not* its double split thresholds.
        x = x.float()
        if x.isinf().any() or (not self.allow_nan and x.isnan().any()):
            raise ValueError("input contains non-finite values unsupported by this forest")
        outputs = self.value.new_empty((x.size(0), self.n_outputs))
        for i in range(0, x.size(0), self.chunk_rows):
            batch = x[i : i + self.chunk_rows]
            node = _packed_leaf_indices(
                batch, self.roots, self.max_depth, self.feature, self.threshold,
                self.missing_left, self.left, self.right,
            )
            leaves = self.value.index_select(0, node).view(batch.size(0), -1, self.n_outputs)
            outputs[i : i + self.chunk_rows] = leaves.mean(dim=1)
        return outputs.squeeze(1)
