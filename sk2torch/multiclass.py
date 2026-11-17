from typing import List, Type

import torch
import torch.nn as nn
from sklearn.base import is_classifier, is_regressor
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor


class _ConstantBinaryPredictor(nn.Module):
    """sklearn substitutes this estimator for constant indicator columns."""

    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_full((x.size(0),), self.value, dtype=torch.long)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p = self.decision_function(x).double()
        return torch.stack([1 - p, p], dim=1)


class _OneVsRestBase(nn.Module):
    def __init__(self, obj: OneVsRestClassifier):
        super().__init__()
        from .wrap import wrap

        self.estimators = nn.ModuleList([
            _ConstantBinaryPredictor(int(e.y_[0])) if isinstance(e, _ConstantPredictor)
            else wrap(e) for e in obj.estimators_
        ])
        # Choose methods from the fitted sklearn estimators, not their wrappers:
        # a wrapper may expose a method even when its fitted model does not.
        self.score_methods = [
            "predict" if is_regressor(e) else
            "decision_function" if hasattr(e, "decision_function") else "predict_proba"
            for e in obj.estimators_
        ]
        self.has_decision = hasattr(obj, "decision_function")
        self.has_proba = hasattr(obj, "predict_proba")
        self.multilabel = obj.multilabel_
        first = obj.estimators_[0]
        self.threshold = 0.0 if is_classifier(first) and hasattr(first, "decision_function") else 0.5

    def _scores(self, x: torch.Tensor) -> torch.Tensor:
        columns = []
        for i, estimator in enumerate(self.estimators):
            method = self.score_methods[i]
            if method == "predict" and hasattr(estimator, "predict"):
                score = estimator.predict(x).reshape(-1)
            elif method == "decision_function" and hasattr(estimator, "decision_function"):
                score = estimator.decision_function(x).reshape(-1)
            elif method == "predict_proba" and hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(x)
                assert isinstance(proba, torch.Tensor)
                score = proba[:, 1]
            else:
                raise RuntimeError("wrapped binary estimator lacks its required scoring method")
            columns.append(score)
        return torch.stack(columns, dim=1)

    def _predict_indices(self, x: torch.Tensor) -> torch.Tensor:
        scores = self._scores(x)
        if self.multilabel:
            return (scores > self.threshold).long()
        if scores.size(1) == 1:
            return (scores[:, 0] > self.threshold).long()
        return scores.argmax(dim=1)

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_decision:
            raise RuntimeError("base estimator does not support decision_function")
        return self._scores(x).squeeze(1)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_proba:
            raise RuntimeError("base estimator does not support predict_proba")
        columns = []
        for estimator in self.estimators:
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(x)
                assert isinstance(proba, torch.Tensor)
                columns.append(proba[:, 1])
            else:
                raise RuntimeError("base estimator does not support predict_proba")
        p = torch.stack(columns, dim=1)
        if p.size(1) == 1:
            p = torch.cat([1 - p, p], dim=1)
        if not self.multilabel:
            totals = p.sum(dim=1, keepdim=True)
            p = p / torch.where(totals == 0, 1.0, totals)
        return p


class TorchOneVsRestClassifier(_OneVsRestBase):
    def __init__(self, obj: OneVsRestClassifier):
        super().__init__(obj)
        self.register_buffer("classes", torch.from_numpy(obj.classes_))

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [OneVsRestClassifier]

    @classmethod
    def wrap(cls, obj: OneVsRestClassifier) -> _OneVsRestBase:
        if all(isinstance(label, str) for label in obj.classes_):
            return _StringOneVsRestClassifier(obj)
        return cls(obj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        indices = self._predict_indices(x)
        return indices if self.multilabel else self.classes[indices]


class _StringOneVsRestClassifier(_OneVsRestBase):
    # Tensors have no string dtype. Keep predict's TorchScript return type
    # concrete for both kinds of classifier instead of returning a union.
    classes: List[str]

    def __init__(self, obj: OneVsRestClassifier):
        super().__init__(obj)
        self.classes = obj.classes_.tolist()

    def forward(self, x: torch.Tensor) -> List[str]:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> List[str]:
        indices: List[int] = self._predict_indices(x).tolist()
        return [self.classes[i] for i in indices]
