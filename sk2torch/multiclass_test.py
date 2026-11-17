import unittest

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits

import sk2torch


def small_hist_classifier():
    return HistGradientBoostingClassifier(
        max_iter=2, max_leaf_nodes=3, min_samples_leaf=2, random_state=0
    )


class OneVsRestClassifierTest(unittest.TestCase):
    @torch.no_grad()
    def assert_ovr_parity(self, estimator, x, y, probe):
        with threadpool_limits(limits=1, user_api="openmp"):
            fitted = OneVsRestClassifier(estimator).fit(x, y)
            labels = fitted.predict(probe)
            scores = {"predict_proba": fitted.predict_proba(probe)}
            if hasattr(fitted, "decision_function"):
                scores["decision_function"] = fitted.decision_function(probe)

        eager = sk2torch.wrap(fitted)
        batch = torch.from_numpy(probe)
        for mode in ("eager", "scripted"):
            with self.subTest(mode=mode):
                model = eager if mode == "eager" else torch.jit.script(eager)
                for name, predict in (("predict", model.predict), ("forward", model)):
                    with self.subTest(method=name):
                        actual = predict(batch)
                        if isinstance(labels[0], str):
                            self.assertIsInstance(actual, list)
                            self.assertListEqual(actual, labels.tolist())
                        else:
                            self.assertIsInstance(actual, torch.Tensor)
                            np.testing.assert_array_equal(actual.numpy(), labels)
                for name, expected in scores.items():
                    with self.subTest(method=name):
                        actual = getattr(model, name)(batch)
                        self.assertIsInstance(actual, torch.Tensor)
                        self.assertEqual(tuple(actual.shape), expected.shape)
                        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-8)

    def test_multiclass_string_labels(self):
        x = np.arange(24, dtype=np.float64)[:, None]
        y = np.repeat(["zinc", "amber", "blue"], 8)
        self.assert_ovr_parity(small_hist_classifier(), x, y, x[[20, 1, 12, 2]])

    def test_binary_numeric_labels(self):
        # The middle leaf is balanced: its zero margin maps to the negative class.
        x = np.repeat([-1.0, 0.0, 1.0], 8)[:, None]
        y = np.array([-7] * 8 + [-7, 42] * 4 + [42] * 8, dtype=np.int32)
        self.assert_ovr_parity(small_hist_classifier(), x, y, x[[8, 0, 16]])

    def test_multilabel_indicators(self):
        probe = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        x = np.tile(probe, (4, 1))
        y = np.column_stack((x[:, 0], x[:, 1], x[:, 0] * x[:, 1])).astype(int)
        self.assert_ovr_parity(small_hist_classifier(), x, y, probe)

    def test_probability_only_multiclass_ties_and_normalization(self):
        x = np.arange(9, dtype=np.float64)[:, None]
        y = np.tile([12, -3, 7], 3)
        # Positive-class probabilities tie in every column. Their sum is 1.5
        # for uniform and zero for constant, exercising both normalization paths.
        for estimator in (DummyClassifier(strategy="uniform"),
                          DummyClassifier(strategy="constant", constant=0)):
            with self.subTest(strategy=estimator.strategy):
                self.assert_ovr_parity(estimator, x, y, x[:2])

    def test_probability_only_multilabel_threshold(self):
        x = np.arange(4, dtype=np.float64)[:, None]
        # Independent marginals of 0.25, 0.5 and 0.75 must not be normalized;
        # the middle label tests strict thresholding at exactly 0.5.
        y = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
        self.assert_ovr_parity(DummyClassifier(strategy="prior"), x, y, x[:2])
