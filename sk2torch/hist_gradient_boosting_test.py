import io
import unittest

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError

from .hist_gradient_boosting import TorchHistGradientBoostingClassifier


class HistGradientBoostingTest(unittest.TestCase):
    def assert_parity(self, fitted, model, x):
        t = torch.from_numpy(x)
        for method in ("decision_function", "predict_proba"):
            expected = getattr(fitted, method)(x)
            actual = getattr(model, method)(t).numpy()
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        labels = fitted.predict(x)
        self.assertEqual(model.predict(t).numpy().dtype, labels.dtype)
        np.testing.assert_array_equal(model.predict(t).numpy(), labels)
        np.testing.assert_array_equal(model(t).numpy(), labels)

    def test_binary_and_multiclass_round_trip(self):
        rng = np.random.RandomState(4)
        x = rng.normal(size=(300, 3))
        signal = x[:, 0] + 0.5 * x[:, 1]
        x[::5, 0] = np.nan
        for cuts in ([0.5], [-0.8, 0.5]):
            labels = np.array([-7, 3, 20], dtype=np.int32)[np.digitize(signal, cuts)]
            fitted = HistGradientBoostingClassifier(
                max_iter=8, max_depth=4, learning_rate=0.3, random_state=0
            ).fit(x, labels)
            eager = TorchHistGradientBoostingClassifier.wrap(fitted)
            stream = io.BytesIO()
            torch.jit.save(torch.jit.script(eager), stream)
            stream.seek(0)
            restored = torch.jit.load(stream)
            for model in (eager, restored):
                for batch in (x[::2], x[:1].astype(np.float32), np.zeros((2, 3), np.int32)):
                    with self.subTest(classes=len(cuts) + 1, dtype=batch.dtype):
                        self.assert_parity(fitted, model, batch)

    def test_split_equality_precision_and_missing_direction(self):
        # Midway between adjacent float32 values; rounding the threshold to
        # float32 would incorrectly send the upper training value to the left.
        eps = np.finfo(np.float32).eps
        low, high = 1.0 + float(eps), 1.0 + 2 * float(eps)
        x = np.repeat([low, high, np.nan], 40)[:, None]
        for missing_left in (False, True):
            y = np.repeat([-2, 9, -2 if missing_left else 9], 40)
            fitted = HistGradientBoostingClassifier(max_iter=1, max_depth=1).fit(x, y)
            root = fitted._predictors[0][0].nodes[0]
            threshold = root["num_threshold"]
            self.assertEqual(threshold, (low + high) / 2)
            self.assertEqual(bool(root["missing_go_to_left"]), missing_left)
            model = torch.jit.script(TorchHistGradientBoostingClassifier.wrap(fitted))
            self.assert_parity(fitted, model, x.astype(np.float32))
            self.assert_parity(fitted, model, np.array([
                -np.inf, np.nextafter(threshold, -np.inf), threshold,
                np.nextafter(threshold, np.inf), np.inf, np.nan,
            ])[:, None])

    def test_stumps_and_zero_score_tie(self):
        x = np.ones((80, 1))
        fitted = HistGradientBoostingClassifier(max_iter=3).fit(x, np.arange(80) % 2 == 1)
        model = torch.jit.script(TorchHistGradientBoostingClassifier.wrap(fitted))
        np.testing.assert_array_equal(fitted.decision_function(x), np.zeros(80))
        self.assert_parity(fitted, model, x)

    def test_reject_unsupported_models(self):
        wrap = TorchHistGradientBoostingClassifier.wrap
        with self.assertRaises(NotFittedError):
            wrap(HistGradientBoostingClassifier())
        x = (np.arange(80) % 2)[:, None]
        y = x[:, 0]
        with self.assertRaisesRegex(ValueError, "categorical"):
            wrap(HistGradientBoostingClassifier(max_iter=1, categorical_features=[0]).fit(x, y))
        with self.assertRaisesRegex(ValueError, "class labels"):
            wrap(HistGradientBoostingClassifier(max_iter=1).fit(x, y.astype(str)))
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            wrap(HistGradientBoostingClassifier(max_iter=1).fit(x, np.zeros_like(y)))

    def test_reject_unsupported_input(self):
        x = np.arange(80)[:, None]
        model = TorchHistGradientBoostingClassifier.wrap(
            HistGradientBoostingClassifier(max_iter=1).fit(x, x[:, 0] % 2)
        )
        with self.assertRaisesRegex(ValueError, "feature count"):
            model(torch.zeros(3, 2))
        with self.assertRaisesRegex(ValueError, "dense real"):
            model(torch.zeros(3, 1, dtype=torch.complex64))
