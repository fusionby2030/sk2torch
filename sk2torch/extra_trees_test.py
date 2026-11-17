import io
import pickle
import unittest

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from . import supported_classes, wrap


def model_variants(fitted):
    eager = wrap(fitted)
    yield "eager", eager

    stream = io.BytesIO()
    torch.save(eager, stream)
    stream.seek(0)
    # Load the whole module, not just weights, from our own in-memory fixture.
    yield "reloaded eager", torch.load(stream, pickle_module=pickle)

    scripted = torch.jit.script(eager)
    yield "scripted", scripted
    stream = io.BytesIO()
    torch.jit.save(scripted, stream)
    stream.seek(0)
    yield "reloaded scripted", torch.jit.load(stream)


class ExtraTreesRegressorTest(unittest.TestCase):
    @torch.no_grad()
    def assert_prediction_parity(self, fitted, *batches):
        cases = [(torch.from_numpy(x), fitted.predict(x)) for x in batches]
        for mode, model in model_variants(fitted):
            for x, expected in cases:
                for name, predict in (("predict", model.predict), ("forward", model)):
                    with self.subTest(mode=mode, method=name, dtype=x.dtype, n=len(x)):
                        actual = predict(x).numpy()
                        self.assertEqual(actual.shape, expected.shape)
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_supported_classes(self):
        self.assertIn(ExtraTreesRegressor, supported_classes())

    def test_single_and_multioutput(self):
        rng = np.random.RandomState(7)
        x = rng.normal(size=(15, 3))
        test_x = rng.normal(size=(11, 3))
        y = x[:, 0] - x[:, 2]
        for targets in (y, np.column_stack((y, 2 + x[:, 1] ** 2))):
            with self.subTest(target_shape=targets.shape):
                # Fifteen distinct targets give fully grown trees unequal leaf depths.
                fitted = ExtraTreesRegressor(n_estimators=5, random_state=3).fit(x, targets)
                self.assert_prediction_parity(fitted, test_x, test_x[:1].astype(np.float32))

    def test_float32_inputs_compared_to_double_thresholds(self):
        low = 4.0
        high = float(np.nextafter(np.float32(low), np.float32(np.inf)))
        midpoint = (low + high) / 2
        x = np.array([[low], [high]])
        fitted = ExtraTreesRegressor(n_estimators=16, random_state=0).fit(x, [-3, 5])

        # Each probe rounds to a different endpoint at float32. A threshold in
        # the upper half catches both comparing the upper probe in double and
        # rounding the threshold up to `high` before comparing float32 inputs.
        probes = np.array([np.nextafter(midpoint, low), np.nextafter(midpoint, high)])[:, None]
        self.assertTrue(any(probes[1, 0] < t.tree_.threshold[0] < high for t in fitted.estimators_))
        np.testing.assert_array_equal(fitted.predict(probes), [-3, 5])
        self.assert_prediction_parity(fitted, probes, x.astype(np.float32))

    def test_learned_nan_routing(self):
        x = np.repeat([-1.0, 1.0, np.nan], [16, 2, 1])[:, None]
        y = np.repeat([0.0, 4.0, 2.0], [16, 2, 1])
        fitted = ExtraTreesRegressor(n_estimators=16, max_depth=1, random_state=0)
        try:
            fitted.fit(x, y)
        except ValueError as exc:
            if "nan" not in str(exc).lower():
                raise
            self.skipTest("installed sklearn does not support ExtraTreesRegressor NaN fits")

        probes = np.array([[-1.0], [np.nan], [1.0]])
        left, missing, right = fitted.predict(probes)
        # Both directions occur across the forest; always taking the larger
        # child (left), or always taking the right child, cannot pass.
        self.assertLess(left, missing)
        self.assertLess(missing, right)
        self.assert_prediction_parity(fitted, probes)

    def test_root_only_trees(self):
        x = np.ones((6, 2))
        fitted = ExtraTreesRegressor(n_estimators=2, random_state=0).fit(x, np.arange(6))
        self.assert_prediction_parity(fitted, np.array([[0.0, 1.0], [2.0, -1.0]]))
