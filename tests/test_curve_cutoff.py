import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from packages import ROMCurve4BayesianInference


class CurveCutoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(plt.close, "all")
        self.root = Path(self.temp.name)
        (self.root / "train").mkdir()
        (self.root / "obs").mkdir()
        # Descending coordinates exercise a non-prefix mask and the x=5 boundary.
        self.x = np.array([9., 7., 5., 3., 1., 0.])
        self.parameters = np.array([[1., .2], [1., .8], [2., .3],
                                    [2., .7], [3., .2], [3., .8]])
        self.curves = (1 + .01 * self.parameters[:, :1] * (1 - np.exp(-self.x))
                       + .02 * self.parameters[:, 1:] * self.x / 5)
        np.savetxt(self.root / "train/curve4_y.txt", self.curves)
        np.savetxt(self.root / "train/parameters.txt", self.parameters)
        np.savetxt(self.root / "obs/curve4_x.txt", self.x)
        np.savetxt(self.root / "obs/curve4_y.txt", self.curves[2:3])

    def make_model(self, thin=1):
        model = ROMCurve4BayesianInference(
            swell_root=self.root, train_data_rels=["train"], model="oldroyd",
            sigma_bias="infer", l_bias="infer", thin=thin,
        )
        model.gpr_restarts = 0
        return model

    def load(self, model, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            model.load_data("obs/curve4_y.txt", u_avg_obs=.1, **kwargs)

    def test_default_cutoff_precedes_noise_calibration(self):
        # An excluded extreme value must not affect the noise or sigma priors.
        observed = self.curves[2:3].copy()
        observed[:, self.x > 5] = 1000
        np.savetxt(self.root / "obs/curve4_y.txt", observed)
        model = self.make_model()
        model.sigma_noise_percent = 2
        self.load(model)
        np.testing.assert_array_equal(model.obs_x_coords, [5, 3, 1, 0])
        np.testing.assert_array_equal(model.y_obs_matrix_clean, observed[:, 2:])
        self.assertAlmostEqual(model.max_displacement, observed[:, 2:].max() - 1)
        self.assertAlmostEqual(model.sigma_noise_target, .02 * model.max_displacement)
        self.assertEqual(model.y_obs_matrix.shape, (1, 4))

    def test_discarded_training_values_do_not_affect_rom(self):
        first = self.make_model()
        self.load(first)
        first.build_rom()
        changed = self.curves.copy()
        changed[:, self.x > 5] += 1000 * self.parameters[:, :1]
        np.savetxt(self.root / "train/curve4_y.txt", changed)
        second = self.make_model()
        self.load(second)
        second.build_rom()
        np.testing.assert_array_equal(first.snapshots_train, self.curves[:, 2:])
        self.assertEqual(first.basis.shape[0], 4)
        np.testing.assert_allclose(first.predict(2, .3), second.predict(2, .3))

    def test_thinned_inference_and_all_curve_plots_use_cropped_rom(self):
        model = self.make_model(thin=2)
        self.load(model)
        model.build_rom()
        np.testing.assert_array_equal(model.obs_x_coords, [5, 1])
        theta = np.array([2., .3, .05, .1, 1.])
        np.testing.assert_allclose(
            model._residual(theta), model.y_obs_matrix[0] - model.predict(2, .3)[::2])
        self.assertTrue(np.isfinite(model.log_posterior(theta)))
        model.samples = np.tile(theta, (6, 1))
        with patch.object(model, "_save_current_figure"), patch.object(plt, "show"), \
                contextlib.redirect_stdout(io.StringIO()):
            model.plot_data()
            model.plot_rom_prediction(lambda_val=2, beta_val=.3)
            model.plot_posterior_predictive(nsamples_pred=6)
        for number in plt.get_fignums():
            ax = plt.figure(number).axes[0]
            self.assertLessEqual(ax.get_xlim()[1], 5)
            for line in ax.lines:
                self.assertTrue(np.all(np.asarray(line.get_xdata()) <= 5))
            for collection in ax.collections:
                for path in collection.get_paths():
                    if isinstance(collection, matplotlib.collections.PolyCollection):
                        self.assertTrue(np.all(path.vertices[:, 0] <= 5))
                if isinstance(collection, matplotlib.collections.PathCollection):
                    np.testing.assert_array_equal(collection.get_offsets()[:, 0], [5, 1])

    def test_changed_cutoff_requires_rebuild_and_none_restores_full_range(self):
        model = self.make_model()
        self.load(model)
        model.build_rom()
        self.load(model, x_max=None)
        self.assertFalse(model.is_trained)
        with self.assertRaisesRegex(RuntimeError, "build_rom"):
            model.predict(2, .3)
        model.build_rom()
        self.assertEqual(model.predict(2, .3).size, self.x.size)
        self.assertEqual(model.obs_x_coords.max(), 9)

    def test_invalid_cutoff_and_mismatched_training_columns(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "no curve points"):
            self.load(model, x_max=-1)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.load(model, x_max=np.nan)
        self.load(model)
        np.savetxt(self.root / "train/curve4_y.txt", self.curves[:, :-1])
        with self.assertRaisesRegex(ValueError, "spatial columns"):
            model.build_rom()


if __name__ == "__main__":
    unittest.main()
