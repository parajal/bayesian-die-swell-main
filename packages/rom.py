"""Compact non-intrusive curve ROM: POD + GPR."""

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MODEL_PARAMETER_NAMES = {
    "ratio": ("Sr",),
    "oldroyd": ("lambda", "beta"),
    "giesekus": ("lambda", "beta", "alpha"),
    "ptt": ("lambda", "beta", "epsilon"),
}
SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler}


def normalize_model_family(model_family):
    name = str(model_family).strip().lower()
    if name not in ("auto", *MODEL_PARAMETER_NAMES):
        raise ValueError(f"model must be one of {['auto', *MODEL_PARAMETER_NAMES]}")
    return name


class ROM:
    """POD basis + GPR from material parameters (and optional U_avg) to POD coefficients."""

    lam_bounds = beta_bounds = third_parameter_bounds = alpha_bounds = epsilon_bounds = sr_bounds = None
    use_uavg = is_trained = False
    pressure_model = pressure_train = None

    def __init__(self, data_dir, filenames_train=("curve4_y.txt", "parameters.txt"),
                 scaler="minmax", eps=1e-6, gpr_restarts=20, random_state=42,
                 model_family="auto"):
        self.data_dir = Path(data_dir)
        self.filenames_train = list(filenames_train)
        self.scaler, self.eps, self.gpr_restarts, self.random_state = scaler, eps, gpr_restarts, random_state
        self.model_family = normalize_model_family(model_family)
        self._set_names("giesekus" if self.model_family == "auto" else self.model_family)

    def _set_names(self, family):
        self.material_parameter_names = list(MODEL_PARAMETER_NAMES[family])
        self.n_material_params = len(self.material_parameter_names)
        self.third_parameter_name = self.material_parameter_names[2] if self.n_material_params == 3 else None

    def _load(self, filenames):
        y, p = (np.loadtxt(self.data_dir / f, ndmin=2) for f in filenames)
        if len(y) != len(p):
            raise ValueError("curve and parameter files must have the same number of rows")
        return y, p

    def train_curve(self):
        Y, P = self._load(self.filenames_train)
        curve_size = Y.shape[1]
        curve_idx = getattr(self, "_curve_indices", None)
        if curve_idx is not None:
            if curve_size != self._curve_size:
                raise ValueError("training curves must share the observation spatial columns")
            Y = Y[:, curve_idx]
        else:
            curve_idx = np.arange(curve_size)
        self.snapshots_train, self.parameters_train = Y, P

        # Parameter columns: material parameters, plus U_avg (column 3) if it varies
        varies = lambda j: P.shape[1] > j and np.ptp(P[:, j]) > 1e-12
        auto = self.model_family == "auto"
        self._set_names(("giesekus" if varies(2) else "oldroyd") if auto else self.model_family)
        if P.shape[1] < self.n_material_params:
            raise ValueError(f"training parameters need columns {self.material_parameter_names}")
        self.use_uavg = bool(auto and varies(3))
        self.param_cols = list(range(self.n_material_params)) + [3] * self.use_uavg

        # POD basis
        self.snap_mean = Y.mean(axis=0)
        centered = Y - self.snap_mean
        U, s, _ = np.linalg.svd(centered.T, full_matrices=False)
        self.basis = U[:, :max(1, int(np.sum(s >= self.eps * s[0])))]

        # Input scaling stored as an affine map: X_scaled = X * a + b
        X = P[:, self.param_cols]
        self._a, self._b = np.ones(X.shape[1]), np.zeros(X.shape[1])
        if self.scaler:
            sc = SCALERS[self.scaler]().fit(X)
            self._a, self._b = ((sc.scale_, sc.min_) if self.scaler == "minmax"
                                else (1 / sc.scale_, -sc.mean_ / sc.scale_))

        # GPR, then precompute a fast predictor: curve = amp * exp(-d²/2) @ W + mean
        self.model = self._gpr(self._scale(X), centered @ self.basis)
        k = self.model.kernel_
        self._amp, self._ls = k.k1.constant_value, k.k2.length_scale
        self._Xtrain = self.model.X_train_ / self._ls
        self._W = self.model.alpha_.reshape(len(X), -1) @ self.basis.T

        self._set_bounds()
        self._training_curve_size = curve_size
        self._training_curve_indices = curve_idx.copy()
        self.is_trained = True
        return self

    def _gpr(self, X, Y):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(X.shape[1]), (1e-3, 1e3))
        gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=self.gpr_restarts,
                                       random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return gpr.fit(X, Y)

    def _scale(self, X):
        return X * self._a + self._b

    def _inputs(self, thetas, u_avg_val=None):
        """Scaled GPR inputs for material-parameter rows, plus U_avg if trained with it."""
        X = np.atleast_2d(np.asarray(thetas, float))[:, :self.n_material_params]
        if self.use_uavg:
            u = getattr(self, "u_avg_obs", None) if u_avg_val is None else u_avg_val
            X = np.column_stack([X, np.full(len(X), float(u))])
        return self._scale(X)

    def _set_bounds(self):
        """Fill any bounds not set by the user from the training range."""
        lo, hi = self.parameters_train.min(axis=0), self.parameters_train.max(axis=0)
        rng = lambda j: (float(lo[j]), float(hi[j]))
        if self.model_family == "ratio":
            self.sr_bounds = self.sr_bounds or rng(0)
            return
        self.lam_bounds = self.lam_bounds or rng(0)
        self.beta_bounds = self.beta_bounds or rng(1)
        if self.third_parameter_name:
            self.third_parameter_bounds = self.alpha_bounds = (
                self.third_parameter_bounds or self.alpha_bounds or rng(2))
            if self.third_parameter_name == "epsilon":
                self.epsilon_bounds = self.third_parameter_bounds

    def _predict_curves(self, thetas, u_avg_val=None):
        """Curves (m, n_points) for material-parameter rows (m, n_material)."""
        if not self.is_trained:
            raise RuntimeError("Call build_rom() or train_curve() after loading the selected x range.")
        X = self._inputs(thetas, u_avg_val) / self._ls
        Y = self._amp * np.exp(-0.5 * cdist(X, self._Xtrain, "sqeuclidean")) @ self._W + self.snap_mean
        if not np.isfinite(Y).all():
            raise FloatingPointError("ROM prediction produced NaN/Inf.")
        return Y

    def predict(self, *theta, u_avg_val=None, alpha_val=None, epsilon_val=None):
        """Single curve; theta is (Sr,), (lambda, beta) or (lambda, beta, third)."""
        third = alpha_val if epsilon_val is None else epsilon_val
        return self._predict_curves([[*theta, third][:self.n_material_params]], u_avg_val)[0]

    # ---------------------------------------------------------------- pressure GPR
    def train_pressure(self, filename):
        """Train a GPR from the material parameters to the wall pressure in ``filename``."""
        P = self.parameters_train
        self.pressure_train = np.loadtxt(self.data_dir / filename, ndmin=2)
        if len(self.pressure_train) != len(P):
            raise ValueError("pressure file rows must match parameters_train rows")
        self.pressure_model = self._gpr(self._scale(P[:, self.param_cols]), self.pressure_train)

    def predict_pressure(self, thetas):
        """GPR-predicted pressure for one (1-D) or many (2-D) material-parameter rows.

        Returns a 1-D vector for a single theta, else an ``(m, n_pressure)`` array.
        """
        if self.pressure_model is None:
            raise RuntimeError("pressure model untrained; use use_pressure=True and build_rom()")
        pred = self.pressure_model.predict(self._inputs(thetas))
        if not np.isfinite(pred).all():
            raise FloatingPointError("pressure GPR prediction produced NaN/Inf.")
        pred = pred.reshape(-1, self.pressure_train.shape[1])
        return pred[0] if np.ndim(thetas) == 1 else pred
