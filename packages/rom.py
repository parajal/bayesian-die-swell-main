import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler}

class ROM:
    """POD basis + GPR from the parameters to the POD coefficients."""

    def __init__(self, scaler="minmax", eps=1e-6, gpr_restarts=100, random_state=42):
        self.x_scaler, self.eps, self.gpr_restarts, self.random_state = scaler, eps, gpr_restarts, random_state

    def train(self, X_train, param_train):
        """Fit the ROM to the curves X_train (one per row) at the parameter rows param_train."""
        self.snap_mean = X_train.mean(axis=0)
        U, s, _ = np.linalg.svd((X_train - self.snap_mean).T, full_matrices=False)
        self.basis = U[:, :max(1, int(np.sum(s >= self.eps * s[0])))]

        self.scaler = SCALERS[self.x_scaler]() if self.x_scaler else None
        scaled_param = self.scaler.fit_transform(param_train) if self.scaler is not None else param_train
        kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(np.ones(scaled_param.shape[1]), (1e-3, 1e3))
        gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=self.gpr_restarts,
                                       random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model = gpr.fit(scaled_param, (X_train - self.snap_mean) @ self.basis)
        return self

    def predict(self, thetas):
        """ROM curve(s) at parameter row(s): one row (1-D) gives one curve, a 2-D
        array gives one curve per row."""
        new_params = np.atleast_2d(np.asarray(thetas, float))
        if self.scaler is not None:
            new_params = self.scaler.transform(new_params)
        curves = self.model.predict(new_params).reshape(len(new_params), -1) @ self.basis.T + self.snap_mean
        return curves[0] if np.ndim(thetas) == 1 else curves
