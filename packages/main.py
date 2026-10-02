"""Bayesian inference wrapper that uses the ROM as the forward model."""

from pathlib import Path

import numpy as np

from .compute_n1 import N1Mixin
from .data_io import DataLoaderMixin
from .likelihood import LikelihoodMixin
from .plotting import PlottingMixin
from .priors import PriorMixin
from .rom import ROM
from .sampler import Sampler

MODEL_PARAMETER_NAMES = {
    "oldroyd": ("lambda", "beta"),
    "giesekus": ("lambda", "beta", "alpha"),
    "ptt": ("lambda", "beta", "epsilon"),
    "tanner": ("N1_2tau_w",),
}

LABELS = {
    "lambda": r"$\lambda$", "beta": r"$\beta$", "alpha": r"$\alpha$",
    "epsilon": r"$\epsilon$",
    "theta1": r"$\theta_1$", "theta2": r"$\theta_2$",
    "log10_theta1": r"$\log_{10}\theta_1$", "log10_theta2": r"$\log_{10}\theta_2$",
    "N1_2tau_w": r"$N_1/(2\tau_w)$",
    "sigma_noise": r"$\sigma_{\mathrm{noise}}$", "sigma_bias": r"$\sigma_{\mathrm{bias}}$",
    "l_bias": r"$\ell_{\mathrm{bias}}$",
}


class ROMCurve4BayesianInference(ROM, DataLoaderMixin, PriorMixin, LikelihoodMixin,
                                 Sampler, PlottingMixin, N1Mixin):
    """Bayesian inference whose forward model is this object's trained curve ROM."""

    y_obs_matrix = y_obs_matrix_clean = obs_x_coords = None
    sigma_noise_prior = sigma_bias_prior = samples = None
    bias_gradient_points = np.linspace(3.5, 5.0, 20)   # delta'(x)=0 here when constrained

    def __init__(self, swell_root=None, train_data_rels=None,
                 filenames_train=("curve4_y.txt", "parameters.txt"),
                 scaler="minmax", eps=1e-6,
                 lambda_bounds=(1.0, 10.0), beta_bounds=(0.1, 0.9), alpha_bounds=(0.1, 0.5),
                 epsilon_bounds=None,
                 true_theta=None, sigma_noise_percent=0.0, sigma_bias=None, l_bias=1.0,
                 constrained_model_error=False,
                 thin=1, model="oldroyd",  radius=1.0,
                 parametrize=False, seed=42, theta=None, U_avg=None, tanner_bounds=(0.01, 5.0),
                 l_bias_bounds=(0.1, 1.0)):

        self.swell_root = self.infer_dir = Path(swell_root or Path.cwd()).expanduser().resolve()
        self.train_data_rels = train_data_rels
        # parametrize: False -> (lambda, beta); True -> (theta1 = (1 - beta) lambda, theta2 = beta), both log10
        if parametrize not in (False, True, None):
            raise ValueError(f"parametrize must be True or False, got {parametrize!r}.")
        self.parametrize = bool(parametrize)
        # mean inlet velocity of the observed run: gammadot_w = 4 U_avg / R in compute_N1
        self.U_avg = None if U_avg is None else float(U_avg)
        self.tanner_bounds = tuple(map(float, tanner_bounds))

        self.model_family = model
        data = self.swell_root / Path((train_data_rels or ["datas/rom_datas"])[0]).expanduser()
        curve, params = filenames_train
        if data.is_file() or data.name.startswith("curve4_y."):
            data, curve = data.parent, data.name
        self.data_dir, self.filenames_train = data.resolve(), (curve, params)
        ROM.__init__(self, scaler=scaler, eps=eps)
        names = ["theta1", "theta2"] if self.parametrize else list(MODEL_PARAMETER_NAMES[model])
        self.lam_bounds, self.beta_bounds = lambda_bounds, beta_bounds
        self.third_parameter_bounds = (epsilon_bounds or alpha_bounds) if model == "ptt" else alpha_bounds
        self.material_parameter_names, self.n_material_params = names, len(names)

        if true_theta is not None and len(true_theta) < self.n_material_params:
            raise ValueError(f"true_theta needs at least {self.n_material_params} values for model='{model}'.")

        # extra values (e.g. Giesekus alpha when fitting Oldroyd-B) only go to the results file
        full_true = None if true_theta is None else tuple(map(float, true_theta))
        self.true_theta = None if full_true is None else full_true[:self.n_material_params]
        if model == "tanner":           # true_theta holds the fluid's (lambda, beta, ...), not N1/(2 tau_w)
            self.true_theta = None
        if self.parametrize and self.true_theta is not None:      # given as (lambda, beta)
            lam, beta = self.true_theta
            self.true_theta = ((1.0 - beta) * lam, beta)
        self.thin, self.radius = thin, radius
        if sigma_bias not in (True, False, None):     # True -> infer the model bias; None -> none
            raise ValueError(f"sigma_bias must be True or None, got {sigma_bias!r}.")
        self.sigma_noise_percent, self.sigma_bias = float(sigma_noise_percent), bool(sigma_bias)

        # l_bias: a fixed bias correlation length, or "infer" to sample it with a U(l_bias_bounds) prior.
        # sigma_bias None/False: no model bias at all, so l_bias and constrained_model_error are ignored.
        self.infer_l_bias = self.sigma_bias and isinstance(l_bias, str) and l_bias.lower() == "infer"
        self.l_bias = float(l_bias) if self.sigma_bias and not self.infer_l_bias else None
        self.l_bias_bounds = tuple(map(float, l_bias_bounds))

        self.constrained_model_error = self.sigma_bias and bool(constrained_model_error)
        self.seed = seed
        # true parameters of the model that generated the data, e.g. Giesekus (lambda, beta, alpha);
        # only written to results/<data folder>.txt (save_results); defaults to the full true_theta
        self.theta = full_true if theta is None else tuple(map(float, theta))
        self.n1_result = None

    def build_rom(self):
        if self.model_family == "tanner":
            print("model='tanner': Tanner's expression is the forward model; no ROM is built.")
            return
        X_train, param_train = self.load_training_data()
        self.train(X_train, param_train)
        names = self.material_parameter_names
        print(f"curve4_y GPR on ({', '.join(names)}), {len(X_train)} curves: {self.model.kernel_}")
        if self.parametrize:
            self._set_parametrize_bounds(param_train)
            print(f"parametrize bounds: {names[0]} {self.theta_bounds[0]}, "
                  f"{names[1]} {self.theta_bounds[1]}, implied lambda {self.lam_bounds}")
        self._mle_discrepancy()

    def _mle_discrepancy(self, n_start=200):
        """MLE fit of the ROM to the clean (noise-free) FEM curve under y_clean = ROM(theta) + delta,
        delta_i ~ N(0, sigma^2) independent: theta_MLE minimises the sum of squares, and
        sigma_MLE = RMS of delta(x) = y_clean - ROM(theta_MLE) (sigma^2 = sum(delta^2) / N).
        With sigma_bias=True, the sigma_bias prior is exponential with mean sigma_MLE
        (rate 1 / sigma_MLE). Runs once both the data (load_data) and the ROM (build_rom) exist."""
        if (self.model_family == "tanner" or self.y_obs_matrix_clean is None
                or getattr(self, "basis", None) is None):
            return
        from scipy.optimize import minimize

        lo, hi = np.asarray(self._get_sampling_bounds(), float).T      # sampler coordinates (log10 if parametrize)
        y_clean = np.asarray(self.y_obs_matrix_clean[0], float)

        def delta(P):                    # clean FEM curve minus ROM, one row per point (sampler coordinates)
            Y = np.atleast_2d(self.predict(self._to_physical(np.atleast_2d(P))))
            if Y.shape[1] > y_clean.size:
                Y = Y[:, self._obs_indices]
            return y_clean - Y

        sse = lambda P: (delta(P) ** 2).sum(axis=1)

        def objective(p):
            if not self._implied_in_bounds(self._to_physical(p)):
                return 1e10
            return float(sse(p)[0])

        starts = np.random.default_rng(self.seed).uniform(lo, hi, size=(n_start, len(lo)))
        starts = starts[[self._implied_in_bounds(t) for t in self._to_physical(starts)]]
        best = minimize(objective, starts[np.argmin(sse(starts))], method="Powell", bounds=list(zip(lo, hi)))

        self.mle_theta = self._to_physical(best.x)
        self.mle_delta = delta(best.x)[0]                                # delta(x) at obs_x_coords
        self.mle_sigma = float(np.sqrt(np.mean(self.mle_delta ** 2)))   # MLE of sigma = RMS(delta)
        dmax = float(np.max(np.abs(self.mle_delta)))
        fit = ", ".join(f"{n} = {v:.4g}" for n, v in zip(self.material_parameter_names, self.mle_theta))
        print(f"MLE fit: {fit}; sigma_MLE = RMS(delta) = {self.mle_sigma:.4g}, max|delta(x)| = {dmax:.4g}")
        if self._infer_sigma_bias():
            self.sigma_bias_prior = 1.0 / self.mle_sigma
            print(f"sigma_bias prior: exponential with mean sigma_MLE = {self.mle_sigma:.4g}")
        self.plot_mle_fit()              # MLE curve over the clean FEM curve and delta(x)

    def _set_parametrize_bounds(self, param_train=None):
        """(theta1, theta2) and the implied lambda = theta1 / (1 - theta2): the range of the training
        runs. Needs only the training parameters, so the prior is available before build_rom
        (e.g. for plot_prior)."""
        if param_train is None:
            param_train = self.load_training_data()[1]
        self.theta_bounds = [(float(lo), float(hi))
                             for lo, hi in zip(param_train.min(axis=0), param_train.max(axis=0))]
        lam = param_train[:, 0] / (1.0 - param_train[:, 1])
        self.lam_bounds = (float(lam.min()), float(lam.max()))

    def _infer_sigma_bias(self):
        return self.sigma_bias

    @staticmethod
    def _tanner_B(s):
        """Tanner's swell ratio B = 0.13 + [1 + (N1/(2 tau_w))^2 / 2]^(1/6), s = N1/(2 tau_w)."""
        return 0.13 + (1.0 + 0.5 * np.asarray(s, float) ** 2) ** (1.0 / 6.0)

    def _get_parameter_bounds(self):
        if self.model_family == "tanner":
            return [self.tanner_bounds]
        if self.parametrize:                    # the range of the training runs
            if getattr(self, "theta_bounds", None) is None:
                self._set_parametrize_bounds()
            return self.theta_bounds
        bounds = [self.lam_bounds, self.beta_bounds] + [self.third_parameter_bounds] * (self.n_material_params == 3)
        return [(float(lo), float(hi)) for lo, hi in bounds]

    def _log10_params(self):
        """Material parameters sampled as log10 (log-uniform prior): theta1 and theta2 with
        parametrize; none otherwise."""
        return (0, 1) if self.parametrize else ()

    def _get_sampling_bounds(self):
        """Material-parameter prior bounds in the sampler's coordinates (log10 for log-sampled)."""
        bounds = list(self._get_parameter_bounds())
        for i in self._log10_params():
            lo, hi = bounds[i]
            if lo <= 0:
                raise ValueError(f"log10 sampling needs a positive lower bound, got {lo}.")
            bounds[i] = (float(np.log10(lo)), float(np.log10(hi)))
        return bounds

    def _implied_in_bounds(self, theta):
        """With parametrize, the implied lambda = theta1 / (1 - theta2) must lie within the training
        range (the (theta1, theta2) box alone reaches values far outside it)."""
        if not getattr(self, "parametrize", False):
            return True
        lo, hi = self.lam_bounds
        return lo <= theta[0] / (1.0 - theta[1]) <= hi

    def _infer_l_bias(self):
        return getattr(self, "infer_l_bias", False)

    def _get_ndim(self):
        return self.n_material_params + 1 + self._infer_sigma_bias() + self._infer_l_bias()

    def _get_parameter_labels(self, latex=True):
        # sampled vector: material parameters, sigma_noise[, sigma_bias][, l_bias]
        names = (self.material_parameter_names + ["sigma_noise"]
                 + ["sigma_bias"] * self._infer_sigma_bias() + ["l_bias"] * self._infer_l_bias())
        return [LABELS[n] for n in names] if latex else names

    def _get_sampling_labels(self):
        """LaTeX labels in the sampler's coordinates (log10 of the log-sampled parameters)."""
        labels = self._get_parameter_labels()
        for i in self._log10_params():
            labels[i] = LABELS["log10_" + self.material_parameter_names[i]]
        return labels

    def _to_physical(self, phi):
        """Sampler coordinates -> physical parameters (10**phi for the log10-sampled ones).
        Works on one point, a flat sample array or a chain (parameters on the last axis)."""
        theta = np.array(phi, float)
        for i in self._log10_params():
            theta[..., i] = 10.0 ** theta[..., i]
        return theta
