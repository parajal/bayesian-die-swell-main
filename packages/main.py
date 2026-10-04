"""Bayesian inference wrapper that uses the ROM as the forward model."""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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

# Material parameters that enter the ROM (GPR) as log10: the Giesekus alpha and PTT epsilon grids are
# roughly log-spaced (0.01 ... 0.4), so a GPR on the linear value picks a short length scale and bulges
# between the coarse nodes at large alpha/epsilon. The prior stays uniform on the parameter itself.
LOG_ROM_INPUTS = {"giesekus": (2,), "ptt": (2,)}

LABELS = {
    "lambda": r"$\lambda$", "beta": r"$\beta$", "alpha": r"$\alpha$", "epsilon": r"$\epsilon$",
    "theta1": r"$\theta_1$", "theta2": r"$\theta_2$", "theta3": r"$\theta_3$",
    "log10_theta1": r"$\log_{10}\theta_1$", "log10_theta2": r"$\log_{10}\theta_2$",
    "log10_theta3": r"$\log_{10}\theta_3$",
    "N1_2tau_w": r"$N_1/(2\tau_w)$",
    "sigma_noise": r"$\sigma_{\mathrm{noise}}$", "sigma_bias": r"$\sigma_{\mathrm{bias}}$",
    "l_bias": r"$\ell_{\mathrm{bias}}$",
}


class ROMCurve4BayesianInference(ROM, DataLoaderMixin, PriorMixin, LikelihoodMixin,
                                 Sampler, PlottingMixin, N1Mixin):
    """Bayesian inference whose forward model is this object's trained curve ROM."""

    y_obs_matrix = y_obs_matrix_clean = obs_x_coords = None
    sigma_noise_prior = sigma_bias_prior = samples = theta_bounds = None
    bias_gradient_points = np.linspace(3.5, 5.0, 20)

    @staticmethod
    def _tanner_B(s):
        """Tanner's swell ratio B = 0.13 + [1 + (N1/(2 tau_w))^2 / 2]^(1/6), s = N1/(2 tau_w)."""
        return 0.13 + (1.0 + 0.5 * np.asarray(s, float) ** 2) ** (1.0 / 6.0)
    
    def __init__(self, swell_root=None, train_data_rels=None,
                 filenames_train=("curve4_y.txt", "parameters.txt"),
                 scaler="minmax", eps=1e-6,
                 lambda_bounds=(1.0, 10.0), beta_bounds=(0.1, 0.9), alpha_bounds=(0.1, 0.5),
                 epsilon_bounds=None,
                 true_theta=None, sigma_noise_percent=0.0, sigma_bias=None, l_bias=1.0,
                 constrained_model_error=False,
                 thin=1, model="oldroyd", radius=1.0,
                 parametrize=False, seed=42, theta=None, U_avg=None, tanner_bounds=(0.01, 5.0),
                 l_bias_bounds=(0.1, 1.5), rabinowitsch_correction=False):

        self.swell_root = self.infer_dir = Path(swell_root or Path.cwd()).expanduser().resolve()        
        self.train_data_rels = train_data_rels
        self.parametrize = bool(parametrize)  # True: theta1 = (1 - beta) lambda, theta2 = beta, theta3 = alpha / epsilon, all log10-sampled
        self.U_avg = None if U_avg is None else float(U_avg)  # mean inlet velocity: gammadot_w = 4 U_avg / R
        # compute_N1: Rabinowitsch-corrected wall shear rate (True) or the Oldroyd-B 4 U_avg / R (False)
        self.rabinowitsch_correction = bool(rabinowitsch_correction)
        self.tanner_bounds = tuple(map(float, tanner_bounds))
        self.model_family = model

        data = self.swell_root / Path((train_data_rels or ["datas/rom_datas"])[0]).expanduser()
        curve, params = filenames_train
        if data.is_file() or data.name.startswith("curve4_y."):  # path points at the curve file itself
            data, curve = data.parent, data.name
        self.data_dir, self.filenames_train = data.resolve(), (curve, params)
        ROM.__init__(self, scaler=scaler, eps=eps)

        names = list(MODEL_PARAMETER_NAMES[model])
        if self.parametrize and model != "tanner":   # theta1 = (1 - beta) lambda, theta2 = beta, theta3 = alpha / epsilon
            names = ["theta1", "theta2", "theta3"][:len(names)]
        self.material_parameter_names, self.n_material_params = names, len(names)
        self.lam_bounds, self.beta_bounds = lambda_bounds, beta_bounds
        self.third_parameter_bounds = (epsilon_bounds or alpha_bounds) if model == "ptt" else alpha_bounds

        full_true = None if true_theta is None else tuple(map(float, true_theta))
        self.true_theta = None if full_true is None or model == "tanner" else full_true[:len(names)]
        if self.parametrize and self.true_theta is not None:   # given as (lambda, beta[, alpha / epsilon])
            lam, beta, *third = self.true_theta
            self.true_theta = ((1.0 - beta) * lam, beta, *third)
        self.theta = full_true if theta is None else tuple(map(float, theta))

        self.sigma_noise_percent = float(sigma_noise_percent)
        self.sigma_bias = bool(sigma_bias) and model != "tanner"   # Tanner: one height, no bias term
        self.infer_l_bias = self.sigma_bias and isinstance(l_bias, str) and l_bias.lower() == "infer"
        self.l_bias = float(l_bias) if self.sigma_bias and not self.infer_l_bias else None
        self.l_bias_bounds = tuple(map(float, l_bias_bounds))
        self.constrained_model_error = self.sigma_bias and bool(constrained_model_error)

        self.thin, self.radius, self.seed = thin, radius, seed
        self.n1_result = None

    def build_rom(self):
        if self.model_family == "tanner":
            print("model='tanner': Tanner's expression is the forward model; no ROM is built.")
            return
        X_train, param_train = self.load_training_data()
        self.train(X_train, self._rom_input(param_train))
        names = self.material_parameter_names
        logs = LOG_ROM_INPUTS.get(self.model_family, ())
        inputs = [f"log10 {n}" if i in logs else n for i, n in enumerate(names)]
        print(f"curve4_y GPR on ({', '.join(inputs)}), {len(X_train)} curves: {self.model.kernel_}")
        if self.parametrize:
            self._set_parametrize_bounds(param_train)
            bounds = ", ".join(f"{n} {b}" for n, b in zip(names, self.theta_bounds))
            print(f"parametrize bounds: {bounds}, implied lambda {self.lam_bounds}")
        self._mle_discrepancy()

    def _mle_discrepancy(self, n_start=200):
        if (self.model_family == "tanner" or self.y_obs_matrix_clean is None
                or getattr(self, "basis", None) is None):
            return
        names = self.material_parameter_names
        lo, hi = np.asarray(self._get_sampling_bounds(), float).T  # sampler coordinates
        y_clean = np.asarray(self.y_obs_matrix_clean[0], float)

        def delta(P): 
            Y = np.atleast_2d(self.predict(self._to_physical(np.atleast_2d(P))))
            return y_clean - (Y[:, self._obs_indices] if Y.shape[1] > y_clean.size else Y)

        sse = lambda P: (delta(P) ** 2).sum(axis=1)
        bounds = list(zip(lo, hi))   # within the prior box (the log10 ROM input needs alpha, epsilon > 0)
        starts = np.random.default_rng(self.seed).uniform(lo, hi, size=(n_start, len(lo)))
        starts = starts[[self._implied_in_bounds(t) for t in self._to_physical(starts)]]
        best = minimize(lambda p: float(sse(p)[0]), starts[np.argmin(sse(starts))],
                        method="Powell", bounds=bounds)

        self.mle_theta = self._to_physical(best.x)
        self.mle_delta = delta(best.x)[0]                  
        self.mle_sigma = float(np.sqrt(np.mean(self.mle_delta ** 2)))  
        fit = ", ".join(f"{n} = {v:.4g}" for n, v in zip(names, self.mle_theta))
        print(f"MLE fit: {fit}; sigma_MLE = RMS(delta) = {self.mle_sigma:.4g}, "
              f"max|delta(x)| = {np.max(np.abs(self.mle_delta)):.4g}")
        if self._infer_sigma_bias():
            self.sigma_bias_prior = 1.0 / self.mle_sigma
            print(f"sigma_bias prior: exponential with mean sigma_MLE = {self.mle_sigma:.4g}")
        self.plot_mle_fit()


    def _rom_input(self, thetas):
        """Physical material parameters -> ROM (GPR) input: log10 of the LOG_ROM_INPUTS columns."""
        x = np.array(thetas, float)
        for i in LOG_ROM_INPUTS.get(self.model_family, ()):
            if np.any(x[..., i] <= 0):
                raise ValueError(f"{self.material_parameter_names[i]} must be > 0: the ROM takes its log10.")
            x[..., i] = np.log10(x[..., i])
        return x

    def predict(self, thetas):
        """ROM curve(s) at physical material parameter row(s); see ROM.predict."""
        return ROM.predict(self, self._rom_input(thetas))

    def _set_parametrize_bounds(self, param_train=None):
        """(theta1, theta2[, theta3]) bounds and the implied lambda = theta1 / (1 - theta2): the training range.
        Needs only the training parameters, so the prior exists before build_rom (e.g. for plot_prior)."""
        if param_train is None:
            param_train = self.load_training_data()[1]
        self.theta_bounds = [(float(lo), float(hi))
                             for lo, hi in zip(param_train.min(axis=0), param_train.max(axis=0))]
        lam = param_train[:, 0] / (1.0 - param_train[:, 1])
        self.lam_bounds = (float(lam.min()), float(lam.max()))


    def _infer_sigma_bias(self):
        return self.sigma_bias

    def _infer_l_bias(self):
        return getattr(self, "infer_l_bias", False)

    def _log10_params(self):
        """Indices sampled as log10 (log-uniform prior): all thetas with parametrize (theta1, theta2 and,
        for Giesekus / PTT, theta3 = alpha / epsilon); none otherwise."""
        return tuple(range(self.n_material_params)) if self.parametrize else ()

    def _get_parameter_bounds(self):
        if self.model_family == "tanner":
            return [self.tanner_bounds]
        if self.parametrize:  # the range of the training runs
            if self.theta_bounds is None:
                self._set_parametrize_bounds()
            return self.theta_bounds
        bounds = [self.lam_bounds, self.beta_bounds, self.third_parameter_bounds][:self.n_material_params]
        return [(float(lo), float(hi)) for lo, hi in bounds]

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
        """With parametrize, the implied lambda = theta1 / (1 - theta2) must lie in the training range
        (the (theta1, theta2) box alone reaches values far outside it)."""
        if not self.parametrize:
            return True
        lo, hi = self.lam_bounds
        return lo <= theta[0] / (1.0 - theta[1]) <= hi

    def _sigma_rates(self):
        """Exponential prior rates of the sampled sigmas: sigma_noise (except for Tanner, where it is
        fixed at the known noise level) and sigma_bias when inferred."""
        return ([self.sigma_noise_prior] * (self.model_family != "tanner")
                + [self.sigma_bias_prior] * self._infer_sigma_bias())

    def _get_ndim(self):
        return self.n_material_params + len(self._sigma_rates()) + self._infer_l_bias()

    def _get_parameter_labels(self, latex=True):
        names = (self.material_parameter_names + ["sigma_noise"] * (self.model_family != "tanner")
                 + ["sigma_bias"] * self._infer_sigma_bias() + ["l_bias"] * self._infer_l_bias())
        return [LABELS[n] for n in names] if latex else names

    def _get_sampling_labels(self):
        labels = self._get_parameter_labels()
        for i in self._log10_params():
            labels[i] = LABELS["log10_" + self.material_parameter_names[i]]
        return labels

    def _to_physical(self, phi):
        theta = np.array(phi, float)
        for i in self._log10_params():
            theta[..., i] = 10.0 ** theta[..., i]
        return theta