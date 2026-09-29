"""Bayesian inference wrapper that uses the ROM as the forward model."""

import importlib
import sys
from pathlib import Path

import numpy as np

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
}

LABELS = {
    "lambda": r"$\lambda$", "beta": r"$\beta$", "alpha": r"$\alpha$",
    "epsilon": r"$\epsilon$", "N1": r"$N_1$",
    "theta1": r"$\theta_1=(1-\beta)\lambda$", "theta2": r"$\theta_2=\beta$",
    "log10_theta1": r"$\log_{10}\theta_1$", "log10_theta2": r"$\log_{10}\theta_2$",
    "log10_lambda": r"$\log_{10}\lambda$",
    "sigma_noise": r"$\sigma_{\mathrm{noise}}$", "sigma_bias": r"$\sigma_{\mathrm{bias}}$",
}

# Constitutive N1 helpers: family -> (module, function, {posterior-mean key: fn kwarg}).
_N1_FUNCS = {
    "oldroyd":  ("compute_n1_oldroyd_b_function", "compute_n1_oldroyd_b", {}),
    "giesekus": ("compute_n1_giesekus_function",  "compute_n1_giesekus",  {"alpha": "alpha"}),
    "ptt":      ("compute_n1_ptt_function",       "compute_n1_ptt",       {"epsilon": "eps"}),
}


class ROMCurve4BayesianInference(ROM, DataLoaderMixin, PriorMixin, LikelihoodMixin,
                                 Sampler, PlottingMixin):
    """Bayesian inference whose forward model is this object's trained curve ROM."""

    y_obs_matrix = y_obs_matrix_clean = obs_x_coords = None
    sigma_noise_prior = sigma_bias_prior = samples = None
    constrained_model_error = False
    bias_gradient_points = np.linspace(3.0, 5.0, 20)   # delta'(x)=0 here when constrained

    def __init__(self, swell_root=None, train_data_rels=None,
                 filenames_train=("curve4_y.txt", "parameters.txt"),
                 scaler="minmax", eps=1e-6,
                 lambda_bounds=(1.0, 10.0), beta_bounds=(0.1, 0.9), alpha_bounds=(0.1, 0.5),
                 epsilon_bounds=None,
                 n1_bounds=None, tanner_ratio_bounds=(0.001, 10.0),
                 true_theta=None, sigma_noise_percent=0.0, sigma_bias=None, l_bias=1.0,
                 constrained_model_error=False,
                 thin=1, model="oldroyd", mode="full_curve", eta0=None, radius=1.0,
                 parametrize=False, seed=42):

        if model == "tanner":
            mode = "swell_height"

        self.swell_root = self.infer_dir = Path(swell_root or Path.cwd()).expanduser().resolve()
        self.train_data_rels = train_data_rels
        # parametrize: False -> (lambda, beta); True -> (theta1, theta2=beta); 'lambda' ->
        # (theta1, lambda) with a log10-input ROM; both sampled as log10. theta1 = (1 - beta) lambda.
        if parametrize not in (False, True, None, "lambda"):
            raise ValueError(f"parametrize must be False, True or 'lambda', got {parametrize!r}.")
        self.parametrize = parametrize if parametrize == "lambda" else bool(parametrize)
        if self.parametrize and model != "oldroyd":
            raise ValueError("parametrize needs model='oldroyd' (theta1 = (1 - beta) lambda).")

        self.model_family = model
        if model == "tanner":
            ratio_bounds = tuple(map(float, tanner_ratio_bounds or (0.0, 10.0)))
            if any(not 0 <= lo < hi for lo, hi in filter(None, (n1_bounds, ratio_bounds))):
                raise ValueError("Tanner bounds must satisfy 0 <= lower < upper.")
            names = ["N1"]
            self.n1_bounds, self.tanner_ratio_bounds = n1_bounds, ratio_bounds
        else:
            # training data: data_dir/filenames_train, read by load_training_data in build_rom
            data = self.swell_root / Path((train_data_rels or ["datas/rom_datas"])[0]).expanduser()
            curve, params = filenames_train
            if data.is_file() or data.name.startswith("curve4_y."):
                data, curve = data.parent, data.name
            self.data_dir, self.filenames_train = data.resolve(), (curve, params)
            ROM.__init__(self, scaler=scaler, eps=eps, random_state=seed)
            names = (["theta1", "lambda"] if self.parametrize == "lambda"
                     else ["theta1", "theta2"] if self.parametrize
                     else list(MODEL_PARAMETER_NAMES[model]))

            # prior bounds of (lambda, beta, alpha/epsilon); with parametrize, build_rom replaces
            # them by the range of the training runs
            self.lam_bounds, self.beta_bounds = lambda_bounds, beta_bounds
            third = (epsilon_bounds or alpha_bounds) if model == "ptt" else alpha_bounds
            self.third_parameter_bounds = third
        self.material_parameter_names, self.n_material_params = names, len(names)
        self.third_parameter_name = names[2] if len(names) == 3 else None

        if true_theta is not None and len(true_theta) != self.n_material_params:
            raise ValueError(f"true_theta needs {self.n_material_params} values for model='{model}'.")

        self.true_theta = None if true_theta is None else tuple(map(float, true_theta))
        if self.parametrize and self.true_theta is not None:      # given as (lambda, beta)
            lam, beta = self.true_theta
            self.true_theta = ((1.0 - beta) * lam, lam if self.parametrize == "lambda" else beta)
        self.mode, self.thin, self.radius, self.eta0 = mode, int(thin), float(radius), eta0
        if sigma_bias not in (True, False, None):     # True -> infer the model bias; None -> none
            raise ValueError(f"sigma_bias must be True or None, got {sigma_bias!r}.")
        self.sigma_noise_percent, self.sigma_bias = float(sigma_noise_percent), bool(sigma_bias)

        self.l_bias = float(l_bias)
        # squared-exponential model bias; True -> delta(0)=0 at the die exit and
        # delta'(x)=0 at bias_gradient_points
        self.constrained_model_error = bool(constrained_model_error)
        self.seed = seed

    def build_rom(self):
        if self.model_family == "tanner":
            print("Tanner analytical forward model; no ROM is built.")
            return
        X_train, param_train = self.load_training_data()
        self.train(X_train, self._rom_input(param_train))
        names = self.material_parameter_names
        inputs = [f"log10 {n}" for n in names] if self.parametrize == "lambda" else names
        print(f"curve4_y GPR on ({', '.join(inputs)}), {len(X_train)} curves: {self.model.kernel_}")
        if self.parametrize:
            self._set_parametrize_bounds(param_train)
            implied = (f"implied beta {self.beta_bounds}" if self.parametrize == "lambda"
                       else f"implied lambda {self.lam_bounds}")
            print(f"parametrize bounds: {names[0]} {self.theta_bounds[0]}, "
                  f"{names[1]} {self.theta_bounds[1]}, {implied}")

    def _set_parametrize_bounds(self, param_train=None):
        """The two parameters and the implied third one (lambda for parametrize=True, beta for
        'lambda'): the range of the training runs. Needs only the training parameters, so the
        prior is available before build_rom (e.g. for plot_prior)."""
        if param_train is None:
            param_train = self.load_training_data()[1]
        self.theta_bounds = [(float(lo), float(hi))
                             for lo, hi in zip(param_train.min(axis=0), param_train.max(axis=0))]
        if self.parametrize == "lambda":          # (theta1, lambda): beta = 1 - theta1 / lambda
            beta = 1.0 - param_train[:, 0] / param_train[:, 1]
            self.beta_bounds = (float(beta.min()), float(beta.max()))
        else:                                     # (theta1, theta2): lambda = theta1 / (1 - theta2)
            lam = param_train[:, 0] / (1.0 - param_train[:, 1])
            self.lam_bounds = (float(lam.min()), float(lam.max()))

    def _rom_input(self, thetas):
        """Physical material parameters -> ROM (GPR) inputs: log10 with parametrize='lambda'."""
        x = np.array(thetas, float)
        if self.parametrize == "lambda":
            x[..., :2] = np.log10(x[..., :2])
        return x

    def predict(self, thetas):
        """ROM curve(s) at physical material parameter row(s) (see ROM.predict)."""
        return ROM.predict(self, self._rom_input(thetas))

    def _infer_sigma_bias(self):
        return self.sigma_bias

    def _get_parameter_bounds(self):
        if self.model_family == "tanner":
            if self.n1_bounds is not None:
                return [tuple(map(float, self.n1_bounds))]
            lo, hi = 2.0 * self._tanner_tau_w() * np.asarray(self.tanner_ratio_bounds)
            return [(float(lo), float(hi))]
        if self.parametrize:                      # the range of the training runs
            if getattr(self, "theta_bounds", None) is None:
                self._set_parametrize_bounds()
            return self.theta_bounds
        bounds = [self.lam_bounds, self.beta_bounds] + [self.third_parameter_bounds] * (self.n_material_params == 3)
        return [(float(lo), float(hi)) for lo, hi in bounds]

    def _log10_params(self):
        """Material parameters sampled as log10 (log-uniform prior): both with parametrize
        (theta1, theta2 or theta1, lambda); none otherwise."""
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
        """With parametrize, the implied third parameter must lie within the training range (the
        two-parameter box alone reaches values far outside it): lambda = theta1 / (1 - theta2) for
        parametrize=True; beta = 1 - theta1 / lambda for 'lambda' (a straight band in log10)."""
        if not getattr(self, "parametrize", False):
            return True
        if self.parametrize == "lambda":
            lo, hi = self.beta_bounds
            return lo <= 1.0 - theta[0] / theta[1] <= hi
        lo, hi = self.lam_bounds
        return lo <= theta[0] / (1.0 - theta[1]) <= hi

    def _get_ndim(self):
        return self.n_material_params + 1 + self._infer_sigma_bias()

    def _get_parameter_labels(self, latex=True):
        names = (self.material_parameter_names + ["sigma_noise"]
                 + ["sigma_bias"] * self._infer_sigma_bias())
        return [LABELS[n] for n in names] if latex else names

    def _get_sampling_labels(self):
        """LaTeX labels in the sampler's coordinates (log10 of the log-sampled parameters)."""
        labels = self._get_parameter_labels()
        for i in self._log10_params():
            labels[i] = LABELS["log10_" + self.material_parameter_names[i]]
        return labels

    def _extract_noise_bias(self, theta):
        n = self.n_material_params
        return float(theta[n]), float(theta[n + 1]) if self._infer_sigma_bias() else None

    def _to_physical(self, phi):
        """Sampler coordinates -> physical parameters (10**phi for the log10-sampled ones).
        Works on one point, a flat sample array or a chain (parameters on the last axis)."""
        theta = np.array(phi, float)
        for i in self._log10_params():
            theta[..., i] = 10.0 ** theta[..., i]
        return theta

    _to_physical_chain = _to_physical

    def _load_n1_function(self, module_name, func_name):
        """Import a compute_n1_* helper from the repo root or swell_root."""
        for root in (Path(__file__).resolve().parents[1], self.swell_root):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        return getattr(importlib.import_module(module_name), func_name)

    def _pressure_drop_dpdx(self):
        """dp/dx read from pressure_drop.txt in the observation folder (None if it is absent)."""
        f = Path(self.infer_dir) / "pressure_drop.txt"
        return float(np.loadtxt(f).ravel()[0]) if f.is_file() else None

    def _wall_shear(self, u_avg, r):
        """(gammadot_w, tau_xy, source): gammadot_w = 4 U_avg / R, and tau_xy = R |dp/dx| / 2
        from the observation's pressure_drop.txt, else eta0 gammadot_w (eta0 argument, or 1)."""
        rate = 4.0 * u_avg / r
        dpdx = self._pressure_drop_dpdx()
        if dpdx is not None:
            return rate, 0.5 * r * abs(dpdx), f"R|dp/dx|/2 from pressure_drop.txt, dp/dx = {dpdx:.6g}"
        eta0 = float(self.eta0 if self.eta0 is not None else 1.0)
        return rate, eta0 * rate, f"eta0 * gammadot_w, eta0 = {eta0:g} (no pressure_drop.txt)"

    def _eta0_for_n1(self, u_avg, r):
        """eta0 for N1. Oldroyd-B (constant viscosity): read the observation's pressure_drop.txt
        and use tau_xy / gammadot_w with tau_xy = R |dp/dx| / 2 and gammadot_w = 4 U_avg / R.
        Without that file (and for the other families): the eta0 argument, else 1."""
        dpdx = self._pressure_drop_dpdx() if self.model_family == "oldroyd" else None
        if dpdx is not None:
            return 0.5 * r * abs(dpdx) / (4.0 * u_avg / r)
        return float(self.eta0 if self.eta0 is not None else 1.0)

    def compute_N1(self, U_avg, radius=None, n_expectation=500, analytic_oldroyd=True, **rheo_kwargs):
        """Posterior distribution of N1 at the die-wall shear rate gammadot_w = 4 U_avg / R
        (e.g. ``model.compute_N1(U_avg=0.1)``): N1(theta_s) for each posterior sample theta_s,
        summarised by E[N1] (their mean), SD and 95% credible interval, and likewise
        S_R = N1 / tau_w.

        Oldroyd-B: closed form over all samples, N1 = 2 theta1 gammadot_w tau_xy with
        theta1 = (1 - beta) lambda (inferred directly with parametrize) and tau_xy = R |dp/dx| / 2
        from the observation's pressure_drop.txt (else eta0 gammadot_w, eta0 argument or 1);
        this is 2 (1 - beta) eta0 lambda gammadot_w^2 with eta0 = tau_xy / gammadot_w.
        Other families (or analytic_oldroyd=False) run one rheology simulation per draw on a
        random ``n_expectation``-sized subset of the samples.
        """
        if self.samples is None:
            raise RuntimeError("no posterior samples yet; call run_mcmc() first")
        u_avg, r = float(U_avg), float(self.radius if radius is None else radius)
        family, names = self.model_family, self.material_parameter_names
        S = np.asarray(self.samples, float)

        if family == "oldroyd" and (self.parametrize or (analytic_oldroyd and not rheo_kwargs)):
            rate, tau_w, src = self._wall_shear(u_avg, r)
            print(f"tau_xy = {tau_w:.6g} ({src}), gammadot_w = {rate:.6g}, "
                  f"eta0 = tau_xy / gammadot_w = {tau_w / rate:.6g}")
            theta1 = S[:, 0] if self.parametrize else (1.0 - S[:, 1]) * S[:, 0]
            n1s = 2.0 * theta1 * rate * tau_w
            tws, rates = np.full(n1s.shape, tau_w), np.full(n1s.shape, rate)
        else:
            if family not in _N1_FUNCS:
                raise ValueError(f"compute_N1() is not defined for model='{family}'.")
            k = min(int(n_expectation), S.shape[0])
            draws = S[np.random.default_rng(0).choice(S.shape[0], size=k, replace=False), :len(names)]
            module, func, extra = _N1_FUNCS[family]
            fn = self._load_n1_function(module, func)
            eta0 = self._eta0_for_n1(u_avg, r)
            n1s, tws, rates = (np.empty(k) for _ in range(3))
            for j, row in enumerate(draws):
                p = dict(zip(names, map(float, row)))
                o = fn(U_avg=u_avg, radius=r, lam=p["lambda"], beta=p["beta"], eta0=eta0,
                       **{kw: p[key] for key, kw in extra.items()}, **rheo_kwargs)
                rates[j] = float(np.atleast_1d(np.asarray(o["rates"], float))[0])
                n1s[j] = float(np.atleast_1d(np.asarray(o["N1"], float))[0])
                tws[j] = float(np.atleast_1d(np.asarray(o.get("tau_xy", eta0 * rates[j]), float))[0])

        En1, sd, ci = float(n1s.mean()), float(n1s.std(ddof=1)), np.percentile(n1s, [2.5, 97.5])
        with np.errstate(divide="ignore", invalid="ignore"):
            sr_s = np.where(tws != 0, n1s / tws, np.nan)
        sr_mean, sr_sd = float(np.nanmean(sr_s)), float(np.nanstd(sr_s, ddof=1))
        sr_ci = np.nanpercentile(sr_s, [2.5, 97.5])
        print(f"E[N1] ({family}) = {En1:.6g} +/- {sd:.6g}  (95% CI [{ci[0]:.6g}, {ci[1]:.6g}], "
              f"n={n1s.size})\nE[S_R] = {sr_mean:.6g} +/- {sr_sd:.6g}  "
              f"(95% CI [{sr_ci[0]:.6g}, {sr_ci[1]:.6g}])  E[tau_w] = {float(np.mean(tws)):.6g}")
        return {"model": family, "N1": En1, "N1_std": sd, "N1_ci": ci, "N1_samples": n1s,
                "Sr": sr_mean, "Sr_std": sr_sd, "Sr_ci": sr_ci, "tau_w": float(np.mean(tws)),
                "rates": float(np.mean(rates)), "n": int(n1s.size)}
