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

LABELS = {
    "lambda": r"$\lambda$", "beta": r"$\beta$", "alpha": r"$\alpha$",
    "epsilon": r"$\epsilon$", "eta_0": r"$\eta_0$", "N1": r"$N_1$",
    "Sr": r"$S_R = N_1/(2\tau_w)$",
    "sigma_noise": r"$\sigma_{\mathrm{noise}}$", "sigma_bias": r"$\sigma_{\mathrm{bias}}$",
    "l_bias": r"$\ell_{\mathrm{bias}}$", "c_bias": r"$c_{\mathrm{bias}}$",
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

    pressure_obs = pressure_obs_clean = None
    y_obs_matrix = y_obs_matrix_clean = observed_uavgs = obs_x_coords = None
    sigma_noise_prior = sigma_bias_prior = c_bias_prior_sd = samples = None
    u_avg_obs = 1.0
    l_bias_pc_lambda = None       # PC-prior rate for l_bias (set from data in load_data)
    constrained_model_error = False  # True -> both endpoint constraints below; False -> neither
    bias_anchor = None            # resolved anchor: 0.0 if constrained_model_error else None (off)
    constrained_gradient = False  # resolved: True -> delta'(5)=0, set from constrained_model_error

    def __init__(self, swell_root=None, train_data_rels=None,
                 filenames_train=("curve4_y.txt", "parameters.txt"),
                 scaler="minmax", eps=1e-6,
                 lambda_bounds=(1.0, 10.0), beta_bounds=(0.1, 0.9), alpha_bounds=(0.1, 0.5),
                 epsilon_bounds=None, eta0_bounds=None, sr_bounds=None,
                 n1_bounds=None, tanner_ratio_bounds=(0.001, 10.0),
                 true_theta=None, sigma_noise_percent=0.0, sigma_bias=None, sigma_bias_scale=0.1,
                 sigma_bias_pc=None, sigma_bias_from_discrepancy=False,
                 mean_bias=None,
                 l_bias=1.0, l_bias_prior="uniform", l_bias_pc=(0.10, 0.05), l_bias_bounds=(0.1, 5.0),
                 correlation_matrix="squared_exp",
                 constrained_model_error=False,
                 thin=1, model="auto", mode="full_curve", eta0=None, radius=1.0,
                 use_pressure=False, pressure_filename="pressure_drop.txt",
                 augment_eta0=False, seed=42):
        
        if model == "tanner":
            mode = "swell_height"
            if use_pressure:
                print("model='tanner': pressure only sets the wall stress (tau_w = -Δp·R/2).")
                use_pressure = False

        self.swell_root = self.infer_dir = Path(swell_root or Path.cwd()).expanduser().resolve()
        self.train_data_rels = train_data_rels
        self._augment_eta0 = model == "oldroyd" and bool(use_pressure or augment_eta0)

        if model == "tanner":
            ratio_bounds = tuple(map(float, tanner_ratio_bounds or (0.0, 10.0)))
            if any(not 0 <= lo < hi for lo, hi in filter(None, (n1_bounds, ratio_bounds))):
                raise ValueError("Tanner bounds must satisfy 0 <= lower < upper.")
            self.model_family, self.data_dir, self.scaler, self.is_trained = "tanner", None, None, True
            self.material_parameter_names, self.n_material_params = ["N1"], 1
            self.third_parameter_name, self.param_cols, self.use_uavg = None, [0], False
            self.n1_bounds, self.tanner_ratio_bounds = n1_bounds, ratio_bounds
        else:
            data = self.swell_root / Path((train_data_rels or ["datas/rom_datas"])[0]).expanduser()
            curve, params = filenames_train
            if model == "ratio" and params == "parameters.txt":
                params = "parameters1.txt"
            if data.is_file() or data.name.startswith("curve4_y."):
                data, curve = data.parent, data.name
            ROM.__init__(self, data.resolve(), (curve, params), scaler=scaler,
                         eps=eps, random_state=seed, model_family=model)

            # User bounds; any left None are filled from the training range in train()
            self.lam_bounds, self.beta_bounds, self.sr_bounds = lambda_bounds, beta_bounds, sr_bounds
            if self._augment_eta0:
                third = eta0_bounds
            elif model == "ptt":
                third = epsilon_bounds or alpha_bounds
            else:
                third = alpha_bounds
            self.third_parameter_bounds = self.alpha_bounds = third

        n_true = (2, 3) if model == "auto" else (self.n_material_params,)
        if true_theta is not None and len(true_theta) not in n_true:
            raise ValueError(f"true_theta needs {n_true} values for model='{model}'.")

        self.true_theta = None if true_theta is None else tuple(map(float, true_theta))
        self.mode, self.thin, self.radius, self.eta0 = mode, int(thin), float(radius), eta0
        self.sigma_noise_percent, self.sigma_bias = float(sigma_noise_percent), sigma_bias
        self._sigma_bias_frac = float(sigma_bias_scale)   # exponential prior mean = frac * disp
        self.sigma_bias_pc = None if sigma_bias_pc is None else tuple(map(float, sigma_bias_pc))
        # True -> set the sigma_bias Exp prior from the discrepancy RMS at run_mcmc
        # (also implied whenever l_bias='fixed', which runs the same routine).
        self.sigma_bias_from_discrepancy = bool(sigma_bias_from_discrepancy)

        # "infer" -> sampled; "fixed" -> fixed to the discrepancy ACF scale at run_mcmc; else a float
        self.l_bias = l_bias if l_bias in ("infer", "fixed") else float(l_bias)
        if l_bias_prior not in ("pc", "uniform"):
            raise ValueError("l_bias_prior must be 'pc' or 'uniform'.")
        self.l_bias_prior = l_bias_prior
        self.l_bias_pc = tuple(map(float, l_bias_pc))   # PC prior: (l0 as fraction of x-range, alpha)
        self.l_bias_bounds = tuple(map(float, l_bias_bounds))   # uniform prior: (lo, hi)
        if correlation_matrix not in ("squared_exp", "matern"):
            raise ValueError("correlation_matrix must be 'squared_exp' or 'matern'.")
        self.correlation_matrix = correlation_matrix
        # constrained_model_error=True pins the discrepancy at both endpoints:
        # delta(0)=0 (anchor) and delta'(5)=0 (flat gradient); False leaves both off.
        self.constrained_model_error = bool(constrained_model_error)
        self.bias_anchor = 0.0 if self.constrained_model_error else None   # delta(0)=0
        self.constrained_gradient = self.constrained_model_error           # delta'(5)=0
        if self.correlation_matrix == "matern" and self.constrained_gradient:
            raise ValueError(
                "constrained_model_error needs delta'(5)=0, but the Matern (nu=1/2, "
                "exponential) correlation is not mean-square differentiable. Use "
                "correlation_matrix='squared_exp' for this derivative constraint."
            )
        self.mean_bias = mean_bias
        if self._infer_mean_bias() and self.mode == "swell_height":
            raise ValueError("mean_bias='infer' is unidentifiable in mode='swell_height'; "
                             "use mode='full_curve'.")
        if self.bias_anchor is not None and self._infer_mean_bias():
            raise ValueError("constrained_model_error and mean_bias='infer' are contradictory: the "
                             "anchor forces delta(x0)=0, but a constant c would move it. Use one.")
        if self._infer_l_bias():
            if not self._infer_sigma_bias():
                raise ValueError("l_bias='infer' requires sigma_bias='infer' (the length scale "
                                 "only enters the likelihood through the discrepancy GP).")
            if self.mode == "swell_height":
                raise ValueError("l_bias='infer' is unidentifiable in mode='swell_height'; "
                                 "use mode='full_curve'.")
        if self.l_bias == "fixed" and self.mode == "swell_height":
            raise ValueError("l_bias='fixed' needs the full discrepancy curve for its ACF; "
                             "it is undefined in mode='swell_height'. Use mode='full_curve'.")
        self.use_pressure, self.pressure_train_filename = bool(use_pressure), pressure_filename
        self.seed = seed

    def _set_names(self, family):
        """Oldroyd-B + pressure adds eta_0 as a third material parameter."""
        super()._set_names(family)
        if getattr(self, "_augment_eta0", False):
            self.material_parameter_names = ["lambda", "beta", "eta_0"]
            self.n_material_params, self.third_parameter_name = 3, "eta_0"

    # ---------------------------------------------------------------- ROM build
    def build_rom(self):
        if self.model_family == "tanner":
            print("Tanner analytical forward model; no ROM is built.")
            return
        self.train_curve()
        if self.use_pressure:
            self.train_pressure(self.pressure_train_filename)

    # ---------------------------------------------------------------- parameter space
    def _infer_sigma_bias(self):
        return self.sigma_bias == "infer"

    def _infer_mean_bias(self):
        return getattr(self, "mean_bias", None) == "infer"

    def _infer_l_bias(self):
        return getattr(self, "l_bias", None) == "infer"

    def _get_parameter_bounds(self):
        if self.model_family == "tanner":
            if self.n1_bounds is not None:
                return [tuple(map(float, self.n1_bounds))]
            lo, hi = 2.0 * self._tanner_tau_w() * np.asarray(self.tanner_ratio_bounds)
            return [(float(lo), float(hi))]
        if not self.is_trained:
            raise ValueError("parameter bounds are not set; call build_rom() first")
        if self.model_family == "ratio":
            bounds = [self.sr_bounds]
        else:
            bounds = [self.lam_bounds, self.beta_bounds] + [self.third_parameter_bounds] * (self.n_material_params == 3)
        return [(float(lo), float(hi)) for lo, hi in bounds]

    def _get_ndim(self):
        return (self.n_material_params + 1 + self._infer_sigma_bias()
                + self._infer_l_bias() + self._infer_mean_bias())

    def _get_parameter_labels(self, latex=True):
        names = (self.material_parameter_names + ["sigma_noise"]
                 + ["sigma_bias"] * self._infer_sigma_bias()
                 + ["l_bias"] * self._infer_l_bias()
                 + ["c_bias"] * self._infer_mean_bias())
        return [LABELS[n] for n in names] if latex else names

    def _extract_noise_bias(self, theta):
        n = self.n_material_params
        return float(theta[n]), float(theta[n + 1]) if self._infer_sigma_bias() else None

    def _extract_l_bias(self, theta):
        """Discrepancy length scale: the inferred value, else the fixed ``l_bias`` float."""
        if not self._infer_l_bias():
            if self.l_bias == "fixed":
                raise RuntimeError("l_bias='fixed' is unresolved; call run_mcmc() (which fixes it "
                                   "from the discrepancy ACF via fix_l_bias_from_acf()) first.")
            return float(self.l_bias)
        return float(theta[self.n_material_params + 1 + self._infer_sigma_bias()])

    def fix_l_bias_from_acf(self, x_filename="curve4_x.txt", maxiter=300,
                            set_sigma_bias_prior=True, plot=True, save=True):
        """Set data-driven discrepancy priors from the model bias delta(x).

        MLE-fits the material parameters to the *clean* observed curve through the
        trained ROM (least squares: global differential-evolution + local polish) and
        takes the residual delta(x) = y_clean - y_model(theta_MLE) as the model bias.
        From that single delta it sets two things:

        * **l_bias** (only when ``l_bias='fixed'``) to delta's correlation length --
          obtained by fitting the *kernel's own* ACF form to the early lags of delta's
          sample autocorrelation (squared-exp exp(-t^2/2l^2), or Matern nu=1/2
          exp(-2t/l)), so the fitted value is directly the kernel length scale.
        * **sigma_bias prior** (when ``set_sigma_bias_prior`` and sigma_bias is
          inferred) to Exp(rate = 1/beta) with beta = sqrt(mean((delta - mean delta)^2)),
          the RMS of the mean-removed discrepancy -- i.e. an exponential whose mean is
          the empirical marginal SD of delta. Stored as ``sigma_bias_beta`` /
          ``sigma_bias_prior``, overwriting the displacement-based prior from load_data.

        Stores ``mle_theta`` and ``l_bias_est``; returns the resolved/estimated l_bias.
        With ``plot=True`` it draws the MLE fit, delta(x) with the +/- sigma_bias band,
        and delta's ACF with the fitted kernel decay and l_bias marked; ``save=True``
        writes that figure via ``_save_current_figure``.
        """
        from scipy.optimize import differential_evolution, minimize, curve_fit
        if not getattr(self, "is_trained", False):
            raise RuntimeError("Call build_rom() before fixing l_bias from the ACF.")
        if getattr(self, "y_obs_matrix_clean", None) is None:
            raise RuntimeError("Call load_data() before fixing l_bias from the ACF.")

        bounds = self._get_parameter_bounds()
        objective = lambda th: float(np.sum(np.asarray(self._residual(th)) ** 2))
        noisy, self.y_obs_matrix = self.y_obs_matrix, self.y_obs_matrix_clean  # MLE vs clean curve
        try:
            de = differential_evolution(objective, bounds, seed=getattr(self, "seed", 42),
                                        maxiter=maxiter, polish=False, tol=1e-12)
            pol = minimize(objective, de.x, method="L-BFGS-B", bounds=bounds)
            theta_mle = pol.x if pol.success and pol.fun <= de.fun else de.x
            delta = np.asarray(self._residual(theta_mle), float)
        finally:
            self.y_obs_matrix = noisy
        self.mle_theta = theta_mle

        # sigma_bias prior from the discrepancy: Exp(rate = 1/beta), beta = RMS of
        # mean-removed delta (the empirical marginal SD of the model bias).
        sb_beta = float(np.sqrt(np.mean((delta - delta.mean()) ** 2)))
        self.sigma_bias_beta = sb_beta
        set_sb = bool(set_sigma_bias_prior) and self._infer_sigma_bias() and sb_beta > 0
        if set_sb:
            self.sigma_bias_prior = 1.0 / sb_beta                  # Exp mean = beta

        x = np.asarray(self._observation_x(delta.size, x_filename), float)
        order = np.argsort(x)                                # curve4_x may be descending
        xs, ds = x[order], delta[order]
        xu = np.linspace(xs.min(), xs.max(), xs.size)        # uniform grid for the ACF
        du = np.interp(xu, xs, ds - ds.mean())
        acf = np.correlate(du, du, "full")[du.size - 1:]
        acf /= acf[0]
        dxu, xr = float(xu[1] - xu[0]), float(xs.max() - xs.min())
        lags = np.arange(acf.size) * dxu

        # e^-1/2 crossing: curve_fit initial guess and fallback if the fit fails
        below = np.where(acf <= np.exp(-0.5))[0]
        if below.size and below[0] > 0:
            i = below[0]
            l_est = float((i - 1 + (np.exp(-0.5) - acf[i - 1]) / (acf[i] - acf[i - 1])) * dxu)
        else:
            l_est = 0.1 * xr

        matern = getattr(self, "correlation_matrix", "squared_exp") == "matern"
        kf = ((lambda t, l: np.exp(-2.0 * t / l)) if matern
              else (lambda t, l: np.exp(-0.5 * (t / l) ** 2)))
        nfit = int(np.clip(round(3 * l_est / dxu), 5, acf.size))   # fit early lags only
        try:
            popt, _ = curve_fit(kf, lags[:nfit], acf[:nfit], p0=[max(l_est, dxu)], maxfev=10000)
            l_hat = float(popt[0])
        except (RuntimeError, ValueError):
            l_hat = l_est
        l_hat = float(np.clip(l_hat, dxu, xr))

        was_fixed = self.l_bias == "fixed"                         # capture before resolving
        self.l_bias_est = l_hat
        if was_fixed:                                              # resolve only the 'fixed' sentinel
            self.l_bias = l_hat
        theta_str = ", ".join(f"{n}={v:.6g}"
                              for n, v in zip(self.material_parameter_names, theta_mle))
        print(f"discrepancy from MLE fit ({self.model_family}): {theta_str}")
        if was_fixed:
            print(f"  l_bias: delta ACF correlation length "
                  f"({'Matern nu=1/2' if matern else 'squared-exp'} kernel) -> fixed to {l_hat:.4g}")
        else:
            print(f"  l_bias left as-is ({self.l_bias!r}); ACF estimate is {l_hat:.4g}")
        if set_sb:
            print(f"  sigma_bias prior: beta = RMS|delta-mean(delta)| = {sb_beta:.4g} "
                  f"-> Exp(rate={1.0 / sb_beta:.4g}, mean={sb_beta:.4g})")

        if plot:
            import matplotlib.pyplot as plt
            y_clean = np.asarray(self.y_obs_matrix_clean[0], float)[:delta.size][order]
            y_fit = y_clean - ds                                   # y_model(theta_MLE)
            nplot = int(np.clip(round(6 * l_hat / dxu), 10, acf.size))
            lvl = np.exp(-2.0) if matern else np.exp(-0.5)         # ACF value at r = l_bias
            kind = "Matern nu=1/2" if matern else "squared-exp"

            fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(24, 6))
            a0.plot(xs, y_clean, color="tab:blue", lw=2, label="observed (clean)")
            a0.plot(xs, y_fit, color="tab:green", lw=2, ls="--",
                    label=f"MLE fit ({self.model_family})")
            a0.set(xlabel=r"$x$", ylabel=r"$h(x)$", title="MLE fit")
            a0.grid(True, alpha=0.3); a0.legend(loc="best", fontsize=18)

            a1.axhline(0.0, color="black", lw=1, alpha=0.5)
            a1.axhline(ds.mean(), color="tab:blue", lw=1, ls=":", alpha=0.8, label=r"mean $\delta$")
            a1.axhspan(ds.mean() - sb_beta, ds.mean() + sb_beta, color="tab:orange", alpha=0.15,
                       label=fr"$\pm\sigma_{{\mathrm{{bias}}}}={sb_beta:.2g}$")
            a1.plot(xs, ds, "o-", color="tab:red", ms=4, label=r"$\delta(x)$")
            sb_title = (fr"model bias $\delta(x)$  $\to$  $\sigma_{{\mathrm{{bias}}}}$ prior "
                        fr"Exp(mean={sb_beta:.2g})") if set_sb else r"model bias $\delta(x)$"
            a1.set(xlabel=r"$x$", ylabel=r"$\delta(x)$", title=sb_title)
            a1.grid(True, alpha=0.3); a1.legend(loc="best", fontsize=16)

            a2.plot(lags[:nplot], acf[:nplot], "o-", color="k", ms=4, label="sample ACF")
            a2.plot(lags[:nplot], kf(lags[:nplot], l_hat), "b--", lw=2,
                    label=fr"fit: {kind}, $\hat\ell={l_hat:.3g}$")
            a2.axhline(lvl, color="grey", lw=1, alpha=0.6)
            a2.plot([l_hat], [lvl], "rs", ms=9)
            a2.annotate(fr"$\ell_{{\mathrm{{bias}}}}={l_hat:.3g}$", xy=(l_hat, lvl),
                        xytext=(l_hat * 1.1, lvl + 0.12), color="tab:red", fontsize=20,
                        arrowprops=dict(arrowstyle="->", color="tab:red"))
            a2.set(xlabel="lag", ylabel="ACF", title="discrepancy autocorrelation")
            a2.grid(True, alpha=0.3); a2.legend(loc="best", fontsize=18)

            fig.tight_layout()
            if save:
                try:
                    self._save_current_figure("l_bias_fixed")
                except Exception:
                    pass
            plt.show()
        return l_hat

    def _extract_mean_bias(self, theta):
        """Constant model-bias c (0.0 unless mean_bias='infer'); last hyperparameter."""
        if not self._infer_mean_bias():
            return 0.0
        n = self.n_material_params + 1 + self._infer_sigma_bias() + self._infer_l_bias()
        return float(theta[n])

    def _to_physical(self, phi):
        return np.array(phi, float)

    _to_physical_chain = _to_physical

    def _load_n1_function(self, module_name, func_name):
        """Import a compute_n1_* helper from the repo root or swell_root."""
        for root in (Path(__file__).resolve().parents[1], self.swell_root):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        return getattr(importlib.import_module(module_name), func_name)

    def posterior_material_means(self, use="mean"):
        """Inferred material parameters (posterior mean or MAP) as a name->value dict."""
        if self.samples is None:
            raise RuntimeError("no posterior samples yet; call run_mcmc() first")
        n = self.n_material_params
        if str(use).lower() == "map":
            vec = np.asarray(self.map_theta, float)[:n]
        else:
            vec = np.asarray(self.samples, float)[:, :n].mean(0)
        return dict(zip(self.material_parameter_names, map(float, vec)))

    def _observed_dp_dx(self):
        """Observed wall pressure gradient dp/dx, if any pressure obs was loaded."""
        for attr in ("pressure_drop_obs", "pressure_obs_clean", "pressure_obs"):
            v = getattr(self, attr, None)
            if v is not None:
                return float(np.atleast_1d(np.asarray(v, float)).ravel()[0])
        return None

    def compute_N1(self, U_avg=None, radius=None, use="mean", **rheo_kwargs):

        family = self.model_family
        means = self.posterior_material_means(use=use)
        u_avg = float(self.u_avg_obs if U_avg is None else U_avg)
        r = float(self.radius if radius is None else radius)

        if family == "ratio":
            sr = float(next(iter(means.values())))
            rate = 4.0 * u_avg / r
            out = {"model": "ratio", "theta_mean": means, "rates": np.atleast_1d(rate),
                   "Sr": np.atleast_1d(sr), "N1_over_tau_w": np.atleast_1d(2.0 * sr)}
            msg = f"S_R = N1/(2 tau_w) (inferred) = {sr:.6g}  at shear rate {rate:.6g}"
            dpdx = self._observed_dp_dx()
            if dpdx is not None:
                tau_w = -0.5 * r * dpdx
                out["tau_w"], out["N1"] = tau_w, np.atleast_1d(2.0 * tau_w * sr)
                msg += (f"\n  tau_w (from observed dp/dx={dpdx:.6g}) = {tau_w:.6g}"
                        f"  ->  N1 = {2.0 * tau_w * sr:.6g}   N1/tau_w = {2.0 * sr:.6g}")
            print(msg)
            return out

        # Constitutive families: run the model at the inferred means.
        if family not in _N1_FUNCS:
            raise ValueError(f"compute_N1() is not defined for model='{family}'.")
        eta0 = float(means.get("eta_0", self.eta0 if self.eta0 is not None else 1.0))
        common = dict(U_avg=u_avg, radius=r, lam=float(means["lambda"]),
                      beta=float(means["beta"]), eta0=eta0)
        module, func, extra = _N1_FUNCS[family]
        fn = self._load_n1_function(module, func)
        out = fn(**common, **{kw: float(means[k]) for k, kw in extra.items()}, **rheo_kwargs)

        out["model"], out["theta_mean"] = family, means
        n1 = np.atleast_1d(np.asarray(out["N1"], float))
        rates = np.atleast_1d(np.asarray(out["rates"], float))
        tau_w = np.atleast_1d(np.asarray(out.get("tau_xy", eta0 * rates), float))
        with np.errstate(divide="ignore", invalid="ignore"):
            out["tau_w"] = tau_w
            out["Sr"] = np.where(tau_w != 0, n1 / (2.0 * tau_w), np.inf)
        for i in range(n1.size):
            print(f"N1 ({family}) = {n1[i]:.6g}  tau_w = {tau_w[i]:.6g}  "
                  f"S_R = {out['Sr'][i]:.6g}  at rate {rates[i]:.6g}")
        return out
