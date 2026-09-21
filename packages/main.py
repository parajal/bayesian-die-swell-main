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
    constrained_model_error = False  # True -> both endpoint constraints below; False -> neither
    bias_anchor = None            # resolved anchor: 0.0 if constrained_model_error else None (off)
    constrained_gradient = False  # resolved: True -> delta'(x)=0 on bias_gradient_points
    bias_gradient_points = np.linspace(3.5, 5.0, 10)  # where delta'(x)=0 is imposed (constrained model error)

    def __init__(self, swell_root=None, train_data_rels=None,
                 filenames_train=("curve4_y.txt", "parameters.txt"),
                 scaler="minmax", eps=1e-6,
                 lambda_bounds=(1.0, 10.0), beta_bounds=(0.1, 0.9), alpha_bounds=(0.1, 0.5),
                 epsilon_bounds=None, eta0_bounds=None, sr_bounds=None,
                 n1_bounds=None, tanner_ratio_bounds=(0.001, 10.0),
                 true_theta=None, sigma_noise_percent=0.0, sigma_bias=None, sigma_bias_scale=0.1,
                 sigma_bias_from_discrepancy=True,
                 mean_bias=None,
                 l_bias=1.0, l_bias_prior="uniform", l_bias_bounds=(0.1, 1.0),
                 correlation_matrix="squared",
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
        # True -> set the sigma_bias Exp prior from the discrepancy RMS at run_mcmc
        # (also implied whenever l_bias='fixed', which runs the same routine).
        self.sigma_bias_from_discrepancy = bool(sigma_bias_from_discrepancy)

        # "infer" -> sampled; "fixed" -> fixed to the discrepancy ACF scale at run_mcmc; else a float
        self.l_bias = l_bias if l_bias in ("infer", "fixed") else float(l_bias)
        if l_bias_prior != "uniform":
            raise ValueError("l_bias_prior must be 'uniform' (l_bias uses a uniform prior).")
        self.l_bias_prior = "uniform"
        self.l_bias_bounds = tuple(map(float, l_bias_bounds))   # uniform prior: (lo, hi)
        # kernel names: 'squared' (squared-exp), 'absolute' (absolute exponential),
        # 'matern' (Matern nu=1/2); old names kept as aliases.
        _kernel_aliases = {"squared": "squared", "squared_exp": "squared",
                           "absolute": "absolute", "exponential": "absolute",
                           "matern": "matern"}
        _k = _kernel_aliases.get(str(correlation_matrix).lower())
        if _k is None:
            raise ValueError("correlation_matrix must be 'squared', 'absolute' or 'matern'.")
        self.correlation_matrix = _k
        # constrained_model_error=True pins the discrepancy at the die exit and flat over
        # the plateau: delta(0)=0 (anchor) and delta'(x)=0 on bias_gradient_points
        # (10 evenly-spaced points in [3.5, 5]). The derivative constraint
        # needs a mean-square-differentiable kernel, so it applies only to 'squared';
        # 'matern'/'absolute' (absolute exponential, not differentiable) keep only the
        # anchor. False leaves both off.
        self.constrained_model_error = bool(constrained_model_error)
        _diff_kernel = self.correlation_matrix == "squared"
        self.bias_anchor = 0.0 if self.constrained_model_error else None          # delta(0)=0
        self.constrained_gradient = self.constrained_model_error and _diff_kernel  # delta'(x)=0 on [3.5,5]
        if self.constrained_model_error and not _diff_kernel:
            print(f"note: correlation_matrix='{self.correlation_matrix}' is not mean-square "
                  "differentiable; keeping the delta(0)=0 anchor but disabling the delta'(5)=0 "
                  "constraint (use 'squared' if you need the derivative constraint).")
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

    def _mle_discrepancy(self, x_filename="curve4_x.txt", maxiter=300):
        """MLE-fit the material params to the *clean* observed curve and return
        ``(theta_mle, x, delta)`` with delta(x) = y_clean - y_model(theta_MLE) at the
        observation x (from ``x_filename``). Sets ``self.mle_theta``. The fit is the
        least-squares (global differential-evolution + local L-BFGS-B polish) through
        the trained ROM against the noise-free curve."""
        from scipy.optimize import differential_evolution, minimize
        if not getattr(self, "is_trained", False):
            raise RuntimeError("Call build_rom() before computing the model bias.")
        if getattr(self, "y_obs_matrix_clean", None) is None:
            raise RuntimeError("Call load_data() before computing the model bias.")

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
        x = np.asarray(self._observation_x(delta.size, x_filename), float)
        return theta_mle, x, delta

    def compute_model_bias(self, x_filename="curve4_x.txt", filename="model_bias.txt",
                           save=True, maxiter=300, plot=True):
        """Compute and save the model bias delta(x) = clean observed curve - MLE fit.

        delta(x) = y_true(x) - y_model(x; theta_MLE) is evaluated at the observation x
        read from ``x_filename`` (the same points as curve4_y.txt). With ``save=True``
        it writes a two-column text file ``x  delta`` (theta_MLE in the header) into the
        observation folder (``self.infer_dir``). With ``plot=True`` (default) it shows two
        panels: the MLE fit (observed vs best-fit curve) and delta(x). Returns ``(x, delta)``.
        """
        theta_mle, x, delta = self._mle_discrepancy(x_filename, maxiter)
        theta_str = ", ".join(f"{n}={v:.6g}"
                              for n, v in zip(self.material_parameter_names, theta_mle))
        print(f"model bias delta(x) = y_clean - y_MLE ({self.model_family}): {theta_str}")
        if save:
            out_path = Path(self.infer_dir) / filename
            header = (f"model={self.model_family}  theta_MLE: {theta_str}\n"
                      f"delta(x) = y_clean(x) - y_model(x; theta_MLE)   (x from {x_filename})\n"
                      "columns: x  delta")
            np.savetxt(out_path, np.column_stack([x, delta]), header=header)
            print(f"saved model bias to {out_path}")
        if plot:
            import matplotlib.pyplot as plt
            order = np.argsort(x)
            xs, ds = x[order], delta[order]
            y_clean = np.asarray(self.y_obs_matrix_clean[0], float)[:delta.size][order]
            y_mle = y_clean - ds                                   # y_model(theta_MLE)
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))
            a0.plot(xs, y_clean, color="tab:blue", lw=2, label="observed (clean)")
            a0.plot(xs, y_mle, color="tab:green", lw=2, ls="--", label=f"MLE fit ({self.model_family})")
            a0.set(xlabel=r"$x$", ylabel=r"$h(x)$", title="MLE fit")
            a0.grid(True, alpha=0.3); a0.legend(loc="best")
            a1.axhline(0.0, color="black", lw=1, alpha=0.5)
            a1.plot(xs, ds, "o-", color="tab:red", ms=4, label=r"$\delta(x)$")
            a1.set(xlabel=r"$x$", ylabel=r"$\delta(x)$", title=r"model bias $\delta(x)$")
            a1.grid(True, alpha=0.3); a1.legend(loc="best")
            fig.suptitle(f"{self.model_family} MLE: {theta_str}", fontsize=16)
            fig.tight_layout()
            try:
                self._save_current_figure("model_bias")
            except Exception:
                pass
            plt.show()
        return x, delta

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
          inferred) to an Exponential whose mean is scaled to the discrepancy:
          - ``sigma_bias_from_discrepancy=True``: mean = max|delta|, the peak model
            bias amplitude, so the prior tracks how large the bias really is instead of
            a fixed fraction of the signal (recommended -- the discrepancy is typically
            ~1% of the swell, an order of magnitude below 10% of the displacement).
          - otherwise (default): mean = ``sigma_bias_scale`` * max_displacement
            (10% of the displacement -- the same scale as the noise prior).
          ``sigma_bias_beta`` keeps delta's peak amplitude max|delta| (for the plot band)
          and ``sigma_bias_rms`` its RMS; ``sigma_bias_prior`` is the exponential rate = 1/mean.

        Stores ``mle_theta`` and ``l_bias_est``; returns the resolved/estimated l_bias.
        With ``plot=True`` it draws the MLE fit, delta(x) with the +/- sigma_bias band,
        and delta's ACF with the fitted kernel decay and l_bias marked; ``save=True``
        writes that figure via ``_save_current_figure``.
        """
        from scipy.optimize import curve_fit
        theta_mle, x, delta = self._mle_discrepancy(x_filename, maxiter)

        # sigma_bias prior: Exponential with mean set from the model bias delta(x).
        #   * default            -> mean = sigma_bias_scale * max_displacement (10% of disp)
        #   * from-discrepancy   -> mean = RMS(delta), i.e. the actual discrepancy amplitude
        #     (self.sigma_bias_from_discrepancy=True), so the prior tracks how big the bias
        #     really is instead of a fixed fraction of the signal.
        # sigma_bias_beta keeps the discrepancy peak amplitude max|delta| for the plot band.
        sb_beta = float(np.max(np.abs(delta)))
        sb_rms = float(np.sqrt(np.mean(np.asarray(delta, float) ** 2)))
        self.sigma_bias_beta = sb_beta
        self.sigma_bias_rms = sb_rms
        set_sb = bool(set_sigma_bias_prior) and self._infer_sigma_bias()
        if set_sb:
            if getattr(self, "sigma_bias_from_discrepancy", False):
                # scale to the discrepancy peak; floor away from 0 for a degenerate bias
                sb_mean = max(sb_beta, 1e-6 * float(self.max_displacement))
                self._sigma_bias_prior_source = "max|delta|"
            else:
                sb_mean = self._sigma_bias_frac * float(self.max_displacement)   # 0.1 * disp
                self._sigma_bias_prior_source = f"{self._sigma_bias_frac:g}*disp"
            self.sigma_bias_prior = 1.0 / sb_mean                 # Exp rate = 1/mean

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

        kern = getattr(self, "correlation_matrix", "squared")       # fit the kernel's own ACF form
        if kern == "matern":                                        # Matern nu=1/2: exp(-2t/l)
            kf = lambda t, l: np.exp(-2.0 * t / l)
            klabel, klvl = "Matern nu=1/2", np.exp(-2.0)
        elif kern == "absolute":                                    # absolute exponential: exp(-t/l)
            kf = lambda t, l: np.exp(-t / l)
            klabel, klvl = "absolute", np.exp(-1.0)
        else:                                                       # squared-exp: exp(-t^2/2l^2)
            kf = lambda t, l: np.exp(-0.5 * (t / l) ** 2)
            klabel, klvl = "squared-exp", np.exp(-0.5)
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
                  f"({klabel} kernel) -> fixed to {l_hat:.4g}")
        else:
            print(f"  l_bias left as-is ({self.l_bias!r}); ACF estimate is {l_hat:.4g}")
        if set_sb:
            sb_mean = 1.0 / self.sigma_bias_prior
            print(f"  sigma_bias prior: Exp(mean={sb_mean:.4g} = {self._sigma_bias_prior_source}, "
                  f"rate={self.sigma_bias_prior:.4g})   "
                  f"[discrepancy RMS={sb_rms:.4g}, max|delta|={sb_beta:.4g}]")

        if plot:
            import matplotlib.pyplot as plt
            y_clean = np.asarray(self.y_obs_matrix_clean[0], float)[:delta.size][order]
            y_fit = y_clean - ds                                   # y_model(theta_MLE)
            nplot = int(np.clip(round(6 * l_hat / dxu), 10, acf.size))
            lvl = klvl                                             # ACF value at r = l_bias
            kind = klabel

            fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(24, 6))
            a0.plot(xs, y_clean, color="tab:blue", lw=2, label="observed (clean)")
            a0.plot(xs, y_fit, color="tab:green", lw=2, ls="--",
                    label=f"MLE fit ({self.model_family})")
            a0.set(xlabel=r"$x$", ylabel=r"$h(x)$", title="MLE fit")
            a0.grid(True, alpha=0.3); a0.legend(loc="best", fontsize=18)

            a1.axhline(0.0, color="black", lw=1, alpha=0.5)
            a1.axhspan(-sb_beta, sb_beta, color="tab:orange", alpha=0.15,
                       label=fr"$\pm\max|\delta|={sb_beta:.2g}$")
            a1.plot(xs, ds, "o-", color="tab:red", ms=4, label=r"$\delta(x)$")
            sb_title = (fr"model bias $\delta(x)$  $\to$  $\sigma_{{\mathrm{{bias}}}}\sim$"
                        fr"Exp(mean$={1.0 / self.sigma_bias_prior:.2g}=${self._sigma_bias_prior_source})"
                        ) if set_sb else r"model bias $\delta(x)$"
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

    def compute_N1(self, U_avg=None, radius=None, use="mean", n_expectation=500,
                   analytic_oldroyd=True, **rheo_kwargs):
        """Compute and print BOTH N1 point estimates and the posterior 95% CI:

        * ``N1(theta_bar)`` -- N1 at the posterior-mean (``use="mean"``) or MAP
          (``use="map"``) parameters (the cheap plug-in);
        * ``E[N1]`` -- the posterior expectation E[N1] = mean of N1(theta) over the
          samples, with its SD and 95% credible interval. These differ because N1 is
          nonlinear in theta (Jensen gap); E[N1] is the recommended estimate.

        For Oldroyd-B the closed form N1 = 2(1-beta) eta0 lam gamma_dot^2 is used, so
        E[N1] is exact over all samples (fast). Other families run one rheology
        simulation per draw, averaged over a random ``n_expectation``-sized subset
        (set ``analytic_oldroyd=False`` to force the simulation path). Returns a dict
        with ``N1_at_mean``, ``N1_expectation``, ``N1_std``, ``N1_ci`` (and ``N1`` = E[N1]).
        """
        u_avg = float(self.u_avg_obs if U_avg is None else U_avg)
        r = float(self.radius if radius is None else radius)
        plug = self._point_N1(u_avg, r, use, **rheo_kwargs)              # prints N1(theta_bar)
        exp = self._expected_N1(u_avg, r, int(n_expectation),
                                analytic_oldroyd, **rheo_kwargs)         # prints E[N1] + CI
        pn1 = plug.get("N1")
        n1_mean = float(np.atleast_1d(pn1)[0]) if pn1 is not None else float("nan")
        lo, hi = exp["N1_ci"]
        print(f"  => N1(mean theta) = {n1_mean:.6g}   |   E[N1] = {exp['N1']:.6g} "
              f"+/- {exp['N1_std']:.6g}   95% CI [{lo:.6g}, {hi:.6g}]")
        return {"model": self.model_family, "use": use, "N1_at_mean": n1_mean,
                "N1_expectation": exp["N1"], "N1_std": exp["N1_std"], "N1_ci": exp["N1_ci"],
                "N1_samples": exp["N1_samples"], "point": plug, "expectation": exp,
                "N1": exp["N1"]}

    def _point_N1(self, u_avg, r, use="mean", **rheo_kwargs):
        """N1 at the posterior-mean (or MAP) parameters -- the plug-in N1(theta_bar)."""
        family = self.model_family
        means = self.posterior_material_means(use=use)

        if family == "ratio":
            sr = float(next(iter(means.values())))
            rate = 4.0 * u_avg / r
            out = {"model": "ratio", "theta_mean": means, "rates": np.atleast_1d(rate),
                   "Sr": np.atleast_1d(sr), "N1_over_tau_w": np.atleast_1d(2.0 * sr)}
            msg = f"N1(mean theta): S_R = N1/(2 tau_w) (inferred) = {sr:.6g}  at shear rate {rate:.6g}"
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
            print(f"N1(mean theta) ({family}) = {n1[i]:.6g}  tau_w = {tau_w[i]:.6g}  "
                  f"S_R = {out['Sr'][i]:.6g}  at rate {rates[i]:.6g}")
        return out

    def _expected_N1(self, u_avg, r, n_samples=500, analytic_oldroyd=True, **rheo_kwargs):
        """Posterior expectation E[N1] = (1/S) sum_s N1(theta_s). For Oldroyd-B uses the
        closed form over ALL samples; otherwise averages a random ``n_samples`` subset.
        Returns the mean, SD, 95% CI and the per-sample N1 array."""
        if self.samples is None:
            raise RuntimeError("no posterior samples yet; call run_mcmc() first")
        family = self.model_family
        S = np.asarray(self.samples, float)
        n = self.n_material_params
        names = self.material_parameter_names
        rate = 4.0 * u_avg / r

        if family == "oldroyd" and analytic_oldroyd and not rheo_kwargs:
            # Oldroyd-B closed form over ALL samples (exact): N1 = 2(1-beta) eta0 lam gdot^2
            lam = S[:, names.index("lambda")]
            beta = S[:, names.index("beta")]
            eta0 = (S[:, names.index("eta_0")] if "eta_0" in names
                    else float(self.eta0 if self.eta0 is not None else 1.0))
            n1s = 2.0 * (1.0 - beta) * eta0 * lam * rate ** 2
            tws = np.broadcast_to(eta0 * rate, n1s.shape).astype(float)
            rates = np.full(n1s.shape, rate)
            k = int(n1s.size)
        elif family == "ratio":                                  # N1 = 2 tau_w S_R (linear in S_R)
            k = min(int(n_samples), S.shape[0])
            draws = S[np.random.default_rng(0).choice(S.shape[0], size=k, replace=False), :n]
            sr = draws[:, 0]
            dpdx = self._observed_dp_dx()
            if dpdx is None:
                raise RuntimeError("E[N1] for model='ratio' needs an observed pressure drop.")
            tau_w = -0.5 * r * dpdx
            n1s = 2.0 * tau_w * sr
            tws = np.full(k, tau_w)
            rates = np.full(k, rate)
        else:
            if family not in _N1_FUNCS:
                raise ValueError(f"compute_N1() is not defined for model='{family}'.")
            k = min(int(n_samples), S.shape[0])
            draws = S[np.random.default_rng(0).choice(S.shape[0], size=k, replace=False), :n]
            module, func, extra = _N1_FUNCS[family]
            fn = self._load_n1_function(module, func)
            n1s, tws, rates = (np.empty(k) for _ in range(3))
            for j in range(k):
                means = dict(zip(names, map(float, draws[j])))
                eta0 = float(means.get("eta_0", self.eta0 if self.eta0 is not None else 1.0))
                common = dict(U_avg=u_avg, radius=r, lam=means["lambda"],
                              beta=means["beta"], eta0=eta0)
                o = fn(**common, **{kw: float(means[key]) for key, kw in extra.items()},
                       **rheo_kwargs)
                rr = float(np.atleast_1d(np.asarray(o["rates"], float))[0])
                n1s[j] = float(np.atleast_1d(np.asarray(o["N1"], float))[0])
                tws[j] = float(np.atleast_1d(np.asarray(o.get("tau_xy", eta0 * rr), float))[0])
                rates[j] = rr

        En1, sd = float(n1s.mean()), float(n1s.std(ddof=1))
        ci = np.percentile(n1s, [2.5, 97.5])
        with np.errstate(divide="ignore", invalid="ignore"):
            sr_s = np.where(tws != 0, n1s / (2.0 * tws), np.nan)
        print(f"E[N1] ({family}) = {En1:.6g} +/- {sd:.6g}  "
              f"(95% CI [{ci[0]:.6g}, {ci[1]:.6g}], n={k})  "
              f"E[S_R] = {np.nanmean(sr_s):.6g}  E[tau_w] = {float(np.mean(tws)):.6g}")
        return {"model": family, "N1": En1, "N1_mean": En1, "N1_std": sd,
                "N1_ci": ci, "N1_samples": n1s, "Sr": float(np.nanmean(sr_s)),
                "tau_w": float(np.mean(tws)), "rates": float(np.mean(rates)), "n": k}
