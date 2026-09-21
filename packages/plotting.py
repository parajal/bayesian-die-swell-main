"""Minimal plotting helpers for ROM Bayesian inference."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
    "lines.linewidth": 2,
    "axes.linewidth": 1,
})


class PlottingMixin:
    """Plots use physical coordinates."""

    CURVE_FIGSIZE = (8, 6)
    CURVE_GRID_ALPHA = 0.3
    CURVE_LEGEND_LOC = "lower right"

    def _curve_figure(self):
        return plt.subplots(figsize=self.CURVE_FIGSIZE)

    @staticmethod
    def _set_curve_labels(ax) -> None:
        ax.set(xlabel=r"$x$", ylabel=r"$h(x)$")

    def _curve_grid(self, ax) -> None:
        ax.grid(True, alpha=self.CURVE_GRID_ALPHA)

    def _curve_legend(self, ax, handles=None, labels=None, loc: Optional[str] = None) -> None:
        loc = self.CURVE_LEGEND_LOC if loc is None else loc
        if handles is None:
            ax.legend(loc=loc, framealpha=0.9)
            return
        ax.legend(handles, labels, loc=loc, framealpha=0.9)

    def _labels(self, latex: bool = True) -> "list[str]":
        labels = list(self._get_parameter_labels(latex=latex))
        samples = getattr(self, "samples", None)
        ndim = np.asarray(samples).shape[-1] if samples is not None else len(labels)
        labels.extend(f"param_{i}" for i in range(len(labels), ndim))
        return labels[:ndim]

    def _observation_x(self, n_pts: int, x_filename: str = "curve4_x.txt") -> np.ndarray:
        obs_x = getattr(self, "obs_x_coords", None)
        if obs_x is None:
            raise RuntimeError("obs_x_coords is unset. Call load_data() with curve4_x.txt present.")
        obs_x = np.asarray(obs_x, dtype=float).ravel()
        if obs_x.shape != (n_pts,) or not np.all(np.isfinite(obs_x)):
            raise ValueError("obs_x_coords must be a finite 1D array matching observed curves.")
        return obs_x

    def _posterior_samples(self) -> np.ndarray:
        if getattr(self, "samples", None) is None:
            raise RuntimeError("Call run_mcmc() first.")
        return np.asarray(self.samples, dtype=float)

    def _plot_output_dir(self) -> Path:
        obs_file = getattr(self, "_obs_filename", None)
        if obs_file:
            folder = f"{Path(obs_file).stem}_{self._noise_label()}"
            out = self.swell_root / "plots" / folder
        else:
            out = Path(self.infer_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _noise_label(self) -> str:
        value = getattr(self, "sigma_noise_percent", None)
        if value is None:
            return "noise_unknown"
        try:
            text = f"{float(value):g}"
        except (TypeError, ValueError):
            text = str(value)
        return "noise_" + text.replace(" ", "_").replace("/", "_").replace("\\", "_")

    def _save_current_figure(self, filename: str) -> None:
        noise_label = self._noise_label()
        stem = filename if filename.endswith(f"_{noise_label}") else f"{filename}_{noise_label}"
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(self._plot_output_dir() / f"{stem}.pdf", dpi=300)

    def _axis_style(self, ax, y_value) -> None:
        """Data-driven x limits and uniform y limits (3 ticks each) for h(x) plots."""
        x = np.asarray(self.obs_x_coords, float)
        xlo, xhi = float(x.min()), float(x.max())
        ymax = float(np.max(y_value))
        ytop = ymax + 0.01 * ymax
        ax.set(
            xlim=(xlo, xhi),
            xticks=np.linspace(xlo, xhi, 3),
            ylim=(1, ytop),
            yticks=np.linspace(1, ytop, 3),
        )

    def _material_theta(self, lambda_val, beta_val, alpha_val=None, epsilon_val=None):
        """Fill (lambda, beta, third) from ``true_theta`` when lambda/beta are unset."""
        if lambda_val is None or beta_val is None:
            theta = getattr(self, "true_theta", None)
            if theta is None:
                raise ValueError("Pass lambda_val and beta_val, or set true_theta on the model.")
            lambda_val, beta_val = float(theta[0]), float(theta[1])
            if alpha_val is None and epsilon_val is None and len(theta) > 2:
                if getattr(self, "third_parameter_name", None) == "epsilon":
                    epsilon_val = float(theta[2])
                else:
                    alpha_val = float(theta[2])
        return lambda_val, beta_val, alpha_val, epsilon_val

    def _select_draws(self, nsamples_pred):
        """Random posterior draws for posterior-predictive replication.

        Returns ``(draws, n_material, rng)``; the sigma_noise column is ``n_material``.
        """
        samples = self._posterior_samples()
        n_material = len(self._get_parameter_bounds())
        rng = np.random.default_rng(0)
        n_draws = min(int(nsamples_pred), samples.shape[0])
        sel = rng.choice(samples.shape[0], size=n_draws, replace=False)
        return samples[sel], n_material, rng

    @staticmethod
    def _band_and_diag(obs, latent_mean, Y_rep, n_sigma):
        """Central band (mean, lo, hi) and coverage diagnostics from replicates."""
        from scipy.stats import norm
        tail = 100 * float(norm.cdf(n_sigma))
        mean_pred = latent_mean.mean(0)
        pred_lo = np.percentile(Y_rep, 100 - tail, axis=0)
        pred_hi = np.percentile(Y_rep, tail, axis=0)
        sigma_total = np.maximum(np.std(Y_rep, axis=0), 1e-8)
        zres = (obs - mean_pred) / sigma_total
        diag = dict(
            coverage=float(np.mean((obs >= pred_lo) & (obs <= pred_hi))),
            rms_z=float(np.sqrt(np.mean(zres ** 2))),
            max_abs_z=float(np.max(np.abs(zres))),
            mean_z=float(np.mean(zres)),
            nominal_coverage=float(2 * norm.cdf(n_sigma) - 1),
        )
        return mean_pred, pred_lo, pred_hi, diag

    @staticmethod
    def _print_pp(name, diag, u_avg=None):
        """Print posterior-predictive coverage diagnostics."""
        pct = int(round(100 * diag["nominal_coverage"]))
        u = "" if u_avg is None else f"U={u_avg:g}, "
        print(f"\n{name} ({u}{pct}% band)")
        print(f"  coverage={diag['coverage']:.1%}  rms_z={diag['rms_z']:.2f}  "
              f"max|z|={diag['max_abs_z']:.2f}  mean_z={diag['mean_z']:.2f}")

    def plot_data(self, x_filename: str = "curve4_x.txt") -> None:
        if getattr(self, "y_obs_matrix", None) is None:
            raise RuntimeError("Call load_data() first.")
        y = np.atleast_2d(self.y_obs_matrix)
        uavgs = None if self.observed_uavgs is None else list(self.observed_uavgs)
        if uavgs is not None and len(uavgs) != y.shape[0]:
            raise ValueError(
                f"Loaded {y.shape[0]} observed curves but only {len(uavgs)} U_avg values. "
                "Use a single-row observation file for one U_avg."
            )
        x = self._observation_x(y.shape[1], x_filename)
        clean = getattr(self, "y_obs_matrix_clean", None)
        clean = None if clean is None else np.atleast_2d(clean)
        fig, ax = self._curve_figure()
        for i, row in enumerate(y):
            u = uavgs[i] if uavgs is not None else self.u_avg_obs
            color = f"C{i}"
            if clean is not None and i < clean.shape[0]:
                ax.plot(x, clean[i], color=color, lw=2, alpha=0.9, label="_nolegend_")
            ax.scatter(x, row, s=16, alpha=0.8, color=color, label=f"U_avg = {u:g}")
        self._set_curve_labels(ax)
        self._axis_style(ax, y)
        self._curve_grid(ax)
        self._curve_legend(ax)
        self._save_current_figure("data")
        plt.show()

    def plot_rom_prediction(
        self,
        lambda_val: Optional[float] = None,
        beta_val: Optional[float] = None,
        tanner_ratio_val: Optional[float] = None,
        n1_val: Optional[float] = None,
        alpha_val: Optional[float] = None,
        epsilon_val: Optional[float] = None,
        u_avg_val: Optional[float] = None,
        x_filename: str = "curve4_x.txt",
    ) -> None:
        """Compare ROM prediction with the loaded noisy FOM curve4."""
        if not getattr(self, "is_trained", False):
            raise RuntimeError("Call train() before plot_rom_prediction().")
        if getattr(self, "y_obs_matrix", None) is None:
            raise RuntimeError("Call load_data() before plot_rom_prediction().")

        if getattr(self, "model_family", None) == "tanner":
            theta = getattr(self, "true_theta", None)
            u_avg = float(u_avg_val if u_avg_val is not None else self.u_avg_obs)
            tau_w = 4*u_avg
            if n1_val is None:
                if tanner_ratio_val is not None:
                    n1_val = 2.0 * tau_w * float(tanner_ratio_val)
                elif theta is not None:
                    n1_val = float(theta[0])
                elif getattr(self, "map_theta", None) is not None:
                    n1_val = float(self.map_theta[0])
                else:
                    raise RuntimeError(
                        "Tanner plotting needs n1_val, tanner_ratio_val, true_theta, "
                        "or posterior MAP samples."
                    )
            y_fom = np.atleast_2d(self.y_obs_matrix)[0]
            x = self._observation_x(len(y_fom), x_filename)
            n1 = float(n1_val)
            ratio = n1 / (2.0 * tau_w)
            pred_height = self._tanner_height(n1, u_avg_val=u_avg)
            obs_height = float(np.max(y_fom))
            i_max = int(np.argmax(y_fom))
            fig, ax = self._curve_figure()
            ax.plot(x, y_fom, color="tab:blue", lw=2, label="FOM + noise")
            ax.scatter([x[i_max]], [obs_height], s=36, color="black", zorder=5, label="observed max")
            ax.axhline(pred_height, color="tab:red", lw=2, label="Tanner")
            self._set_curve_labels(ax)
            self._axis_style(ax, np.append(y_fom, pred_height))
            self._curve_grid(ax)
            self._curve_legend(ax)
            self._save_current_figure("tanner_prediction")
            plt.show()
            print(f"N1: {n1:.6e}")
            print(f"tau_w from U_avg={u_avg:g}: {tau_w:.6e}")
            print(f"N1_over_2tau_w: {ratio:.6e}")
            print(f"Tanner predicted swell height: {pred_height:.6e}")
            print(f"Observed max curve4: {obs_height:.6e}")
            print(f"Residual: {obs_height - pred_height:.6e}")
            return

        lambda_val, beta_val, alpha_val, epsilon_val = self._material_theta(
            lambda_val, beta_val, alpha_val, epsilon_val)

        u_avg = float(u_avg_val if u_avg_val is not None else self.u_avg_obs)
        y_fom = np.atleast_2d(self.y_obs_matrix)[0]
        has_third = self.n_material_params == 3
        y_rom = self.predict(
            float(lambda_val),
            float(beta_val),
            u_avg_val=u_avg,
            alpha_val=(alpha_val if has_third else None),
            epsilon_val=(epsilon_val if has_third else None),
        )
        idx = getattr(self, "_obs_indices", None)
        if idx is not None and y_rom.shape != y_fom.shape:
            idx = np.asarray(idx, dtype=int)
            if y_rom.size >= int(idx[-1]) + 1:
                y_rom = y_rom[idx]
        x = self._observation_x(len(y_fom), x_filename)
        mean_rel_err = np.mean(np.abs(y_rom - y_fom) / (np.abs(y_fom) + 1e-14))
        rel_l2_curve = float(np.linalg.norm(y_rom - y_fom) / (np.linalg.norm(y_fom) + 1e-14))
        fig, ax = self._curve_figure()
        ax.scatter(x, y_fom, s=24, alpha=0.9, color="tab:blue", label="FOM + noise")
        ax.scatter(x, y_rom, s=24, alpha=0.9, color="tab:red", label="ROM")
        self._set_curve_labels(ax)
        self._axis_style(ax, np.concatenate([y_fom, y_rom]))
        self._curve_grid(ax)
        self._curve_legend(ax)
        self._save_current_figure("rom_prediction")
        plt.show()
        print(f"ROM prediction mean relative error against noisy curve4: {mean_rel_err:.6e}")
        print(f"ROM prediction relative L2 error against noisy curve4:    {rel_l2_curve:.6e}")
        if getattr(self, "use_pressure", False) and getattr(self, "pressure_obs", None) is not None:
            self.plot_pressure_prediction(
                lambda_val=float(lambda_val),
                beta_val=float(beta_val),
                alpha_val=alpha_val,
                epsilon_val=epsilon_val,
            )

    def plot_pressure_prediction(
        self,
        lambda_val: Optional[float] = None,
        beta_val: Optional[float] = None,
        alpha_val: Optional[float] = None,
        epsilon_val: Optional[float] = None,
    ) -> float:

        if not getattr(self, "use_pressure", False):
            raise RuntimeError("plot_pressure_prediction() requires use_pressure=True.")
        if getattr(self, "pressure_model", None) is None:
            raise RuntimeError("Call build_rom() before plot_pressure_prediction().")
        p_obs = getattr(self, "pressure_obs", None)
        if p_obs is None:
            raise RuntimeError(
                "Call load_data(..., pressure_filename=...) before plot_pressure_prediction()."
            )
        p_obs = np.asarray(p_obs, dtype=float).ravel()

        lambda_val, beta_val, alpha_val, epsilon_val = self._material_theta(
            lambda_val, beta_val, alpha_val, epsilon_val)

        theta = [float(lambda_val), float(beta_val)]
        if self.n_material_params == 3:
            third = epsilon_val if epsilon_val is not None else alpha_val
            if third is None:
                raise ValueError("The third material parameter is required for this model.")
            theta.append(float(third))
        p_rom = np.asarray(self.predict_pressure(theta), dtype=float).ravel()

        rel_l2 = float(np.linalg.norm(p_rom - p_obs) / (np.linalg.norm(p_obs) + 1e-14))
        mean_rel = float(np.mean(np.abs(p_rom - p_obs) / (np.abs(p_obs) + 1e-14)))
        idx = np.arange(1, p_obs.size + 1)
        fig, ax = self._curve_figure()
        ax.plot(idx, p_obs, "o-", color="tab:blue", ms=10, label="FOM + noise")
        ax.plot(idx, p_rom, "s--", color="tab:red", ms=10, label="GPR")
        ax.set(xlabel=r"pressure point", ylabel=r"$p$")
        ax.set_xticks(idx)
        self._curve_grid(ax)
        self._curve_legend(ax, loc="best")
        self._save_current_figure("pressure_prediction")
        plt.show()
        print(f"Pressure GPR mean relative error against observed pressure: {mean_rel:.6e}")
        print(f"Pressure GPR relative L2 error against observed pressure:    {rel_l2:.6e}")
        return rel_l2

    def _prior_curves(self, n: int = 5000) -> "list[tuple[np.ndarray, np.ndarray]]":
        """(x, density) for every inferred parameter, in vector order.

        material -> uniform; sigma_noise/sigma_bias -> exponential;
        l_bias -> uniform; c_bias -> Normal(0, sd).
        """
        curves = []
        for lo, hi in self._get_parameter_bounds():                 # material params
            x = np.linspace(lo, hi, n)
            curves.append((x, np.full_like(x, 1.0 / (hi - lo))))
        rates = [self.sigma_noise_prior] + [self.sigma_bias_prior] * self._infer_sigma_bias()
        for rate in rates:                                          # sigma_noise (+ sigma_bias): exponential
            x = np.linspace(0, 5.0 / rate, n)
            curves.append((x, rate * np.exp(-rate * x)))
        if self._infer_l_bias():                                     # l_bias: Uniform(lo, hi)
            lo_l, hi_l = self.l_bias_bounds
            x = np.linspace(lo_l, hi_l, n)
            curves.append((x, np.full_like(x, 1.0 / (hi_l - lo_l))))
        if self._infer_mean_bias():                                # c_bias: Normal(0, sd)
            sd = self.c_bias_prior_sd
            x = np.linspace(-4 * sd, 4 * sd, n)
            curves.append((x, np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi))))
        return curves

    def _prior_pdf_list(self):
        """Vectorized prior pdf callables in posterior-column order: material (uniform),
        sigma_noise (+ sigma_bias) exponential, l_bias uniform, c_bias Normal(0, sd).
        Aligned with ``_get_parameter_labels`` / the sample columns, one per inferred dim."""
        def uniform(lo, hi):
            return lambda x: np.where((x >= lo) & (x <= hi), 1.0 / (hi - lo), 0.0)
        def expon(rate):
            return lambda x: np.where(x >= 0, rate * np.exp(-rate * np.clip(x, 0.0, None)), 0.0)
        pdfs = [uniform(float(lo), float(hi)) for lo, hi in self._get_parameter_bounds()]
        pdfs.append(expon(float(self.sigma_noise_prior)))              # sigma_noise
        if self._infer_sigma_bias():
            pdfs.append(expon(float(self.sigma_bias_prior)))           # sigma_bias
        if self._infer_l_bias():                                       # l_bias: Uniform(lo, hi)
            pdfs.append(uniform(*map(float, self.l_bias_bounds)))
        if self._infer_mean_bias():                                    # c_bias: Normal(0, sd)
            sd = float(self.c_bias_prior_sd)
            pdfs.append(lambda x: np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi)))
        return pdfs

    def plot_prior(self, n: int = 5000) -> None:
        labels = self._get_parameter_labels()
        curves = self._prior_curves(n)
        m = len(curves)
        fig, axes = plt.subplots(1, m, figsize=(8 * m, 6), squeeze=False)
        axes = axes.ravel()
        for ax, (x, dens), label in zip(axes, curves, labels):
            ax.plot(x, dens, lw=2)
            ax.set(xlabel=label, ylim=(0, 1.2 * float(np.max(dens))))
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Prior density")
        self._save_current_figure("prior")
        plt.show()

    def plot_bias_realizations(self, n_real: int = 20, l_bias=None, sigma: float = 1.0,
                               n_grid: int = 400, x_max: float = 5.0, band: float = 2.0,
                               save: bool = True):
        """Draw and plot ``n_real`` realizations of the discrepancy GP prior delta(x).

        Uses the same ``_bias_correlation_matrix`` as the likelihood, so when
        ``constrained_model_error=True`` the constraints delta(0)=0 and delta'(x)=0 on
        ``self.bias_gradient_points`` are enforced exactly. ``sigma`` is the marginal SD
        of delta (sigma_bias); ``l_bias`` defaults to the model's length scale (the fixed
        value, else ``l_bias_est``, else 1.0). Shaded band is +/- ``band``*sigma(x)."""
        if l_bias is None:
            l_bias = (float(self.l_bias) if isinstance(self.l_bias, (int, float))
                      else float(getattr(self, "l_bias_est", 0.5) or 0.5))
        obs = getattr(self, "obs_x_coords", None)
        hi = float(np.max(obs)) if obs is not None else float(x_max)
        xg = np.linspace(0.0, hi, n_grid)
        K = float(sigma) ** 2 * self._bias_correlation_matrix(xg, l_bias)
        K = 0.5 * (K + K.T)
        w, V = np.linalg.eigh(K)
        L = V * np.sqrt(np.clip(w, 0.0, None))
        rng = np.random.default_rng(getattr(self, "seed", 42))
        samples = rng.standard_normal((int(n_real), len(xg))) @ L.T
        std = np.sqrt(np.clip(np.diag(K), 0.0, None))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.fill_between(xg, -band * std, band * std, color="0.85", lw=0,
                        label=fr"$\pm{band:g}\sigma(x)$")
        for s in samples:
            ax.plot(xg, s, color="tab:blue", lw=0.8, alpha=0.5)
        ax.axhline(0.0, color="0.4", lw=0.6)
        constrained = getattr(self, "constrained_gradient", False)
        if getattr(self, "bias_anchor", None) is not None:
            ax.plot(float(self.bias_anchor), 0.0, "ko", ms=6, label=r"$\delta(0)=0$")
        if constrained:
            dp = np.asarray(self.bias_gradient_points, float)
            ax.axvspan(dp[0], dp[-1], color="tab:red", alpha=0.10, lw=0, label=r"$\delta'(x)=0$")
        title = ("constrained" if (constrained or self.bias_anchor is not None) else "prior")
        ax.set(xlabel=r"$x$", ylabel=r"$\delta(x)$", xlim=(0.0, hi),
               title=fr"{int(n_real)} draws of the {title} discrepancy GP "
                     fr"($\ell_{{bias}}={l_bias:.3g}$, $\sigma_{{bias}}={sigma:g}$)")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        if save:
            try:
                self._save_current_figure("bias_realizations")
            except Exception:
                pass
        plt.show()
        return xg, samples

    def plot_trace_all(self, burn_in_ratio: float = 0.6) -> None:
        if getattr(self, "chain", None) is None:
            raise RuntimeError("Call run_mcmc() first.")
        chain = self._to_physical_chain(np.asarray(self.chain, dtype=float))
        burn = int(burn_in_ratio * chain.shape[0])
        labels = self._labels()[:chain.shape[2]]
        fig, axes = plt.subplots(
            chain.shape[2], 1, sharex=True, figsize=(9, 2.4 * chain.shape[2]), squeeze=False
        )
        for i, ax in enumerate(axes.ravel()):
            ax.plot(chain[:, :, i], lw=0.7, alpha=0.55)
            ax.axvline(burn, color="black", ls="--", lw=1)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.2)
        axes.ravel()[-1].set_xlabel("MCMC step")
        self._save_current_figure("trace_all")
        plt.show()

    def plot_corner(
        self,
        theta_true=None,
        log_scale: bool = False,
        cred_level: float = 0.95,
        label_size: float = 40,
        physical_only: bool = True,
        show_plot: bool = True,
        true_param: bool = True,
    ) -> None:

        try:
            import corner
        except ImportError as exc:
            raise ImportError("corner is required for plot_corner().") from exc

        samples_full = self._posterior_samples()
        bounds = self._get_parameter_bounds()
        n_phys = len(bounds)
        ndim = n_phys if physical_only else samples_full.shape[1]
        samples = samples_full[:, :ndim].astype(float).copy()
        labels = list(self._get_parameter_labels()[:ndim])
        if true_param:
            theta_true = theta_true if theta_true is not None else getattr(self, "true_theta", None)
        else:
            theta_true = None

        truths = [None] * ndim
        if theta_true is not None:
            vals = np.atleast_1d(np.asarray(theta_true, dtype=float))
            for i in range(min(n_phys, len(vals))):
                if not np.isfinite(vals[i]):
                    continue
                if not vals[i] > 0:
                    continue
                truths[i] = float(vals[i])
        if true_param and not physical_only:
            names = self._get_parameter_labels(latex=False)[:ndim]

            def _set_truth(pname, value):
                if value is not None and pname in names:
                    truths[names.index(pname)] = float(value)

            curve_sig = getattr(self, "sigma_noise_realized", None)
            _set_truth("sigma_noise", curve_sig)
        fig = plt.figure(figsize=(8 * ndim, 8 * ndim))
        corner.corner(
            samples,
            fig=fig,
            labels=labels,
            label_kwargs=(dict(fontsize=label_size) if label_size else None),
            levels=(0.68, 0.95),
            color="black",
            hist_kwargs=dict(histtype="step", linewidth=2, density=True, color="black"),
            data_kwargs=dict(ms=1.5, alpha=0.2, color="gray"),
        )
        q_lo = 50 * (1 - cred_level)
        q_hi = 50 * (1 + cred_level)
        pdfs = self._prior_pdf_list()                    # prior for every inferred parameter
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ylim = ax.get_ylim()
            xlo, xhi = ax.get_xlim()
            if truths[i] is not None:
                pad = 0.05 * (xhi - xlo)
                xlo = min(xlo, truths[i] - pad)
                xhi = max(xhi, truths[i] + pad)
                for j in range(i, ndim):
                    axes[j, i].set_xlim(xlo, xhi)
            if i < len(pdfs):                            # overlay prior on all diagonals
                xs = np.linspace(xlo, xhi, 400)
                ax.plot(xs, pdfs[i](xs), color="red", lw=1.5, label="prior")
            ci = np.percentile(samples[:, i], [q_lo, q_hi])
            ax.axvline(ci[0], color="black", ls=":", lw=1.5)
            ax.axvline(ci[1], color="black", ls=":", lw=1.5)
            if truths[i] is not None:
                ax.axvline(truths[i], color="blue", ls=":", lw=2, label="ground truth")
                ax.legend(loc="best")
            ax.set_ylim(ylim)

        base = "corner" if log_scale else "corner_physical"
        self._save_current_figure(base if physical_only else base + "_all")
        if show_plot:
            plt.show()
            return
        plt.close(fig)

    def plot_corner_all(self, **kwargs) -> None:
        """Corner plot including the noise/bias hyperparameters."""
        kwargs.setdefault("physical_only", False)
        self.plot_corner(**kwargs)

    def _bias_correlation_matrix(self, x, l_bias) -> np.ndarray:
        """Discrepancy correlation (squared-exponential, Matern nu=1/2, or absolute
        exponential, per ``self.correlation_matrix``), optionally GP-conditioned on
        linear constraints (Brynjarsdottir & O'Hagan 2014; derivatives of a GP are
        jointly Gaussian):

        * ``bias_anchor=True`` -> delta(0)=0     (die-exit value constraint)
        * ``constrained_gradient=True`` -> delta'(x)=0 at ``self.bias_gradient_points``
          (10 evenly-spaced points on [3.5, 5] by default); squared-exponential only,
          since the absolute-exponential kernels ('matern' nu=1/2, 'absolute') are not
          mean-square differentiable.

        Returns ``K' = K - C A^{-1} C^T`` (PSD; zero variance along the constraints),
        with ``C``/``A`` the value/derivative cross- and auto-covariances of the
        kernel. With no constraints it is the plain correlation matrix.
        """
        x = np.asarray(x, dtype=float).ravel()
        l2 = float(l_bias) ** 2
        kind = getattr(self, "correlation_matrix", "squared")
        if kind == "matern":
            # Matern, nu=1/2: c(r) = exp(-sqrt(8*0.5) r / l) = exp(-2r/l) (Fuglstad et al. 2019,
            # Definition 2.3), the "range" convention.
            kf = lambda a, b: np.exp(-2.0 * np.abs(np.subtract.outer(a, b)) / l_bias)
        elif kind == "absolute":
            # Absolute exponential: c(r) = exp(-|r|/l)  (correlation = e^-1 at r = l).
            kf = lambda a, b: np.exp(-np.abs(np.subtract.outer(a, b)) / l_bias)
        else:
            kf = lambda a, b: np.exp(-np.subtract.outer(a, b) ** 2 / (2.0 * l2))
        K = kf(x, x)

        Vc, Dc = [], []
        if getattr(self, "bias_anchor", None) is not None:
            Vc = [float(self.bias_anchor)]                                   # delta(x0)=0
        if getattr(self, "constrained_gradient", False):
            if kind != "squared":
                raise RuntimeError("constrained_gradient requires correlation_matrix='squared' "
                                   "(the derivative covariances used here are squared-exponential).")
            Dc = np.asarray(getattr(self, "bias_gradient_points",
                                    np.linspace(3.5, 5.0, 10)), float)  # delta'(x)=0 on [3.5,5]
        if not Vc and not Dc:
            return K
        Vc, Dc = np.asarray(Vc, float), np.asarray(Dc, float)

        C_blocks = []
        if Vc.size:
            C_blocks.append(kf(x, Vc))                                       # Cov(d(x),  d(Vc))
        if Dc.size:
            C_blocks.append(np.subtract.outer(x, Dc) / l2 * kf(x, Dc))       # Cov(d(x),  d'(Dc))
        C = np.hstack(C_blocks)

        nv, nd = Vc.size, Dc.size
        A = np.zeros((nv + nd, nv + nd))
        if nv:
            A[:nv, :nv] = kf(Vc, Vc)
        if nd:
            dd = np.subtract.outer(Dc, Dc)
            A[nv:, nv:] = kf(Dc, Dc) / l2 * (1.0 - dd ** 2 / l2)             # Cov(d'(Dc), d'(Dc))
        if nv and nd:
            gVD = np.subtract.outer(Vc, Dc) / l2 * kf(Vc, Dc)               # Cov(d(Vc), d'(Dc))
            A[:nv, nv:], A[nv:, :nv] = gVD, gVD.T
        A[np.diag_indices_from(A)] += 1e-10
        # Least-squares (pseudo-inverse) solve: robust if A is near-singular; symmetrize
        # to remove floating-point asymmetry.
        sol = np.linalg.lstsq(A, C.T, rcond=None)[0]
        Kc = K - C @ sol
        return 0.5 * (Kc + Kc.T)

    @staticmethod
    def _correlated_normal(rng, corr, sigma) -> np.ndarray:
        """Draw a zero-mean sample with covariance ``sigma**2 * corr``."""
        n = corr.shape[0]
        if sigma <= 0:
            return np.zeros(n)
        try:
            L = np.linalg.cholesky(corr + 1e-12 * np.eye(n))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(0.5 * (corr + corr.T))
            L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
        return sigma * (L @ rng.standard_normal(n))

    def _predictive_summary(self, u_avg, x, obs, n_sigma, nsamples_pred, condition_discrepancy):

        draws, n_material, rng = self._select_draws(nsamples_pred)
        n_draws, i_sn = len(draws), n_material
        infer_bias = self._infer_sigma_bias()
        infer_l = self._infer_l_bias()
        sn_draws = np.abs(draws[:, i_sn])
        sb_draws = np.abs(draws[:, i_sn + 1]) if infer_bias else np.zeros(n_draws)
        i_l = i_sn + 1 + int(infer_bias)          # l_bias column (if inferred)
        l_draws = draws[:, i_l] if infer_l else None
        c_draws = (draws[:, i_l + int(infer_l)] if self._infer_mean_bias()
                   else np.zeros(n_draws))
        corr = (self._bias_correlation_matrix(x, float(self.l_bias))
                if infer_bias and not infer_l else None)
        obs_idx = getattr(self, "_obs_indices", None)
        if obs_idx is None:
            raise RuntimeError("_obs_indices is unset. Call load_data() first.")
        obs_idx = np.asarray(obs_idx, dtype=int)

        curves = self._predict_curves(draws[:, :n_material], u_avg_val=u_avg)
        if curves.shape[1] != obs.size and curves.shape[1] >= int(obs_idx[-1]) + 1:
            curves = curves[:, obs_idx]

        latent_mean = np.empty((n_draws, obs.size), dtype=float)
        Y_rep = np.empty((n_draws, obs.size), dtype=float)
        for k in range(n_draws):
            g = curves[k]
            sn, sb = float(sn_draws[k]), float(sb_draws[k])
            corr_k = (self._bias_correlation_matrix(x, float(l_draws[k]))
                      if infer_bias and infer_l else corr)
            if not infer_bias or sb <= 0:
                delta = delta_mean = np.zeros_like(g)
            elif condition_discrepancy:
                A = sb * sb * corr_k
                Sigma = 0.5 * (A + A.T) + sn * sn * np.eye(len(g))
                Sinv = np.linalg.pinv(Sigma, hermitian=True)
                delta_mean = A @ Sinv @ (obs - g)
                delta = delta_mean + self._correlated_normal(rng, A - A @ Sinv @ A, 1)
            else:
                delta = self._correlated_normal(rng, corr_k, sb)
                delta_mean = np.zeros_like(g)
            latent_mean[k] = g + c_draws[k] + delta_mean
            Y_rep[k] = g + c_draws[k] + delta + rng.standard_normal(len(g)) * sn

        return self._band_and_diag(obs, latent_mean, Y_rep, n_sigma)

    def plot_posterior_predictive(
        self,
        n_sigma: float = 1.96,
        nsamples_pred: int = 5000,
        u_avg_val: Optional[float] = None,
        x_filename: str = "curve4_x.txt",
        condition_discrepancy: bool = False,
    ) -> None:

        from matplotlib.lines import Line2D

        if getattr(self, "y_obs_matrix", None) is None:
            raise RuntimeError("Call load_data() first.")
        obs = np.atleast_2d(self.y_obs_matrix)[0]
        x = self._observation_x(obs.size, x_filename)
        u_avg = float(u_avg_val if u_avg_val is not None else self.u_avg_obs)

        if getattr(self, "model_family", None) == "tanner":
            import math

            draws, n_material, rng = self._select_draws(nsamples_pred)
            n_draws, i_sn = len(draws), n_material
            heights = np.asarray(
                [self._tanner_height(float(s[0]), u_avg_val=u_avg) for s in draws],
                dtype=float,
            )
            sn = np.abs(draws[:, i_sn])
            if self._infer_sigma_bias():
                sn = np.sqrt(sn ** 2 + np.abs(draws[:, i_sn + 1]) ** 2)
            y_rep = heights + rng.standard_normal(n_draws) * sn
            tail = 50 * (1 + math.erf(float(n_sigma) / math.sqrt(2)))
            pred_lo = float(np.percentile(y_rep, 100 - tail))
            pred_hi = float(np.percentile(y_rep, tail))
            mean_pred = float(np.mean(heights))
            sigma_total = max(float(np.std(y_rep, ddof=1)), 1e-8)
            obs_height = float(np.max(obs))
            zres = (obs_height - mean_pred) / sigma_total
            diag = dict(
                coverage=float(pred_lo <= obs_height <= pred_hi),
                rms_z=float(abs(zres)),
                max_abs_z=float(abs(zres)),
                mean_z=float(zres),
                nominal_coverage=float(2 * tail / 100 - 1),
            )
            self.pp_diagnostics = diag
            i_max = int(np.argmax(obs))
            fig, ax = self._curve_figure()
            ax.plot(x, obs, color="black", lw=1.8, alpha=0.8, label="_nolegend_")
            ax.scatter([x[i_max]], [obs_height], color="black", s=34, zorder=5, label="observed max")
            ax.axhspan(pred_lo, pred_hi, color="steelblue", alpha=0.25)
            ax.axhline(mean_pred, color="steelblue", lw=1.8)
            self._set_curve_labels(ax)
            self._axis_style(ax, np.append(obs, pred_hi))
            self._curve_grid(ax)
            band = Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="Tanner posterior predictive")
            h, lbl = ax.get_legend_handles_labels()
            self._curve_legend(ax, h + [band], lbl + ["Tanner posterior predictive"])
            self._print_pp("Tanner posterior predictive", diag, u_avg)
            self._save_current_figure("posterior_predictive")
            plt.show()
            return

        mean_pred, pred_lo, pred_hi, diag = self._predictive_summary(
            u_avg, x, obs, n_sigma, nsamples_pred, condition_discrepancy
        )
        self.pp_diagnostics = diag
        fig, ax = self._curve_figure()
        ax.fill_between(x, pred_lo, pred_hi, color="steelblue", alpha=0.25)
        ax.plot(x, mean_pred, color="steelblue", lw=1.5, zorder=4)
        ax.scatter(x, obs, color="black", s=12, zorder=5, alpha=0.8,
                   edgecolors="black", linewidths=0.5, label="_nolegend_")
        self._set_curve_labels(ax)
        self._axis_style(ax, np.concatenate([obs, pred_hi]))
        self._curve_grid(ax)
        band = Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="Posterior predictive")
        h, lbl = ax.get_legend_handles_labels()
        self._curve_legend(ax, h + [band], lbl + ["Posterior predictive"])
        self._print_pp("Posterior predictive", diag, u_avg)
        self._save_current_figure("posterior_predictive")
        plt.show()
        if getattr(self, "use_pressure", False) and getattr(self, "pressure_obs", None) is not None:
            self.plot_pressure_posterior_predictive(n_sigma=n_sigma, nsamples_pred=nsamples_pred)

    def _pressure_predictive_summary(self, n_sigma, nsamples_pred):

        p_obs = np.asarray(getattr(self, "pressure_obs", None), dtype=float).ravel()
        if p_obs.size == 0 or not np.all(np.isfinite(p_obs)):
            raise RuntimeError(
                "Observed pressure is missing; call load_data(..., pressure_filename=...)."
            )
        draws, n_material, rng = self._select_draws(nsamples_pred)
        sn_draws = np.abs(draws[:, n_material])

        # One GPR evaluation for all draws; i.i.d. Gaussian noise added per draw.
        latent_mean = self.predict_pressure(draws[:, :n_material])
        Y_rep = latent_mean + rng.standard_normal(latent_mean.shape) * sn_draws[:, None]
        return (p_obs, *self._band_and_diag(p_obs, latent_mean, Y_rep, n_sigma))

    def plot_pressure_posterior_predictive(
        self,
        n_sigma: float = 1.96,
        nsamples_pred: int = 5000,
    ) -> None:

        from matplotlib.lines import Line2D

        if not getattr(self, "use_pressure", False):
            raise RuntimeError("plot_pressure_posterior_predictive() requires use_pressure=True.")
        if getattr(self, "pressure_model", None) is None:
            raise RuntimeError("Call build_rom() before plot_pressure_posterior_predictive().")

        p_obs, mean_pred, pred_lo, pred_hi, diag = self._pressure_predictive_summary(
            n_sigma, nsamples_pred
        )
        self.pressure_pp_diagnostics = diag
        idx = np.arange(1, p_obs.size + 1)
        fig, ax = self._curve_figure()
        ax.fill_between(idx, pred_lo, pred_hi, color="steelblue", alpha=0.25)
        ax.plot(idx, mean_pred, color="steelblue", lw=1.5, zorder=4)
        ax.scatter(idx, p_obs, color="black", s=34, zorder=5, alpha=0.8,
                   edgecolors="black", linewidths=0.5, label="_nolegend_")
        ax.set(xlabel=r"pressure point", ylabel=r"$p$")
        ax.set_xticks(idx)
        self._curve_grid(ax)
        band = Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="Posterior predictive")
        h, lbl = ax.get_legend_handles_labels()
        self._curve_legend(ax, h + [band], lbl + ["Posterior predictive"], loc="best")
        self._print_pp("Pressure posterior predictive", diag)
        self._save_current_figure("pressure_posterior_predictive")
        plt.show()
