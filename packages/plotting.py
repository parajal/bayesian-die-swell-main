"""Plotting helpers for ROM Bayesian inference: data, prior, trace, corner, posterior predictive."""

from pathlib import Path

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
    """Plots use physical coordinates; figures are saved to ``infer_dir``."""

    def _obs_matrix(self) -> np.ndarray:
        if getattr(self, "y_obs_matrix", None) is None:
            raise RuntimeError("Call load_data() first.")
        return np.atleast_2d(self.y_obs_matrix)

    def _posterior_samples(self) -> np.ndarray:
        if getattr(self, "samples", None) is None:
            raise RuntimeError("Call run_mcmc() first.")
        return np.asarray(self.samples, dtype=float)

    def _save_current_figure(self, filename: str) -> None:
        """Save as ``<infer_dir>/<filename>_noise_<pct>.pdf``."""
        out = Path(self.infer_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(out / f"{filename}_noise_{self.sigma_noise_percent:g}.pdf", dpi=300)

    def _finish_curve(self, ax, y, name, handles=None, labels=None) -> None:
        """Shared h(x) styling (data-driven x, y from 1, 3 ticks each), legend, save, show."""
        x = np.asarray(self.obs_x_coords, float)
        xlo, xhi = float(x.min()), float(x.max())
        ytop = 1.01 * float(np.max(y))
        ax.set(xlabel=r"$x$", ylabel=r"$h(x)$",
               xlim=(xlo, xhi), xticks=np.linspace(xlo, xhi, 3),
               ylim=(1, ytop), yticks=np.linspace(1, ytop, 3))
        ax.grid(True, alpha=0.3)
        ax.legend(*(() if handles is None else (handles, labels)), loc="lower right", framealpha=0.9)
        self._save_current_figure(name)
        plt.show()

    def plot_data(self) -> None:
        """Noisy observations (points) over the noise-free curve (line)."""
        y = self._obs_matrix()
        x = np.asarray(self.obs_x_coords, float)
        fig, ax = plt.subplots(figsize=(8, 6))
        clean = getattr(self, "y_obs_matrix_clean", None)
        if clean is not None:
            for i, row in enumerate(np.atleast_2d(clean)):
                ax.plot(x, row, color=f"C{i}", lw=2, alpha=0.9,
                        label="noise-free" if i == 0 else "_nolegend_")
        for i, row in enumerate(y):
            ax.scatter(x, row, s=16, alpha=0.8, color=f"C{i}",
                       label="noisy data" if i == 0 else "_nolegend_")
        self._finish_curve(ax, y, "data")

    def _prior_curves(self, n: int = 5000) -> "list[tuple[np.ndarray, np.ndarray]]":
        """(x, density) for every inferred parameter, in vector order, in the sampler's coordinates.

        material -> uniform (log10 theta1, theta2 with parametrize); sigma_noise/sigma_bias -> exponential.
        """
        curves = []
        for lo, hi in self._get_sampling_bounds():                  # material params
            x = np.linspace(lo, hi, n)
            curves.append((x, np.full_like(x, 1.0 / (hi - lo))))
        rates = [self.sigma_noise_prior] + [self.sigma_bias_prior] * self._infer_sigma_bias()
        for rate in rates:                                          # sigma_noise (+ sigma_bias): exponential
            x = np.linspace(0, 5.0 / rate, n)
            curves.append((x, rate * np.exp(-rate * x)))
        return curves

    def plot_prior(self, n: int = 5000) -> None:
        curves = self._prior_curves(n)
        fig, axes = plt.subplots(1, len(curves), figsize=(8 * len(curves), 6), squeeze=False)
        axes = axes.ravel()
        for ax, (x, dens), label in zip(axes, curves, self._get_sampling_labels()):
            ax.plot(x, dens, lw=2)
            ax.set(xlabel=label, ylim=(0, 1.2 * float(np.max(dens))))
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Prior density")
        self._save_current_figure("prior")
        plt.show()

    def plot_trace_all(self, burn_in_ratio: float = 0.6) -> None:
        if getattr(self, "chain", None) is None:
            raise RuntimeError("Call run_mcmc() first.")
        chain = self._to_physical_chain(np.asarray(self.chain, dtype=float))
        n = chain.shape[2]
        labels = list(self._get_parameter_labels())
        labels += [f"param_{i}" for i in range(len(labels), n)]
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(9, 2.4 * n), squeeze=False)
        for i, ax in enumerate(axes.ravel()):
            ax.plot(chain[:, :, i], lw=0.7, alpha=0.55)
            ax.axvline(int(burn_in_ratio * chain.shape[0]), color="black", ls="--", lw=1)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.2)
        axes.ravel()[-1].set_xlabel("MCMC step")
        self._save_current_figure("trace_all")
        plt.show()

    def plot_corner(
        self,
        theta_true=None,
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
                if np.isfinite(vals[i]) and vals[i] > 0:
                    truths[i] = float(vals[i])
        if true_param and not physical_only:
            names = self._get_parameter_labels(latex=False)[:ndim]
            sig = getattr(self, "sigma_noise_realized", None)
            if sig is not None and "sigma_noise" in names:
                truths[names.index("sigma_noise")] = float(sig)
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
        q_lo, q_hi = 50 * (1 - cred_level), 50 * (1 + cred_level)
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
            if i < n_phys:                                   # prior density in physical units
                lo, hi = bounds[i]
                xs = np.linspace(xlo, xhi, 400)
                inside = (xs >= lo) & (xs <= hi)
                dens = np.zeros_like(xs)
                if i in self._log10_params():                # log-uniform: 1 / (x ln(hi/lo))
                    dens[inside] = 1.0 / (xs[inside] * np.log(hi / lo))
                else:
                    dens[inside] = 1.0 / (hi - lo)
                ax.plot(xs, dens, color="red", lw=1.5)
            for c in np.percentile(samples[:, i], [q_lo, q_hi]):
                ax.axvline(c, color="black", ls=":", lw=1.5)
            if truths[i] is not None:
                ax.axvline(truths[i], color="blue", ls=":", lw=2, label="ground truth")
                ax.legend(loc="best")
            ax.set_ylim(ylim)

        self._save_current_figure("corner_physical" if physical_only else "corner_physical_all")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_corner_all(self, **kwargs) -> None:
        """Corner plot including the noise/bias hyperparameters."""
        kwargs.setdefault("physical_only", False)
        self.plot_corner(**kwargs)

    def _bias_correlation_matrix(self, x, l_bias) -> np.ndarray:
        """Squared-exponential discrepancy correlation c(r) = exp(-r^2 / 2 l^2).

        With ``constrained_model_error=True`` it is GP-conditioned (Brynjarsdottir & O'Hagan
        2014; derivatives of a GP are jointly Gaussian) on delta(0)=0 at the die exit and
        delta'(x)=0 at ``bias_gradient_points`` (flat discrepancy in the plateau):
        ``K' = K - C A^{-1} C^T`` (PSD; zero variance along the constraints), with ``C``/``A``
        the value/derivative cross- and auto-covariances of the kernel.

        With ``orthogonality_constraint=True`` it is further conditioned on G^T delta = 0, with
        G the model sensitivities dy/dtheta at the no-bias best fit (Plumlee 2017; see
        ``_orthogonal_directions``): ``K'' = K' - K'G (G^T K'G)^{-1} G^T K'``.

        Also used by the likelihood. K' depends only on the grid, l_bias and the constraints,
        so the last result is cached (read-only) and reused while those are unchanged.
        """
        x = np.asarray(x, dtype=float).ravel()
        constrained = bool(getattr(self, "constrained_model_error", False))
        Dc = np.asarray(self.bias_gradient_points, float) if constrained else np.empty(0)
        orth = bool(getattr(self, "orthogonality_constraint", False))
        G = self._orthogonal_directions() if orth else np.empty((x.size, 0))
        if G.shape[0] != x.size:
            raise ValueError("orthogonality_constraint needs the full observation grid (mode='full_curve').")
        key = (float(l_bias), constrained, tuple(Dc), x.tobytes(), G.tobytes())
        cached = getattr(self, "_bias_corr_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        l2 = float(l_bias) ** 2
        kf = lambda a, b: np.exp(-np.subtract.outer(a, b) ** 2 / (2.0 * l2))
        K = kf(x, x)
        if constrained:
            V = np.zeros(1)                                                   # delta(0)=0
            C = np.hstack([kf(x, V),                                          # Cov(d(x), d(0))
                           np.subtract.outer(x, Dc) / l2 * kf(x, Dc)])        # Cov(d(x), d'(Dc))
            g = np.subtract.outer(V, Dc) / l2 * kf(V, Dc)                     # Cov(d(0), d'(Dc))
            dd = np.subtract.outer(Dc, Dc)
            A = np.block([[kf(V, V), g],
                          [g.T, kf(Dc, Dc) / l2 * (1.0 - dd ** 2 / l2)]])     # Cov(d'(Dc), d'(Dc))
            A[np.diag_indices_from(A)] += 1e-10
            K = K - C @ np.linalg.solve(A, C.T)
        if orth:                                         # G^T delta = 0 (Plumlee 2017)
            KG = K @ G
            B = G.T @ KG
            B[np.diag_indices_from(B)] += 1e-10
            K = K - KG @ np.linalg.solve(B, KG.T)

        K.setflags(write=False)
        self._bias_corr_cache = (key, K)
        return K

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

    @staticmethod
    def _band_and_diag(obs, latent_mean, Y_rep, n_sigma):
        """Central band (mean, lo, hi) and coverage diagnostics from replicates."""
        from scipy.stats import norm
        tail = 100 * float(norm.cdf(n_sigma))
        mean_pred = latent_mean.mean(0)
        pred_lo = np.percentile(Y_rep, 100 - tail, axis=0)
        pred_hi = np.percentile(Y_rep, tail, axis=0)
        zres = (obs - mean_pred) / np.maximum(np.std(Y_rep, axis=0), 1e-8)
        diag = dict(
            coverage=float(np.mean((obs >= pred_lo) & (obs <= pred_hi))),
            rms_z=float(np.sqrt(np.mean(zres ** 2))),
            max_abs_z=float(np.max(np.abs(zres))),
            mean_z=float(np.mean(zres)),
            nominal_coverage=float(2 * norm.cdf(n_sigma) - 1),
        )
        return mean_pred, pred_lo, pred_hi, diag

    def _predictive_summary(self, x, obs, n_sigma, nsamples_pred, condition_discrepancy):
        """Posterior-predictive replicates g(theta) + delta + noise over random posterior draws."""
        samples = self._posterior_samples()
        n_material = len(self._get_parameter_bounds())         # sigma_noise column index
        rng = np.random.default_rng(0)
        n_draws = min(int(nsamples_pred), samples.shape[0])
        draws = samples[rng.choice(samples.shape[0], size=n_draws, replace=False)]

        infer_bias = self._infer_sigma_bias()
        sn_draws = np.abs(draws[:, n_material])
        sb_draws = np.abs(draws[:, n_material + 1]) if infer_bias else np.zeros(n_draws)
        corr = self._bias_correlation_matrix(x, self.l_bias) if infer_bias else None

        curves = self.predict(draws[:, :n_material])
        if curves.shape[1] != obs.size:
            curves = curves[:, self._obs_indices]

        latent_mean = np.empty((n_draws, obs.size), dtype=float)
        Y_rep = np.empty((n_draws, obs.size), dtype=float)
        for k in range(n_draws):
            g = curves[k]
            sn, sb = float(sn_draws[k]), float(sb_draws[k])
            if not infer_bias or sb <= 0:
                delta = delta_mean = np.zeros_like(g)
            elif condition_discrepancy:
                A = sb * sb * corr
                Sigma = 0.5 * (A + A.T) + sn * sn * np.eye(len(g))
                Sinv = np.linalg.pinv(Sigma, hermitian=True)
                delta_mean = A @ Sinv @ (obs - g)
                delta = delta_mean + self._correlated_normal(rng, A - A @ Sinv @ A, 1)
            else:
                delta = self._correlated_normal(rng, corr, sb)
                delta_mean = np.zeros_like(g)
            latent_mean[k] = g + delta_mean
            Y_rep[k] = g + delta + rng.standard_normal(len(g)) * sn

        return self._band_and_diag(obs, latent_mean, Y_rep, n_sigma)

    def plot_posterior_predictive(
        self,
        n_sigma: float = 1.96,
        nsamples_pred: int = 5000,
        condition_discrepancy: bool = False,
    ) -> None:
        from matplotlib.lines import Line2D

        obs = self._obs_matrix()[0]
        x = np.asarray(self.obs_x_coords, float)
        mean_pred, pred_lo, pred_hi, diag = self._predictive_summary(
            x, obs, n_sigma, nsamples_pred, condition_discrepancy
        )
        self.pp_diagnostics = diag
        print(f"\nPosterior predictive ({round(100 * diag['nominal_coverage'])}% band)")
        print(f"  coverage={diag['coverage']:.1%}  rms_z={diag['rms_z']:.2f}  "
              f"max|z|={diag['max_abs_z']:.2f}  mean_z={diag['mean_z']:.2f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.fill_between(x, pred_lo, pred_hi, color="steelblue", alpha=0.25)
        ax.plot(x, mean_pred, color="steelblue", lw=1.5, zorder=4)
        ax.scatter(x, obs, color="black", s=12, zorder=5, alpha=0.8,
                   edgecolors="black", linewidths=0.5, label="_nolegend_")
        band = Line2D([0], [0], color="steelblue", lw=6, alpha=0.25)
        self._finish_curve(ax, np.concatenate([obs, pred_hi]), "posterior_predictive",
                           [band], ["Posterior predictive"])
