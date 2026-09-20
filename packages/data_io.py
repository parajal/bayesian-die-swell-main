from pathlib import Path
import numpy as np


class DataLoaderMixin:
    """Load observed swell curves, add synthetic noise to the curve, set sigma priors."""

    def load_data(self, filename, u_avg_obs, x_max=5.0):
        """Keep x <= x_max for observations, plots and subsequent ROM training.

        Set x_max=None to retain the full curve. Training snapshots must use
        the same spatial columns as the observation file.
        """
        path = Path(self.swell_root, filename).resolve()
        self.infer_dir = folder = path.parent
        self.u_avg_obs, self.observed_uavgs = u_avg_obs, [u_avg_obs]

        y = np.loadtxt(path, ndmin=2)
        x_full = np.loadtxt(folder / "curve4_x.txt").ravel()
        if x_full.size != y.shape[1] or not np.isfinite(x_full).all():
            raise ValueError("curve4_x.txt must contain one finite coordinate per curve column")
        x_max = None if x_max is None else float(x_max)
        if x_max is not None and not np.isfinite(x_max):
            raise ValueError("x_max must be finite or None")
        curve_idx = np.flatnonzero(x_full <= x_max) if x_max is not None else np.arange(x_full.size)
        if curve_idx.size == 0:
            raise ValueError("no curve points remain at or below x_max")
        if self.thin < 1:
            raise ValueError("thin must be at least 1")

        # A trained ROM must be rebuilt if the retained spatial columns change.
        if (getattr(self, "is_trained", False) and self.model_family != "tanner"
                and (getattr(self, "_training_curve_size", None) != x_full.size
                     or not np.array_equal(getattr(self, "_training_curve_indices", None), curve_idx))):
            self.is_trained = False
        self.x_max, self._curve_size, self._curve_indices = x_max, x_full.size, curve_idx
        # Prediction indices refer to the cropped ROM, not the original files.
        self._obs_indices = idx = np.arange(0, curve_idx.size, self.thin)
        selected = curve_idx[idx]
        self.y_obs_matrix_clean = y = y[:, selected]
        self.obs_x_coords = x = x_full[selected]

        disp = y.max() - 1.0
        if not (np.isfinite(disp) and disp > 0):
            raise ValueError("no swelling in observed curve (max height <= die radius)")
        self.max_displacement = disp
        self.beta = self.sigma_noise_prior = rate = 1.0 / (0.10 * disp)
        if self._infer_sigma_bias() and self.sigma_bias_pc is not None:
            # PC elicitation (Fuglstad et al. 2019): P(sigma_bias > sigma0) = alpha,
            # sigma0 a fraction of the displacement. rate = -ln(alpha) / sigma0.
            frac, alpha = self.sigma_bias_pc
            self._sigma0 = frac * disp
            self.sigma_bias_prior = -np.log(alpha) / self._sigma0
        elif self._infer_sigma_bias():
            self.sigma_bias_prior = 1.0 / (self._sigma_bias_frac * disp)
        else:
            self.sigma_bias_prior = None
        self.c_bias_prior_sd = 0.10 * disp if self._infer_mean_bias() else None
        if self._infer_l_bias() and self.l_bias_prior == "pc":
            # PC prior (Fuglstad/Simpson) on the length scale: penalizes short (rough)
            # scales toward the smooth base model, set by P(l < l0) = alpha with l0 a
            # fraction of the x-range. 1-D range: lambda = -ln(alpha) * l0**0.5.
            frac, alpha = self.l_bias_pc
            self._l0 = frac * float(x.max() - x.min())
            self.l_bias_pc_lambda = -np.log(alpha) * self._l0 ** 0.5

        self.sigma_noise_target = self.sigma_noise_percent / 100 * disp
        noise = np.random.default_rng(0).normal(0, self.sigma_noise_target, y.shape)
        self.sigma_noise_realized = noise.std()
        self.y_obs_matrix = y + noise

        info = {
            "folder": folder,
            "curve file": path.name,
            "U_avg": u_avg_obs,
            "points": f"{idx.size} (thin={self.thin})",
            "x range": f"[{x.min():.4g}, {x.max():.4g}]",
            "height range": f"[{y.min():.4g}, {y.max():.4g}]",
            "max displacement": f"{disp:.4g}",
            "sigma_noise prior": f"Exp(rate={rate:.4g}), mean {1 / rate:.4g}",
            "sigma_bias prior": (
                f"PC: P(sigma>{self._sigma0:.4g})={self.sigma_bias_pc[1]:g}, rate={self.sigma_bias_prior:.4g}"
                if self.sigma_bias_prior and self.sigma_bias_pc is not None else
                f"Exp(mean={self._sigma_bias_frac * disp:.4g})" if self.sigma_bias_prior else "not inferred"),
            "c_bias prior": f"Normal(0, {self.c_bias_prior_sd:.4g})" if self.c_bias_prior_sd else "not inferred",
            "l_bias prior": (
                f"Uniform({self.l_bias_bounds[0]:g}, {self.l_bias_bounds[1]:g})"
                if self._infer_l_bias() and self.l_bias_prior == "uniform" else
                f"PC: P(l<{self._l0:.4g})={self.l_bias_pc[1]:g}, lambda={self.l_bias_pc_lambda:.4g}"
                if self._infer_l_bias() else
                "fixed (from discrepancy ACF, set at run_mcmc)" if self.l_bias == "fixed"
                else f"fixed {self.l_bias:g}"),
            "bias kernel": self.correlation_matrix,
            "bias anchor": f"delta({self.bias_anchor:g})=0" if self.bias_anchor is not None else "none",
            "bias derivative": "delta'(5)=0" if self.constrained_gradient else "none",
            "curve noise": f"{self.sigma_noise_percent}% of disp, sigma {self.sigma_noise_target:.4g} "
                           f"(realized {self.sigma_noise_realized:.4g})",
        }

        if self.use_pressure:
            self.pressure_obs = self.pressure_obs_clean = np.loadtxt(folder / "pressure_drop.txt").ravel()
            self.pressure_drop_obs = float(self.pressure_obs[0])
            info["pressure"] = f"{np.array2string(self.pressure_obs, precision=4)} (no noise)"

        width = max(map(len, info))
        print("\n".join(f"{k:<{width}} : {v}" for k, v in info.items()))
