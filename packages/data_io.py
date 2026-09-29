from pathlib import Path
import numpy as np

class DataLoaderMixin:
    """Load the training data and the observed swell curve (with synthetic noise), set sigma priors."""

    def load_training_data(self):
        Y = np.loadtxt(self.data_dir / self.filenames_train[0], ndmin=2)
        P = np.loadtxt(self.data_dir / self.filenames_train[1], ndmin=2)[:, :self.n_material_params]
        if self.parametrize:                   
            P[:, 0] *= 1.0 - P[:, 1]
        return Y, P

    def load_data(self, filename):
        path = Path(self.swell_root, filename).resolve()
        self.infer_dir = path.parent

        y = np.loadtxt(path, ndmin=2)
        x_full = np.loadtxt(self.infer_dir / "curve4_x.txt")
        self._curve_size = x_full.size

        self._obs_indices = np.arange(0, x_full.size, self.thin)
        self.y_obs_matrix_clean = y[:, self._obs_indices]
        self.obs_x_coords = x_full[self._obs_indices]

        self.max_displacement = self.y_obs_matrix_clean.max() - 1.0
        self.sigma_noise_prior = 1.0 / (0.10 * self.max_displacement)
        self.sigma_bias_prior = self.sigma_noise_prior if self._infer_sigma_bias() else None

        sigma = (self.sigma_noise_percent / 100) * self.max_displacement
        noise = np.random.default_rng(self.seed).normal(0, sigma, self.y_obs_matrix_clean.shape)
        self.sigma_noise_realized = noise.std()
        self.y_obs_matrix = self.y_obs_matrix_clean + noise

        info = {
            "folder": self.infer_dir,
            "points": f"{self._obs_indices.size} (thin={self.thin})",
            "curve noise": f"{self.sigma_noise_percent}% of disp, sigma {sigma:.4g} "
                           f"(realized {self.sigma_noise_realized:.4g})"}
        w = max(map(len, info))
        print("\n".join(f"{k:<{w}} : {v}" for k, v in info.items()))