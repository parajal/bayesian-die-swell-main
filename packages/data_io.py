from pathlib import Path
import numpy as np

class DataLoaderMixin:
    """Load the ROM training set and the observed curve; set the sigma_noise prior."""

    def load_training_data(self):
        curves_file, params_file = self.filenames_train
        Y = np.loadtxt(self.data_dir / curves_file, ndmin=2)[:, ::-1]
        P = np.loadtxt(self.data_dir / params_file, ndmin=2)[:, :self.n_material_params]
        if self.parametrize:
            P[:, 0] = (1.0 - P[:, 1]) * P[:, 0]
        return Y, P

    def load_data(self, filename):
        rng = np.random.default_rng(self.seed)
        path = Path(self.swell_root, filename).resolve()
        self.infer_dir = path.parent
        y = np.loadtxt(path, ndmin=2)[:, ::-1]
        x_full = np.loadtxt(self.infer_dir / "curve4_x.txt")[::-1]
        self._obs_indices = np.arange(0, x_full.size, self.thin)
        self.y_obs_matrix_clean = y[:, self._obs_indices]
        self.obs_x_coords = x_full[self._obs_indices]
        self.max_displacement = self.y_obs_matrix_clean.max() - 1.0
        self.sigma_noise_prior = 1.0 / (0.10 * self.max_displacement)  
        sigma_noise = self.sigma_noise_percent / 100 * self.max_displacement
        noise = rng.normal(0, sigma_noise, self.y_obs_matrix_clean.shape)
        self.sigma_noise_realized = noise.std()
        self.y_obs_matrix = self.y_obs_matrix_clean + noise
        self._mle_discrepancy()
