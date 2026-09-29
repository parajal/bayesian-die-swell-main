"""Prior distributions."""

import numpy as np
class PriorMixin:

    def log_prior(self, phi):
        """Uniform on the material parameters in the sampler's coordinates (log10 of both with
        parametrize, i.e. log-uniform); exponential on sigma_noise (and sigma_bias)."""
        phi = np.asarray(phi, float)
        lo, hi = np.asarray(self._get_sampling_bounds(), float).T
        rates = np.array([self.sigma_noise_prior]
                         + [self.sigma_bias_prior] * self._infer_sigma_bias())
        n, k = len(lo), len(rates)
        if not np.isfinite(phi).all() or np.any((phi[:n] < lo) | (phi[:n] > hi)):
            return -np.inf
        theta = self._to_physical(phi)
        sigmas = theta[n:n + k]
        if np.any(sigmas <= 0) or not self._implied_in_bounds(theta[:n]):
            return -np.inf
        logp = -np.log(hi - lo).sum() + np.sum(np.log(rates) - rates * sigmas)
        return float(logp)