"""Prior distributions."""

import numpy as np
class PriorMixin:

    def log_prior(self, phi):
        """Uniform on physical params; exponential on sigmas; PC or uniform on l_bias; Normal(0, sd) on c."""
        theta = np.asarray(self._to_physical(phi), float)
        lo, hi = np.asarray(self._get_parameter_bounds(), float).T
        rates = np.array([self.sigma_noise_prior]
                         + [self.sigma_bias_prior] * self._infer_sigma_bias())
        n, k = len(lo), len(rates)
        params, sigmas = theta[:n], theta[n:n + k]

        if (not np.isfinite(theta).all() or np.any((params < lo) | (params > hi))
                or np.any(sigmas <= 0)):
            return -np.inf
        logp = -np.log(hi - lo).sum() + np.sum(np.log(rates) - rates * sigmas)
        j = n + k
        if self._infer_l_bias():
            l = theta[j]
            if self.l_bias_prior == "uniform":            # l_bias ~ Uniform(lo, hi)
                lo_l, hi_l = self.l_bias_bounds
                if not (lo_l <= l <= hi_l):
                    return -np.inf
                logp += -np.log(hi_l - lo_l)
            else:                                          # l_bias ~ PC prior (range, 1-D)
                lam = self.l_bias_pc_lambda
                if l <= 0:
                    return -np.inf
                logp += np.log(lam / 2) - 1.5 * np.log(l) - lam * l ** -0.5
            j += 1
        if self._infer_mean_bias():                      # c ~ Normal(0, sd)
            c, sd = theta[j], self.c_bias_prior_sd
            logp += -0.5 * (c / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)
        return float(logp)

    def _log_prior(self, rng, n_samples):
        """Draw n_samples from the prior."""
        lo, hi = np.asarray(self._get_parameter_bounds(), float).T
        rates = np.array([self.sigma_noise_prior]
                         + [self.sigma_bias_prior] * self._infer_sigma_bias())
        n, k = len(lo), len(rates)

        samples = np.zeros((n_samples, self._get_ndim()))
        samples[:, :n] = rng.uniform(lo, hi, size=(n_samples, n))
        samples[:, n:n + k] = rng.exponential(1 / rates, size=(n_samples, k))
        j = n + k
        if self._infer_l_bias():
            if self.l_bias_prior == "uniform":
                lo_l, hi_l = self.l_bias_bounds
                samples[:, j] = rng.uniform(lo_l, hi_l, size=n_samples)
            else:                                          # PC prior via inverse CDF: l = (lambda / -ln u)^2
                u = np.clip(rng.uniform(size=n_samples), 1e-12, 1 - 1e-12)
                samples[:, j] = (self.l_bias_pc_lambda / -np.log(u)) ** 2
            j += 1
        if self._infer_mean_bias():
            samples[:, j] = rng.normal(0.0, self.c_bias_prior_sd, size=n_samples)
        return samples