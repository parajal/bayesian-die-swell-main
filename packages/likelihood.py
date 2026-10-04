"""Gaussian likelihood and posterior."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class LikelihoodMixin:
    """Gaussian likelihood and posterior."""

    @staticmethod
    def _residual_logprob(residual, sigma, sigma_bias=None, eig=None, K=None):
        n = len(residual)
        if K is not None:
            c = cho_factor(sigma**2 * np.eye(n) + sigma_bias**2 * K, lower=True)
            r2 = residual @ cho_solve(c, residual)
            logdet = 2 * np.sum(np.log(np.diag(c[0])))
        elif eig is None:
            r2 = residual @ residual / sigma**2
            logdet = 2 * n * np.log(sigma)
        else:
            w, V = eig
            d = sigma**2 + sigma_bias**2 * w
            r2 = np.sum((residual @ V) ** 2 / d)
            logdet = np.sum(np.log(d))
        return -0.5 * (r2 + logdet + n * np.log(2 * np.pi))

    def _bias_correlation_matrix(self, x, l_bias) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        constrained = bool(getattr(self, "constrained_model_error", False))
        Dc = np.asarray(self.bias_gradient_points, float) if constrained else np.empty(0)
        key = (float(l_bias), constrained, tuple(Dc), x.tobytes())
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

        K.setflags(write=False)
        self._bias_corr_cache = (key, K)
        return K

    def _bias_eigh(self, x, l_bias):
        """Eigen-decomposition (w, V) of K', cached: K' is fixed during sampling (x, l_bias, constraints)."""
        K = self._bias_correlation_matrix(x, l_bias)
        cached = getattr(self, "_bias_eig_cache", None)
        if cached is None or cached[0] is not K:
            w, V = np.linalg.eigh(K)
            cached = self._bias_eig_cache = (K, (np.clip(w, 0.0, None), V))
        return cached[1]

    def _residuals(self, theta):
        theta = np.asarray(theta, float)
        y_obs = np.asarray(self.y_obs_matrix[0], float)
        if self.model_family == "tanner":    
            R = (self.h_max_obs - self._tanner_B(np.atleast_2d(theta)[:, 0]))[:, None]
        else:
            Y = np.atleast_2d(self.predict(np.atleast_2d(theta)[:, :self.n_material_params]))
            if Y.shape[1] > y_obs.size:
                Y = Y[:, self._obs_indices]
            R = y_obs - Y
        return R[0] if theta.ndim == 1 else R

    def log_likelihood(self, phi, residual=None):
        """Log likelihood of one point; residual can be passed in when already computed."""
        theta = self._to_physical(phi)
        n = self.n_material_params
        sigma_noise = self.sigma_noise_known if self.model_family == "tanner" else theta[n]
        sigma_bias = theta[n + 1] if self._infer_sigma_bias() else None
        if sigma_noise <= 0 or (sigma_bias is not None and sigma_bias <= 0):
            return -np.inf
        if residual is None:
            residual = self._residuals(theta)

        eig = K = None
        if sigma_bias is not None:      
            x = self.obs_x_coords[:len(residual)]
            if self._infer_l_bias():          
                K = self._bias_correlation_matrix(x, theta[n + 2])
            else:
                eig = self._bias_eigh(x, self.l_bias)
        value = self._residual_logprob(residual, sigma_noise, sigma_bias, eig, K)
        return value if np.isfinite(value) else -np.inf

    def log_posterior(self, phi):
        phi = np.asarray(phi, float)
        if phi.ndim > 1:    
            lp = np.array([self.log_prior(p) for p in phi])
            ok = np.flatnonzero(np.isfinite(lp))
            if ok.size:
                R = self._residuals(self._to_physical(phi[ok]))
                for i, r in zip(ok, R):
                    ll = self.log_likelihood(phi[i], residual=r)
                    lp[i] = lp[i] + ll if np.isfinite(ll) else -np.inf
            return lp

        return self.log_prior(phi) + self.log_likelihood(phi)
