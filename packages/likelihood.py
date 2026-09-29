"""Gaussian likelihood and posterior, evaluated for one point or many (emcee vectorize=True)."""
import numpy as np
class LikelihoodMixin:

    def _residuals(self, theta):
        theta = np.atleast_2d(np.asarray(theta, float))[:, :self.n_material_params]
        y_obs = np.asarray(self.y_obs_matrix[0], float)
        if getattr(self, "model_family", None) == "tanner":
            return np.array([[y_obs.max() - self._tanner_height(t[0])] for t in theta])
        Y = np.atleast_2d(self.predict(theta))
        if Y.shape[1] > y_obs.size:
            Y = Y[:, self._obs_indices]
        if getattr(self, "mode", "full_curve") == "swell_height":
            return (y_obs.max() - Y.max(axis=1))[:, None]
        return y_obs - Y

    def _bias_eigh(self, x, l_bias):
        K = self._bias_correlation_matrix(x, l_bias)
        cached = getattr(self, "_bias_eig_cache", None)
        if cached is None or cached[0] is not K:
            w, V = np.linalg.eigh(K)
            cached = self._bias_eig_cache = (K, (np.clip(w, 0.0, None), V))
        return cached[1]

    def log_likelihood(self, phi):
        phi = np.asarray(phi, float)
        theta = self._to_physical(np.atleast_2d(phi))
        n = self.n_material_params
        s_noise = theta[:, n]
        s_bias = theta[:, n + 1] if self._infer_sigma_bias() else None

        ll = np.full(len(theta), -np.inf)
        ok = np.isfinite(theta).all(axis=1) & (s_noise > 0)
        if s_bias is not None:
            ok &= s_bias > 0
        if ok.any():
            R = self._residuals(theta[ok])
            N = R.shape[1]
            if s_bias is None:
                var = s_noise[ok] ** 2
                quad, logdet = (R ** 2).sum(axis=1) / var, N * np.log(var)
            else:
                w, V = self._bias_eigh(self.obs_x_coords[:N], self.l_bias)
                d = s_noise[ok, None] ** 2 + s_bias[ok, None] ** 2 * w
                quad, logdet = ((R @ V) ** 2 / d).sum(axis=1), np.log(d).sum(axis=1)
            ll[ok] = -0.5 * (quad + logdet + N * np.log(2 * np.pi))
        ll[~np.isfinite(ll)] = -np.inf
        return ll if phi.ndim > 1 else float(ll[0])

    def log_posterior(self, phi):
        phi = np.asarray(phi, float)
        Phi = np.atleast_2d(phi)
        lp = np.array([self.log_prior(p) for p in Phi])
        ok = np.isfinite(lp)
        if ok.any():
            lp[ok] += self.log_likelihood(Phi[ok])
        return lp if phi.ndim > 1 else float(lp[0])
