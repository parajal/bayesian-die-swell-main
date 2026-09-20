"""Gaussian likelihood and posterior."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

_FAILURES = (FloatingPointError, ValueError, RuntimeError, np.linalg.LinAlgError)


class LikelihoodMixin:

    @staticmethod
    def _gauss_ll(r, cov):
        """log N(r | 0, cov); cov is a scalar variance or a full matrix."""
        r = np.ravel(r)
        if np.ndim(cov) == 0:
            quad, logdet = r @ r / cov, r.size * np.log(cov)
        else:
            c = cho_factor(cov, lower=True)
            quad, logdet = r @ cho_solve(c, r), 2 * np.log(np.diag(c[0])).sum()
        return -0.5 * (quad + logdet + r.size * np.log(2 * np.pi))

    def _residual(self, theta):
        y_obs = np.asarray(self.y_obs_matrix[0], float)
        family = getattr(self, "model_family", None)
        if family == "tanner":
            return np.array([y_obs.max() - self._tanner_height(theta[0], self.u_avg_obs)])

        if family == "ratio":
            y = self.predict(float(theta[0]))
        else:
            kw = {}
            if len(theta) > 2:
                eps = getattr(self, "third_parameter_name", "") == "epsilon"
                kw["epsilon_val" if eps else "alpha_val"] = theta[2]
            y = self.predict(theta[0], theta[1], u_avg_val=self.u_avg_obs, **kw)

        y = np.asarray(y, float).ravel()
        if y.size > y_obs.size:
            y = y[self._obs_indices]
        if getattr(self, "mode", "full_curve") == "swell_height":
            return np.array([y_obs.max() - y.max()])
        return y_obs - y

    def log_likelihood(self, phi):
        phi = np.asarray(phi, float)
        if not np.isfinite(phi).all():
            return -np.inf
        try:
            theta = self._to_physical(phi)
            s_noise, s_bias = self._extract_noise_bias(theta)
            if not all(np.isfinite(s) and s > 0 for s in (s_noise, s_bias) if s is not None):
                return -np.inf

            r = self._residual(theta) - self._extract_mean_bias(theta)  # curve channel only
            if s_bias is None:
                cov = s_noise**2
            else:
                x = self.obs_x_coords[:r.size]
                K = self._bias_correlation_matrix(x, self._extract_l_bias(theta))
                cov = s_noise**2 * np.eye(r.size) + s_bias**2 * K
            ll = self._gauss_ll(r, cov)

            if getattr(self, "use_pressure", False):
                ll += self._gauss_ll(self.pressure_obs - self.predict_pressure(theta), s_noise**2)
        except _FAILURES:
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf

    def log_posterior(self, phi):
        lp = self.log_prior(phi)
        return lp + self.log_likelihood(phi) if np.isfinite(lp) else -np.inf