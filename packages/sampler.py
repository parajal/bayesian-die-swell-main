"""Simple MCMC sampling with emcee."""

import emcee
import numpy as np
from sklearn.cluster import KMeans


class Sampler:
    """Run MCMC and compute basic diagnostics."""

    @staticmethod
    def gelman_rubin(chain):
        """R-hat per dimension; chain has emcee's shape (nsteps, nwalkers, ndim)."""
        n = chain.shape[0]
        W = chain.var(axis=0, ddof=1).mean(axis=0)
        B = n * chain.mean(axis=0).var(axis=0, ddof=1)
        return np.sqrt(((n - 1) / n * W + B / n) / W)

    @staticmethod
    def _moves(points):
        """0.6 stretch + 0.2 Gaussian random walk + 0.2 DE snooker. The Gaussian proposal
        covariance is that of ``points`` (sampler coordinates) scaled by 2.38^2 / ndim."""
        points = np.atleast_2d(points)
        cov = np.cov(points, rowvar=False) * 2.38 ** 2 / points.shape[1]
        return [(emcee.moves.StretchMove(), 0.6),
                (emcee.moves.GaussianMove(cov), 0.2),
                (emcee.moves.DESnookerMove(), 0.2)]

    def run_mcmc(self, nwalkers=10, nsteps=5000, burn_fraction=0.3,
                 warmup_fraction=0.1, ball_scale=0.2):
        np.random.seed(self.seed)
        ndim = self._get_ndim()
        p0 = self.sample_starting_points(nwalkers)
        self._print_walkers("Initial walkers", p0, n = nwalkers)
        # warm-up: Gaussian covariance from the initial walkers
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=self._moves(p0))

        nwarm, nburn = int(warmup_fraction * nsteps), int(burn_fraction * nsteps)
        if nwarm:
            state = sampler.run_mcmc(p0, nwarm, progress=True)
            best = state.coords[np.argmax(state.log_prob)]
            warm = sampler.get_chain(discard=nwarm // 2, flat=True)    # second half of the warm-up
            # restart ball: ball_scale x each parameter's spread in the warm-up (not absolute)
            p0 = self.sample_starting_points(nwalkers, center=best, scale=ball_scale * warm.std(axis=0))
            self._print_walkers("Walkers restarted after warm-up", p0, n = nwalkers)
            # main run: Gaussian covariance from the warm-up samples
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=self._moves(warm))

        sampler.run_mcmc(p0, nsteps, progress=True)
        self.chain = sampler.get_chain()
        self.log_prob = sampler.get_log_prob(discard=nburn, flat=True)
        flat = sampler.get_chain(discard=nburn, flat=True)
        self.samples = self._to_physical(flat)
        self.map_theta = self._to_physical(flat[np.argmax(self.log_prob)])
        self.results = self.print_inference_results(nburn)
        return self.samples

    def _print_walkers(self, title, p0, n = 10):
        """Print the first n walker positions in physical space."""
        names = self._get_parameter_labels(latex=False)
        print(f"{title} (first {min(n, len(p0))} of {len(p0)}, physical space):")
        print("  " + "".join(f"{name:>14}" for name in names))
        for w in self._to_physical(np.asarray(p0)[:n]):
            print("  " + "".join(f"{v:14.5g}" for v in w))

    def sample_starting_points(self, nwalkers, center=None, scale=1e-3, pool_factor=20):
        """Walker starts: KMeans centers of prior draws, or a Gaussian ball around center."""
        rng = np.random.default_rng(self.seed)
        n = pool_factor * nwalkers
        if center is None:        # prior draws (sampler coordinates): uniform material, exponential sigmas
            lo, hi = np.asarray(self._get_sampling_bounds(), float).T
            rates = np.array([self.sigma_noise_prior] + [self.sigma_bias_prior] * self._infer_sigma_bias())
            pool = np.hstack([rng.uniform(lo, hi, size=(n, len(lo))),
                              rng.exponential(1 / rates, size=(n, len(rates)))])
        else:
            pool = center + scale * rng.standard_normal((n, len(center)))
        pool = pool[[np.isfinite(self.log_posterior(p)) for p in pool]]
        if len(pool) < nwalkers:
            raise RuntimeError("not enough valid starting points to initialize walkers")
        if center is not None:
            return pool[:nwalkers]
        return KMeans(nwalkers, n_init=10, random_state=self.seed).fit(pool).cluster_centers_

    def print_inference_results(self, nburn):
        """Print and return posterior mean, std, R-hat and 95% CI per parameter."""
        x = self.samples
        stats = dict(mean=x.mean(0), std=x.std(0, ddof=1),
                     rhat=self.gelman_rubin(self.chain[nburn:]),
                     ci_low=np.percentile(x, 2.5, axis=0),
                     ci_high=np.percentile(x, 97.5, axis=0))

        print(f"{'Parameter':<15}{'Mean':>12}{'Std':>12}{'Rhat':>10}"
              f"{'95% CI Low':>15}{'95% CI High':>15}")
        results = {}
        for i, name in enumerate(self._get_parameter_labels(latex=False)):
            r = results[name] = {k: float(v[i]) for k, v in stats.items()}
            print(f"{name:<15}{r['mean']:>12.4e}{r['std']:>12.4e}{r['rhat']:>10.3f}"
                  f"{r['ci_low']:>15.4e}{r['ci_high']:>15.4e}")
        return results
