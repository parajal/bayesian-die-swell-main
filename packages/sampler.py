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

    def run_mcmc(self, nwalkers=10, nsteps=5000, burn_fraction=0.3,
                 warmup_fraction=0.1, ball_scale=0.2):
        np.random.seed(self.seed)
        # Resolve l_bias='fixed' and/or set the sigma_bias prior from the discrepancy delta(x).
        if (getattr(self, "l_bias", None) == "fixed"
                or (self._infer_sigma_bias() and getattr(self, "sigma_bias_from_discrepancy", False))):
            self.fix_l_bias_from_acf()
        moves = [(emcee.moves.StretchMove(), 0.8),
                (emcee.moves.DEMove(), 0.1),
                (emcee.moves.GaussianMove(0.1), 0.1)]
        moves = [(emcee.moves.StretchMove(), 1.0)]
        sampler = emcee.EnsembleSampler(nwalkers, self._get_ndim(), self.log_posterior, moves=moves)
        p0 = self.sample_starting_points(nwalkers)

        nwarm, nburn = int(warmup_fraction * nsteps), int(burn_fraction * nsteps)
        if nwarm:
            state = sampler.run_mcmc(p0, nwarm, progress=True)
            best = state.coords[np.argmax(state.log_prob)]
            sampler.reset()
            p0 = self.sample_starting_points(nwalkers, center=best, scale=ball_scale)

        sampler.run_mcmc(p0, nsteps, progress=True)
        self.chain = sampler.get_chain()
        self.log_prob = sampler.get_log_prob(discard=nburn, flat=True)
        flat = sampler.get_chain(discard=nburn, flat=True)
        self.samples = self._to_physical(flat)
        self.map_theta = self._to_physical(flat[np.argmax(self.log_prob)])
        self.results = self.print_inference_results(nburn)
        return self.samples

    def sample_starting_points(self, nwalkers, center=None, scale=1e-3, pool_factor=20):
        """Walker starts: KMeans centers of prior draws, or a Gaussian ball around center."""
        rng = np.random.default_rng(self.seed)
        n = pool_factor * nwalkers
        if center is None:
            pool = self._log_prior(rng, n)
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
