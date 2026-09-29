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
                 warmup_fraction=0.1, ball_scale=0.2, n_tau=50, check_every=100):
        np.random.seed(self.seed)
        ndim = self._get_ndim()
        p0 = self.sample_starting_points(nwalkers)
        self._print_walkers("Initial walkers", p0, n = nwalkers)
        moves = [(emcee.moves.DEMove(), 0.6), (emcee.moves.DESnookerMove(), 0.2), (emcee.moves.StretchMove(), 0.2)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=moves,
                                        vectorize=True)

        nwarm = int(warmup_fraction * nsteps)
        if nwarm:
            state = sampler.run_mcmc(p0, nwarm, progress=True)
            best = state.coords[np.argmax(state.log_prob)]
            warm = sampler.get_chain(discard=nwarm // 2, flat=True)
            p0 = self.sample_starting_points(nwalkers, center=best, scale=ball_scale * warm.std(axis=0))
            self._print_walkers("Walkers restarted after warm-up", p0, n = nwalkers)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=moves,
                                            vectorize=True)

        # sample until converged (n_tau * tau < steps and tau stable to 1%), at most nsteps
        converged, old_tau = False, np.inf
        for _ in sampler.sample(p0, iterations=nsteps, progress=True):
            if sampler.iteration % check_every:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            converged = bool(np.all(n_tau * tau < sampler.iteration)
                             and np.all(np.abs(old_tau - tau) / tau < 0.01))
            if converged:
                break
            old_tau = tau

        tau = sampler.get_autocorr_time(tol=0)
        if converged:
            nburn, thin = int(2 * np.max(tau)), max(1, int(0.5 * np.min(tau)))
            print(f"converged after {sampler.iteration} steps (> {n_tau} tau, tau stable to 1%): "
                  f"burn-in {nburn}, thin {thin}")
        else:
            nburn, thin = int(burn_fraction * sampler.iteration), 1
            print(f"not converged within {nsteps} steps (needs > {n_tau} x max tau = "
                  f"{n_tau * np.max(tau):.0f} steps and a stable tau): burn-in {nburn} "
                  f"(burn_fraction), no thinning")

        self.sampler = sampler               # kept for diagnostics (autocorrelation, acceptance, ...)
        self.nburn, self.thin, self.converged = nburn, thin, converged
        self.chain = sampler.get_chain()
        self.log_prob = sampler.get_log_prob(discard=nburn, thin=thin, flat=True)
        flat = sampler.get_chain(discard=nburn, thin=thin, flat=True)
        self.samples = self._to_physical(flat)
        self.map_theta = self._to_physical(flat[np.argmax(self.log_prob)])
        self.results = self.print_inference_results(nburn)
        self.print_diagnostics(nburn)
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
        pool = pool[np.isfinite(self.log_posterior(pool))]
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

    def print_diagnostics(self, nburn):
        """Print and return the mean acceptance fraction, the integrated autocorrelation time
        per parameter (sampler coordinates, after burn-in) with the kept-steps / tau ratio, and
        the walkers whose mean log posterior lies more than 5 below the median (stuck walkers)."""
        s = self.sampler
        names = self._get_parameter_labels(latex=False)
        acc = float(np.mean(s.acceptance_fraction))
        tau = s.get_autocorr_time(discard=nburn, quiet=True)
        n_over_tau = float((s.iteration - nburn) / np.nanmax(tau))
        lp = s.get_log_prob(discard=nburn).mean(axis=0)          # mean log posterior per walker
        med = float(np.median(lp))
        stuck = np.flatnonzero(lp < med - 5.0).tolist()

        print(f"acceptance fraction {acc:.3f} (aim ~0.2-0.5)")
        print("autocorrelation time (steps): "
              + ", ".join(f"{n} {t:.0f}" for n, t in zip(names, tau))
              + f"  |  kept steps / max tau = {n_over_tau:.0f} (aim >= 50)")
        print(f"walker mean log posterior: median {med:.1f}, lowest {lp.min():.1f} "
              f"(walker {int(lp.argmin())})"
              + (f"; more than 5 below the median: walkers {stuck}" if stuck else ""))
        self.diagnostics = dict(acceptance=acc, tau=dict(zip(names, map(float, tau))),
                                n_over_tau=n_over_tau, walker_mean_logp=lp, stuck_walkers=stuck)
        return self.diagnostics
