"""Simple MCMC sampling with emcee."""

from pathlib import Path
import emcee
import numpy as np

class Sampler:
    """Run MCMC and save the results."""

    @staticmethod
    def gelman_rubin(chain):
        """R-hat per dimension; chain has emcee's shape (nsteps, nwalkers, ndim)."""
        n = chain.shape[0]
        W = chain.var(axis=0, ddof=1).mean(axis=0)
        B = n * chain.mean(axis=0).var(axis=0, ddof=1)
        return np.sqrt(((n - 1) / n * W + B / n) / W)

    def run_mcmc(self, nwalkers=10, nsteps=5000, burn_fraction=0.3,
                 warmup_fraction=0.1, ball_scale=0.5):
        ndim = self._get_ndim()
        p0 = self.sample_starting_points(nwalkers)
        self._print_walkers("Initial walkers", p0, n = nwalkers)
        moves = [(emcee.moves.DEMove(), 0.6), (emcee.moves.DESnookerMove(), 0.2), (emcee.moves.StretchMove(), 0.2)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=moves, vectorize=True)
        sampler.random_state = np.random.RandomState(self.seed).get_state()   # emcee's own RNG (moves)

        nwarm = int(warmup_fraction * nsteps)
        if nwarm:
            state = sampler.run_mcmc(p0, nwarm, progress=True)
            best = state.coords[np.argmax(state.log_prob)]
            warm = sampler.get_chain(discard=nwarm // 2, flat=True)
            p0 = self.sample_starting_points(nwalkers, center=best, scale=ball_scale * warm.std(axis=0))
            self._print_walkers("Walkers restarted after warm-up", p0, n = nwalkers)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=moves, vectorize=True)

        sampler.run_mcmc(p0, nsteps, progress=True)
        nburn = int(burn_fraction * nsteps)

        self.sampler, self.nburn = sampler, nburn
        self.chain = sampler.get_chain()
        self.log_prob = sampler.get_log_prob(discard=nburn, flat=True)
        self.samples = self._to_physical(sampler.get_chain(discard=nburn, flat=True))
        self.results = self.print_inference_results(nburn)
        self.n1_result = None        
        self.save_results()
        return self.samples

    def _print_walkers(self, title, p0, n = 10):
        """Print the first n walker positions in physical space."""
        names = self._get_parameter_labels(latex=False)
        print(f"{title} (first {min(n, len(p0))} of {len(p0)}, physical space):")
        print("  " + "".join(f"{name:>14}" for name in names))
        for w in self._to_physical(np.asarray(p0)[:n]):
            print("  " + "".join(f"{v:14.5g}" for v in w))

    def sample_starting_points(self, nwalkers, center=None, scale=1e-3, pool_factor=20):
        """Walker starts: direct prior draws, or a Gaussian ball around center (draws with a
        finite posterior are kept, e.g. inside the implied-parameter range with parametrize)."""
        rng = np.random.default_rng(self.seed)
        n = pool_factor * nwalkers
        if center is None:    
            lo, hi = np.asarray(self._get_sampling_bounds(), float).T
            rates = np.array([self.sigma_noise_prior] + [self.sigma_bias_prior] * self._infer_sigma_bias())
            pool = np.hstack([rng.uniform(lo, hi, size=(n, len(lo))),
                              rng.exponential(1 / rates, size=(n, len(rates)))]
                             + [rng.uniform(*self.l_bias_bounds, size=(n, 1))] * self._infer_l_bias())
        else:
            pool = center + scale * rng.standard_normal((n, len(center)))
        pool = pool[np.isfinite(self.log_posterior(pool))]
        if len(pool) < nwalkers:
            raise RuntimeError("not enough valid starting points to initialize walkers")
        return pool[:nwalkers]

    def print_inference_results(self, nburn):
        """Print and return posterior mean, std, R-hat and 90% CI per parameter."""
        x = self.samples
        stats = dict(mean=x.mean(0), std=x.std(0, ddof=1),
                     rhat=self.gelman_rubin(self.chain[nburn:]),
                     ci_low=np.percentile(x, 5.0, axis=0),
                     ci_high=np.percentile(x, 95.0, axis=0))

        print(f"{'Parameter':<15}{'Mean':>12}{'Std':>12}{'Rhat':>10}"
              f"{'90% CI Low':>15}{'90% CI High':>15}")
        results = {}
        for i, name in enumerate(self._get_parameter_labels(latex=False)):
            r = results[name] = {k: float(v[i]) for k, v in stats.items()}
            print(f"{name:<15}{r['mean']:>12.4e}{r['std']:>12.4e}{r['rhat']:>10.3f}"
                  f"{r['ci_low']:>15.4e}{r['ci_high']:>15.4e}")
        return results

    def save_results(self):

        root, data = Path(self.swell_root), Path(self.infer_dir)
        out_dir = root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        bias = str(bool(self.sigma_bias)).lower()
        tanner = self.model_family == "tanner"
        prefix = "tanner_" if tanner else ""          # keeps Tanner and ROM results of the same data apart
        out = out_dir / f"{prefix}bias_{bias}_{data.parent.name}_{data.name}.txt"
        try:
            data_rel = data.relative_to(root)
        except ValueError:
            data_rel = data

        lines = [f"# data         : {data_rel.as_posix()}",
                 "# fitted model : "
                 + ("tanner, B = 0.13 + [1 + (N1/(2 tau_w))^2 / 2]^(1/6) = max height" if tanner
                    else f"{self.model_family}, parametrize={self.parametrize}")
                 + f", sigma_bias={self.sigma_bias}"
                 + (f", l_bias inferred U{self.l_bias_bounds}" if self._infer_l_bias()
                    else f", l_bias={self.l_bias}" if self.sigma_bias else "")
                 + f", noise {self.sigma_noise_percent:g} %",
                 "#", "# true parameters"]
        theta = getattr(self, "theta", None)
        if theta is None:
            lines.append("# (not given)")
        else:
            names = ["lambda", "beta", "alpha"] + [f"p{i + 1}" for i in range(3, len(theta))]
            lines += [f"{n:<14}{v:g}" for n, v in zip(names, theta)]
        lines += ["#", "# inferred parameters",
                  f"# {'name':<12}{'mean':>13}{'Rhat':>9}{'ci90_low':>14}{'ci90_high':>14}"]
        for name, r in self.results.items():
            lines.append(f"{name:<14}{r['mean']:>13.4e}{r['rhat']:>9.3f}"
                         f"{r['ci_low']:>14.4e}{r['ci_high']:>14.4e}")
        n1 = getattr(self, "n1_result", None)
        if n1 is not None:
            sr = ("2 N1/(2 tau_w) (Tanner)" if tanner else "2 (1 - beta) lambda gammadot_w")
            lines += ["#", f"# derived quantities at U_avg = {n1['U_avg']:g}  (S_R = N1 / tau_w = "
                      f"{sr}, N1 = S_R tau_w, tau_w = R |dp/dx| / 2)",
                      f"# {'name':<12}{'mean':>13}{'ci90_low':>14}{'ci90_high':>14}"]
            for name, key in (("E[N1]", "N1"), ("E[S_R]", "Sr")):
                lo, hi = n1[f"{key}_ci"]
                lines.append(f"{name:<14}{n1[key]:>13.4e}{lo:>14.4e}{hi:>14.4e}")
            if n1.get("gammadot_w") is not None:
                lines.append(f"{'gammadot_w':<14}{n1['gammadot_w']:>13.4e}    # {n1['gammadot_src']} "
                             f"(4 U_avg / R = {n1['gammadot_nominal']:g})")
            if n1.get("N1_true") is not None:
                lo, hi = self.true_n1_x
                lines.append(f"{'N1_true':<14}{n1['N1_true']:>13.4e}    # FEM wall N1 from "
                             f"{self.true_n1_file}, mean over {lo:g} <= x <= {hi:g}")
            if n1.get("Sr_true") is not None:
                lines.append(f"{'S_R_true':<14}{n1['Sr_true']:>13.4e}    # N1_true / tau_w, "
                             f"tau_w = R |dp/dx| / 2 from pressure_drop.txt")
            if n1.get("gammadot_true") is not None:
                lo, hi = self.true_n1_x
                lines.append(f"{'gammadot_w_true':<15}{n1['gammadot_true']:>12.4e}    # FEM wall shear "
                             f"rate from {self.true_n1_file}, mean over {lo:g} <= x <= {hi:g}")
        out.write_text("\n".join(lines) + "\n")
        print(f"saved results to {out}")
        return out
