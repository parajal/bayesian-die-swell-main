"""N1 from the posterior: wall shear stress from the pressure drop, wall shear rate,
E[N1] / E[S_R] with credible intervals, and the FEM reference (true) wall values."""

import importlib
import re
import sys
from pathlib import Path

import numpy as np

# Constitutive N1 helpers: family -> (module, function, {posterior-mean key: fn kwarg}).
_N1_FUNCS = {
    "oldroyd":  ("compute_n1_oldroyd_b_function", "compute_n1_oldroyd_b", {}),
    "giesekus": ("compute_n1_giesekus_function",  "compute_n1_giesekus",  {"alpha": "alpha"}),
    "ptt":      ("compute_n1_ptt_function",       "compute_n1_ptt",       {"epsilon": "eps"}),
}
# Default helper options per family, overridable through compute_N1(**rheo_kwargs): the PTT training runs
# use TFEM model = 5, the linear-factor PTT (model = 6 is the exponential one: compute_N1(ptt="exponential")).
_N1_DEFAULTS = {"ptt": {"ptt": "linear"}}


class N1Mixin:
    """compute_N1 and the pressure-drop / FEM-wall helpers it needs."""

    true_n1_file, true_n1_x = "curve5_0002.txt", (-3.0, -1.0)

    def _load_n1_function(self, module_name, func_name):
        """Import a compute_n1_* helper from the repo root or swell_root."""
        for root in (Path(__file__).resolve().parents[1], self.swell_root):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        return getattr(importlib.import_module(module_name), func_name)

    def _pressure_drop_dpdx(self):
        """dp/dx read from pressure_drop.txt in the observation folder (None if it is absent)."""
        f = Path(self.infer_dir) / "pressure_drop.txt"
        return float(np.loadtxt(f).ravel()[0]) if f.is_file() else None

    def _true_wall(self):
        """FEM wall N1 and gammadot (columns named in the header of curve5_0002.txt in the
        observation folder), averaged over the fully developed part of the die wall true_n1_x:
        {'N1': ..., 'gammadot': ...}, or None if the file is absent."""
        f = Path(self.infer_dir) / self.true_n1_file
        if not f.is_file():
            return None
        cols = [ln.lstrip("#").split() for ln in f.read_text().splitlines() if ln.startswith("#")][-1]
        d = np.loadtxt(f, comments="#", ndmin=2)
        lo, hi = self.true_n1_x
        mean = d[(d[:, 0] >= lo) & (d[:, 0] <= hi)].mean(axis=0)
        return {name: float(mean[cols.index(name)]) for name in ("N1", "gammadot")}

    def _tau_w(self, r):
        """Wall shear stress tau_w = R |dp/dx| / 2 from the observation's pressure_drop.txt (None if absent)."""
        dpdx = self._pressure_drop_dpdx()
        return None if dpdx is None else 0.5 * r * abs(dpdx)

    def _wall_shear_rate(self, u_avg, r):
        """(gammadot_w, source). rabinowitsch_correction=False: the Oldroyd-B (Newtonian-profile) value
        4 U_avg / R. True: the Rabinowitsch-Mooney value (3n' + 1) / (4n') 4 U_avg / R, with
        n' = dln tau_w / dln(4 U_avg / R) from the flow curve of the same fluid: tau_w = R |dp/dx| / 2 of
        every folder <root>/*/<observation folder name> with pressure_drop.txt, U_avg from its input.txt
        (log-log quadratic fit, linear with two flow rates)."""
        nominal = 4.0 * u_avg / r
        if not getattr(self, "rabinowitsch_correction", False):
            return nominal, "4 U_avg / R"
        tau = {}
        name = Path(self.infer_dir).name
        for d in Path(self.infer_dir).parent.parent.glob(f"*/{name}"):
            f, inp = d / "pressure_drop.txt", d / "input.txt"
            if f.is_file() and inp.is_file():
                u = re.search(r"^\s*U_avg\s*=\s*([-+0-9.eEdD]+)", inp.read_text(), re.M)
                if u:
                    tau[float(u.group(1).replace("d", "e"))] = 0.5 * r * abs(float(np.loadtxt(f).ravel()[0]))
        if len(tau) < 2:
            raise ValueError(f"rabinowitsch_correction needs pressure_drop.txt at two or more flow rates in "
                             f"{Path(self.infer_dir).parent.parent}/*/{name}; found {sorted(tau)}")
        U = np.array(sorted(tau))
        c = np.polyfit(np.log(4.0 * U / r), np.log([tau[u] for u in U]), min(2, len(U) - 1))
        n = float(np.polyval(np.polyder(c), np.log(nominal)))
        return nominal * (3.0 * n + 1.0) / (4.0 * n), \
            f"Rabinowitsch, n' = {n:.4f} from U_avg = {', '.join(f'{u:g}' for u in U)}"

    def compute_N1(self, radius=None, n_expectation=500, **rheo_kwargs):
        """E[N1] and E[S_R = N1 / tau_w] with 90% credible intervals at the U_avg given to the
        constructor. Oldroyd-B: S_R = 2 (1 - beta) lambda gammadot_w and
        N1 = 2 (1 - beta) lambda tau_w gammadot_w, with tau_w = R |dp/dx| / 2 (pressure_drop.txt);
        gammadot_w is 4 U_avg / R, or its Rabinowitsch correction with rabinowitsch_correction=True
        (see _wall_shear_rate). model='tanner': S_R = 2 N1/(2 tau_w) and N1 = S_R tau_w.
        Giesekus / PTT: the compute_n1_* helpers in steady shear at gammadot_w."""
        if self.U_avg is None:
            raise ValueError("compute_N1 needs U_avg: pass U_avg=... to ROMCurve4BayesianInference.")
        u_avg, r = self.U_avg, float(self.radius if radius is None else radius)
        family, names = self.model_family, self.material_parameter_names
        S = np.asarray(self.samples, float)
        tau_w = self._tau_w(r)
        rate = rate_src = None

        if family == "tanner":
            if tau_w is None:
                raise FileNotFoundError(f"compute_N1 needs pressure_drop.txt in {self.infer_dir}")
            print(f"tau_w = R|dp/dx|/2 = {tau_w:.6g}, S_R = 2 N1/(2 tau_w) from Tanner")
            sr_s = 2.0 * S[:, 0]                      # S_R = N1 / tau_w = 2 [N1 / (2 tau_w)]
            n1s = sr_s * tau_w                        # N1 = S_R tau_w
            tw_mean = tau_w
        elif family == "oldroyd":
            if tau_w is None:
                raise FileNotFoundError(f"compute_N1 needs pressure_drop.txt in {self.infer_dir}")
            rate, rate_src = self._wall_shear_rate(u_avg, r)
            print(f"tau_w = R|dp/dx|/2 = {tau_w:.6g}, gammadot_w = {rate:.6g} ({rate_src})")
            theta1 = S[:, 0] if self.parametrize else (1.0 - S[:, 1]) * S[:, 0]
            sr_s = 2.0 * theta1 * rate                # S_R = N1 / tau_w = 2 (1 - beta) lambda gammadot_w
            n1s = sr_s * tau_w                        # N1 = 2 (1 - beta) lambda tau_w gammadot_w
            tw_mean = tau_w
        else:
            if family not in _N1_FUNCS:
                raise ValueError(f"compute_N1() is not defined for model='{family}'.")
            k = min(int(n_expectation), S.shape[0])
            draws = S[np.random.default_rng(self.seed).choice(S.shape[0], size=k, replace=False), :len(names)]
            module, func, extra = _N1_FUNCS[family]
            fn = self._load_n1_function(module, func)
            gd_w, gd_src = self._wall_shear_rate(u_avg, r)
            print(f"steady shear ({module}) at gammadot_w = {gd_w:.6g} ({gd_src})")
            n1s, tws, rates = (np.empty(k) for _ in range(3))
            for j, row in enumerate(draws):
                p = dict(zip(names, map(float, row)))
                if self.parametrize:   # (theta1, theta2, theta3) -> (lambda, beta, alpha / epsilon)
                    p = {"lambda": p["theta1"] / (1.0 - p["theta2"]), "beta": p["theta2"],
                         **{key: p["theta3"] for key in extra}}
                o = fn(rates=gd_w, lam=p["lambda"], beta=p["beta"],
                       **{kw: p[key] for key, kw in extra.items()},
                       **{**_N1_DEFAULTS.get(family, {}), **rheo_kwargs})
                rates[j] = float(np.atleast_1d(np.asarray(o["rates"], float))[0])
                n1s[j] = float(np.atleast_1d(np.asarray(o["N1"], float))[0])
                tws[j] = float(np.atleast_1d(np.asarray(o.get("tau_xy", rates[j]), float))[0])
            with np.errstate(divide="ignore", invalid="ignore"):
                sr_s = np.where(tws != 0, n1s / tws, np.nan)
            rate, rate_src, tw_mean = float(np.mean(rates)), gd_src, float(np.mean(tws))

        En1, sd, ci = float(n1s.mean()), float(n1s.std(ddof=1)), np.percentile(n1s, [5.0, 95.0])
        sr_mean, sr_sd = float(np.nanmean(sr_s)), float(np.nanstd(sr_s, ddof=1))
        sr_ci = np.nanpercentile(sr_s, [5.0, 95.0])
        print(f"E[N1] ({family}) = {En1:.6g} +/- {sd:.6g}  (90% CI [{ci[0]:.6g}, {ci[1]:.6g}], "
              f"n={n1s.size})\nE[S_R] = {sr_mean:.6g} +/- {sr_sd:.6g}  "
              f"(90% CI [{sr_ci[0]:.6g}, {sr_ci[1]:.6g}])  tau_w = {tw_mean:.6g}")
        wall = self._true_wall()
        n1_true = gd_true = sr_true = None
        if wall is not None:
            n1_true, gd_true = wall["N1"], wall["gammadot"]
            sr_true = n1_true / tau_w if tau_w else None
            print(f"true N1 (FEM, {self.true_n1_file}) = {n1_true:.6g}, true S_R = N1_true / tau_w = "
                  + (f"{sr_true:.6g}" if sr_true is not None else "n/a")
                  + f", true gammadot_w = {gd_true:.6g}")
        self.n1_result = dict(U_avg=u_avg, N1=En1, N1_ci=tuple(map(float, ci)),
                              Sr=sr_mean, Sr_ci=tuple(map(float, sr_ci)), N1_true=n1_true,
                              Sr_true=sr_true, gammadot_w=rate, gammadot_src=rate_src,
                              gammadot_true=gd_true, gammadot_nominal=4.0 * u_avg / r)
        if getattr(self, "results", None) is not None:        # add E[N1], E[S_R] to the results file
            self.save_results()
        return {"model": family, "N1": En1, "N1_std": sd, "N1_ci": ci, "N1_samples": n1s,
                "Sr": sr_mean, "Sr_std": sr_sd, "Sr_ci": sr_ci, "Sr_samples": sr_s, "tau_w": tw_mean,
                "rates": rate, "n": int(n1s.size)}
