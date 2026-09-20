from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VEModel:
    """Material parameters for the viscoelastic constitutive model."""

    model: int = 2
    alam: int = 2
    lam: float = 0.1
    G: float = 10.0
    eta_s: float = 0.1
    alpha: float = 0.1
    eps: float = 0.1
    tauy: float = 10.0
    Kfac: float = 10.0
    nexp: float = 0.5


@dataclass
class RheoData:
    """Container matching the MATLAB rheodata structure."""

    rates: np.ndarray | None = None
    rate_for_startup: float | None = None
    stress_imp: float | None = None
    stress: np.ndarray | None = None
    time: np.ndarray | None = None
    strain: np.ndarray | None = None
    giesekus_error: float | None = None


def cvec_to_tensor(cvec: np.ndarray) -> np.ndarray:
    cxx, cxy, cxz, cyy, cyz, czz = np.asarray(cvec, dtype=float)
    return np.array(
        [[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]],
        dtype=float,
    )


def tensor_to_cvec(tensor: np.ndarray) -> np.ndarray:
    return np.array(
        [
            tensor[0, 0],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 1],
            tensor[1, 2],
            tensor[2, 2],
        ],
        dtype=float,
    )


def fill_l(rate: float, flowtype: int) -> np.ndarray:
    """Fill the velocity-gradient tensor."""
    L = np.zeros((3, 3), dtype=float)

    if flowtype == 1:
        L[0, 1] = rate
    elif flowtype == 2:
        L[0, 0] = rate
        L[1, 1] = -rate
    elif flowtype == 3:
        L[0, 0] = rate
        L[1, 1] = -rate / 2.0
        L[2, 2] = -rate / 2.0
    else:
        raise ValueError("flowtype must be 1=shear, 2=planar extension, or 3=uniaxial extension")

    return L


def stress_viscoelastic_3d(cvec: np.ndarray, vemodel: VEModel) -> np.ndarray:
    """Calculate the viscoelastic stress vector."""
    cvec = np.asarray(cvec, dtype=float)
    return vemodel.G * np.array(
        [
            cvec[0] - 1.0,
            cvec[1],
            cvec[2],
            cvec[3] - 1.0,
            cvec[4],
            cvec[5] - 1.0,
        ],
        dtype=float,
    )


def stress_solvent_3d(vemodel: VEModel, rate: float, flowtype: int) -> np.ndarray:
    """Calculate the solvent stress vector."""
    L = fill_l(rate, flowtype)
    return tensor_to_cvec(vemodel.eta_s * (L + L.T))


def von_mises(svec: np.ndarray) -> float:
    """Calculate the von Mises equivalent shear stress."""
    sxx, sxy, sxz, syy, syz, szz = np.asarray(svec, dtype=float)
    return float(
        np.sqrt(
            ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) / 6.0
            + sxy**2
            + sxz**2
            + syz**2
        )
    )


def rhs_viscoelastic(cvec: np.ndarray, L: np.ndarray, vemodel: VEModel) -> np.ndarray:
    """Right-hand side for the conformation tensor evolution."""
    I = np.eye(3)
    cc = cvec_to_tensor(cvec)

    ucd_part = L @ cc + cc @ L.T
    diff = cc - I

    if vemodel.model == 1:
        rlx_part = -(1.0 / vemodel.lam) * diff
    elif vemodel.model == 2:
        rlx_part = -(1.0 / vemodel.lam) * (diff + vemodel.alpha * (diff @ diff))
    elif vemodel.model == 3:
        rlx_part = (
            -(1.0 / vemodel.lam)
            * (1.0 + vemodel.eps * (np.trace(cc) - 3.0))
            * diff
        )
    elif vemodel.model == 4:
        rlx_part = (
            -(1.0 / vemodel.lam)
            * np.exp(vemodel.eps * (np.trace(cc) - 3.0))
            * diff
        )
    else:
        raise ValueError("model must be 1=UCM, 2=Giesekus, 3=PTTlin, or 4=PTTexp")

    if vemodel.alam in (2, 3):
        stress = stress_viscoelastic_3d(cvec, vemodel)
        taud = von_mises(stress)

    if vemodel.alam == 1:
        rlx_part = np.zeros_like(rlx_part)
    elif vemodel.alam == 2:
        fac = 0.0 if taud == 0.0 else max(0.0, (taud - vemodel.tauy) / taud)
        rlx_part = rlx_part * fac
    elif vemodel.alam == 3:
        if taud <= vemodel.tauy:
            fac = 0.0
        else:
            fac = 1.0 / (
                (taud / vemodel.G)
                * ((taud - vemodel.tauy) / vemodel.Kfac) ** (-1.0 / vemodel.nexp)
            )
        rlx_part = fac * rlx_part * vemodel.lam
    elif vemodel.alam != 0:
        raise ValueError("alam must be 0=none, 1=elastic, 2=SRM1, or 3=SRM2")

    return tensor_to_cvec(ucd_part + rlx_part)


def solve_root_newton(
    func,
    x0: np.ndarray,
    *,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    jacobian_step: float = 1e-6,
) -> np.ndarray:
    """Small finite-difference Newton solver used as a Python fsolve stand-in."""
    x = np.asarray(x0, dtype=float).copy()

    for _ in range(max_iter):
        fx = np.asarray(func(x), dtype=float)
        norm_fx = np.linalg.norm(fx, ord=2)
        if norm_fx < tolerance:
            return x

        jacobian = np.zeros((len(x), len(x)), dtype=float)
        for j in range(len(x)):
            step = jacobian_step * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += step
            jacobian[:, j] = (np.asarray(func(xp), dtype=float) - fx) / step

        try:
            dx = np.linalg.solve(jacobian, -fx)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(jacobian, -fx, rcond=None)[0]

        damping = 1.0
        accepted = False
        while damping > 1e-8:
            candidate = x + damping * dx
            candidate_norm = np.linalg.norm(func(candidate), ord=2)
            if candidate_norm < norm_fx:
                x = candidate
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            x = x + dx

        if np.linalg.norm(damping * dx, ord=2) < tolerance * (1.0 + np.linalg.norm(x, ord=2)):
            if np.linalg.norm(func(x), ord=2) < 10.0 * tolerance:
                return x

    raise RuntimeError(f"No steady solution found; final residual is {np.linalg.norm(func(x), ord=2):g}")


def solve_steady_conformation(
    vemodel: VEModel,
    rate: float,
    *,
    flowtype: int = 1,
    initial_guess: np.ndarray | None = None,
    function_tolerance: float = 1e-8,
) -> np.ndarray:
    """Solve the steady conformation tensor for one imposed rate."""
    if initial_guess is None:
        initial_guess = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=float)

    L = fill_l(rate, flowtype)
    return solve_root_newton(
        lambda c: rhs_viscoelastic(c, L, vemodel),
        initial_guess,
        tolerance=function_tolerance,
    )


def rate_for_stress(
    cvec: np.ndarray,
    vemodel: VEModel,
    rheodata: RheoData,
    flowtype: int,
) -> float:
    """Calculate the deformation rate needed to impose the requested stress."""
    tauvec = stress_viscoelastic_3d(cvec, vemodel)

    if rheodata.stress_imp is None:
        raise ValueError("rheodata.stress_imp must be set")

    if flowtype == 1:
        return float((rheodata.stress_imp - tauvec[1]) / vemodel.eta_s)
    if flowtype == 2:
        return float((rheodata.stress_imp - (tauvec[0] - tauvec[3])) / (4.0 * vemodel.eta_s))
    if flowtype == 3:
        return float((rheodata.stress_imp - (tauvec[0] - tauvec[3])) / (3.0 * vemodel.eta_s))

    raise ValueError("flowtype must be 1=shear, 2=planar extension, or 3=uniaxial extension")


def run_rate_controlled(
    *,
    vemodel: VEModel | None = None,
    rates: np.ndarray | None = None,
    only_startup: bool = False,
    flowtype: int = 1,
    rate_for_startup: float | None = None,
    time_startup: float | None = None,
    numsteps: int = 1000,
    function_tolerance: float = 1e-6,
) -> RheoData:
    """Run the rate-controlled startup and optional steady-state simulations."""
    vemodel = vemodel or VEModel()
    rates = np.asarray(rates if rates is not None else np.logspace(-3, 2, 500), dtype=float)
    rate_for_startup = float(rate_for_startup if rate_for_startup is not None else rates[-1])
    time_startup = float(time_startup if time_startup is not None else 40.0 * vemodel.lam)
    deltat = time_startup / numsteps

    if not only_startup and not np.isclose(rate_for_startup, rates[-1]):
        raise ValueError("Performing steady simulations but rate_for_startup != rates[-1]")

    rheodata = RheoData(rates=rates, rate_for_startup=rate_for_startup)
    cn = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    rheodata.stress = np.zeros((6, numsteps + 1), dtype=float)
    rheodata.time = deltat * np.arange(numsteps + 1, dtype=float)

    tau = stress_viscoelastic_3d(cn, vemodel)
    solvent_stress = stress_solvent_3d(vemodel, rate_for_startup, flowtype)
    rheodata.stress[:, 0] = tau + solvent_stress

    for n in range(numsteps):
        L = fill_l(rate_for_startup, flowtype)
        k1 = rhs_viscoelastic(cn, L, vemodel)

        L = fill_l(rate_for_startup, flowtype)
        k2 = rhs_viscoelastic(cn + deltat * k1, L, vemodel)

        cnp1 = cn + deltat * (k1 + k2) / 2.0

        tau = stress_viscoelastic_3d(cnp1, vemodel)
        solvent_stress = stress_solvent_3d(vemodel, rate_for_startup, flowtype)
        rheodata.stress[:, n + 1] = tau + solvent_stress

        cn = cnp1

    if only_startup:
        return rheodata

    steady_stress = np.zeros((6, len(rates)), dtype=float)
    c0 = cn

    for i in range(len(rates) - 1, -1, -1):
        L = fill_l(rates[i], flowtype)

        try:
            cvec = solve_root_newton(
                lambda c: rhs_viscoelastic(c, L, vemodel),
                c0,
                tolerance=function_tolerance,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"No steady solution found at rate {rates[i]}") from exc

        tau = stress_viscoelastic_3d(cvec, vemodel)
        solvent_stress = stress_solvent_3d(vemodel, rates[i], flowtype)
        steady_stress[:, i] = tau + solvent_stress
        c0 = cvec

    rheodata.stress = steady_stress

    if vemodel.model == 2 and vemodel.alam == 0 and flowtype == 1:
        eta = vemodel.G * vemodel.lam
        final_rate = rates[-1]
        chik = (
            (
                np.sqrt(
                    1.0
                    + 16.0
                    * vemodel.alpha
                    * (1.0 - vemodel.alpha)
                    * (vemodel.lam * final_rate) ** 2
                )
                - 1.0
            )
            / (
                8.0
                * vemodel.alpha
                * (1.0 - vemodel.alpha)
                * (vemodel.lam * final_rate) ** 2
            )
        ) ** 0.5
        fk = (1.0 - chik) / (1.0 + (1.0 - 2.0 * vemodel.alpha) * chik)
        rheodata.giesekus_error = abs(
            eta * (1.0 - fk) ** 2 / (1.0 + (1.0 - 2.0 * vemodel.alpha) * fk)
            + vemodel.eta_s
            - rheodata.stress[1, -1] / final_rate
        )

    return rheodata


def run_stress_controlled(
    *,
    vemodel: VEModel | None = None,
    flowtype: int = 1,
    stress_imp: float = 12.0,
    numtimesteps1: int = 40,
    numtimesteps2: int = 1000,
    time1: float = 4e-3,
    time2: float = 1.0,
) -> RheoData:
    """Run the stress-controlled startup simulation."""
    vemodel = vemodel or VEModel()
    rheodata = RheoData(stress_imp=stress_imp)

    cvec = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    strain = 0.0

    numsteps = numtimesteps1 + numtimesteps2
    rheodata.stress = np.zeros((6, numsteps), dtype=float)
    rheodata.strain = np.zeros(numsteps, dtype=float)
    rheodata.rates = np.zeros(numsteps, dtype=float)
    rheodata.time = np.zeros(numsteps, dtype=float)

    deltat1 = time1 / numtimesteps1
    deltat2 = time2 / numtimesteps2
    deltat = deltat1
    time = 0.0

    for idx in range(numsteps):
        n = idx + 1
        time += deltat

        if n == numtimesteps1 + 1:
            deltat = deltat2

        rate1 = rate_for_stress(cvec, vemodel, rheodata, flowtype)
        L = fill_l(rate1, flowtype)
        k1 = rhs_viscoelastic(cvec, L, vemodel)

        rate2 = rate_for_stress(cvec + k1 * deltat, vemodel, rheodata, flowtype)
        L = fill_l(rate1, flowtype)
        k2 = rhs_viscoelastic(cvec + k1 * deltat, L, vemodel)

        cvec = cvec + deltat * (k1 + k2) / 2.0
        strain += deltat * (rate1 + rate2) / 2.0

        tau = stress_viscoelastic_3d(cvec, vemodel)
        ratenp1 = rate_for_stress(cvec, vemodel, rheodata, flowtype)
        solvent_stress = stress_solvent_3d(vemodel, ratenp1, flowtype)

        rheodata.stress[:, idx] = tau + solvent_stress
        rheodata.strain[idx] = strain
        rheodata.rates[idx] = ratenp1
        rheodata.time[idx] = time

    return rheodata
