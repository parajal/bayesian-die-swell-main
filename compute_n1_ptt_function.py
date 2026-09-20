import numpy as np

from rheology import VEModel, run_rate_controlled


def axisymmetric_wall_shear_rate(U_avg, radius=None, diameter=None):
    """Wall shear rate for a circular axisymmetric die/nozzle."""
    if radius is None:
        if diameter is None:
            raise ValueError("Provide either radius or diameter.")
        radius = diameter / 2.0

    return 4.0 * U_avg / radius


def compute_n1_ptt(
    rates=None,
    U_avg=None,
    radius=None,
    diameter=None,
    lam=2.5,
    beta=0.6,
    eps=0.2,
    eta0=1.0,
    ptt="exponential",
    G=None,
    eta_s=None,
    flowtype=1,
):
    """Compute N1 = tau_xx - tau_yy for the PTT model.

    ptt can be "linear" or "exponential". You can provide rates directly,
    or provide U_avg and radius/diameter for an axisymmetric die. You can
    provide beta and eta0, or G and eta_s directly.
    """
    if rates is None:
        if U_avg is not None:
            rates = axisymmetric_wall_shear_rate(U_avg, radius=radius, diameter=diameter)
        else:
            rates = 1.0
    radius_used = radius if radius is not None else diameter / 2.0 if diameter is not None else None

    rates = np.atleast_1d(np.array(rates, dtype=float))
    rates = np.sort(rates)

    if G is None or eta_s is None:
        eta_s = beta * eta0
        eta_p = (1.0 - beta) * eta0
        G = eta_p / lam

    if ptt == "linear":
        model = 3
    elif ptt == "exponential":
        model = 4
    else:
        raise ValueError('ptt must be "linear" or "exponential".')

    vemodel = VEModel(
        model=model,
        alam=0,       # standard PTT model
        lam=lam,
        G=G,
        eta_s=eta_s,
        eps=eps,
    )

    rheodata = run_rate_controlled(
        vemodel=vemodel,
        rates=rates,
        only_startup=False,
        flowtype=flowtype,
        rate_for_startup=rates[-1],
        time_startup=40.0 * lam,
        numsteps=1000,
    )

    n1 = rheodata.stress[0, :] - rheodata.stress[3, :]
    tau_xy = rheodata.stress[1, :]
    eta_app = tau_xy / rates
    psi1 = n1 / rates**2

    return {
        "rates": rates,
        "N1": n1,
        "tau_xy": tau_xy,
        "eta_app": eta_app,
        "Psi1": psi1,
        "G": G,
        "eta_s": eta_s,
        "lambda": lam,
        "beta": beta,
        "eta0": eta0,
        "eps": eps,
        "ptt": ptt,
        "U_avg": U_avg,
        "radius": radius,
        "diameter": diameter,
        "radius_used": radius_used,
    }


if __name__ == "__main__":
    result = compute_n1_ptt(
        U_avg=0.1,
        radius=1.0,
        lam=5.5,
        beta=0.4,
        eta0=1.0,
        eps=0.3,
        ptt="exponential",
    )

    print("G =", result["G"])
    print("eta_s =", result["eta_s"])
    print("lambda =", result["lambda"])
    print("eps =", result["eps"])
    print("ptt =", result["ptt"])
    print("U_avg =", result["U_avg"])
    print("radius_used =", result["radius_used"])
    print("rate = 4 * U_avg / radius_used =", result["rates"][0])
    print()
    print("rate,N1,tau_xy,eta_app,Psi1")

    for rate, n1, tau_xy, eta_app, psi1 in zip(
        result["rates"],
        result["N1"],
        result["tau_xy"],
        result["eta_app"],
        result["Psi1"],
    ):
        print(f"{rate:.12g},{n1:.12g},{tau_xy:.12g},{eta_app:.12g},{psi1:.12g}")
