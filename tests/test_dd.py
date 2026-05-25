"""Unit tests for pylabdd.dislocations.

The tests are intentionally split into two kinds:

1. lightweight unit tests that patch the force routines and therefore do not
   depend on the numerical details of the Fortran backend;
2. one numerical consistency test that compares the package force routine with
   a pure-Python reference implementation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch

import numpy as np
import pytest

import pylabdd as dd


# Material parameters used by the legacy test.
# Units: stress: MPa; length: micron; sim_time: microseconds.
MU = 80.0e3
NU = 0.3
B0 = 0.2e-3
C = MU * B0 / (2.0 * np.pi * (1.0 - NU))
F0 = 10.0
LX = 100.0
LY = 100.0
DT0 = 0.02


def calc_fpk_py(tau0: float, dsl: dd.Dislocations) -> np.ndarray:
    """Pure-Python reference calculation for the Peach-Koehler force."""
    sigdxx = np.zeros(dsl.Ntot)
    sigdyy = np.zeros(dsl.Ntot)
    sigdxy = np.zeros(dsl.Ntot)

    for i in range(dsl.Ntot):
        jl = list(range(0, i)) + list(range(i + 1, dsl.Ntot))
        xpi = dsl.xpos[i]
        ypi = dsl.ypos[i]
        h11 = 0.0
        h22 = 0.0
        h12 = 0.0
        for j in jl:
            x = xpi - dsl.xpos[j]
            y = ypi - dsl.ypos[j]
            h11 += dsl.bx[j] * dsl.sig_xx(x, y)
            h11 += dsl.by[j] * dsl.sig_yy(y, x)
            h22 += dsl.bx[j] * dsl.sig_yy(x, y)
            h22 += dsl.by[j] * dsl.sig_xx(y, x)
            h12 += dsl.bx[j] * dsl.sig_xy(x, y)
            h12 += dsl.by[j] * dsl.sig_xy(y, x)
        sigdxx[i] = h11
        sigdyy[i] = h22
        sigdxy[i] = h12

    sigdxy += tau0
    hh1 = sigdxy * dsl.bx[: dsl.Nmob] + sigdyy * dsl.by[: dsl.Nmob]
    hh2 = sigdxx * dsl.bx[: dsl.Nmob] + sigdxy * dsl.by[: dsl.Nmob]
    return np.array([hh1, -hh2])*dsl.b0


@pytest.fixture
def deterministic_dislocations() -> dd.Dislocations:
    rng_state = np.random.get_state()
    np.random.seed(110)
    try:
        dsl = dd.Dislocations(5, 5, 0.0, C, B0, LX=LX, LY=LY, bc="pbc", dt0=DT0)
        dsl.positions()
    finally:
        np.random.set_state(rng_state)
    return dsl


@pytest.fixture
def simple_dislocations() -> dd.Dislocations:
    return dd.Dislocations(
        4,
        3,
        0.0,
        C,
        B0,
        xpos=np.array([1.0, 3.0, 5.0, 7.0]),
        ypos=np.array([2.0, 4.0, 6.0, 8.0]),
        LX=10.0,
        LY=10.0,
        bc="pbc",
        dt0=DT0,
    )


def fake_force_constant(
    xp: np.ndarray,
    yp: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    tau0: float,
    *args: object,
) -> np.ndarray:
    """Return a backend-like force array before multiplication by C."""
    # Nm is the penultimate argument for both cfpk(..., Nm, Ntot) and
    # cfpk_pbc(..., lx, ly, Nm, Ntot).
    nm = int(args[-2])
    out = np.zeros((2, nm))
    out[0, :] = 1.0 + tau0
    out[1, :] = 0.0
    return out


def fake_force_sign_change(
    xp: np.ndarray,
    yp: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    tau0: float,
    *args: object,
) -> np.ndarray:
    nm = int(args[-2])
    out = np.zeros((2, nm))
    # The force changes sign after the first predictor step. This exercises the
    # branch that halves dx/dy to avoid crossing an equilibrium point.
    out[0, :] = np.where(np.asarray(xp[:nm]) > 1.0, -1.0, 1.0)
    return out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Nd": 0},
        {"Nd": 3, "Nm": -1},
        {"Nd": 3, "Nm": 4},
        {"Nd": 3, "Nm": 2, "LX": 0.0},
        {"Nd": 3, "Nm": 2, "LY": 0.0},
        {"Nd": 3, "Nm": 2, "dmax": 0.0},
        {"Nd": 3, "Nm": 2, "dt0": 0.0},
        {"Nd": 3, "Nm": 2, "bc": "unknown"},
    ],
)
def test_init_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    params = {"Nd": 3, "Nm": 2, "spi1": 0.0, "C": C, "b0": B0}
    params.update(kwargs)
    with pytest.raises(ValueError):
        dd.Dislocations(**params)


@pytest.mark.parametrize("name", ["xpos", "ypos"])
def test_init_rejects_wrong_position_length(name: str) -> None:
    kwargs = {name: np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match=name):
        dd.Dislocations(3, 2, 0.0, C, B0, **kwargs)


def test_positions_initializes_arrays_and_burgers_signs() -> None:
    rng_state = np.random.get_state()
    np.random.seed(123)
    try:
        dsl = dd.Dislocations(6, 4, 0.0, C, B0, LX=20.0, LY=10.0)
        dsl.positions(stol=0.05)
    finally:
        np.random.set_state(rng_state)

    assert dsl.xpos.shape == (6,)
    assert dsl.ypos.shape == (6,)
    assert np.all((dsl.xpos >= 0.0) & (dsl.xpos <= dsl.lx))
    assert np.all((dsl.ypos >= 0.0) & (dsl.ypos <= dsl.ly))
    # For spi1=0, bx starts as +1 and positions() flips every second sign.
    np.testing.assert_allclose(dsl.bx, np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))


def test_positions_rejects_impossible_spacing() -> None:
    dsl = dd.Dislocations(5, 5, 0.0, C, B0, LX=10.0, LY=1.0)
    with pytest.raises(RuntimeError, match="Could not place"):
        dsl.positions(stol=10.0, max_attempts=20)


def test_calc_force_matches_python_reference_for_fixed_bc(
    deterministic_dislocations: dd.Dislocations,
) -> None:
    dsl = deterministic_dislocations
    tau0 = 0.0
    dsl.bc = "fixed"

    force_from_backend = dsl.calc_force(tau0=tau0)
    force_from_python = calc_fpk_py(tau0, dsl)

    np.testing.assert_allclose(force_from_backend, force_from_python, rtol=1.0e-7, atol=1.0e-7)


def test_calc_force_periodic_with_huge_box_approaches_fixed_bc(
    deterministic_dislocations: dd.Dislocations,
) -> None:
    dsl = deterministic_dislocations
    tau0 = 0.0

    dsl.bc = "pbc"
    force_pbc = dsl.calc_force(tau0=tau0, lx=1.0e6 * LX, ly=1.0e6 * LY)

    dsl.bc = "fixed"
    force_fixed = dsl.calc_force(tau0=tau0)

    np.testing.assert_allclose(force_pbc, force_fixed, rtol=1.0e-7, atol=1.0e-7)


def test_calc_force_uses_instance_box_size_by_default(simple_dislocations: dd.Dislocations) -> None:
    calls: list[tuple[float, float]] = []

    def fake_pbc(xp, yp, bx, by, tau0, lx, ly, nm, ntot):
        calls.append((lx, ly))
        return np.zeros((2, nm))

    simple_dislocations.cfpk_pbc = fake_pbc
    simple_dislocations.calc_force(tau0=0.0)

    assert calls == [(simple_dislocations.lx, simple_dislocations.ly)]


def test_fnorm_does_not_modify_positions(simple_dislocations: dd.Dislocations) -> None:
    simple_dislocations.cfpk_pbc = fake_force_constant
    xpos_before = simple_dislocations.xpos.copy()
    ypos_before = simple_dislocations.ypos.copy()

    value = simple_dislocations.fnorm(np.array([0.1, 0.2, 0.3]), tau0=0.0, Nm=3)

    assert value >= 0.0
    np.testing.assert_allclose(simple_dislocations.xpos, xpos_before)
    np.testing.assert_allclose(simple_dislocations.ypos, ypos_before)


def test_fnorm_rejects_wrong_dr_length(simple_dislocations: dd.Dislocations) -> None:
    with pytest.raises(ValueError, match="dr"):
        simple_dislocations.fnorm(np.array([0.1, 0.2]), tau0=0.0, Nm=3)


@pytest.mark.parametrize(
    ("law", "expected"),
    [
        ("viscous", np.array([-2.0 / F0, 0.0, 3.0 / F0])),
        ("powerlaw", np.array([-(2.0 / F0) ** 7, 0.0, (3.0 / F0) ** 7])),
    ],
)
def test_dvel_supported_mobility_laws(law: str, expected: np.ndarray) -> None:
    dsl = dd.Dislocations(3, 3, 0.0, C, B0, dmob=1.0, f0=F0, m=7.0)
    np.testing.assert_allclose(dsl.dvel(np.array([-2.0, 0.0, 3.0]), law), expected)


def test_dvel_rejects_unknown_law(simple_dislocations: dd.Dislocations) -> None:
    with pytest.raises(ValueError, match="not supported"):
        simple_dislocations.dvel(np.array([1.0]), "unknown")


def test_move_disl_updates_only_mobile_dislocations(simple_dislocations: dd.Dislocations) -> None:
    simple_dislocations.cfpk_pbc = fake_force_constant
    xpos_before = simple_dislocations.xpos.copy()
    ypos_before = simple_dislocations.ypos.copy()

    fsp, dt_new = simple_dislocations.move_disl(tau0=0.0, Nm=3, ml="viscous", dt=0.01)

    assert fsp.shape == (3,)
    assert dt_new > 0.0
    assert np.any(simple_dislocations.xpos[:3] != xpos_before[:3])
    np.testing.assert_allclose(simple_dislocations.xpos[3:], xpos_before[3:])
    np.testing.assert_allclose(simple_dislocations.ypos[3:], ypos_before[3:])


def test_move_disl_rejects_invalid_dt(simple_dislocations: dd.Dislocations) -> None:
    with pytest.raises(ValueError, match="dt"):
        simple_dislocations.move_disl(tau0=0.0, Nm=3, ml="viscous", dt=0.0)


def test_move_disl_applies_periodic_wrapping() -> None:
    dsl = dd.Dislocations(
        2,
        2,
        0.0,
        C,
        B0,
        xpos=np.array([9.99, 0.01]),
        ypos=np.array([5.0, 5.0]),
        LX=10.0,
        LY=10.0,
        bc="pbc",
        dmax=1.0,
    )
    dsl.cfpk_pbc = fake_force_constant

    dsl.move_disl(tau0=100.0, Nm=2, ml="viscous", dt=1.0)

    assert np.all((dsl.xpos >= 0.0) & (dsl.xpos <= dsl.lx))


def test_move_disl_fixed_bc_clips_positions() -> None:
    dsl = dd.Dislocations(
        2,
        2,
        0.0,
        C,
        B0,
        xpos=np.array([9.99, 0.01]),
        ypos=np.array([5.0, 5.0]),
        LX=10.0,
        LY=10.0,
        bc="fixed",
        dmax=1.0,
    )
    dsl.cfpk = fake_force_constant

    dsl.move_disl(tau0=100.0, Nm=2, ml="viscous", dt=1.0)

    assert np.all((dsl.xpos >= 0.0) & (dsl.xpos <= dsl.lx))
    assert np.all((dsl.ypos >= 0.0) & (dsl.ypos <= dsl.ly))


def test_relax_disl_stores_equilibrium_positions_as_copies(simple_dislocations: dd.Dislocations) -> None:
    simple_dislocations.cfpk_pbc = lambda xp, yp, bx, by, tau0, lx, ly, nm, ntot: np.zeros((2, nm))

    with patch("matplotlib.pyplot.show"):
        simple_dislocations.relax_disl(ftol=1.0e-6, plot_relax=False, plot_conf=False)

    assert simple_dislocations.xpeq is not simple_dislocations.xpos
    assert simple_dislocations.ypeq is not simple_dislocations.ypos
    np.testing.assert_allclose(simple_dislocations.xpeq, simple_dislocations.xpos)
    np.testing.assert_allclose(simple_dislocations.ypeq, simple_dislocations.ypos)

    old_xpeq = simple_dislocations.xpeq.copy()
    simple_dislocations.xpos += 1.0
    np.testing.assert_allclose(simple_dislocations.xpeq, old_xpeq)


def test_calc_pkforce_returns_expected_shape(simple_dislocations: dd.Dislocations) -> None:
    fpk, xp = simple_dislocations.calc_PKforce(hy=5.0, ngp=25)

    assert fpk.shape == (25,)
    assert xp.shape == (25,)
    assert np.all(np.isfinite(xp))


@pytest.mark.parametrize(
    "call",
    [
        lambda dsl: dsl.calc_PKforce(hy=5.0, ngp=1),
        lambda dsl: dsl.calc_PKforce(hy=5.0, x1=2.0, x2=1.0),
        lambda dsl: dsl.plot_stress(ngp=1),
    ],
)
def test_grid_based_methods_reject_invalid_arguments(simple_dislocations: dd.Dislocations, call) -> None:
    with pytest.raises(ValueError):
        call(simple_dislocations)
