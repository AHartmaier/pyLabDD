# Module pylabdd.dislocations
"""Module pylabdd.dislocations.

Defines :class:`Dislocations`, which stores and evolves a 2D dislocation
configuration.

Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, December 2023
Email: alexander.hartmaier@rub.de
Distributed under GNU General Public License (GPLv3)
August 2025
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias
import logging
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

FloatArray: TypeAlias = NDArray[np.float64]
BoundaryCondition: TypeAlias = Literal["pbc", "fixed"]
MobilityLaw: TypeAlias = Literal["viscous", "powerlaw"]
PKForceFunction: TypeAlias = Callable[..., FloatArray]


def _as_1d_float_array(
    values: ArrayLike | None,
    *,
    length: int,
    default: float = 0.0,
    name: str,
) -> FloatArray:
    """Convert scalar/array input to a 1D float array of a prescribed length."""
    if values is None:
        return np.full(length, default, dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=np.float64)
    else:
        arr = np.ravel(arr).astype(np.float64, copy=False)

    if arr.size != length:
        raise ValueError(f"{name} must contain exactly {length} values, got {arr.size}.")
    return arr.copy()


# define class for dislocations
class Dislocations:
    """Dislocation configuration and simple dislocation dynamics operations.

    Parameters
    ----------
    Nd
        Total number of dislocations.
    Nm
        Number of mobile dislocations.
    spi1
        Slip-plane inclination angle in radians.
    C
        Elastic parameter ``C = mu*b0/(2*pi*(1-nu))``.
    b0
        Norm of Burgers vector.
    dmob
        Dislocation mobility prefactor.
    f0
        Lattice friction stress used by the power-law mobility.
    m
        Stress exponent used by the power-law mobility.
    dmax
        Maximum distance a dislocation can move in one sim_time step.
    xpos, ypos
        Optional initial dislocation positions. Scalars are broadcast to all
        dislocations; arrays must have length ``Nd``.
    LX, LY
        Domain dimensions.
    bc
        Boundary condition, either ``"pbc"`` or ``"fixed"``.
    dt0
        Reference sim_time step.
    """

    def __init__(
        self,
        Nd: int,  # total number of dislocations, stored in self.Ntot
        Nm: int,  # number of mobile dislocations, stored in self.Nmob
        spi1: float,  # slip plane inclination angle in radians
        C: float,  # elastic parameter
        b0: float,  # Burgers vector norm
        dmob: float = 1.0,  # mobility
        f0: float = 0.8,
        m: float = 7.0,
        dmax: float = 0.002,
        xpos: ArrayLike | None = None,
        ypos: ArrayLike | None = None,
        LX: float = 10.0,
        LY: float = 10.0,
        bc: BoundaryCondition = "pbc",
        dt0: float = 0.02,
    ) -> None:
        # Import locally from installed package. Either F90 subroutines from
        # PK_force may be used or slower Python subroutines from PK_force_py as
        # fallback in case of compilation issues.
        from pylabdd import calc_fpk, calc_fpk_pbc

        if Nd <= 0:
            raise ValueError(f"Nd must be positive, got {Nd}.")
        if Nm < 0 or Nm > Nd:
            raise ValueError(f"Nm must satisfy 0 <= Nm <= Nd, got Nm={Nm}, Nd={Nd}.")
        if LX <= 0.0 or LY <= 0.0:
            raise ValueError(f"LX and LY must be positive, got LX={LX}, LY={LY}.")
        if dmax <= 0.0:
            raise ValueError(f"dmax must be positive, got {dmax}.")
        if dt0 <= 0.0:
            raise ValueError(f"dt0 must be positive, got {dt0}.")
        if bc not in ("pbc", "fixed"):
            raise ValueError(f"BC not defined: {bc}")

        self.cfpk: PKForceFunction = calc_fpk
        self.cfpk_pbc: PKForceFunction = calc_fpk_pbc

        self.Ntot: int = int(Nd)  # total number of dislocations
        self.Nmob: int = int(Nm)  # number of mobile dislocations

        # dislocation positions
        self.xpos: FloatArray = _as_1d_float_array(xpos, length=Nd, name="xpos")
        self.ypos: FloatArray = _as_1d_float_array(ypos, length=Nd, name="ypos")
        self.dx: FloatArray = np.zeros(Nd, dtype=np.float64)
        self.dy: FloatArray = np.zeros(Nd, dtype=np.float64)
        self.xpeq: FloatArray | None = None  # equilibrium positions, set in relax_disl
        self.ypeq: FloatArray | None = None

        # slip plane inclination angles
        self.sp_inc: FloatArray = np.ones(Nd, dtype=np.float64) * float(spi1)

        # Burgers vectors
        self.bx: FloatArray = np.cos(self.sp_inc)
        self.by: FloatArray = np.sin(self.sp_inc)

        # calculate dislocation densities
        self.rho: float = Nd / (LX * LY)
        self.rho_m: float = Nm / (LX * LY)

        # dislocation mobility parameters
        self.b0: float = float(b0)
        self.C: float = float(C)
        self.dmob: float = float(dmob)
        self.f0: float = float(f0)
        self.m: float = float(m)
        self.dmax: float = float(dmax)

        # geometry of the domain
        self.lx: float = float(LX)
        self.ly: float = float(LY)
        self.bc: BoundaryCondition = bc

        # numerical parameters
        self.dt0: float = float(dt0)

    # define functions for stress field evaluation
    def sig_xx(self, X: ArrayLike, Y: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        hx = np.multiply(X_arr, X_arr)
        hy = np.multiply(Y_arr, Y_arr)
        hh = hx + hy
        return -self.C * Y_arr * (3.0 * hx + hy) / (hh * hh)

    def sig_yy(self, X: ArrayLike, Y: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        hx = np.multiply(X_arr, X_arr)
        hy = np.multiply(Y_arr, Y_arr)
        hh = hx + hy
        return self.C * Y_arr * (hx - hy) / (hh * hh)

    def sig_xy(self, X: ArrayLike, Y: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        hx = np.multiply(X_arr, X_arr)
        hy = np.multiply(Y_arr, Y_arr)
        hh = hx + hy
        return self.C * X_arr * (hx - hy) / (hh * hh)

    def calc_force(
        self,
        xp: ArrayLike | None = None,
        yp: ArrayLike | None = None,
        Nm: int | None = None,
        tau0: float | None = None,
        lx: float | None = None,
        ly: float | None = None,
    ) -> FloatArray:
        xp_arr = self.xpos if xp is None else np.asarray(xp, dtype=np.float64)
        yp_arr = self.ypos if yp is None else np.asarray(yp, dtype=np.float64)
        Nm_eff = self.Nmob if Nm is None else int(Nm)
        tau0_eff = 0.0 if tau0 is None else float(tau0)
        lx_eff = self.lx if lx is None else float(lx)
        ly_eff = self.ly if ly is None else float(ly)
        self._validate_mobile_count(Nm_eff)

        if self.bc == "pbc":
            fslf = self.cfpk_pbc(
                 xp_arr, yp_arr, self.bx, self.by, 0.0, lx_eff, ly_eff, Nm_eff, self.Ntot
            )
        else:
            #fpk = self.cfpk(xp_arr, yp_arr, self.bx, self.by, tau0_eff, Nm_eff, self.Ntot)
            fslf = self.cfpk(xp_arr, yp_arr, self.bx, self.by, 0.0, Nm_eff, self.Ntot)
        fpk = (self.C * np.asarray(fslf, dtype=np.float64) +
              tau0_eff*np.asarray([self.bx[:Nm_eff], self.by[:Nm_eff]], dtype=np.float64)) * self.b0

        return fpk

    # initialize random dislocation positions
    def positions(self, stol: float = 0.25, *, max_attempts: int = 100_000) -> None:
        """Initialize random dislocation positions.

        Slip planes are selected sequentially with a minimum approximate spacing
        ``stol``. ``max_attempts`` prevents an infinite loop if the requested
        spacing is incompatible with ``Ntot`` and ``ly``.
        """
        if stol < 0.0:
            raise ValueError(f"stol must be non-negative, got {stol}.")

        # select slip planes first by random sequential algorithm
        self.ypos[0] = self.ly * np.random.rand()
        isl = 1
        attempts = 0
        while isl < self.Ntot:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Could not place all slip planes. Reduce stol or Ntot, or increase LY."
                )
            hy = self.ly * np.random.rand()
            flag = np.logical_and(self.ypos[0:isl] < hy + stol, self.ypos[0:isl] > hy - stol)
            if not np.any(flag):
                self.ypos[isl] = hy
                isl += 1

        # place dislocations randomly on slip planes
        hh = np.random.rand(self.Ntot)
        self.xpos = self.lx * hh
        self.ypos += np.sin(self.sp_inc) * hh * self.ly
        ih = np.nonzero(self.ypos < 0.0)[0]
        self.ypos[ih] += self.ly
        ih = np.nonzero(self.ypos > self.ly)[0]
        self.ypos[ih] -= self.ly

        # random positive and negative Burgers vectors could be used here.
        # Change sign of every second dislocation.
        self.bx[0 : self.Ntot : 2] *= -1.0
        self.by[0 : self.Ntot : 2] *= -1.0

    # define force norm for relaxation with L-BFGS-B method
    def fnorm(self, dr: ArrayLike, tau0: float, Nm: int) -> float:
        self._validate_mobile_count(Nm)
        dr_arr = np.asarray(dr, dtype=np.float64)
        if dr_arr.size != Nm:
            raise ValueError(f"dr must contain {Nm} entries, got {dr_arr.size}.")

        # Work on copies: force-norm evaluations must not modify the current configuration.
        xp = self.xpos.copy()
        yp = self.ypos.copy()
        dx = np.multiply(dr_arr, np.abs(self.bx[0:Nm]))
        dy = np.multiply(dr_arr, np.abs(self.by[0:Nm]))
        xp[0:Nm] += dx
        yp[0:Nm] += dy
        if self.bc == "pbc":
            fpk = self.cfpk_pbc(xp, yp, self.bx, self.by, tau0, self.lx, self.ly, Nm, self.Ntot)
        else:
            fpk = self.cfpk(xp, yp, self.bx, self.by, tau0, Nm, self.Ntot)
        fpk_arr = self.C * np.asarray(fpk, dtype=np.float64)
        fsp = np.sum(
            np.multiply(fpk_arr, np.abs(np.array([self.bx[0:Nm], self.by[0:Nm]]))), axis=0
        )
        return float(np.sum(np.abs(fsp)) / Nm)

    # calculate dislocation velocity
    def dvel(self, fsp: ArrayLike, ml: MobilityLaw) -> FloatArray:
        fsp_arr = np.asarray(fsp, dtype=np.float64)
        if ml == "viscous":
            hh = fsp_arr
        elif ml == "powerlaw":
            hh = np.multiply(np.abs(fsp_arr / self.f0) ** self.m, np.sign(fsp_arr))
        else:
            raise ValueError(f'Dislocation mobility "{ml}" not supported.')
        return np.asarray(hh * self.dmob, dtype=np.float64)

    # update dislocation positions
    def move_disl(
        self,
        tau0: float,
        Nm: int,
        ml: MobilityLaw,
        dt: float,
        bc: BoundaryCondition | None = None,
    ) -> tuple[FloatArray, float]:
        bc_eff: BoundaryCondition = self.bc if bc is None else bc
        self._validate_mobile_count(Nm)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")

        if bc_eff == "pbc":
            fpk = self.cfpk_pbc(
                self.xpos, self.ypos, self.bx, self.by, tau0, self.lx, self.ly, Nm, self.Ntot
            )
            fpk_arr = self.C * np.asarray(fpk, dtype=np.float64)
            fpk_arr[1, :] *= -1.0
            # define maximum dislocation displacement
            lb: float | FloatArray = -self.dmax
            ub: float | FloatArray = self.dmax
        elif bc_eff == "fixed":
            fpk = self.cfpk(self.xpos, self.ypos, self.bx, self.by, tau0, Nm, self.Ntot)
            fpk_arr = self.C * np.asarray(fpk, dtype=np.float64)
            # define possible range to move a dislocation within box
            with np.errstate(divide="ignore", invalid="ignore"):
                lower_bound = np.abs(self.xpos[0:Nm] / self.bx[0:Nm])
                upper_bound = np.abs((self.lx - self.xpos[0:Nm]) / self.bx[0:Nm])
            lb = -np.minimum(lower_bound, np.ones(Nm, dtype=np.float64) * self.dmax)
            ub = np.minimum(upper_bound, np.ones(Nm, dtype=np.float64) * self.dmax)
        else:
            raise ValueError(f"BC not defined: {bc_eff}")

        fsp = np.sum(
            np.multiply(fpk_arr, np.abs(np.array([self.bx[0:Nm], self.by[0:Nm]]))), axis=0
        )
        drp = self.dvel(fsp, ml) * dt  # predictor for simple forward Euler integration dr = v*dt
        drp = np.clip(drp, lb, ub)  # speed limit and fixed-box constraint
        dr = np.zeros(self.Ntot, dtype=np.float64)
        dr[0:Nm] += drp  # only Nm dislocations are moved, the rest is fixed

        # sim_time step control diagnostics
        hh = np.abs(drp)
        dr_max = float(np.amax(hh)) if hh.size else 0.0
        nmax = np.nonzero(hh >= self.dmax)[0]
        self.dx = np.multiply(dr, np.abs(self.bx))  # projection on slip plane
        self.dy = np.multiply(dr, np.abs(self.by))
        xp = self.xpos + self.dx
        yp = self.ypos + self.dy

        # verify if force after predictor step has same sign as before
        # if not, a dislocation passes a minimum and needs a reduced sim_time step
        ih = np.array([1, 1], dtype=np.int64)  # initialize such that while is performed at least once
        jc = 0
        while len(ih) > 0 and jc < 5:
            if bc_eff == "pbc":
                fpk = self.cfpk_pbc(xp, yp, self.bx, self.by, tau0, self.lx, self.ly, Nm, self.Ntot)
            else:
                fpk = self.cfpk(xp, yp, self.bx, self.by, tau0, Nm, self.Ntot)
            fpk_arr = self.C * np.asarray(fpk, dtype=np.float64)
            fsp2 = np.sum(
                np.multiply(fpk_arr, np.abs(np.array([self.bx[0:Nm], self.by[0:Nm]]))),
                axis=0,
            )
            sign_product = fsp * fsp2
            ih = np.nonzero(sign_product < 0.0)[0]
            # dislocations with indices ih traversed a minimum and need special treatment
            if jc == 4:
                self.dx[ih] = 0.0
                self.dy[ih] = 0.0
                fsp[ih] = 0.0
            self.dx[ih] *= 0.5
            self.dy[ih] *= 0.5
            xp[ih] = self.xpos[ih] + self.dx[ih]
            yp[ih] = self.ypos[ih] + self.dy[ih]
            jc += 1

        # update positions according to boundary conditions
        if bc_eff == "fixed":
            self.xpos = np.clip(xp, 0.0, self.lx)
            self.ypos = np.clip(yp, 0.0, self.ly)
            bc1 = np.logical_or(self.xpos == 0.0, self.ypos == 0.0)
            bc2 = np.logical_or(self.xpos == self.lx, self.ypos == self.ly)
            ih = np.nonzero(np.logical_or(bc1, bc2))[0]
            fsp[ih] = 0.0
            self.dx[ih] = 0.0
            self.dy[ih] = 0.0
        elif bc_eff == "pbc":
            self.xpos = xp
            self.ypos = yp
            ih = np.nonzero(self.xpos < 0.0)[0]
            self.xpos[ih] += self.lx
            ih = np.nonzero(self.ypos < 0.0)[0]
            self.ypos[ih] += self.ly
            ih = np.nonzero(self.xpos > self.lx)[0]
            self.xpos[ih] -= self.lx
            ih = np.nonzero(self.ypos > self.ly)[0]
            self.ypos[ih] -= self.ly

        # sim_time step control
        if len(nmax) > 2:
            dt = float(np.maximum(self.dt0 * 0.02, dt * 0.9))
        elif dr_max < self.dmax * 0.9:
            dt = float(np.minimum(self.dt0 * 50.0, dt * 1.1))
        return fsp, dt

    # relax all dislocation if True, otherwise only mobile dislocations are relaxed
    def relax_disl(
        self,
        relax_all: bool = False,
        ftol: float = 5.0e-2,
        dt: float = 0.02,
        plot_conf: bool = False,
        plot_relax: bool = True,
    ) -> None:
        # ftol acceptable residual error in force relaxation
        Nm = self.Ntot if relax_all else self.Nmob
        self._validate_mobile_count(Nm)
        if ftol <= 0.0:
            raise ValueError(f"ftol must be positive, got {ftol}.")

        # initialize parameters for relaxation
        fn = 2.0 * ftol
        nl = 0
        nout = 1000
        fout = int(50000 / nout)
        fd: list[float] = []
        while fn > ftol and nl < 50000:
            fsp, dt = self.move_disl(
                0.0, Nm, "viscous", dt
            )  # move dislocations w/o ext. stress
            fn = float(np.sum(np.abs(fsp)) / Nm)
            nl += 1
            if plot_relax and np.mod(nl, fout) == 0:
                fd.append(fn)
            if plot_conf and np.mod(nl, 5000) == 0:
                self.plot_stress()
                print("Iteration:", nl, ", residual force:", fn)
        self.xpeq = self.xpos.copy()  # store equilibrium positions
        self.ypeq = self.ypos.copy()
        if plot_conf:
            self.plot_stress()
            print("Final configuration", nl, fn)
        if plot_relax:
            fd.append(fn)
            fd_arr = np.array(fd, dtype=np.float64)
            plt.semilogy(fd_arr)
            plt.title("Dislocation structure relaxation")
            plt.xlabel("iteration")
            plt.ylabel("PK force norm")
            plt.show()

    # calculate and plot stress field on grid
    def plot_stress(self, ngp: int = 150,
                    show_arrows: bool = True) -> None:
        if ngp <= 1:
            raise ValueError(f"ngp must be larger than 1, got {ngp}.")

        xp = np.linspace(0.0, self.lx, ngp)
        yp = np.linspace(0.0, self.ly, ngp)
        XP, YP = np.meshgrid(xp, yp)
        s11 = np.zeros((ngp, ngp), dtype=np.float64)
        s22 = np.zeros((ngp, ngp), dtype=np.float64)
        s12 = np.zeros((ngp, ngp), dtype=np.float64)
        for i in range(self.Ntot):
            s11 += self.bx[i] * self.sig_xx(XP - self.xpos[i], YP - self.ypos[i])
            s11 -= self.by[i] * self.sig_yy(YP - self.ypos[i], XP - self.xpos[i])
            s22 += self.bx[i] * self.sig_yy(XP - self.xpos[i], YP - self.ypos[i])
            s22 -= self.by[i] * self.sig_xx(YP - self.ypos[i], XP - self.xpos[i])
            s12 += self.bx[i] * self.sig_xy(XP - self.xpos[i], YP - self.ypos[i])
            s12 -= self.by[i] * self.sig_xy(YP - self.ypos[i], XP - self.xpos[i])

        extent = (0.0, self.lx, 0.0, self.ly)
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        fig.subplots_adjust(hspace=0.2)

        for ax in axs:
            ax.set_xlabel(r"x ($\mu$m)")
            ax.set_ylabel(r"y ($\mu$m)")
        axs[0].set_title(r"$\sigma_{xx}$ (MPa)")
        axs[1].set_title(r"$\sigma_{yy}$ (MPa)")
        axs[2].set_title(r"$\sigma_{xy}$ (MPa)")
        axs[0].imshow(s11, origin="lower", extent=extent, vmin=-8.0, vmax=8.0, cmap=cm.RdBu)
        axs[1].imshow(s22, origin="lower", extent=extent, vmin=-8.0, vmax=8.0, cmap=cm.RdBu)
        im = axs[2].imshow(s12, origin="lower", extent=extent, vmin=-8.0, vmax=8.0, cmap=cm.RdBu)
        fig.colorbar(im, ax=axs[2])

        # plot markers for dislocations if not too many
        if self.Ntot < 10:
            for ax in axs:
                ax.scatter(self.xpos, self.ypos, s=30, c="yellow", marker="o")

        # plot arrows for mobile dislocations
        if show_arrows:
            for i in range(self.Nmob):
                dx = float(self.dx[i])
                dy = float(self.dy[i])
                hh = dx * dx + dy * dy
                if hh < self.b0:
                    dx = float(self.bx[i])
                    dy = float(self.by[i])
                for ax in axs:
                    ax.arrow(
                        self.xpos[i],
                        self.ypos[i],
                        4.0 * dx,
                        4.0 * dy,
                        head_width=1.5,
                        width=0.5,
                        head_length=2.0,
                        color="#20ff00",
                    )
        fig.tight_layout()
        plt.show()

    # create line plot with Peach Koehler force
    def calc_PKforce(
        self,
        hy: float,
        ngp: int = 150,
        x1: float = 0.01,
        x2: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Calculate Peach-Koehler force along a given plane.

        Parameters
        ----------
        hy
            y-offset of the line for which the PK force is calculated.
        ngp
            Number of grid points.
        x1
            Start point of the line plot.
        x2
            End point of the line plot. Defaults to ``self.lx``.

        Returns
        -------
        fpk
            PK force in mN/m.
        xp
            x-positions at which the PK force is evaluated, in microns.
        """
        if ngp <= 1:
            raise ValueError(f"ngp must be larger than 1, got {ngp}.")
        if x2 is None:
            x2 = self.lx
        if x2 <= x1:
            raise ValueError(f"x2 must be larger than x1, got x1={x1}, x2={x2}.")

        nd = len(self.xpos)  # number of dislocations in group
        xp = np.linspace(x1, x2, num=ngp)
        yp = np.ones(ngp, dtype=np.float64) * hy
        fpk = np.zeros(ngp, dtype=np.float64)
        for i in range(nd):
            fpk += self.b0 * self.bx[i] * self.sig_xy(xp - self.xpos[i], yp - self.ypos[i])
            fpk -= self.b0 * self.by[i] * self.sig_xy(yp - self.ypos[i], xp - self.xpos[i])
        return fpk * 1000.0, xp

    def _validate_mobile_count(self, Nm: int) -> None:
        if Nm < 0 or Nm > self.Ntot:
            raise ValueError(f"Nm must satisfy 0 <= Nm <= Ntot, got Nm={Nm}, Ntot={self.Ntot}.")
