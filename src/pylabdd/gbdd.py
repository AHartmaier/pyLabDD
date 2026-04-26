from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
NameOrNames: TypeAlias = str | Sequence[str] | None


class GB_dislocations:
    """Grain-boundary dislocation simulation wrapper.

    The numerical work is performed by the Fortran routine
    ``pylabdd.mod_gbdd.calc_gbdd``. This class prepares the Fortran-order
    arrays, stores the returned data, and provides basic plotting helpers.
    """

    def __init__(
        self,
        tau0: float = 0.0,
        temp: float = 900.0,
        len_gb_seg: float = 0.1,
        grains_size: float = 10.0,
        Ngbn: int = 21,
        maxdis: int = 5000,
        maxout: int = 1000,
        niter: int = 5_000_000,
        tfin: float = 5000.0e6,
        dtmax: float = 400.0,
        screenout: bool = False,
    ) -> None:
        from pylabdd.mod_gbdd import calc_gbdd

        self.calc_gbdd: Callable[..., tuple[Any, ...]] = calc_gbdd

        self.tau0: float = float(tau0)
        self.temp: float = float(temp)
        self.Dgp: float = float(len_gb_seg)
        self.D2: float = float(grains_size)
        self.Ngbn: int = int(Ngbn)
        self.maxdis: int = int(maxdis)
        self.maxout: int = int(maxout)
        self.niter: int = int(niter)
        self.tfin: float = float(tfin)
        self.dtmax: float = float(dtmax)
        self.screenout: bool = bool(screenout)

        if self.Ngbn <= 0:
            raise ValueError("Ngbn must be positive.")
        if self.maxdis <= 0:
            raise ValueError("maxdis must be positive.")
        if self.maxout <= 0:
            raise ValueError("maxout must be positive.")
        if self.niter <= 0:
            raise ValueError("niter must be positive.")

        # Define names for output quantities. These names must match the order
        # used by the Fortran subroutine.
        self.field_names: list[str] = [
            "flux",
            "dGdb",
            "bdot",
            "bfield",
            "displacement",
        ]
        self.glob_names: list[str] = [
            "iteration",
            "timestep",
            "plast_slip_rate",
            "psr_av",
            "plastic_slip",
            "appl_stress",
            "max_bdot",
            "max_velocity",
            "av_velocity",
            "n_slip_dis",
            "n_absorbed",
            "n_gbdis_eff",
        ]
        self.dis_names: list[str] = ["position", "force", "velocity"]

        self.nfield: int = len(self.field_names)
        self.nglob: int = len(self.glob_names)
        self.npuval: int = len(self.dis_names)

        # Results are initialized by run_sim().
        self.field_data: dict[str, FloatArray] | None = None
        self.glob_data: dict[str, FloatArray] | None = None
        self.pu_dis: dict[str, FloatArray] | None = None
        self.gb_node_pos: FloatArray | None = None
        self.sim_time: FloatArray | None = None
        self.it_done: int | None = None
        self.npu_max: int | None = None
        self.nout: int | None = None

    def run_sim(self) -> None:
        """Run the Fortran GB-dislocation simulation and store the results."""
        time_out: FloatArray = np.zeros(self.maxout, dtype=np.float64, order="F")
        xout: FloatArray = np.zeros(self.Ngbn, dtype=np.float64, order="F")
        vout: FloatArray = np.zeros(
            (self.maxout, self.nfield, self.Ngbn), dtype=np.float64, order="F"
        )
        pu_out: FloatArray = np.zeros(
            (self.maxout, self.npuval, self.maxdis), dtype=np.float64, order="F"
        )
        globout: FloatArray = np.zeros(
            (self.maxout, self.nglob), dtype=np.float64, order="F"
        )

        it_done, npu_max, nout, time_out, xout, vout, pu_out, globout = self.calc_gbdd(
            self.tau0,
            self.temp,
            self.Dgp,
            self.D2,
            self.Ngbn,
            self.maxdis,
            self.tfin,
            self.niter,
            self.dtmax,
            time_out,
            xout,
            vout,
            pu_out,
            globout,
            self.screenout,
        )

        nout_i = int(nout)
        npu_max_i = int(npu_max)
        if not 0 <= nout_i <= self.maxout:
            raise ValueError(f"Fortran returned invalid nout={nout_i}.")
        if not 0 <= npu_max_i <= self.maxdis:
            raise ValueError(f"Fortran returned invalid npu_max={npu_max_i}.")

        self.field_data = {
            name: vout[:nout_i, i, :].copy() for i, name in enumerate(self.field_names)
        }
        self.glob_data = {
            name: globout[:nout_i, i].copy() for i, name in enumerate(self.glob_names)
        }
        self.pu_dis = {
            name: pu_out[:nout_i, i, :npu_max_i].copy()
            for i, name in enumerate(self.dis_names)
        }
        self.gb_node_pos = xout.copy()
        self.sim_time = time_out[:nout_i].copy()
        self.it_done = int(it_done)
        self.npu_max = npu_max_i
        self.nout = nout_i

    def _require_results(self) -> tuple[
        dict[str, FloatArray],
        dict[str, FloatArray],
        dict[str, FloatArray],
        FloatArray,
        FloatArray,
        int,
        int,
    ]:
        """Return simulation results or raise a clear error if none exist."""
        if (
            self.field_data is None
            or self.glob_data is None
            or self.pu_dis is None
            or self.gb_node_pos is None
            or self.sim_time is None
            or self.nout is None
            or self.npu_max is None
        ):
            raise RuntimeError("Run run_sim() before accessing or plotting results.")
        return (
            self.field_data,
            self.glob_data,
            self.pu_dis,
            self.gb_node_pos,
            self.sim_time,
            self.nout,
            self.npu_max,
        )

    @staticmethod
    def _normalize_names(names: NameOrNames, valid_names: Sequence[str]) -> list[str]:
        if names is None:
            return []
        if isinstance(names, str):
            if names.lower() in {"all", "a"}:
                return list(valid_names)
            return [names]
        return list(names)

    def plot_time_series(self, names: NameOrNames = None, semi_log: bool = False) -> None:
        """Plot sim_time series of selected global values."""
        _, glob_data, _, _, time, _, _ = self._require_results()
        selected_names = self._normalize_names(names, self.glob_names)
        if not selected_names:
            return

        missing = [name for name in selected_names if name not in glob_data]
        if missing:
            raise KeyError(f"Unknown global field(s): {missing}")

        ts = time * 1.0e-6
        ylabel = "data" if len(selected_names) > 1 else selected_names[0]
        colors = ["k", "r", "b", "g", "c", "m", "orange"]

        for i, field in enumerate(selected_names):
            gv = glob_data[field][1:].copy()
            field_semi_log = semi_log
            if field == "timestep":
                max_gv = np.max(gv) if gv.size else 0.0
                if max_gv != 0.0:
                    gv /= max_gv
            elif field == "max_bdot":
                gv *= 1.0e6
                ylabel = r"$\dot{b}_{max}$ (1/s)"
                field_semi_log = True

            plot_func = plt.semilogy if field_semi_log else plt.plot
            plot_func(
                ts[1:],
                gv,
                marker="none",
                linestyle="-",
                color=colors[i % len(colors)],
                label=field,
            )

        if len(selected_names) > 1:
            plt.legend()
        plt.xlabel("sim_time (s)")
        plt.ylabel(ylabel)
        plt.show()

    def plot_field(self, names: NameOrNames = None, nplot: int = 10) -> None:
        """Plot selected field data over the grain boundary."""
        field_data, _, _, gb_node_pos, time, nout, _ = self._require_results()
        selected_names = self._normalize_names(names, self.field_names)
        if not selected_names:
            return
        if nplot <= 0:
            raise ValueError("nplot must be positive.")

        missing = [name for name in selected_names if name not in field_data]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")

        dt = max(1, nout // nplot)
        ts = time * 1.0e-6

        for field in selected_names:
            for i in range(0, nout, dt):
                yv = field_data[field][i, :].copy()
                if field == "displacement":
                    yv -= np.mean(yv)
                plt.plot(
                    gb_node_pos,
                    yv,
                    marker="none",
                    linestyle="-",
                    color=plt.cm.viridis(i / max(1, nout)),
                    label=f"{field} t={ts[i]:.2f}s",
                )
            plt.legend()
            plt.ylabel(field)
            plt.xlabel(r"x ($\mu$m)")
            plt.show()

    def plot_pile_up(self, nplot: int = 10) -> None:
        """Plot pile-up dislocation positions as a function of sim_time."""
        _, _, pu_dis, _, time, nout, npu_max = self._require_results()
        if nplot <= 0:
            raise ValueError("nplot must be positive.")
        if npu_max <= 0:
            return

        positions = pu_dis["position"]
        dt = max(1, nout // nplot)
        ts = time * 1.0e-6

        for i in range(0, nout, dt):
            active = np.nonzero(positions[i, :])[0]
            if active.size == 0:
                continue
            plt.plot(
                positions[i, active],
                np.full(active.size, ts[i]),
                marker="o",
                linestyle="none",
                color=plt.cm.viridis(i / max(1, nout)),
                label=f"pile-up@t={ts[i]:.2f}s",
            )

        plt.xlabel(r"dislocation position ($\mu$m)")
        plt.ylabel("sim_time (s)")
        plt.xlim((0.0, self.D2 * 1.05))
        plt.show()
