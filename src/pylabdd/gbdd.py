from __future__ import annotations

from collections.abc import Callable, Sequence
from numpy.typing import ArrayLike, NDArray
from typing import Any, TypeAlias
from pathlib import Path
import h5py
import numpy as np
from matplotlib import pyplot as plt

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
        # model parameters
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
        #constitutive parameters
        self.mu: float = 44.e3  # shear modulus (MPa)
        self.nu: float = 0.3  # Poisson ration
        self.B: float = 0.25e-3  # bulk Burgers vector norm (micron)
        self.delta: float = 5.e-4  # GB thickness (micron)
        self.Qact: float = 57.e3  # activation energy for GB diffusion (J/mol)
        self.drag: float = 500.  # dislocation drag coefficient; mobility = B / drag
        self.Dif_gb: float = 10.  # GB diffusion coeff (micron^2/micro s)

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
        self.GBDD_PARAM_KEYS = (
            "mu",
            "nu",
            "B",
            "delta",
            "Qact",
            "drag",
            "Dif_gb",
        )

        self.dis_names: list[str] = ["position", "force", "velocity"]

        self.nfield: int = len(self.field_names)
        self.nglob: int = len(self.glob_names)
        self.npuval: int = len(self.dis_names)

        # Results are initialized by run_sim().
        self.time_out = None
        self.xout = None
        self.vout = None
        self.pu_out = None
        self.globout = None
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
        self.time_out: FloatArray = np.zeros(self.maxout, dtype=np.float64, order="F")
        self.xout: FloatArray = np.zeros(self.Ngbn, dtype=np.float64, order="F")
        self.vout: FloatArray = np.zeros(
            (self.maxout, self.nfield, self.Ngbn), dtype=np.float64, order="F"
        )
        self.pu_out: FloatArray = np.zeros(
            (self.maxout, self.npuval, self.maxdis), dtype=np.float64, order="F"
        )
        self.globout: FloatArray = np.zeros(
            (self.maxout, self.nglob), dtype=np.float64, order="F"
        )

        pv = self._parameter_vector()
        npv = len(pv)

        it_done, npu_max, nout, time_out, xout, vout, pu_out, globout = self.calc_gbdd(
            pv,
            npv,
            self.tau0,
            self.temp,
            self.Dgp,
            self.D2,
            self.Ngbn,
            self.maxdis,
            self.tfin,
            self.niter,
            self.dtmax,
            self.time_out,
            self.xout,
            self.vout,
            self.pu_out,
            self.globout,
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
                    label=f"t={ts[i]:.2f}s",
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

    def save_hdf5(
            self,
            filename: str | Path,
            *,
            nout: int | None = None,
            overwrite: bool = True,
    ) -> Path:
        """
        Save GBDD simulation results to HDF5.

        Assumes the following arrays exist on self:
            self.time_out,
            self.xout,
            self.vout,
            self.pu_out,
            self.globout,

        and simulation/material parameters are stored on self.
        """

        path = Path(filename)
        mode = "w" if overwrite else "x"

        time_out = np.asarray(self.time_out, dtype=np.float64)
        xout = np.asarray(self.xout, dtype=np.float64)
        vout = np.asarray(self.vout, dtype=np.float64)
        pu_out = np.asarray(self.pu_out, dtype=np.float64)
        globout = np.asarray(self.globout, dtype=np.float64)

        if vout.ndim != 3:
            raise ValueError(f"self.vout must have shape (nt, nfields, Ngbn), got {vout.shape}")
        if pu_out.ndim != 3:
            raise ValueError(f"self.pu_out must have shape (nt, 3, maxdis), got {pu_out.shape}")
        if globout.ndim != 2:
            raise ValueError(f"self.globout must have shape (nt, nglob), got {globout.shape}")

        nt = vout.shape[0]

        if nout is None:
            nout = self.nout
        if nout is None or nout <= 0:
            nonzero = np.flatnonzero(time_out)
            nout = int(nonzero[-1] + 1) if nonzero.size else 1
        if nout < 1 or nout > nt:
            raise ValueError(f"nout must be between 1 and {nt}, got {nout}")

        time_out = time_out[:nout]
        vout = vout[:nout, :, :]
        pu_out = pu_out[:nout, :, :]
        globout = globout[:nout, :]

        metadata = {
            "title": "GB dislocation dynamics",
            "version": "2.2.0",
            "author": "Alexander Hartmaier",
            "institution": "Ruhr-Universitaet Bochum, ICAMS",
            "copyright": "Copyright (c) 2013-2026 by the Author. All rights reserved.",
            "license": "GNU General Public License version 3 (GNU GPL-3.0)",
        }

        material = {
            "shear_modulus": self.mu,
            "poisson_ratio": self.nu,
            "temperature": self.temp,
            "dislocation_drag": self.drag,
            "grain_size": self.D2,
            "length_grain_boundary": self.Dgp,
            "burgers_vector_norm_B_micron": self.B,
            "gb_cell_size": self.Dgp / (self.Ngbn - 1),
            "gb_thickness": self.delta,
            "activation_energy": self.Qact,
        }

        simulation = {
            "tau0": self.tau0,
            "tfin": self.tfin,
            "dtmax": self.dtmax,
            "niter": self.niter,
            "it": self.it_done,
            "nout": nout,
            "Ngbn": self.Ngbn,
            "maxdis": self.maxdis,
            "Npu_max": self.npu_max,
            "center_node": int((self.Ngbn + 1) / 2),
            "nfields": vout.shape[1],
            "nglob": globout.shape[1],
            "maxout": nt,
        }

        with h5py.File(path, mode) as h5:
            h5.attrs["format"] = np.bytes_("pyLabDD GBDD result")
            h5.attrs["format_version"] = np.bytes_("1.0")

            meta_group = h5.create_group("metadata")
            self._write_attrs(meta_group, metadata)

            material_group = h5.create_group("material")
            self._write_attrs(material_group, material)

            simulation_group = h5.create_group("simulation")
            self._write_attrs(simulation_group, simulation)

            h5.create_dataset("time", data=time_out, compression="gzip")
            h5.create_dataset("x", data=xout, compression="gzip")

            gb_group = h5.create_group("gb")
            gb_group.attrs["field_names"] = np.asarray(self.field_names, dtype="S")
            gb_group.create_dataset("fields", data=vout, compression="gzip")

            for i, name in enumerate(self.field_names):
                if i < vout.shape[1]:
                    gb_group.create_dataset(name, data=vout[:, i, :], compression="gzip")

            pileup_group = h5.create_group("pileup")
            pileup_group.attrs["field_names"] = np.asarray(self.dis_names, dtype="S")
            pileup_group.create_dataset("fields", data=pu_out, compression="gzip")

            for i, name in enumerate(self.dis_names):
                if i < pu_out.shape[1]:
                    pileup_group.create_dataset(name, data=pu_out[:, i, :], compression="gzip")

            global_group = h5.create_group("global")
            global_group.attrs["field_names"] = np.asarray(self.glob_names, dtype="S")
            global_group.create_dataset("fields", data=globout, compression="gzip")

            for i, name in enumerate(self.glob_names):
                if i < globout.shape[1]:
                    global_group.create_dataset(name, data=globout[:, i], compression="gzip")

        return path

    def read_hdf5(self, filename: str | Path) -> dict[str, Any]:
        """
        Read GBDD HDF5 result file and restore the main arrays on self.

        Sets:
            self.time_out
            self.xout
            self.vout
            self.pu_out
            self.globout
            self.nout

        Returns a dictionary with arrays and metadata.
        """

        path = Path(filename)

        with h5py.File(path, "r") as h5:
            time_out = h5["time"][()]
            xout = h5["x"][()]
            vout = h5["gb/fields"][()]
            pu_out = h5["pileup/fields"][()]
            globout = h5["global/fields"][()]

            metadata = self._read_attrs(h5["metadata"]) if "metadata" in h5 else {}
            material = self._read_attrs(h5["material"]) if "material" in h5 else {}
            simulation = self._read_attrs(h5["simulation"]) if "simulation" in h5 else {}
            attrs = self._read_attrs(h5)

            gb = {
                name: h5[f"gb/{name}"][()]
                for name in self.field_names
                if f"gb/{name}" in h5
            }

            pileup = {
                name: h5[f"pileup/{name}"][()]
                for name in self.dis_names
                if f"pileup/{name}" in h5
            }

            global_data = {
                name: h5[f"global/{name}"][()]
                for name in self.glob_names
                if f"global/{name}" in h5
            }

        self.time_out = time_out
        self.xout = xout
        self.vout = vout
        self.pu_out = pu_out
        self.globout = globout
        self.nout = int(simulation.get("nout", len(time_out)))

        return {
            "time": time_out,
            "x": xout,
            "vout": vout,
            "pu_out": pu_out,
            "globout": globout,
            "gb": gb,
            "pileup": pileup,
            "global": global_data,
            "metadata": metadata,
            "material": material,
            "simulation": simulation,
            "attrs": attrs,
        }

    @staticmethod
    def _normalize_names(names: NameOrNames, valid_names: Sequence[str]) -> list[str]:
        if names is None:
            return []
        if isinstance(names, str):
            if names.lower() in {"all", "a"}:
                return list(valid_names)
            return [names]
        return list(names)

    def _parameter_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, key) for key in self.GBDD_PARAM_KEYS],
            dtype=np.float64,
        )

    def _write_attrs(self, group: h5py.Group, attrs: dict[str, Any]) -> None:
        """Write scalar metadata attributes safely to an HDF5 group."""
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, Path):
                value = str(value)
            if isinstance(value, str):
                group.attrs[key] = np.bytes_(value)
            elif np.isscalar(value):
                group.attrs[key] = value
            else:
                group.attrs[key] = np.asarray(value)

    def _read_attrs(self, group: h5py.Group) -> dict[str, Any]:
        """Read HDF5 attributes and decode byte strings."""
        out: dict[str, Any] = {}
        for key, value in group.attrs.items():
            if isinstance(value, bytes):
                out[key] = value.decode()
            elif isinstance(value, np.bytes_):
                out[key] = value.decode()
            else:
                out[key] = value
        return out
