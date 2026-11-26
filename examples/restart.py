"""Restart a current GBDD HDF5 result with its saved model parameters.

PYTHONPATH=src python examples/restart.py data/result.h5 --preview
PYTHONPATH=src python examples/restart.py data/result.h5 --duration 6e7

Duration is additional microseconds; default: the saved chunk's duration.
Iteration/output budgets default to saved values. Existing files are never
 overwritten. This continues the physical state, not a bitwise solver checkpoint.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from pylabdd import GB_dislocations
from pylabdd.gbdd_steady import SteadyCriteria, SteadyHistory, save_steady_metadata


def load_restart(path):
    """Validate a restart, refusing missing physical metadata or invalid state."""
    path = Path(path)
    required = ("version", "temperature", "tau0", "Ngbn", "length_grain_boundary",
                "grain_size", "maxdis", "maxout", "niter", "dtmax", "shear_modulus",
                "poisson_ratio", "burgers_vector_norm", "gb_thickness",
                "activation_energy", "dislocation_drag", "diffusion_coeff",
                "nucleation_barrier", "lambda", "Z", "nout", "nabs")
    with h5py.File(path, "r") as f:
        meta = dict(f["metadata"].attrs)
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError("Missing restart metadata: " + ", ".join(missing))
        version = meta["version"]
        if isinstance(version, bytes):
            version = version.decode()
        if version != "2.2.6":
            raise ValueError(f"Unsupported model version {version}; expected 2.2.6.")
    sim = GB_dislocations(from_file=path)
    if sim.glob_names != ["iteration", "timestep", "plast_slip_rate", "psr_av", "plastic_slip",
                          "appl_stress", "max_bdot", "max_velocity", "av_velocity",
                          "n_slip_dis", "n_absorbed", "n_gbdis_eff"]:
        raise ValueError("Unexpected global field order.")
    if sim.field_names not in (["flux", "dGdb", "bdot", "bfield", "displacement"],
                               ["flux", "dGdb", "bdot", "bfield", "displacement", "beta"]):
        raise ValueError("Unexpected GB field order; raw b_i/B is required.")
    if not 1 <= sim.nout <= min(len(sim.sim_time), len(sim.globout), len(sim.vout), len(sim.pu_out)):
        raise ValueError("nout is inconsistent with saved arrays.")
    last = sim.nout-1
    row = sim.globout[last].copy()
    if not np.all(np.isfinite(row)):
        raise ValueError("Nonfinite final global state.")
    count = int(round(row[9]))
    if count != row[9] or not 0 <= count <= min(sim.pu_out.shape[2], sim.maxdis):
        raise ValueError("Invalid active pile-up count.")
    positions = sim.pu_dis["position"][last, :count].copy()
    frame = sim.vout[last].copy()
    time = float(sim.sim_time[last])
    if (not np.isfinite(time) or time <= 0 or not np.all(np.isfinite(frame))
            or not np.all(np.isfinite(positions)) or np.any(positions <= 0)
            or np.any(positions > sim.D2) or row[1] <= 0):
        raise ValueError("Invalid restart time or final state.")
    if row[10] != sim.nabs:
        raise ValueError("Metadata nabs disagrees with the final absorption counter.")
    return sim, dict(pudis=positions, vout=frame, r_time=time,
                     nabs=sim.nabs, global_state=row)


def restart_file(source, *, duration=None, output=None, niter=None, maxout=None,
                 screenout=False, preview=False, criteria=None):
    source = Path(source).resolve()
    sim, state = load_restart(source)
    old_duration = state["r_time"]-float(sim.sim_time[0])
    duration = old_duration if duration is None else duration
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite (microseconds).")
    if niter is not None:
        if niter < 2:
            raise ValueError("niter must be at least 2.")
        sim.niter = niter
    if maxout is not None:
        sim.maxout = maxout
    if not 2 <= sim.maxout <= 1000:
        raise ValueError("maxout must be between 2 and 1000.")
    sim.tfin = state["r_time"]+duration
    sim.screenout = screenout
    print(f"Restart: {source}\n"
          f"t={state['r_time']*1e-6:.9g} s, target={sim.tfin*1e-6:.9g} s; "
          f"Npu={len(state['pudis'])}, Nabs={sim.nabs}\n"
          f"T={sim.temp:g} K, tau={sim.tau0:g} MPa, H={sim.D2:g} um, L={sim.Dgp:g} um; "
          f"lambda={sim.lambda_:g} um, Z={sim.Z:g} um, fcrit={sim.fcrit:g}\n"
          f"dtmax={sim.dtmax:g} us, saved dt={state['global_state'][1]:g} us; "
          f"niter={sim.niter}, maxout={sim.maxout}")
    if preview:
        return sim
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = source.with_name(f"{source.stem}_restart_{stamp}.h5")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(source, "r") as f:
        if "steady_state/history" in f:
            h = f["steady_state/history"]
            saved_criteria = SteadyCriteria(**json.loads(h.attrs["criteria_json"]))
            history = SteadyHistory(sim.D2, criteria or saved_criteria)
            history.time = h["time"][:]
            history.global_fields = h["global_fields"][:]
            history.positions = h["positions"][:]
            if not len(history.time) or history.time[-1] != state["r_time"]:
                raise ValueError("Saved convergence history does not end at the restart time.")
        else:
            history = SteadyHistory(sim.D2, criteria or SteadyCriteria(window_time=duration/3))
            history.append(sim)
        anticipation_json = f["steady_state"].attrs.get("anticipation_json") if "steady_state" in f else None
    sim.run_sim(**state)
    if sim.nout < 2 or sim.sim_time[-1] <= state["r_time"]:
        raise RuntimeError("Restart made no physical-time progress.")
    history.append(sim)
    sim.steady_history = history
    sim.steady_report = history.assess()
    output.parent.mkdir(parents=True, exist_ok=True)
    sim.save_hdf5(output, overwrite=False)
    save_steady_metadata(output, sim.steady_report, history=history)
    with h5py.File(output, "a") as f:
        if anticipation_json is not None:
            f["steady_state"].attrs["anticipation_json"] = anticipation_json
        group = f.create_group("restart")
        group.attrs.update(source=str(source), start_time=state["r_time"], requested_duration=duration)
    print(f"Saved {output}; t={sim.sim_time[-1]*1e-6:.9g} s; "
          f"steady={sim.steady_report.converged}: {', '.join(sim.steady_report.reasons)}")
    return sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="HDF5 result to continue")
    parser.add_argument("--duration", type=float, help="additional microseconds; default: saved chunk duration")
    parser.add_argument("--output", type=Path, help="new output file")
    parser.add_argument("--niter", type=int, help="override iteration budget only")
    parser.add_argument("--maxout", type=int, help="override output frame budget only")
    parser.add_argument("--screenout", action="store_true")
    parser.add_argument("--preview", action="store_true", help="validate and print without running")
    parser.add_argument("--criteria-file", type=Path, help="JSON SteadyCriteria override (batch continuation)")
    args = parser.parse_args()
    restart_file(args.source, duration=args.duration, output=args.output,
                 niter=args.niter, maxout=args.maxout, screenout=args.screenout, preview=args.preview,
                 criteria=SteadyCriteria(**json.loads(args.criteria_file.read_text())) if args.criteria_file else None)


if __name__ == "__main__":
    main()
