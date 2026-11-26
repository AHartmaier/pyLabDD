"""Audit GBDD results and run a resumable Linux restart queue (at most 10 jobs).

PYTHONPATH=src python examples/restart_jobs.py audit --database data --queue restart_queue
PYTHONPATH=src python examples/restart_jobs.py run --database data --queue restart_queue --workers 10

Queue paths are relative to --database so data and queue can be copied to Linux.
Run resumes queue.json; use audit --refresh to deliberately rebuild the queue.
No simulations run during audit. Failed jobs stay open until run --retry-failed.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py
import numpy as np

from pylabdd.gbdd_steady import SteadyCriteria, assess_steady_state, _read_case, _quality_score
from restart import load_restart


PARAMETERS = ("temperature", "tau0", "grain_size", "length_grain_boundary", "Ngbn",
              "shear_modulus", "poisson_ratio", "burgers_vector_norm", "gb_thickness",
              "activation_energy", "dislocation_drag", "diffusion_coeff",
              "nucleation_barrier", "lambda", "Z", "dtmax", "version")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def inspect_file(path, database, criteria=None):
    """Recompute convergence from saved arrays; never trust a saved boolean."""
    m, t, g, p, b, x = _read_case(path)
    notes = []
    # Pre-fix backend files may label their restart frame t=0, followed by
    # outputs at a large absolute time. Its actual timestamp is unrecoverable;
    # assess only subsequent correctly timed samples, never the fictitious gap.
    if len(t) > 3 and t[0] == 0 and t[1]-t[0] > 5*np.median(np.diff(t[1:])):
        t, g, p = t[1:], g[1:], p[1:]
        notes.append("Excluded initial zero-time frame inconsistent with later output spacing")
    params = {}
    for key in PARAMETERS:
        value = m[key]  # Missing physical information must not merge distinct cases.
        params[key] = value.decode() if isinstance(value, bytes) else (float(value) if np.isscalar(value) and not isinstance(value, str) else value)
    case_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    with h5py.File(path, "r") as f:
        if "steady_state/history" in f:
            h = f["steady_state/history"]
            selected = criteria or SteadyCriteria(**json.loads(h.attrs["criteria_json"]))
            ht, hg, hp = h["time"][:], h["global_fields"][:], h["positions"][:]
            if not len(ht) or ht[-1] != t[-1]:
                raise ValueError("convergence history ends at a different time than the result")
        else:
            selected = criteria or SteadyCriteria(window_time=float(t[-1]-t[0])/3)
            ht, hg, hp = t, g, p
    report = assess_steady_state(ht, hg, hp, m["grain_size"], selected)
    restart_error = None
    try:
        load_restart(path)
    except (ValueError, KeyError, OSError, IndexError) as exc:
        restart_error = str(exc)
    return dict(path=path.relative_to(database).as_posix(), case_id=case_id,
                parameters=params, time_us=float(t[-1]), chunk_time_us=float(t[-1]-t[0]),
                criteria=asdict(selected), report=report.to_dict(),
                quality=_quality_score(report, selected), restart_error=restart_error,
                modified=path.stat().st_mtime, notes=notes)


def export_lists(queue, state):
    """Keep human-readable case lists beside the authoritative JSON queue."""
    columns = ["case_id", "status", "temperature_K", "stress_MPa", "H_um", "L_um",
               "Ngbn", "time_s", "creep_rate_s-1", "source", "reasons"]
    def row(case):
        p, report = case["parameters"], case["report"]
        rates = report.get("metrics", {}).get("creep_rates", [])
        return [case["case_id"], case["status"], p["temperature"], p["tau0"], p["grain_size"],
                p["length_grain_boundary"], p["Ngbn"], case["time_us"]*1.e-6,
                float(np.mean(rates))*1.e6 if rates else "", case["source"],
                "; ".join(report["reasons"] + ([case["error"]] if case.get("error") else []))]
    for name, cases in (("successful_cases.csv", [c for c in state["cases"] if c["status"] == "converged"]),
                        ("open_jobs.csv", [c for c in state["cases"] if c["status"] != "converged"])):
        temporary = queue/(name+".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(row(c) for c in cases)
        temporary.replace(queue/name)
    with (queue/"all_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "case_id", "converged", "time_s", "reasons", "notes", "restart_error"])
        for r in state["files"]:
            writer.writerow([r["path"], r["case_id"], r["report"]["converged"], r["time_us"]*1.e-6,
                             "; ".join(r["report"]["reasons"]), "; ".join(r["notes"]), r["restart_error"] or ""])
    successes = [c for c in state["cases"] if c["status"] == "converged"]
    lines = ["# Converged GBDD cases", "", f"Audit/update: {state['updated']}", "",
             f"{len(successes)} successful cases; {len(state['cases'])-len(successes)} open cases.", "",
             "Criteria: three physical-time windows; creep range <= 5%, absorption range <= 15%,",
             "mean count range <= 2 + 2%, position quantile range <= 2% of grain size;",
             "at least 5 samples and 10 absorbed dislocations per window (unless explicitly overridden).", "",
             "Full parameters, individual file assessments and exact per-case criteria are in queue.json.", "",
             "| T (K) | stress (MPa) | H (um) | L (um) | creep rate (s^-1) | Result |",
             "|---:|---:|---:|---:|---:|---|"]
    for c in successes:
        p = c["parameters"]
        rate = np.mean(c["report"]["metrics"]["creep_rates"])*1.e6
        lines.append(f"| {p['temperature']:g} | {p['tau0']:g} | {p['grain_size']:g} | {p['length_grain_boundary']:g} | {rate:.6g} | {c['source']} |")
    if not successes:
        lines += ["", "No case currently passes all criteria. This includes cases with insufficient event evidence."]
    (queue/"successful_cases.md").write_text("\n".join(lines)+"\n")


def save_queue(queue, state):
    state["updated"] = utc_now()
    atomic_json(queue/"queue.json", state)
    export_lists(queue, state)


def audit(database, queue, criteria=None):
    records, errors = [], []
    for path in sorted(database.rglob("*.h5")):
        # Incomplete atomic output files end in .partial, not .h5.
        try:
            records.append(inspect_file(path, database, criteria))
        except (ValueError, KeyError, OSError, IndexError, TypeError) as exc:
            errors.append(dict(path=path.relative_to(database).as_posix(), error=str(exc)))
    grouped = {}
    for record in records:
        grouped.setdefault(record["case_id"], []).append(record)
    cases = []
    for case_id, candidates in grouped.items():
        successful = [r for r in candidates if r["report"]["converged"]]
        valid = [r for r in candidates if not r["restart_error"]]
        pool = successful or valid or candidates
        chosen = min(pool, key=lambda r: (r["quality"], -r["time_us"], -r["modified"], r["path"]))
        case = dict(case_id=case_id, parameters=chosen["parameters"], source=chosen["path"],
                    criteria=chosen["criteria"], report=chosen["report"], time_us=chosen["time_us"],
                    chunk_time_us=chosen["chunk_time_us"], iterations=0, attempts=0,
                    status="converged" if successful else ("failed" if chosen["restart_error"] else "pending"),
                    error=chosen["restart_error"], output=None)
        cases.append(case)
    cases.sort(key=lambda c: tuple(c["parameters"][k] for k in ("temperature", "tau0", "grain_size", "length_grain_boundary")))
    state = dict(format_version=1, created=utc_now(), database_hint=str(database),
                 files=records, unreadable_files=errors, cases=cases)
    save_queue(queue, state)
    print(f"Audited {len(records)+len(errors)} files: {len(records)} readable, {len(errors)} unassessable; "
          f"{sum(c['status']=='converged' for c in cases)} converged / {len(cases)} distinct cases.")
    return state


def adjust_window(case):
    """Expand sampling windows when event evidence is inadequate, not tolerances."""
    metrics = case["report"].get("metrics", {})
    if "insufficient absorption events per window" not in case["report"]["reasons"] or not metrics:
        return
    current = case["criteria"]["window_time"]
    rate = float(np.mean(metrics["absorption_rates"]))
    required = 1.5*case["criteria"]["min_absorptions"]/rate if rate > 0 else 2*current
    case["criteria"]["window_time"] = max(current, min(2*current, required))


def accept_result(case, record):
    if record["case_id"] != case["case_id"]:
        raise ValueError("Restart output parameters do not match queued case")
    if record["time_us"] <= case["time_us"]:
        raise ValueError("Restart made no physical-time progress")
    case.update(source=record["path"], time_us=record["time_us"], report=record["report"],
                status="converged" if record["report"]["converged"] else "pending", error=None,
                iterations=case["iterations"]+1)
    case["output"] = None


def parent_guard(parent):
    """Linux workers die even if the queue manager receives SIGKILL."""
    if sys.platform.startswith("linux"):
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
            os._exit(127)
        if os.getppid() != parent:
            os.kill(os.getpid(), signal.SIGKILL)


def stop_children(active):
    for process, stream, case in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+5
    while any(p.poll() is None for p, _, _ in active.values()) and time.monotonic() < deadline:
        time.sleep(.1)
    for process, stream, case in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        stream.close()
        case["status"] = "pending"


def run_queue(database, queue, state, *, workers=10, duration=None, niter=None,
              maxout=None, max_rounds=None, retry_failed=False):
    if not 1 <= workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    for case in state["cases"]:
        # A completed atomic result can outlive an interrupted parent process.
        if case["status"] == "running":
            output = database/case["output"] if case.get("output") else None
            if output and output.exists():
                try:
                    accept_result(case, inspect_file(output, database, SteadyCriteria(**case["criteria"])))
                except Exception as exc:
                    case.update(status="failed", error=f"Recovery: {exc}")
            else:
                case["status"] = "pending"
        if retry_failed and case["status"] == "failed":
            case.update(status="pending", error=None)
    active, round_counts = {}, {}
    script = Path(__file__).resolve().with_name("restart.py")
    environment = os.environ.copy()
    # Avoid ten workers each starting threaded BLAS/OpenMP pools.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    old_handler = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while True:
            # Round-robin by restarts in this invocation prevents slow cases
            # occupying the same ten slots forever while later cases starve.
            for case in sorted(state["cases"], key=lambda c: (round_counts.get(c["case_id"], 0), c["iterations"], c["case_id"])):
                if len(active) >= workers:
                    break
                if case["status"] != "pending" or (max_rounds is not None and round_counts.get(case["case_id"], 0) >= max_rounds):
                    continue
                adjust_window(case)
                case["attempts"] += 1
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                relative = Path("batch_restarts")/case["case_id"]/f"restart_{case['attempts']}_{stamp}.h5"
                output = database/relative
                output.parent.mkdir(parents=True, exist_ok=True)
                criteria_path = output.with_suffix(".criteria.json")
                atomic_json(criteria_path, case["criteria"])
                log = output.with_suffix(".log")
                command = [sys.executable, str(script), str(database/case["source"]),
                           "--output", str(output.with_suffix(".partial")),
                           "--duration", str(duration or case["chunk_time_us"]), "--criteria-file", str(criteria_path)]
                if niter is not None:
                    command += ["--niter", str(niter)]
                if maxout is not None:
                    command += ["--maxout", str(maxout)]
                case.update(status="running", output=relative.as_posix(), log=log.relative_to(database).as_posix())
                # Persist intent before launch. Result is renamed only after a
                # successful child exit, so crash debris cannot enter an audit.
                save_queue(queue, state)
                stream = log.open("a")
                try:
                    parent = os.getpid()
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=environment, start_new_session=True,
                                               preexec_fn=lambda: parent_guard(parent))
                except OSError as exc:
                    stream.close()
                    case.update(status="failed", error=str(exc))
                    save_queue(queue, state)
                    continue
                active[case["case_id"]] = (process, stream, case)
                round_counts[case["case_id"]] = round_counts.get(case["case_id"], 0)+1
                print(f"Started {case['case_id']} (active {len(active)}/{workers})", flush=True)
            if not active:
                break
            time.sleep(.5)
            for key, (process, stream, case) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[key]
                output = database/case["output"]
                try:
                    if code:
                        raise RuntimeError(f"Restart exited {code}; see {case['log']}")
                    record = inspect_file(output.with_suffix(".partial"), database, SteadyCriteria(**case["criteria"]))
                    output.with_suffix(".partial").replace(output)
                    record["path"] = output.relative_to(database).as_posix()
                    accept_result(case, record)
                except Exception as exc:
                    case.update(status="failed", error=str(exc))
                save_queue(queue, state)
                print(f"Finished {key}: {case['status']}", flush=True)
    except KeyboardInterrupt:
        print("Stopping workers; preserving open jobs.", flush=True)
    finally:
        stop_children(active)
        signal.signal(signal.SIGTERM, old_handler)
        save_queue(queue, state)
    open_count = sum(c["status"] != "converged" for c in state["cases"])
    print(f"Queue saved: {open_count} open cases, {sum(c['status']=='failed' for c in state['cases'])} failed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "run"])
    parser.add_argument("--database", type=Path, default=Path("data"))
    parser.add_argument("--queue", type=Path, default=Path("restart_queue"))
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--duration", type=float, help="additional microseconds per restart; default saved chunk duration")
    parser.add_argument("--niter", type=int)
    parser.add_argument("--maxout", type=int)
    parser.add_argument("--max-rounds", type=int, help="optional restarts per case in this invocation; default unlimited")
    parser.add_argument("--criteria-file", type=Path, help="audit criteria JSON override")
    parser.add_argument("--refresh", action="store_true", help="rebuild queue from all results")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    for name in ("duration", "niter", "maxout", "max_rounds"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive finite")
    if args.niter is not None and args.niter < 2:
        parser.error("--niter must be at least 2")
    if args.maxout is not None and not 2 <= args.maxout <= 1000:
        parser.error("--maxout must be between 2 and 1000")
    database, queue = args.database.resolve(), args.queue.resolve()
    if not database.is_dir():
        parser.error("database directory does not exist")
    queue.mkdir(parents=True, exist_ok=True)
    with (queue/"queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another audit/runner already holds this queue lock")
        if args.action == "audit" or not (queue/"queue.json").exists():
            if (queue/"queue.json").exists() and not args.refresh:
                parser.error("queue exists; use --refresh to rebuild, or run to resume")
            criteria = SteadyCriteria(**json.loads(args.criteria_file.read_text())) if args.criteria_file else None
            state = audit(database, queue, criteria)
        else:
            if args.refresh or args.criteria_file:
                parser.error("use audit --refresh to change the saved queue/criteria")
            state = json.loads((queue/"queue.json").read_text())
            if state.get("format_version") != 1:
                parser.error("unsupported queue format")
        if args.action == "run":
            run_queue(database, queue, state, workers=args.workers, duration=args.duration,
                      niter=args.niter, maxout=args.maxout, max_rounds=args.max_rounds,
                      retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
