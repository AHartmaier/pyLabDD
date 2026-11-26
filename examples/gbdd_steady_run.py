"""Database continuation: use --preview to inspect without running dynamics.

Example:
    python examples/gbdd_steady_run.py --preview --grain-size 30 --temp 450
    python examples/gbdd_steady_run.py --grain-size 30 --temp 450 --chunks 12
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from pylabdd import GB_dislocations
from pylabdd.gbdd_steady import (
    AnticipationOptions, GBDDDatabase, SteadyCriteria, analytic_pileup,
    run_until_steady, save_steady_metadata,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path(__file__).resolve().parents[1]/"data")
    parser.add_argument("--grain-size", type=float, default=30.)
    parser.add_argument("--temp", type=float, default=450.)
    parser.add_argument("--length", type=float, default=.5)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--window-time", type=float, default=6.e7, help="microseconds")
    parser.add_argument("--count-exponent", type=float, default=1.)
    parser.add_argument("--strict-donors", action="store_true", help="require measured donor convergence")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    nodes = max(3, round(args.length/.005)+1)
    if nodes % 2 == 0:
        nodes += 1
    settings = dict(tau0=150., temp=args.temp, len_gb_seg=args.length,
                    grain_size=args.grain_size, Ngbn=nodes, maxdis=4000,
                    maxout=400, niter=70_000_000, dtmax=1., screenout=False,
                    mu=44000., nu=.3, B=.00025, delta=.005,
                    Qact=95000., drag=200., Dif_gb=10., lambda_=.0025, Z=10.)
    criteria = SteadyCriteria(window_time=args.window_time, windows=3,
                              min_absorptions=10, creep_rtol=.05,
                              absorption_rtol=.15, count_atol=2., spatial_atol=.02)
    options = AnticipationOptions(neighbors=3, max_distance=1.5,
                                 count_grain_exponent=args.count_exponent,
                                 allow_unconverged=not args.strict_donors)
    database = GBDDDatabase(args.database)
    if args.preview:
        target = GB_dislocations(**settings)
        state = database.anticipate(target, options=options, criteria=criteria)
        if state is None:
            print(f"No admissible donor: analytic fallback with {len(analytic_pileup(target))} dislocations.")
        else:
            print(f"Anticipated {len(state.positions)} dislocations; b_i/B range "
                  f"[{state.coefficients.min():.4g}, {state.coefficients.max():.4g}]")
            for donor in state.provenance["donors"]:
                print(f"  weight={donor['weight']:.3f}, converged={donor['converged']}: {donor['path']}")
                print("   "+", ".join(donor["report"]["reasons"]))
        return
    # Individual chunk files and a final file remain discoverable by the next
    # database scan. Metadata distinguishes guesses from converged results.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = args.database/"continuation"/(
        f"tau150_T{args.temp:g}_H{args.grain_size:g}_L{args.length:g}_{stamp}")
    results = run_until_steady(settings, time_inc=args.window_time,
                max_chunks=args.chunks, criteria=criteria, database=database,
                anticipation_options=options, output_prefix=prefix)
    final = results[-1]
    path = Path(f"{prefix}_final.h5")
    final.save_hdf5(path)
    save_steady_metadata(path, final.steady_report, final.anticipation, final.steady_history)
    print(f"Saved {path}; converged={final.steady_report.converged}")


if __name__ == "__main__":
    main()
