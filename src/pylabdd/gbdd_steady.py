"""Database continuation and physical-time convergence checks for GBDD.

Times and rates use the solver units: microseconds and inverse microseconds.
The analysis helpers use NumPy and h5py; the runner uses the existing backend.
"""
from dataclasses import asdict, dataclass, field
from copy import copy
import json
from pathlib import Path
import warnings

import h5py
import numpy as np


@dataclass(frozen=True)
class SteadyCriteria:
    """Compare the last ``windows`` equal-duration windows.

    ``window_time=None`` uses the entire supplied history divided into windows.
    Spatial tolerance is the maximum change in mean pile-up quantiles, divided
    by grain size. Counts may fluctuate within windows; their means must agree.
    Zero-event arrest does not pass the default minimum-event requirement.
    """
    window_time: float | None = None
    windows: int = 3
    min_samples: int = 5
    min_absorptions: int = 10
    creep_rtol: float = 0.05
    creep_atol: float = 1.e-15
    absorption_rtol: float = 0.15
    absorption_atol: float = 0.0
    count_rtol: float = 0.02
    count_atol: float = 2.0
    spatial_atol: float = 0.02
    quantiles: int = 21

    def __post_init__(self):
        for name in ("windows", "min_samples", "min_absorptions", "quantiles"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.windows < 2 or self.min_samples < 2 or self.quantiles < 2:
            raise ValueError("Need at least two windows, samples, and quantiles.")
        if self.window_time is not None and (not np.isfinite(self.window_time) or self.window_time <= 0):
            raise ValueError("window_time must be positive and finite.")
        for name in ("min_absorptions", "creep_rtol", "creep_atol", "absorption_rtol",
                     "absorption_atol", "count_rtol", "count_atol", "spatial_atol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")


@dataclass
class SteadyReport:
    converged: bool
    reasons: list[str]
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _mean(t, values, start, end):
    """Time-weighted mean with interpolated boundaries, also for vector data."""
    inside = (t > start) & (t < end)
    tx = np.r_[start, t[inside], end]
    if values.ndim == 1:
        vx = np.r_[np.interp(start, t, values), values[inside], np.interp(end, t, values)]
    else:
        vx = np.vstack(([np.interp(start, t, col) for col in values.T],
                        values[inside], [np.interp(end, t, col) for col in values.T]))
    # np.trapezoid is unavailable on older supported NumPy releases.
    return np.sum((vx[1:] + vx[:-1]) * np.diff(tx).reshape((-1,) + (1,) * (vx.ndim-1)) / 2, axis=0) / (end-start)


def assess_steady_state(time, global_fields, positions, grain_size, criteria=None):
    """Assess arrays in the standard GBDD field order, ignoring no padded rows.

    Pass valid rows only (``[:nout]``). Rates are slopes of cumulative slip and
    absorption count, not averages of instantaneous output. Quantiles are
    computed from the first n_slip_dis active entries, excluding padding.
    """
    c = criteria or SteadyCriteria()
    t, g, p = np.asarray(time), np.asarray(global_fields), np.asarray(positions)
    if (t.ndim != 1 or g.ndim != 2 or p.ndim != 2 or len(g) != len(t)
            or len(p) != len(t) or g.shape[1] < 11):
        raise ValueError("Expected time (n,), globals (n, >=11), positions (n, maxdis).")
    if not np.isfinite(grain_size) or grain_size <= 0:
        raise ValueError("grain_size must be positive and finite.")
    if len(t) < 2:
        return SteadyReport(False, ["insufficient history"])
    if (not np.all(np.isfinite(t)) or not np.all(np.isfinite(g[:, [4, 9, 10]]))
            or not np.all(np.isfinite(p)) or np.any(np.diff(t) <= 0)):
        return SteadyReport(False, ["nonfinite data or non-increasing time"])
    counts = np.rint(g[:, 9]).astype(int)
    if np.any(counts < 0) or np.any(counts > p.shape[1]) or np.any(np.diff(g[:, 10]) < 0):
        return SteadyReport(False, ["invalid active counts or decreasing absorption counter"])
    duration = c.window_time or (t[-1]-t[0])/c.windows
    start = t[-1] - duration*c.windows
    if start < t[0] - 1.e-8*duration:
        return SteadyReport(False, ["insufficient physical time"])
    start = max(start, t[0])
    edges = np.linspace(start, t[-1], c.windows+1)
    qlevels = np.linspace(0, 1, c.quantiles)
    q = np.zeros((len(t), c.quantiles))
    for i, count in enumerate(counts):
        if count:
            active = p[i, :count]
            if np.any(active <= 0) or np.any(active > grain_size*(1+1.e-8)):
                return SteadyReport(False, ["invalid active pile-up positions"])
            q[i] = np.quantile(active/grain_size, qlevels)
    creep, absorption, events, number, spatial, samples = [], [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        samples.append(int(np.count_nonzero((t >= a) & (t <= b))))
        creep.append(float((np.interp(b, t, g[:, 4])-np.interp(a, t, g[:, 4]))/(b-a)))
        # Cumulative absorption is a step function, not a continuous signal.
        ia, ib = np.searchsorted(t, [a, b], side="right")-1
        events.append(float(g[ib, 10]-g[ia, 10]))
        absorption.append(events[-1]/(b-a))
        number.append(float(_mean(t, g[:, 9], a, b)))
        spatial.append(_mean(t, q, a, b))
    reasons = []
    if min(samples) < c.min_samples:
        reasons.append("insufficient samples per window")
    if min(events) < c.min_absorptions:
        reasons.append("insufficient absorption events per window")
    metrics = dict(window_edges=edges.tolist(), creep_rates=creep,
                   absorption_rates=absorption, absorption_events=events,
                   mean_counts=number, samples=samples)
    for name, values, rtol, atol in (
        ("creep", creep, c.creep_rtol, c.creep_atol),
        ("absorption", absorption, c.absorption_rtol, c.absorption_atol),
        ("count", number, c.count_rtol, c.count_atol),
    ):
        spread = float(np.ptp(values))
        limit = atol + rtol*max(abs(np.asarray(values)))
        metrics[name+"_spread"] = spread
        metrics[name+"_limit"] = float(limit)
        if spread > limit:
            reasons.append(name+" drift")
    spatial_spread = float(np.max(np.ptp(spatial, axis=0)))
    metrics["spatial_spread"] = spatial_spread
    if spatial_spread > c.spatial_atol:
        reasons.append("spatial distribution drift")
    return SteadyReport(not reasons, reasons, metrics)


class SteadyHistory:
    """Accumulate consecutive chunks; drop their duplicate restart frames."""
    def __init__(self, grain_size, criteria=None):
        self.grain_size = grain_size
        self.criteria = criteria or SteadyCriteria()
        self.time = np.empty(0)
        self.global_fields = np.empty((0, 12))
        self.positions = np.empty((0, 0))

    def append(self, simulation):
        n = simulation.nout
        t = np.asarray(simulation.sim_time[:n])
        if not len(t):
            raise ValueError("Simulation has no valid output.")
        if len(self.time) and t[0] < self.time[-1]:
            raise ValueError("Chunks must be consecutive, without overlapping histories.")
        keep = t > self.time[-1] if len(self.time) else np.ones(len(t), dtype=bool)
        p = np.asarray(simulation.pu_out[:n, 0, :])[keep]
        width = max(p.shape[1], self.positions.shape[1])
        self.positions = np.vstack((np.pad(self.positions, ((0, 0), (0, width-self.positions.shape[1]))),
                                    np.pad(p, ((0, 0), (0, width-p.shape[1])))))
        self.time = np.r_[self.time, t[keep]]
        self.global_fields = np.vstack((self.global_fields, simulation.globout[:n][keep]))
        if self.criteria.window_time and len(self.time):
            start = self.time[-1]-self.criteria.windows*self.criteria.window_time
            i = max(0, np.searchsorted(self.time, start)-1)
            self.time, self.global_fields, self.positions = self.time[i:], self.global_fields[i:], self.positions[i:]

    def assess(self):
        return assess_steady_state(self.time, self.global_fields, self.positions,
                                   self.grain_size, self.criteria)


@dataclass(frozen=True)
class AnticipationOptions:
    """Control donor selection and empirical continuation (not equilibrium).

    Distances are weighted log ratios of H, L, T and positive applied stress.
    Count scaling is (H/H_donor)**count_grain_exponent * tau/tau_donor.
    The default exponent 1 preserves continuum pile-up filling fraction;
    choose e.g. 0.5 for weaker large-grain extrapolation. No extrapolation of
    temperature-dependent b-field amplitude is assumed. ``bfield_scale`` and
    ``count_scale`` allow manual corrections. Unconverged donors are fallback
    guesses only and are always identified in provenance.
    """
    neighbors: int = 3
    allow_unconverged: bool = True
    max_distance: float = 1.5
    distance_weights: tuple = (1.0, 2.0, 4.0, 2.0)
    count_grain_exponent: float = 1.0
    count_scale: float = 1.0
    bfield_scale: float = 1.0
    tail_fraction: float = 0.25
    projection_rcond: float = 1.e-10
    quality_penalty: float = 0.25

    def __post_init__(self):
        if not isinstance(self.neighbors, (int, np.integer)) or self.neighbors < 1 or not 0 < self.tail_fraction <= 1:
            raise ValueError("neighbors >= 1 and 0 < tail_fraction <= 1 required.")
        values = [self.max_distance, self.count_scale, self.projection_rcond]
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Distance, count scale and projection cutoff must be positive finite.")
        if not np.isfinite(self.bfield_scale) or not np.isfinite(self.count_grain_exponent):
            raise ValueError("Scaling parameters must be finite.")
        if not np.isfinite(self.quality_penalty) or self.quality_penalty < 0:
            raise ValueError("quality_penalty must be nonnegative finite.")
        if len(self.distance_weights) != 4 or any(not np.isfinite(v) or v <= 0 for v in self.distance_weights):
            raise ValueError("Four positive finite distance weights required.")


@dataclass
class AnticipatedState:
    positions: np.ndarray
    coefficients: np.ndarray  # raw b_i/B, not reconstructed beta/B
    provenance: dict

    def run_kwargs(self, nfield=6, time_offset=1.e-6):
        """Use the existing restart path as a synthetic initial condition.

        Offset must be positive because Fortran uses time>0 as restart flag.
        Counters use an arbitrary synthetic slip reference, not donor history.
        All derivative/flux channels start at zero and are recomputed by the
        solver. This is an anticipated state, not an exact donor restart.
        """
        if not np.isfinite(time_offset) or time_offset <= 0 or nfield < 4:
            raise ValueError("Positive finite time_offset and at least four channels required.")
        frame = np.zeros((nfield, len(self.coefficients)))
        frame[3] = self.coefficients
        return dict(pudis=self.positions.copy(), vout=frame, r_time=time_offset,
                    nabs=max(0, int(round(float(np.sum(self.coefficients))))))


def _names(group):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in group.attrs["field_names"]]


def _read_case(path):
    with h5py.File(path, "r") as f:
        m = dict(f["metadata"].attrs)
        n = int(m["nout"])
        if n < 2 or n > len(f["time"]):
            raise ValueError("Invalid or insufficient nout.")
        names = _names(f["global"])
        order = [names.index(k) for k in ("iteration", "timestep", "plast_slip_rate", "psr_av",
                 "plastic_slip", "appl_stress", "max_bdot", "max_velocity", "av_velocity",
                 "n_slip_dis", "n_absorbed", "n_gbdis_eff")]
        g = f["global/fields"][:n][:, order]
        p = f["pileup/fields"][:n, _names(f["pileup"]).index("position"), :]
        b = f["gb/fields"][:n, _names(f["gb"]).index("bfield"), :]
        return m, f["time"][:n], g, p, b, f["xval"][:]


def _quality_score(report, criteria):
    """Penalty for measured drift and under-sampling; zero for passing donors."""
    if report.converged:
        return 0.
    m = report.metrics
    if not m:
        return 3.
    score = 0.
    for name in ("creep", "absorption", "count"):
        spread, limit = m[name+"_spread"], m[name+"_limit"]
        score += np.log1p(max(0., spread/max(limit, 1.e-30)-1))
    score += np.log1p(max(0., m["spatial_spread"]/max(criteria.spatial_atol, 1.e-30)-1))
    score += max(0., 1.-min(m["absorption_events"])/max(criteria.min_absorptions, 1))
    score += max(0., 1.-min(m["samples"])/criteria.min_samples)
    return float(score)


def _project_coefficients(x, coefficients, source_width, target_x, target_width, rcond=1.e-10):
    """Fit reconstructed u/B, with zero endpoint coefficients and fixed sum.

    Sample in normalized segment coordinates; widths scale with segment length.
    Retaining the sum preserves the far-field displacement jump. Raw nodal
    coefficients cannot be interpolated as a density when changing the mesh.
    """
    if np.array_equal(x, target_x) and np.isclose(source_width, target_width, rtol=1.e-12, atol=0):
        return coefficients.copy()
    points = np.linspace(-1.5, 1.5, max(201, 4*len(target_x)))
    kernel = lambda nodes, width: 0.5+np.arctan((points[:, None]-nodes[None, :])/width)/np.pi
    values = kernel(x, source_width) @ coefficients
    basis = kernel(target_x[1:-1], target_width)
    size = basis.shape[1]
    if size < 1:
        raise ValueError("Need interior GB nodes for field projection.")
    # Eliminate the final interior coefficient to enforce the sum exactly.
    total = np.sum(coefficients)
    if size == 1:
        solution = np.array([total])
    else:
        partial = np.linalg.lstsq(basis[:, :-1]-basis[:, -1:],
                                  values-total*basis[:, -1], rcond=rcond)[0]
        solution = np.r_[partial, total-partial.sum()]
    return np.r_[0., solution, 0.]


class GBDDDatabase:
    """Rescan HDF5 results on each call; no stale catalog or convergence flags.

    Incompatible material parameters, kernel parameters and model versions are
    excluded. H, L, T, stress and mesh may differ. Only positive-stress cases
    are supported by the current continuation rule.
    """
    def __init__(self, directory, *, recursive=True):
        self.directory = Path(directory)
        self.recursive = recursive
        self.last_scan = []

    def anticipate(self, target, *, options=None, criteria=None):
        o = options or AnticipationOptions()
        c = criteria or SteadyCriteria()
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        if target.tau0 <= 0 or target.D2 <= 10*target.B or target.Dgp <= 0 or target.temp <= 0:
            raise ValueError("Anticipation requires positive stress, lengths and temperature, H > 10B.")
        if target.Ngbn < 3 or target.Ngbn % 2 != 1:
            raise ValueError("Anticipation requires an odd GB mesh with at least three nodes.")
        compatible = dict(shear_modulus=target.mu, poisson_ratio=target.nu,
                          burgers_vector_norm=target.B, gb_thickness=target.delta,
                          activation_energy=target.Qact, dislocation_drag=target.drag,
                          diffusion_coeff=target.Dif_gb, nucleation_barrier=target.fcrit,
                          Z=target.Z, **{"lambda": target.lambda_})
        self.last_scan = []
        donors = []
        pattern = "**/*.h5" if self.recursive else "*.h5"
        for path in sorted(self.directory.glob(pattern)):
            entry = dict(path=str(path.resolve()))
            try:
                m, t, g, p, b, x = _read_case(path)
                version = m.get("version", "")
                version = version.decode() if isinstance(version, bytes) else str(version)
                if version != "2.2.6":
                    raise ValueError("incompatible or unknown model version (requires 2.2.6)")
                for key, value in compatible.items():
                    if key not in m or value is None or not np.isclose(m[key], value, rtol=1.e-6, atol=1.e-12):
                        raise ValueError(f"incompatible or missing {key}")
                source = np.array([m["grain_size"], m["length_grain_boundary"], m["temperature"], m["tau0"]])
                if np.any(source <= 0) or not np.all(np.isfinite(source)):
                    raise ValueError("nonpositive donor parameters")
                distance = float(np.linalg.norm(np.log(np.array([target.D2, target.Dgp, target.temp, target.tau0])/source)
                                                * np.sqrt(o.distance_weights)))
                # New driver outputs retain multi-chunk convergence history.
                # Recompute it with today's criteria rather than trusting a flag.
                with h5py.File(path, "r") as f:
                    if "steady_state/history" in f:
                        h = f["steady_state/history"]
                        report = assess_steady_state(h["time"][:], h["global_fields"][:],
                                                     h["positions"][:], source[0], c)
                    else:
                        report = assess_steady_state(t, g, p, source[0], c)
                score = _quality_score(report, c)
                entry.update(distance=distance, converged=report.converged, reasons=report.reasons,
                             quality_score=score)
                if (not np.all(np.isfinite(b)) or not np.all(np.isfinite(x))
                        or x.shape != b.shape[1:] or np.any(np.diff(x) <= 0)):
                    raise ValueError("invalid GB data")
                if report.reasons and any(r.startswith(("invalid", "nonfinite")) for r in report.reasons):
                    raise ValueError("invalid donor history")
                if distance > o.max_distance or (not report.converged and not o.allow_unconverged):
                    entry["excluded"] = "distance limit or convergence requirement"
                else:
                    donors.append((report.converged, distance, path.stat().st_mtime, path, m, t, g, p, b, x, report, score))
            except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
                entry["excluded"] = str(exc)
            self.last_scan.append(entry)
        if not donors:
            return None
        # Prefer converged donors within the permitted neighborhood; within
        # identical conditions prefer the latest converged/improved file.
        donors.sort(key=lambda d: (not d[0], d[1]+o.quality_penalty*d[11], -d[2], str(d[3])))
        unique, seen = [], set()
        for donor in donors:
            m = donor[4]
            key = tuple(m[k] for k in ("grain_size", "length_grain_boundary", "temperature", "tau0"))
            if key not in seen:
                seen.add(key)
                unique.append(donor)
        if unique[0][0]:
            unique = [d for d in unique if d[0]]
        selected = unique[:o.neighbors]
        if selected[0][1] < 1.e-10:
            selected = selected[:1]
        weights = np.array([1/max(d[1]+o.quality_penalty*d[11], 1.e-6)**2 for d in selected])
        weights /= weights.sum()
        if not all(d[0] for d in selected):
            warnings.warn("Using unconverged database states as initial guesses; see provenance.", RuntimeWarning, stacklevel=2)
        levels = np.linspace(0, 1, 101)
        shape, count, coefficients = np.zeros(101), 0., np.zeros(target.Ngbn)
        provenance = []
        for weight, donor in zip(weights, selected):
            _, distance, _, path, m, t, g, p, b, x, report, score = donor
            a = t[-1]-o.tail_fraction*(t[-1]-t[0])
            active_counts = np.rint(g[:, 9]).astype(int)
            qs = np.zeros((len(t), len(levels)))
            for i, n in enumerate(active_counts):
                if n:
                    qs[i] = np.quantile(p[i, :n]/m["grain_size"], levels)
            scale = (target.D2/m["grain_size"])**o.count_grain_exponent * target.tau0/m["tau0"]
            count += weight*float(_mean(t, g[:, 9], a, t[-1]))*scale
            shape += weight*_mean(t, qs, a, t[-1])
            mean_b = _mean(t, b, a, t[-1])
            source_L = m["length_grain_boundary"]
            coefficients += weight*_project_coefficients(x/source_L, mean_b, m["lambda"]/source_L,
                            np.linspace(-.5, .5, target.Ngbn), target.lambda_/target.Dgp, o.projection_rcond)
            provenance.append(dict(path=str(path.resolve()), weight=float(weight), distance=distance, quality_score=score,
                                   converged=report.converged, report=report.to_dict()))
        n = int(round(count*o.count_scale))
        if n > target.maxdis:
            raise ValueError(f"Anticipated {n} dislocations exceeds maxdis={target.maxdis}; adjust count scaling.")
        positions = target.D2*np.interp((np.arange(n)+.5)/max(n, 1), levels, shape)
        positions = np.clip(positions, np.nextafter(10*target.B, np.inf), target.D2)
        if len(positions) > 1 and np.any(np.diff(positions) <= 0):
            raise ValueError("Predicted positions coincide; choose a different donor or count scale.")
        return AnticipatedState(positions, coefficients*o.bfield_scale,
                               dict(donors=provenance, options=asdict(o),
                                    scan_summary=dict(files=len(self.last_scan),
                                        excluded=sum("excluded" in entry for entry in self.last_scan))))


def save_steady_metadata(path, report, anticipated=None, history=None):
    """Attach reviewable diagnostics to an already saved standard GBDD file."""
    with h5py.File(path, "a") as f:
        group = f.require_group("steady_state")
        group.attrs["converged"] = report.converged
        group.attrs["report_json"] = json.dumps(report.to_dict())
        if anticipated is not None:
            group.attrs["anticipation_json"] = json.dumps(anticipated.provenance)
        if history is not None:
            if "history" in group:
                del group["history"]
            h = group.create_group("history")
            h.attrs["criteria_json"] = json.dumps(asdict(history.criteria))
            h.create_dataset("time", data=history.time, compression="gzip")
            h.create_dataset("global_fields", data=history.global_fields, compression="gzip")
            h.create_dataset("positions", data=history.positions, compression="gzip")


def analytic_pileup(target, fill_fraction=0.3):
    """Continuum fallback with positive, separated positions outside absorption.

    Invert the single-ended pile-up CDF (2 theta + sin(2 theta))/pi.
    The requested filling fraction controls both count and occupied length,
    following the local iterative driver's convention.
    """
    if not 0 < fill_fraction <= 1 or target.tau0 <= 0:
        raise ValueError("Require 0 < fill_fraction <= 1 and positive applied stress.")
    n = int(fill_fraction*np.pi*(1-target.nu)*target.D2*target.tau0/(target.mu*target.B))
    if n > target.maxdis:
        raise ValueError(f"Analytic pile-up count {n} exceeds maxdis={target.maxdis}.")
    cutoff = 10*target.B
    length = fill_fraction*target.D2
    if length <= cutoff:
        return np.empty(0)
    theta = np.linspace(0, np.pi/2, 10001)
    cdf = (2*theta+np.sin(2*theta))/np.pi
    angles = np.interp((np.arange(n)+.5)/max(n, 1), cdf, theta)
    return cutoff+(length-cutoff)*np.sin(angles)**2


def run_until_steady(simulation_kwargs, *, time_inc=1.e8, max_chunks=6,
                     criteria=None, database=None, anticipation_options=None,
                     fill_fraction=0.3, initial_positions=None, initial_time=0.,
                     output_prefix=None, plot=False):
    """Run consecutive chunks, returning simulations with ``steady_report``.

    A fresh database scan supplies a synthetic start; otherwise use the analytic
    fallback (or explicit initial_positions). Only newly simulated history is
    assessed. max_chunks is a budget, not evidence of convergence: exhaustion
    warns and the final report remains false. Each chunk has at most the given
    niter, maxout and time_inc; continuation uses the actual last saved time.
    Reports and donor provenance are saved when output_prefix is supplied.
    """
    from .gbdd import GB_dislocations

    if (not np.isfinite(time_inc) or time_inc <= 0
            or not isinstance(max_chunks, (int, np.integer)) or max_chunks < 1):
        raise ValueError("Positive finite time_inc and max_chunks >= 1 required.")
    if not np.isfinite(initial_time) or initial_time < 0:
        raise ValueError("initial_time must be nonnegative finite.")
    settings = dict(simulation_kwargs)
    settings.pop("tfin", None)
    target = GB_dislocations(tfin=initial_time+time_inc, **settings)
    c = criteria or SteadyCriteria(window_time=time_inc)
    if c.window_time is None:
        raise ValueError("Runner needs an explicit window_time to avoid changing convergence windows.")
    history = SteadyHistory(target.D2, c)
    anticipated = None
    if database is not None and initial_positions is None:
        db = database if isinstance(database, GBDDDatabase) else GBDDDatabase(database)
        anticipated = db.anticipate(target, options=anticipation_options, criteria=c)
    if anticipated is not None:
        state = anticipated.run_kwargs(nfield=target.nfield, time_offset=max(initial_time, 1.e-6))
    else:
        positions = analytic_pileup(target, fill_fraction) if initial_positions is None else initial_positions
        state = dict(pudis=np.asarray(positions, dtype=float))
        if initial_time:
            raise ValueError("Nonzero initial_time requires a database anticipated state.")
    results = []
    for chunk in range(max_chunks):
        start = state.get("r_time", 0.)
        sim = target if chunk == 0 else GB_dislocations(tfin=start+time_inc, **settings)
        sim.tfin = start+time_inc
        sim.run_sim(**state)
        if sim.nout < 2 or sim.sim_time[-1] <= start:
            raise RuntimeError("Simulation made no physical-time progress; increase niter or check the backend.")
        history.append(sim)
        sim.steady_report = history.assess()
        # append() replaces arrays, so a shallow snapshot remains tied to this
        # chunk even when the rolling history advances in subsequent chunks.
        sim.steady_history = copy(history)
        sim.anticipation = anticipated
        results.append(sim)
        print(f"Chunk {chunk+1}: t={sim.sim_time[-1]*1.e-6:.6g} s; "
              f"steady={sim.steady_report.converged}; "
              + ", ".join(sim.steady_report.reasons))
        if plot:
            sim.plot_time_series(["n_slip_dis", "n_absorbed", "psr_av"])
        if output_prefix is not None:
            path = Path(f"{output_prefix}_{chunk+1}.h5")
            path.parent.mkdir(parents=True, exist_ok=True)
            sim.save_hdf5(path)
            save_steady_metadata(path, sim.steady_report, anticipated, history)
        if sim.steady_report.converged:
            break
        last = sim.nout-1
        count = int(round(sim.globout[last, 9]))
        state = dict(pudis=sim.pu_out[last, 0, :count].copy(),
                     vout=sim.vout[last].copy(), r_time=float(sim.sim_time[last]), nabs=sim.nabs)
    if not results[-1].steady_report.converged:
        warnings.warn("Chunk budget exhausted before steady-state criteria passed. "
                      "Inspect steady_report and extend the run/windows; this result is not converged.",
                      RuntimeWarning, stacklevel=2)
    return results
