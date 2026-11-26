"""Numerical and workflow tests for database continuation and steady creep."""
from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest

from pylabdd.gbdd_steady import (
    AnticipatedState, AnticipationOptions, GBDDDatabase, SteadyCriteria,
    SteadyHistory, _project_coefficients, analytic_pileup, assess_steady_state,
    run_until_steady, save_steady_metadata,
)


def target(**overrides):
    values = dict(tau0=150., temp=450., D2=1., Dgp=.1, Ngbn=21,
                  mu=44000., nu=.3, B=.00025, delta=.005, Qact=95000.,
                  drag=200., Dif_gb=10., fcrit=.01, Z=10., lambda_=.0025,
                  maxdis=4000)
    values.update(overrides)
    return SimpleNamespace(**values)


def history():
    t = np.linspace(0., 300., 301)
    g = np.zeros((len(t), 12))
    g[:, 4] = t*1.e-5
    g[:, 9] = 4
    g[:, 10] = np.floor(t)
    p = np.tile([.05, .2, .4, .8], (len(t), 1))
    return t, g, p


def write_case(path, *, drift=False, H=1., T=450., width=.0025):
    t, g, p = history()
    if drift:
        g[:, 4] += 1.e-7*t**2
    p *= H
    with h5py.File(path, "w") as f:
        m = f.create_group("metadata")
        m.attrs.update(dict(nout=len(t), version="2.2.6", grain_size=H,
            length_grain_boundary=.1, temperature=T, tau0=150., shear_modulus=44000.,
            poisson_ratio=.3, burgers_vector_norm=.00025, gb_thickness=.005,
            activation_energy=95000., dislocation_drag=200., diffusion_coeff=10.,
            nucleation_barrier=.01, Z=10., **{"lambda": width}))
        f.create_dataset("time", data=t)
        x = np.linspace(-.05, .05, 21)
        f.create_dataset("xval", data=x)
        gl = f.create_group("global")
        gl.attrs["field_names"] = np.array(["iteration", "timestep", "plast_slip_rate", "psr_av",
            "plastic_slip", "appl_stress", "max_bdot", "max_velocity", "av_velocity",
            "n_slip_dis", "n_absorbed", "n_gbdis_eff"], dtype="S")
        gl.create_dataset("fields", data=g)
        pu = f.create_group("pileup")
        pu.attrs["field_names"] = np.array(["position"], dtype="S")
        pu.create_dataset("fields", data=p[:, None, :])
        gb = f.create_group("gb")
        gb.attrs["field_names"] = np.array(["flux", "dGdb", "bdot", "bfield", "displacement", "beta"], dtype="S")
        b = np.zeros((len(t), 6, len(x)))
        b[:, 3, 1:-1] = np.exp(-(x[1:-1]/.02)**2)
        # Reconstructed channels deliberately differ, catching wrong-channel transfer.
        b[:, 4:] = 123456.
        gb.create_dataset("fields", data=b)


def test_steady_rates_use_cumulative_slopes_and_ignore_instantaneous_noise():
    t, g, p = history()
    g[:, 2:4] = np.random.default_rng(0).normal(size=(len(t), 2))*1000
    report = assess_steady_state(t, g, p, 1., SteadyCriteria(window_time=100.))
    assert report.converged, report.reasons
    np.testing.assert_allclose(report.metrics["creep_rates"], 1.e-5)
    np.testing.assert_allclose(report.metrics["absorption_rates"], 1.)


@pytest.mark.parametrize("change,reason", [
    ("creep", "creep drift"), ("count", "count drift"),
    ("spatial", "spatial distribution drift"), ("absorption", "absorption drift"),
    ("events", "insufficient absorption events per window"),
])
def test_each_criterion_can_prevent_false_convergence(change, reason):
    t, g, p = history()
    if change == "creep":
        g[:, 4] += t*t*1.e-7
    elif change == "count":
        p = np.pad(p, ((0, 0), (0, 4)), constant_values=.9)
        g[t > 150, 9] = 8
    elif change == "spatial":
        p[t > 150] *= .5
    elif change == "absorption":
        g[:, 10] += np.floor(np.maximum(0, t-150))
    else:
        g[:, 10] = 0
    report = assess_steady_state(t, g, p, 1., SteadyCriteria(window_time=100.))
    assert not report.converged
    assert reason in report.reasons


def test_time_weighting_interpolation_padding_and_truncated_history():
    t, g, p = history()
    # Irregular samples; append zero padding to positions, never treat it as active.
    use = np.r_[np.arange(0, 100, 3), np.arange(100, 200, 2), np.arange(200, 301)]
    p = np.pad(p, ((0, 0), (0, 10)))
    report = assess_steady_state(t[use], g[use], p[use], 1., SteadyCriteria(window_time=100.))
    assert report.converged
    assert report.metrics["spatial_spread"] < 1.e-12
    assert not assess_steady_state(t[:200], g[:200], p[:200], 1., SteadyCriteria(window_time=100.)).converged


def test_sparse_absorption_boundaries_do_not_invent_fractional_events():
    t, g, p = history()
    t = t[::10] + .5
    g, p = g[::10], p[::10]
    g[:, 10] = np.floor(t/50)
    report = assess_steady_state(t, g, p, 1., SteadyCriteria(window_time=99., min_absorptions=3))
    assert all(float(n).is_integer() for n in report.metrics["absorption_events"])
    assert not report.converged


def test_history_drops_restart_frame_and_handles_changing_storage_width():
    t, g, p = history()
    h = SteadyHistory(1., SteadyCriteria(window_time=100.))
    for a, b, pad in [(0, 151, 0), (150, 301, 10)]:
        chunk = SimpleNamespace(nout=b-a, sim_time=t[a:b], globout=g[a:b],
                                pu_out=np.pad(p[a:b], ((0, 0), (0, pad)))[:, None, :])
        h.append(chunk)
    assert len(h.time) == 301
    assert h.positions.shape == (301, 14)
    assert h.assess().converged


def test_database_rescans_changed_files_and_prefers_converged_cases(tmp_path):
    write_case(tmp_path/"exact.h5", drift=True)
    write_case(tmp_path/"nearby.h5", H=1.1)
    db = GBDDDatabase(tmp_path)
    state = db.anticipate(target(), options=AnticipationOptions(neighbors=1))
    assert state.provenance["donors"][0]["path"].endswith("nearby.h5")
    # A corrected result at the same path must be discovered without refreshing a catalog.
    write_case(tmp_path/"exact.h5")
    state = db.anticipate(target())
    assert state.provenance["donors"][0]["path"].endswith("exact.h5")
    assert len(state.provenance["donors"]) == 1
    assert len(state.positions) == 4
    assert np.all(np.diff(state.positions) > 0)
    assert state.coefficients.max() < 2  # not displacement/beta


def test_database_fallback_policies_distance_and_incompatible_material(tmp_path):
    write_case(tmp_path/"unstable.h5", drift=True)
    db = GBDDDatabase(tmp_path)
    assert db.anticipate(target(), options=AnticipationOptions(allow_unconverged=False)) is None
    with pytest.warns(RuntimeWarning, match="unconverged"):
        assert db.anticipate(target()) is not None
    assert db.anticipate(target(drag=500.)) is None
    assert "dislocation_drag" in db.last_scan[0]["excluded"]
    assert db.anticipate(target(D2=100.)) is None
    (tmp_path/"broken.h5").write_bytes(b"not hdf5")
    assert db.anticipate(target(drag=500.)) is None
    assert len(db.last_scan) == 2


def test_mesh_projection_preserves_jump_and_reconstructed_displacement():
    x = np.linspace(-.5, .5, 21)
    b = np.exp(-(x/.2)**2)
    b[[0, -1]] = 0
    xx = np.linspace(-.5, .5, 41)
    projected = _project_coefficients(x, b, .025, xx, .025)
    assert projected[0] == projected[-1] == 0
    assert projected.sum() == pytest.approx(b.sum())
    sample = np.linspace(-1., 1., 300)
    kernel = lambda nodes: .5+np.arctan((sample[:, None]-nodes)/.025)/np.pi
    np.testing.assert_allclose(kernel(xx)@projected, kernel(x)@b, atol=1.e-8)


def test_continuation_scaling_capacity_and_synthetic_restart(tmp_path):
    write_case(tmp_path/"source.h5")
    db = GBDDDatabase(tmp_path)
    state = db.anticipate(target(D2=2., Ngbn=41), options=AnticipationOptions(count_grain_exponent=1.))
    assert len(state.positions) == 8
    assert len(state.coefficients) == 41
    assert np.all(state.positions > .0025)
    kwargs = state.run_kwargs()
    assert kwargs["r_time"] > 0
    np.testing.assert_allclose(kwargs["vout"][3], state.coefficients)
    assert np.count_nonzero(kwargs["vout"][:3]) == 0
    with pytest.raises(ValueError, match="maxdis"):
        db.anticipate(target(maxdis=2))


def test_analytic_guess_is_separated_and_outside_absorption_zone():
    p = analytic_pileup(target(D2=100.))
    assert p.min() > .0025
    assert p.max() < 30.
    assert np.all(np.diff(p) > 0)


def test_runner_accumulates_chunks_and_restarts_only_active_positions(monkeypatch):
    from pylabdd.gbdd import GB_dislocations
    calls = []

    def fake_run(self, **state):
        calls.append(state)
        start = state.get("r_time", 0.)
        # Stop early as if hitting niter; next chunk must start here, not at tfin.
        t = np.linspace(start, start+50., 51)
        self.sim_time, self.nout = t, len(t)
        self.globout = np.zeros((len(t), 12))
        self.globout[:, 4] = t*1.e-5
        self.globout[:, 9] = 2
        self.globout[:, 10] = t
        self.nabs = int(t[-1])
        self.pu_out = np.zeros((len(t), 3, 10))
        self.pu_out[:, 0, :2] = [.2, .8]
        self.vout = np.zeros((len(t), 6, self.Ngbn))

    monkeypatch.setattr(GB_dislocations, "run_sim", fake_run)
    results = run_until_steady(dict(grain_size=1., Ngbn=21, maxout=51),
                initial_positions=[.2, .8], time_inc=100., max_chunks=8,
                criteria=SteadyCriteria(window_time=100.))
    assert len(results) == 6
    assert results[-1].steady_report.converged
    assert results[0].steady_history.time[-1] == 50.
    assert calls[1]["r_time"] == 50.
    assert len(calls[1]["pudis"]) == 2
    with pytest.warns(RuntimeWarning, match="budget exhausted"):
        failed = run_until_steady(dict(grain_size=1., Ngbn=21, maxout=51),
                    initial_positions=[.2, .8], time_inc=100., max_chunks=1)
    assert not failed[-1].steady_report.converged


def test_report_and_provenance_persist(tmp_path):
    path = tmp_path/"result.h5"
    write_case(path)
    report = assess_steady_state(*history(), 1.)
    state = AnticipatedState(np.array([.2]), np.zeros(21), {"donors": []})
    save_steady_metadata(path, report, state)
    with h5py.File(path) as f:
        assert f["steady_state"].attrs["converged"]
        assert json.loads(f["steady_state"].attrs["report_json"])["converged"]
        assert "anticipation_json" in f["steady_state"].attrs


def test_database_reassesses_saved_multichunk_history_instead_of_flag(tmp_path):
    path = tmp_path/"chunk.h5"
    write_case(path)
    t, g, p = history()
    h = SteadyHistory(1., SteadyCriteria(window_time=100.))
    h.time, h.global_fields, h.positions = t, g, p
    # The standard arrays contain just the last chunk, too short for three
    # physical windows. Only the saved rolling history establishes convergence.
    with h5py.File(path, "a") as f:
        for name in ("time", "global/fields", "pileup/fields", "gb/fields"):
            data = f[name][-101:]
            del f[name]
            f.create_dataset(name, data=data)
        f["metadata"].attrs["nout"] = 101
    save_steady_metadata(path, h.assess(), history=h)
    db = GBDDDatabase(tmp_path)
    options = AnticipationOptions(allow_unconverged=False)
    assert db.anticipate(target(), criteria=h.criteria, options=options) is not None
    with h5py.File(path, "a") as f:
        f["steady_state"].attrs["converged"] = False
    assert db.anticipate(target(), criteria=h.criteria, options=options) is not None
    with h5py.File(path, "a") as f:
        f["steady_state/history/global_fields"][:, 4] = g[:, 4]+t*t*1.e-7
    assert db.anticipate(target(), criteria=h.criteria, options=options) is None


def test_unconverged_donor_quality_can_outweigh_parameter_distance(tmp_path):
    write_case(tmp_path/"bad_exact.h5", drift=True)
    write_case(tmp_path/"under_sampled_near.h5", H=1.01)
    # Require more events than either run supplies, keeping both in the
    # unconverged class while the second still has stationary rates.
    criteria = SteadyCriteria(min_absorptions=101)
    with pytest.warns(RuntimeWarning):
        state = GBDDDatabase(tmp_path).anticipate(target(), criteria=criteria,
                                                   options=AnticipationOptions(neighbors=1))
    assert state.provenance["donors"][0]["path"].endswith("under_sampled_near.h5")
