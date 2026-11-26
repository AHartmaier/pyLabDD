"""Restart CLI restores physical parameters instead of substituting defaults."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from pylabdd import GB_dislocations
from pylabdd.gbdd_steady import SteadyHistory, SteadyCriteria, save_steady_metadata


@pytest.fixture
def restart_module():
    path = Path(__file__).resolve().parents[1]/"examples/restart.py"
    spec = importlib.util.spec_from_file_location("restart_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    sim = GB_dislocations(tau0=150., temp=450., Ngbn=5, maxdis=10, maxout=3,
                         Z=10., lambda_=.0025, fcrit=.03, delta=.005, Qact=95000.)
    sim.nout, sim.npu_max, sim.nabs, sim.it_done = 3, 3, 5, 100
    sim.sim_time = np.array([10., 20., 30.])
    sim.gb_node_pos = np.linspace(-.05, .05, 5)
    sim.vout = np.zeros((3, 6, 5))
    sim.vout[:, 3, 1:-1] = [1., 2., 1.]
    sim.pu_out = np.zeros((3, 3, 3))
    sim.pu_out[:, 0, :2] = [.2, .8]
    sim.globout = np.zeros((3, 12))
    sim.globout[:, 1] = .25
    sim.globout[:, 4] = [.01, .02, .03]
    sim.globout[:, 9] = 2
    sim.globout[:, 10] = 5
    path = tmp_path/"source.h5"
    sim.save_hdf5(path)
    history = SteadyHistory(sim.D2, SteadyCriteria(window_time=10., min_absorptions=0))
    history.append(sim)
    save_steady_metadata(path, history.assess(), history=history)
    return path


def test_load_restores_kernel_source_and_only_active_positions(restart_module, checkpoint):
    sim, state = restart_module.load_restart(checkpoint)
    assert sim.Z == 10.
    assert sim.lambda_ == .0025
    assert sim.fcrit == .03
    assert sim.Qact == 95000.
    np.testing.assert_array_equal(state["pudis"], [.2, .8])
    assert state["r_time"] == 30.
    assert state["nabs"] == 5
    assert state["global_state"][4] == .03


def test_missing_metadata_is_not_replaced_by_defaults(restart_module, checkpoint):
    with h5py.File(checkpoint, "a") as f:
        del f["metadata"].attrs["lambda"]
    with pytest.raises(ValueError, match="Missing restart metadata: lambda"):
        restart_module.load_restart(checkpoint)


def test_restart_writes_history_and_provenance_without_overwriting(restart_module, checkpoint, tmp_path, monkeypatch):
    def fake_run(self, **state):
        assert self.Z == 10. and self.lambda_ == .0025 and self.fcrit == .03
        assert state["global_state"][1] == .25
        assert state["global_state"][4] == .03
        self.sim_time = np.array([state["r_time"], self.tfin])
        self.nout, self.npu_max = 2, 2
        self.vout = np.tile(state["vout"], (2, 1, 1))
        self.pu_out = np.zeros((2, 3, 2))
        self.pu_out[:, 0] = state["pudis"]
        self.globout = np.tile(state["global_state"], (2, 1))
        self.globout[1, 4] += .001

    monkeypatch.setattr(GB_dislocations, "run_sim", fake_run)
    output = tmp_path/"continued.h5"
    sim = restart_module.restart_file(checkpoint, duration=10., output=output)
    assert sim.sim_time[-1] == 40.
    with h5py.File(output) as f:
        assert f["restart"].attrs["source"] == str(checkpoint.resolve())
        assert f["metadata"].attrs["Z"] == 10.
        np.testing.assert_array_equal(f["steady_state/history/time"][:], [10., 20., 30., 40.])
    with pytest.raises(FileExistsError):
        restart_module.restart_file(checkpoint, duration=10., output=output)
    with h5py.File(checkpoint) as f:
        assert f["time"][-1] == 30.


def test_preview_does_not_execute_backend(restart_module, checkpoint, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not run dynamics")
    monkeypatch.setattr(GB_dislocations, "run_sim", forbidden)
    sim = restart_module.restart_file(checkpoint, preview=True)
    assert sim.tfin == 50.
