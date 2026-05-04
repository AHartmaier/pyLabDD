import matplotlib
matplotlib.use("Agg")
import pytest
import numpy as np
from unittest.mock import patch
from pylabdd import GB_dislocations


def test_init_sets_basic_parameters():
    gb = GB_dislocations(Ngbn=11, maxdis=100, maxout=20)
    assert gb.Ngbn == 11
    assert gb.maxdis == 100
    assert gb.maxout == 20
    assert gb.field_data is None


@pytest.mark.parametrize("kwargs", [
    {"Ngbn": 0},
    {"maxdis": 0},
    {"maxout": 0},
    {"niter": 0},
])
def test_init_rejects_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        GB_dislocations(**kwargs)

@pytest.mark.parametrize("method,args", [
    ("plot_time_series", ("iteration",)),
    ("plot_field", ("flux",)),
    ("plot_pile_up", ()),
])
def test_plot_methods_require_simulation(method, args):
    gb = GB_dislocations()
    with pytest.raises(RuntimeError, match="Run run_sim"):
        getattr(gb, method)(*args)

def test_normalize_names_none():
    assert GB_dislocations._normalize_names(None, ["a", "b"]) == []

def test_normalize_names_single():
    assert GB_dislocations._normalize_names("flux", ["flux"]) == ["flux"]

def test_normalize_names_all():
    assert GB_dislocations._normalize_names("all", ["a", "b"]) == ["a", "b"]

def test_normalize_names_sequence():
    assert GB_dislocations._normalize_names(["a", "b"], ["a", "b"]) == ["a", "b"]

def fake_calc_gbdd(
    param, nparam, tau0, temp, Dgp, D2, Ngbn, maxdis, tfin, niter, dtmax,
    time_out, xout, vout, pu_out, globout, screenout
):
    nout = 3
    npu_max = 2
    nabs = 1
    it_done = 123

    time_out[:nout] = [0.0, 1.0, 2.0]
    xout[:] = np.arange(Ngbn)

    vout[:nout, 0, :] = 10.0       # flux
    vout[:nout, 1, :] = 20.0       # dGdb

    globout[:nout, 0] = [1, 2, 3]  # iteration
    globout[:nout, 1] = [0.1, 0.2, 0.3]

    pu_out[:nout, 0, :npu_max] = [[1, 2], [3, 4], [5, 6]]
    pu_out[:nout, 1, :npu_max] = 100.0
    pu_out[:nout, 2, :npu_max] = 200.0

    return it_done, npu_max, nout, nabs, time_out, xout, vout, pu_out, globout

def test_run_sim_maps_fortran_output():
    gb = GB_dislocations(Ngbn=5, maxdis=4, maxout=10)
    gb.calc_gbdd = fake_calc_gbdd

    gb.run_sim()

    assert gb.it_done == 123
    assert gb.nout == 3
    assert gb.npu_max == 2

    np.testing.assert_allclose(gb.sim_time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(gb.gb_node_pos, [0, 1, 2, 3, 4])

    assert gb.field_data["flux"].shape == (3, 5)
    assert gb.glob_data["iteration"].shape == (3,)
    assert gb.pu_dis["position"].shape == (3, 2)

    np.testing.assert_allclose(gb.field_data["flux"], 10.0)
    np.testing.assert_allclose(gb.pu_dis["position"], [[1, 2], [3, 4], [5, 6]])

def fake_bad_nout(*args):
    time_out, xout, vout, pu_out, globout = args[-5:]
    return 0, 0, 999999, 1, time_out, xout, vout, pu_out, globout

def fake_bad_npu(*args):
    time_out, xout, vout, pu_out, globout = args[-5:]
    return 0, 999999, 1, 1, time_out, xout, vout, pu_out, globout

def test_run_sim_rejects_invalid_nout():
    gb = GB_dislocations(maxout=5)
    gb.calc_gbdd = fake_bad_nout
    with pytest.raises(ValueError, match="nout"):
        gb.run_sim()

def test_run_sim_rejects_invalid_npu_max():
    gb = GB_dislocations(maxdis=5)
    gb.calc_gbdd = fake_bad_npu
    with pytest.raises(ValueError, match="npu_max"):
        gb.run_sim()

def make_simulated_gb():
    gb = GB_dislocations(Ngbn=5, maxdis=4, maxout=10)
    gb.calc_gbdd = fake_calc_gbdd
    gb.run_sim()
    return gb

def test_plot_time_series_runs():
    gb = make_simulated_gb()
    with patch("matplotlib.pyplot.show"):
        gb.plot_time_series("iteration")

def test_plot_field_runs():
    gb = make_simulated_gb()
    with patch("matplotlib.pyplot.show"):
        gb.plot_field("flux", nplot=2)

def test_plot_pile_up_runs():
    gb = make_simulated_gb()
    with patch("matplotlib.pyplot.show"):
        gb.plot_pile_up(nplot=2)

def test_plot_time_series_rejects_unknown_name():
    gb = make_simulated_gb()
    with pytest.raises(KeyError):
        gb.plot_time_series("does_not_exist")

def test_plot_field_rejects_unknown_name():
    gb = make_simulated_gb()
    with pytest.raises(KeyError):
        gb.plot_field("does_not_exist")