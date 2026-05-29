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
    param, nparam, tau0, temp, Dgp, D2, Ngbn, maxdis, tfin, niter, dtmax, npu_max,
    time_out, xout, vout, pu_out, globout, screenout
):
    nout = 3
    npu_max = 2
    nabs = int(globout[0, 10]) + 1
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

    np.testing.assert_allclose(gb.sim_time[:gb.nout], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(gb.gb_node_pos, [0, 1, 2, 3, 4])

    assert gb.field_data["flux"].shape == (gb.maxout, 5)
    assert gb.glob_data["iteration"].shape == (gb.maxout,)
    assert gb.pu_dis["position"].shape == (gb.maxout, 2)

    np.testing.assert_allclose(gb.field_data["flux"][:gb.nout], 10.0)
    np.testing.assert_allclose(gb.pu_dis["position"][:gb.nout], [[1, 2], [3, 4], [5, 6]])


def fake_restart_calc_gbdd(
    param, nparam, tau0, temp, Dgp, D2, Ngbn, maxdis, tfin, niter, dtmax, npu_max,
    time_out, xout, vout, pu_out, globout, screenout
):
    nout = 4
    npu_max = 2
    nabs = int(globout[0, 10]) + 1
    it_done = 321

    time_out[:nout] = [0.0, 1.0, 2.0, 3.0] + time_out[0]
    xout[:] = np.arange(Ngbn)

    vout[1:nout, 0, :] = 10.0
    vout[1:nout, 3, :] = 20.0

    pu_out[1:nout, 0, :npu_max] = [[3, 4], [5, 6], [7, 8]]

    return it_done, npu_max, nout, nabs, time_out, xout, vout, pu_out, globout


def test_run_sim_restart_keeps_initial_state_at_index_zero_and_continues_at_one():
    gb = GB_dislocations(Ngbn=5, maxdis=4, maxout=10)
    gb.calc_gbdd = fake_restart_calc_gbdd
    restart_vout = np.zeros((gb.nfield, gb.Ngbn))
    restart_vout[0, :] = 99.0
    restart_vout[3, :] = np.arange(gb.Ngbn)

    gb.run_sim(
        pudis=np.array([1.0, 2.0]),
        vout=restart_vout,
        r_time=8.0,
        nabs=5,
    )

    assert gb.nout == 4
    np.testing.assert_allclose(gb.sim_time[:gb.nout], [8.0, 9.0, 10.0, 11.0])
    np.testing.assert_allclose(gb.vout[0, 0, :], 99.0)
    np.testing.assert_allclose(gb.vout[1, 3, :], 20.0)
    np.testing.assert_allclose(gb.pu_out[0, 0, :2], [1.0, 2.0])
    np.testing.assert_allclose(gb.field_data["flux"][1:gb.nout], 10.0)
    np.testing.assert_allclose(gb.pu_dis["position"][1:gb.nout], [[3, 4], [5, 6], [7, 8]])
    assert gb.nabs == 6

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
# -----------------------------------------------------------------------------
# HDF5 I/O tests
# -----------------------------------------------------------------------------

def test_save_hdf5_writes_expected_structure_and_metadata(tmp_path):
    gb = make_simulated_gb()
    out_file = tmp_path / "gbdd_result.h5"

    returned_path = gb.save_hdf5(out_file, nout=2)

    assert returned_path == out_file
    assert out_file.exists()

    import h5py
    with h5py.File(out_file, "r") as h5:
        assert h5.attrs["format"].decode() == "pyLabDD GBDD result"
        assert h5.attrs["format_version"].decode() == "1.0"

        assert set(h5.keys()) == {"metadata", "time", "xval", "gb", "pileup", "global"}
        assert h5["time"].shape == (2,)
        assert h5["xval"].shape == (gb.Ngbn,)
        assert h5["gb/fields"].shape == (2, len(gb.field_names), gb.Ngbn)
        assert h5["pileup/fields"].shape == (2, len(gb.dis_names), gb.npu_max)
        assert h5["global/fields"].shape == (2, len(gb.glob_names))

        gb_names = [name.decode() for name in h5["gb"].attrs["field_names"]]
        pu_names = [name.decode() for name in h5["pileup"].attrs["field_names"]]
        glob_names = [name.decode() for name in h5["global"].attrs["field_names"]]
        assert gb_names == gb.field_names
        assert pu_names == gb.dis_names
        assert glob_names == gb.glob_names

        meta = h5["metadata"].attrs
        assert meta["nout"] == 2
        assert meta["npu_max"] == gb.npu_max
        assert meta["Ngbn"] == gb.Ngbn
        assert meta["maxout"] == gb.maxout
        assert meta["tau0"] == gb.tau0
        assert meta["temperature"] == gb.temp

        np.testing.assert_allclose(h5["time"][()], [0.0, 1.0])
        np.testing.assert_allclose(h5["gb/fields"][:, 0, :], 10.0)
        np.testing.assert_allclose(h5["pileup/fields"][:, 0, :], [[1, 2], [3, 4]])
        np.testing.assert_allclose(h5["global/fields"][:, 0], [1, 2])


def test_read_hdf5_roundtrip_restores_arrays_metadata_and_dicts(tmp_path):
    gb = make_simulated_gb()
    out_file = tmp_path / "roundtrip.h5"
    gb.save_hdf5(out_file, nout=3)

    loaded = GB_dislocations()
    loaded.read_hdf5(out_file)

    assert loaded.nout == 3
    assert loaded.npu_max == gb.npu_max
    assert loaded.nabs == gb.nabs
    assert loaded.it_done == gb.it_done
    assert loaded.Ngbn == gb.Ngbn
    assert loaded.maxdis == gb.maxdis
    assert loaded.maxout == gb.maxout

    assert loaded.field_names == gb.field_names
    assert loaded.dis_names == gb.dis_names
    assert loaded.glob_names == gb.glob_names

    np.testing.assert_allclose(loaded.sim_time, gb.sim_time[:3])
    np.testing.assert_allclose(loaded.gb_node_pos, gb.gb_node_pos)
    np.testing.assert_allclose(loaded.vout, gb.vout[:3, :, :])
    np.testing.assert_allclose(loaded.pu_out, gb.pu_out[:3, :, :gb.npu_max])
    np.testing.assert_allclose(loaded.globout, gb.globout[:3, :])

    np.testing.assert_allclose(loaded.field_data["flux"], gb.vout[:3, 0, :])
    np.testing.assert_allclose(loaded.pu_dis["position"], gb.pu_out[:3, 0, :gb.npu_max])
    np.testing.assert_allclose(loaded.glob_data["iteration"], gb.globout[:3, 0])


def test_save_hdf5_does_not_overwrite_when_requested(tmp_path):
    gb = make_simulated_gb()
    out_file = tmp_path / "existing.h5"
    gb.save_hdf5(out_file)

    with pytest.raises(FileExistsError):
        gb.save_hdf5(out_file, overwrite=False)


def test_save_hdf5_rejects_invalid_nout(tmp_path):
    gb = make_simulated_gb()

    with pytest.raises(ValueError, match="nout must be between"):
        gb.save_hdf5(tmp_path / "bad_nout.h5", nout=gb.maxout + 1)


@pytest.mark.parametrize(
    "attribute,bad_value,match",
    [
        ("vout", np.zeros((3, 5)), "self.vout must have shape"),
        ("pu_out", np.zeros((3, 4)), "self.pu_out must have shape"),
        ("globout", np.zeros((3, 4, 5)), "self.globout must have shape"),
    ],
)
def test_save_hdf5_rejects_invalid_array_dimensions(tmp_path, attribute, bad_value, match):
    gb = make_simulated_gb()
    setattr(gb, attribute, bad_value)

    with pytest.raises(ValueError, match=match):
        gb.save_hdf5(tmp_path / f"bad_{attribute}.h5")


def test_save_hdf5_can_infer_nout_from_nonzero_time_when_nout_is_missing(tmp_path):
    gb = make_simulated_gb()
    gb.nout = None
    gb.sim_time[:] = 0.0
    gb.sim_time[:3] = [0.0, 0.5, 1.0]

    out_file = tmp_path / "inferred_nout.h5"
    gb.save_hdf5(out_file)

    import h5py
    with h5py.File(out_file, "r") as h5:
        assert h5["time"].shape == (3,)
        assert h5["metadata"].attrs["nout"] == 3


def test_read_hdf5_accepts_legacy_file_without_nucleation_barrier(tmp_path):
    gb = make_simulated_gb()
    out_file = tmp_path / "legacy_without_fcrit.h5"
    gb.save_hdf5(out_file)

    import h5py
    with h5py.File(out_file, "a") as h5:
        del h5["metadata"].attrs["nucleation_barrier"]

    loaded = GB_dislocations()
    loaded.read_hdf5(out_file)

    assert loaded.fcrit is None
    np.testing.assert_allclose(loaded.field_data["flux"], gb.vout[:gb.nout, 0, :])
