import matplotlib
matplotlib.use("Agg")

import h5py
import numpy as np

from pylabdd.hdf5_catalog import HDF5Catalog, inspect_hdf5


def make_result(path, temperature=900.0):
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = np.bytes_("pyLabDD GBDD result")
        metadata = h5.create_group("metadata")
        metadata.attrs["temperature"] = temperature
        metadata.attrs["grain_size"] = 10.0
        metadata.attrs["tau0"] = 150.0
        h5.create_dataset("time", data=[0.0, 1.0, 2.0])
        h5.create_dataset("xval", data=[0.0, 0.5, 1.0])
        glob = h5.create_group("global")
        glob.attrs["field_names"] = np.asarray(["iteration", "psr_av"], dtype="S")
        glob.create_dataset("fields", data=[[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        gb = h5.create_group("gb")
        gb.attrs["field_names"] = np.asarray(["bfield"], dtype="S")
        gb.create_dataset("fields", data=np.arange(9.0).reshape(3, 1, 3))


def test_inspect_hdf5_extracts_attributes_and_structure(tmp_path):
    result_file = tmp_path / "gb_tau150_T900_H10_L1.h5"
    make_result(result_file)

    result = inspect_hdf5(result_file)

    assert result["filename_parameters"] == {"tau": 150, "T": 900, "H": 10, "L": 1}
    assert result["attributes"]["/metadata"]["temperature"] == 900.0
    assert result["datasets"]["/global/fields"]["shape"] == [3, 2]


def test_catalog_search_load_and_plot(tmp_path):
    make_result(tmp_path / "first_T900.h5", temperature=900.0)
    make_result(tmp_path / "second_T800.h5", temperature=800.0)

    with HDF5Catalog(tmp_path / "catalog.sqlite") as catalog:
        records = catalog.scan(tmp_path)
        matches = catalog.query(
            {"temperature": 900.0, "tau0": 150.0},
            dataset="global/psr_av",
        )
        metadata = catalog.metadata_dict()

        assert len(records) == 2
        assert len(matches) == 1
        assert len(metadata) == 2
        np.testing.assert_allclose(catalog.load_dataset(matches[0]["path"], "global/psr_av"), [0.1, 0.2, 0.3])
        np.testing.assert_allclose(catalog.load_dataset(matches[0]["path"], "gb/bfield"), np.arange(9.0).reshape(3, 3))
        assert catalog.plot_dataset(matches[0]["path"], "gb/bfield", nplot=2) is not None
        assert catalog.plot_group("psr_av", {"temperature": 900.0}) is not None
