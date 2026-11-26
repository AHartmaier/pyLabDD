"""Audit grouping, legacy times, queue limits and failure/recovery semantics."""
import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pylabdd import GB_dislocations


@pytest.fixture
def jobs(monkeypatch):
    examples = Path(__file__).resolve().parents[1]/"examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("restart_jobs_test", examples/"restart_jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(path, *, temp=500., drift=False, broken_first_time=False):
    sim = GB_dislocations(tau0=150., temp=temp, grain_size=1., Ngbn=5,
                         maxout=301, niter=1000, dtmax=1.)
    sim.nout, sim.npu_max, sim.nabs, sim.it_done = 301, 2, 300, 1000
    sim.sim_time = np.arange(301, dtype=float)+1000.
    if broken_first_time:
        sim.sim_time[0] = 0
    sim.gb_node_pos = np.linspace(-.05, .05, 5)
    sim.vout = np.zeros((301, 6, 5))
    sim.pu_out = np.zeros((301, 3, 2))
    sim.pu_out[:, 0] = [.2, .8]
    sim.globout = np.zeros((301, 12))
    sim.globout[:, 1] = 1.
    t = np.arange(301)
    sim.globout[:, 4] = 1.e-5*t + (t*t*1.e-7 if drift else 0)
    sim.globout[:, 9] = 2
    sim.globout[:, 10] = t
    sim.save_hdf5(path)


def test_audit_groups_cases_and_excludes_mislabelled_frame(jobs, tmp_path):
    database, queue = tmp_path/"data", tmp_path/"queue"
    database.mkdir(); queue.mkdir()
    checkpoint(database/"steady.h5", broken_first_time=True)
    checkpoint(database/"same_case.h5", drift=True)
    checkpoint(database/"open.h5", temp=450., drift=True)
    (database/"bad.h5").write_bytes(b"broken")
    state = jobs.audit(database, queue)
    assert len(state["cases"]) == 2
    assert sum(c["status"] == "converged" for c in state["cases"]) == 1
    assert len(state["unreadable_files"]) == 1
    successful = next(c for c in state["cases"] if c["status"] == "converged")
    assert successful["source"] == "steady.h5"
    record = next(r for r in state["files"] if r["path"] == "steady.h5")
    assert record["notes"]
    assert record["criteria"]["window_time"] == pytest.approx(299./3)
    assert (queue/"successful_cases.csv").read_text().count("steady.h5") == 1
    assert "open.h5" in (queue/"open_jobs.csv").read_text()
    assert json.loads((queue/"queue.json").read_text())["format_version"] == 1


def base_case(index):
    return dict(case_id=str(index), status="pending", source=f"source{index}.h5", error=None,
                parameters=dict(temperature=500., tau0=150., grain_size=1., length_grain_boundary=.1, Ngbn=5),
                time_us=100., chunk_time_us=100., iterations=0, attempts=0, output=None,
                criteria=dict(window_time=30., windows=3, min_absorptions=10),
                report=dict(converged=False, reasons=["creep drift"], metrics={}))


def test_scheduler_caps_workers_and_iterates_until_pass(jobs, tmp_path, monkeypatch):
    database, queue = tmp_path/"data", tmp_path/"queue"
    database.mkdir(); queue.mkdir()
    state = dict(cases=[base_case(i) for i in range(13)], updated="", files=[], unreadable_files=[])
    active, peak, attempts, launched = [], [0], {}, []

    class Child:
        def __init__(self, command, **kwargs):
            source = Path(command[2]).name
            index = source.split("source")[-1].split(".")[0] if source.startswith("source") else Path(command[2]).parent.name
            output = Path(command[command.index("--output")+1])
            # Case id is also the output directory name.
            index = output.parent.name
            launched.append(index)
            attempts[index] = attempts.get(index, 0)+1
            output.write_text(json.dumps(dict(case_id=index, time_us=100.+100.*attempts[index])))
            self.pid, self.code = 12345, None
            active.append(self)
            peak[0] = max(peak[0], len(active))
            assert kwargs["start_new_session"]
        def poll(self):
            if self.code is None:
                self.code = 0
                active.remove(self)
            return self.code
        def wait(self):
            return self.poll()

    def inspect(path, database, criteria):
        row = json.loads(path.read_text())
        done = attempts[row["case_id"]] >= 2
        row.update(path=path.relative_to(database).as_posix(), report=dict(converged=done,
                   reasons=[] if done else ["creep drift"], metrics={"creep_rates": [1.e-5]*3}))
        return row

    monkeypatch.setattr(jobs.subprocess, "Popen", Child)
    monkeypatch.setattr(jobs, "inspect_file", inspect)
    monkeypatch.setattr(jobs.time, "sleep", lambda seconds: None)
    jobs.run_queue(database, queue, state, workers=10)
    assert peak[0] == 10
    assert len(set(launched[:13])) == 13  # every case gets a turn before repeats
    assert not active
    assert all(c["status"] == "converged" and c["iterations"] == 2 for c in state["cases"])
    assert all((database/c["source"]).exists() for c in state["cases"])
    with pytest.raises(ValueError, match="between 1 and 10"):
        jobs.run_queue(database, queue, state, workers=11)


def test_failed_jobs_stay_open_and_are_not_infinite_retries(jobs, tmp_path, monkeypatch):
    database, queue = tmp_path/"data", tmp_path/"queue"
    database.mkdir(); queue.mkdir()
    state = dict(cases=[base_case(0)], files=[], unreadable_files=[])
    class Failed:
        pid = 12345
        def __init__(self, *args, **kwargs): pass
        def poll(self): return 1
        def wait(self): return 1
    monkeypatch.setattr(jobs.subprocess, "Popen", Failed)
    monkeypatch.setattr(jobs.time, "sleep", lambda seconds: None)
    jobs.run_queue(database, queue, state, workers=1)
    assert state["cases"][0]["status"] == "failed"
    assert state["cases"][0]["attempts"] == 1
    assert "exited 1" in state["cases"][0]["error"]
    assert "failed" in (queue/"open_jobs.csv").read_text()
    jobs.run_queue(database, queue, state, workers=1, retry_failed=True)
    assert state["cases"][0]["attempts"] == 2


def test_sampling_window_growth_preserves_tolerances(jobs):
    case = base_case(0)
    before = copy.deepcopy(case["criteria"])
    case["report"] = dict(reasons=["insufficient absorption events per window"],
                           metrics=dict(absorption_rates=[.01, .01, .01]))
    jobs.adjust_window(case)
    assert case["criteria"]["window_time"] == 60.
    assert {k:v for k,v in case["criteria"].items() if k != "window_time"} == {k:v for k,v in before.items() if k != "window_time"}


def test_recovers_completed_result_from_previous_manager(jobs, tmp_path, monkeypatch):
    database, queue = tmp_path/"data", tmp_path/"queue"
    database.mkdir(); queue.mkdir()
    case = base_case(0)
    case.update(status="running", output="completed.h5")
    (database/"completed.h5").touch()
    state = dict(cases=[case], files=[], unreadable_files=[])
    monkeypatch.setattr(jobs, "inspect_file", lambda *args: dict(case_id="0", path="completed.h5", time_us=200.,
        report=dict(converged=True, reasons=[], metrics={"creep_rates": [1.e-5]*3})))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must recover without relaunch"))
    jobs.run_queue(database, queue, state)
    assert case["status"] == "converged"
    assert case["source"] == "completed.h5"
