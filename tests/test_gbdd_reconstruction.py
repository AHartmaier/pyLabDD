"""Check reconstructed observables against the actual Fortran backend source."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def reconstruction_backend(tmp_path_factory):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is required for the backend reconstruction checks")
    directory = tmp_path_factory.mktemp("gbdd_reconstruction")
    source = Path(__file__).resolve().parents[1] / "src/pylabdd/mod_gbdd.f90"
    driver = """
program reconstruction_check
use backend
implicit none
real(8) :: params(10), time(3), xpos(7), pu(3,3,4), glob(3,12)
real(8), allocatable :: fields(:,:,:)
integer :: channels, it, npu, nout, nabs, frame, node, nparams
character(len=16) :: argument, mode
call get_command_argument(1,argument)
read(argument,*) channels
call get_command_argument(2,mode)
allocate(fields(3,channels,7))
params(:8)=[44.d3,.3d0,.25d-3,5.d-4,57.d3,500.d0,10.d0,1.d-2]
nparams=8
call get_command_argument(3,argument)
if (len_trim(argument)>0) read(argument,*) nparams
params(9)=2.d0
params(10)=0.012299093542812717d0
call get_command_argument(4,argument)
if (len_trim(argument)>0) read(argument,*) params(9)
call get_command_argument(5,argument)
if (len_trim(argument)>0) read(argument,*) params(10)
time=0.d0
fields=0.d0
pu=0.d0
glob=0.d0
npu=0
if (trim(mode)=='restart') then
    time(1)=1.d0
    fields(1,4,:)=[0.d0,1.d0,-2.d0,3.d0,-2.d0,1.d0,0.d0]
end if
call calc_gbdd(params(:nparams),nparams,0.d0,600.d0,.1d0,1.d0,7,4,time(1)+1.d-5,100,1.d-6, &
    it,npu,nout,nabs,time,xpos,fields,pu,glob,.false.)
do frame=1,nout
    do node=1,7
        write(*,*) 'ROW',frame,node,xpos(node),fields(frame,4:channels,node)
    end do
end do
end program
"""
    combined = directory / "check.f90"
    combined.write_text("module backend\ncontains\n" + source.read_text()
                        + "\nend module\n" + driver)
    executable = directory / "check"
    subprocess.run(
        [compiler, "-ffree-line-length-none", "-fcheck=all", str(combined), "-o", str(executable)],
        cwd=directory, check=True, capture_output=True, text=True,
    )
    return executable


@pytest.mark.parametrize("channels", [5, 6])
@pytest.mark.parametrize("mode", ["single", "restart"])
@pytest.mark.parametrize("nparams,Z,width", [(8, 2.0, None), (9, 1.0, None),
                                          (10, 1.0, 0.008), (10, 3.0, 0.02)])
def test_fortran_reconstructs_initial_and_evolved_fields(reconstruction_backend, channels, mode,
                                                       nparams, Z, width):
    result = subprocess.run(
        [str(reconstruction_backend), str(channels), mode, str(nparams), str(Z), str(width or 0)],
        check=True, capture_output=True, text=True,
    )
    rows = np.array([
        [float(value) for value in line.split()[1:]]
        for line in result.stdout.splitlines() if line.split()[:1] == ["ROW"]
    ])
    lam = width if width is not None else Z / 2 * np.exp(-2 * np.pi * (1 - 0.3))
    frames = np.unique(rows[:, 0])
    assert len(frames) >= 2
    for frame in frames:
        data = rows[rows[:, 0] == frame]
        x, coefficients = data[:, 2], data[:, 3]
        distance = x[:, None] - x[None, :]
        expected_u = (0.5 + np.arctan(distance / lam) / np.pi) @ coefficients
        np.testing.assert_allclose(data[:, 4], expected_u, rtol=1.e-12, atol=1.e-14)
        if channels == 6:
            expected_beta = (lam / (np.pi * (distance**2 + lam**2))) @ coefficients
            np.testing.assert_allclose(data[:, 5], expected_beta, rtol=1.e-12, atol=1.e-14)
    initial = rows[rows[:, 0] == 1]
    expected_coefficients = ([0, 0, 0, 1, 0, 0, 0] if mode == "single"
                             else [0, 1, -2, 3, -2, 1, 0])
    np.testing.assert_allclose(initial[:, 3], expected_coefficients)
    # A centered single Cauchy core has u(0)/B=1/2, rather than a sharp step.
    if mode == "single":
        assert initial[3, 4] == pytest.approx(0.5)
        if channels == 6:
            assert initial[3, 5] == pytest.approx(1 / (np.pi * lam))
