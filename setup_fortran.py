from setuptools.command.build_ext import build_ext
import subprocess
import sys
from pathlib import Path


class BuildFortran(build_ext):
    def run(self):
        try:
            import fmodpy
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fmodpy"])
            import fmodpy

        fortran_dir = Path(__file__).parent / "src" / "pylabdd"
        ffile = fortran_dir / "PK_force.f90"

        if not ffile.exists():
            raise FileNotFoundError(f"Fortran file not found: {ffile}")

        print(f"[build_ext] Compiling Fortran file: {ffile}")
        fmodpy.fimport(
            str(ffile),
            output_dir=str(fortran_dir),
            rebuild=True
        )

        super().run()
