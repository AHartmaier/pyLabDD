#from setuptools import Extension
#from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as _build_py
import subprocess
import sys
from pathlib import Path

# Dummy Extension, damit setuptools garantiert build_ext ausführt
#dummy_extension = Extension("pylabdd.PK_force", sources=[])

class BuildFortran(_build_py):
    def run(self):
        print("=" * 80)
        print("[BuildFortran] Starting Fortran compilation with fmodpy")
        print("=" * 80)

        # Ensure gfortran is available
        try:
            subprocess.run(["gfortran", "--version"], check=True)
            print("[BuildFortran] gfortran found.")
        except Exception as e:
            print("[BuildFortran] gfortran not found! Install via: conda install -c conda-forge gfortran")
            raise e

        # Ensure fmodpy is installed
        try:
            import fmodpy
            print("[BuildFortran] fmodpy imported successfully.")
        except ImportError:
            print("[BuildFortran] fmodpy not found – installing it.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fmodpy"])
            import fmodpy

        # Path to Fortran source
        fortran_dir = Path(__file__).parent.parent  # point to src/pylabdd
        ffile = fortran_dir / "PK_force.f90"
        if not ffile.exists():
            raise FileNotFoundError(f"[BuildFortran] Fortran source not found: {ffile}")

        print(f"[BuildFortran] Compiling {ffile}")

        try:
            # Let fmodpy build into its own subdirectory PK_force/
            fmodpy.fimport(
                str(ffile),
                output_dir=str(fortran_dir),
                rebuild=True,
                verbose=True
            )
        except Exception as e:
            print("[BuildFortran] Fortran compilation failed!")
            raise e

        # Check if PK_force folder exists
        pk_dir = fortran_dir / "PK_force"
        if pk_dir.exists():
            print(f"[BuildFortran] PK_force directory created: {pk_dir}")
        else:
            print("[BuildFortran] WARNING: PK_force directory not found!")

        super().run()

# Wird von setuptools in pyproject.toml importiert
#ext_modules = [dummy_extension]

