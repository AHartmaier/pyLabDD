from setuptools.command.build_py import build_py as _build_py
import subprocess
import sys
import os
from pathlib import Path


class BuildFortran(_build_py):
    def run(self):
        
        print("=" * 80)
        print("[BuildFortran] Starting Fortran compilation with fmodpy")
        print("=" * 80)
        
        fc = os.environ.get("FC")
        if not fc:
            print("No environment variable 'FC' set. Using 'gfortran' as fortran compiler.")
            fc = 'gfortran'

        # Ensure gfortran is available
        try:
            subprocess.run([fc, "--version"], check=True)
            print("[BuildFortran] gfortran found.")
        except Exception as e:
            print("[BuildFortran] gfortran not found! Install via: conda install -c conda-forge gfortran")
            #raise e

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
        fflags = os.environ.get("FFLAGS", "").split()
        print(f"[BuildFortran] Using FFLAGS: {fflags}")
        ldlibs = os.environ.get("LDFLAGS", "").split()
        print(f"[BuildFortran] Libraries: {ldlibs}")
        libpath = os.environ.get("LIBRARY_PATH", "").split()
        print(f"[BuildFortran] Library Path: {libpath}")
        cargs = fflags + ldlibs
        try:
            # Let fmodpy build into its own subdirectory PK_force/
            fmodpy.fimport(
                str(ffile),
                f_compiler=str(fc),
                f_compiler_args=cargs,
                libraries=libpath,
                library_extensions=['so', 'dylib', 'dll', '.5.dylib'],
                output_dir=str(fortran_dir),
                rebuild=False,
                verbose=True
            )
        except Exception as e:
            print("[BuildFortran] Fortran compilation failed!")
            print(e)
            # Try to wrap gfortran to call it with libraries
            import shutil, tempfile, stat

            prefix = os.environ.get("PREFIX") or os.environ.get("CONDA_PREFIX")
            real_fc = str(fc)
            
            wrap_dir = tempfile.mkdtemp(prefix="fcwrap_")
            wrapper  = os.path.join(wrap_dir, "gfortran")
            
            script = f"""#!/usr/bin/env bash
            # Forward to the real gfortran, appending conda-forge paths.
            exec "{real_fc}" "$@" -L"{prefix}/lib" -Wl,-rpath,"{prefix}/lib"
            """
            with open(wrapper, "w") as f:
                f.write(script)
            os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            fmodpy.fimport(
                str(ffile),
                f_compiler=wrapper,
                f_compiler_args=fflags,
                libraries=libpath,
                library_extensions=['so', 'dylib', 'dll', '.5.dylib'],
                output_dir=str(fortran_dir),
                rebuild=False,
                verbose=True
            )

        # Check if PK_force folder exists
        pk_dir = fortran_dir / "PK_force"
        if pk_dir.exists():
            print(f"[BuildFortran] PK_force directory created: {pk_dir}")
        else:
            print("[BuildFortran] WARNING: PK_force directory not found!")

        super().run()
