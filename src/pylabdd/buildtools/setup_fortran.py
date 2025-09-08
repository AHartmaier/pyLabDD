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
            
        if os.environ.get("CONDA_BUILD_CROSS_COMPILATION") == "1":
            # cross-compilation for osx_arm64 build on conda-forge is active
            # patch fmodpy to run test on build-env and build code for host-env
            import tempfile, stat

            PREFIX = os.environ["PREFIX"]
            BUILD_PREFIX = os.environ["BUILD_PREFIX"]
            arm_fc = str(fc)
            x86_fc = os.environ["FC_FOR_BUILD"]
            
            wrap_dir = tempfile.mkdtemp(prefix="fcwrap_")
            wrapper = os.path.join(wrap_dir, "gfortran")
            
            script = f"""#!/usr/bin/env bash
            # If building the fmodpy size-probe, compile/link with BUILD (x86_64) compiler so it can run.
            for i in "$@"; do
              if [[ "$i" == *"fmodpy_get_size"* ]]; then
                exec "{x86_fc}" "$@" -L"{BUILD_PREFIX}/lib" -Wl,-rpath,"{BUILD_PREFIX}/lib"
              fi
            done
            # Otherwise, use the ARM compiler and link against HOST libs.
            exec "{arm_fc}" "$@" -L"{PREFIX}/lib" -Wl,-rpath,"{PREFIX}/lib"
            """
            with open(wrapper, "w") as f:
                f.write(script)
            os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            
            #os.environ["FC"] = wrapper
            #os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")

            print("[BuildFortran-CrossCompiling] Script:")
            print(script)
            print(f"[BuildFortran-CrossCompiling] arm_gfortran: {arm_fc}")
            print(f"[BuildFortran-CrossCompiling] x86_gfortran: {x86_fc}")
            fc = str(wrapper)
            cargs = []  # f"-I{plib}", f"-L{plib}", f"-Wl,-rpath,{plib}"]
            print(f"[BuildFortran-CrossCompiling] ARM64 cargs: {cargs}")
        else:
            fc = str(fc)
            cargs = []
            
        try:
            # Let fmodpy build into its own subdirectory PK_force/
            fmodpy.fimport(
                str(ffile),
                f_compiler=fc,
                #f_compiler_args=cargs,
                #libraries=libpath,
                output_dir=str(fortran_dir),
                rebuild=False,
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
