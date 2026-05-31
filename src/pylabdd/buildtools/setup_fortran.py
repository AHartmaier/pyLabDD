from setuptools.command.build_py import build_py as _build_py
import sys
import os
from pathlib import Path


def _drop_arm_fortran_flags(flags):
    """Remove ARM CPU flags that an x86_64 build compiler cannot parse."""
    cleaned = []
    arm_prefixes = ("arm", "armv", "aarch64", "apple-m", "cortex-")
    flags = list(flags)
    i = 0

    while i < len(flags):
        flag = flags[i]
        if flag in ("-march", "-mcpu", "-mtune"):
            value = flags[i + 1].lower() if i + 1 < len(flags) else ""
            if value.startswith(arm_prefixes):
                i += 2
                continue
            cleaned.append(flag)
            i += 1
            continue

        if flag.startswith(("-march=", "-mcpu=", "-mtune=")):
            value = flag.split("=", 1)[1].lower()
            if value.startswith(arm_prefixes):
                i += 1
                continue

        cleaned.append(flag)
        i += 1

    return cleaned


def _for_build_env():
    env = os.environ.copy()
    fflags = env.get("FFLAGS")
    if fflags:
        env["FFLAGS"] = " ".join(_drop_arm_fortran_flags(fflags.split()))
    return env


def _is_macos_build():
    return sys.platform == "darwin" or os.environ.get("target_platform", "").startswith("osx-")


def _with_osx_headerpad_ldflags(ldflags):
    flags = ldflags.split()
    headerpad_flag = "-Wl,-headerpad_max_install_names"
    if _is_macos_build() and headerpad_flag not in flags:
        flags.append(headerpad_flag)
    return " ".join(flags)


class BuildFortran(_build_py):
    def run(self):
        import subprocess
        print("=" * 80)
        print("[BuildFortran] Starting Fortran compilation with fmodpy")
        print("=" * 80)
        
        fc = os.environ.get("FC")
        if not fc:
            print("No environment variable 'FC' set. Using 'gfortran' as fortran compiler.")
            fc = 'gfortran'
        print('F90 compiler: ', fc)

        # Ensure gfortran is available
        try:
            subprocess.run([fc, "--version"], check=True)
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
        fortran_sources = ["PK_force.f90", "mod_gbdd.f90"]
        cross = os.environ.get("CONDA_BUILD_CROSS_COMPILATION") == "1"
        arm_fc = str(fc)
        x86_fc = os.environ.get("FC_FOR_BUILD")
        for source in fortran_sources:
            ffile = fortran_dir / source
            if not ffile.exists():
                raise FileNotFoundError(f"[BuildFortran] Fortran source not found: {ffile}")
            print(f"[BuildFortran] Compiling {ffile}")
            fimport_fc = fc
            
            if cross:
                # cross-compilation for osx_arm64 build on conda-forge is active
                # patch fmodpy to run test on build-env and build code for host-env
                import tempfile, stat

                BUILD_PREFIX = os.environ["BUILD_PREFIX"]
                if not x86_fc:
                    raise RuntimeError("[BuildFortran] FC_FOR_BUILD must be set for cross-compilation")
                
                wrap_dir = tempfile.mkdtemp(prefix="fcwrap_")
                wrapper = os.path.join(wrap_dir, "gfortran")
                
                script = f"""#!/usr/bin/env bash
# fmodpy probes the generated module with the build-arch compiler.  Conda's
# osx-arm64 host FFLAGS can contain ARM-only options (for example
# -march=armv8.3-a), which x86_64 gfortran rejects, so drop those here.
args=()
pending_arch_flag=""
for arg in "$@"; do
    if [[ -n "$pending_arch_flag" ]]; then
        case "$arg" in
            arm*|aarch64*|apple-m*|cortex-*)
                pending_arch_flag=""
                continue
                ;;
        esac
        args+=("$pending_arch_flag" "$arg")
        pending_arch_flag=""
        continue
    fi
    case "$arg" in
        -march|-mcpu|-mtune)
            pending_arch_flag="$arg"
            continue
            ;;
        -march=arm*|-mcpu=arm*|-mtune=arm*|-march=aarch64*|-mcpu=aarch64*|-mtune=aarch64*|-march=apple-m*|-mcpu=apple-m*|-mtune=apple-m*|-march=cortex-*|-mcpu=cortex-*|-mtune=cortex-*)
            continue
            ;;
    esac
    args+=("$arg")
done
if [[ -n "$pending_arch_flag" ]]; then
    args+=("$pending_arch_flag")
fi
exec "{x86_fc}" "${{args[@]}}" -L"{BUILD_PREFIX}/lib" -Wl,-rpath,"{BUILD_PREFIX}/lib"
"""
                with open(wrapper, "w") as f:
                    f.write(script)
                os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

                print("[BuildFortran-CrossCompiling] Script:")
                print(script)
                print(f"[BuildFortran-CrossCompiling] arm_gfortran: {arm_fc}")
                print(f"[BuildFortran-CrossCompiling] x86_gfortran: {x86_fc}")
                fimport_fc = str(wrapper)
                
            try:
                # Let fmodpy build into its own subdirectory PK_force/
                old_fflags = os.environ.get("FFLAGS")
                old_ldflags = os.environ.get("LDFLAGS")
                if cross:
                    build_env = _for_build_env()
                    os.environ["FFLAGS"] = build_env.get("FFLAGS", "")
                os.environ["LDFLAGS"] = _with_osx_headerpad_ldflags(os.environ.get("LDFLAGS", ""))
                fmodpy.fimport(
                    str(ffile),
                    f_compiler=fimport_fc,
                    output_dir=str(fortran_dir),
                    rebuild=False,
                    verbose=True
                )
                print(f"[BuildFortran] fimport completed successfully for module {source}")
            except Exception as e:
                print("[BuildFortran] Fortran compilation failed!")
                raise e
            finally:
                if cross:
                    if old_fflags is None:
                        os.environ.pop("FFLAGS", None)
                    else:
                        os.environ["FFLAGS"] = old_fflags
                if old_ldflags is None:
                    os.environ.pop("LDFLAGS", None)
                else:
                    os.environ["LDFLAGS"] = old_ldflags
        if cross:
            print(f"[BuildFortran] Cross-compilation active: building arm64 libraries with {arm_fc} and x86_64 libraries with {x86_fc}")
            for source in fortran_sources:
                mod_name = Path(source).stem
                lib_path = fortran_dir / mod_name
                wrapper_name = f"{mod_name}_c_wrapper.f90"
                lib_name = lib_path / f"{mod_name}.arm64.so"
                fflags  = os.environ.get("FFLAGS", "").split()
                ldflags = _with_osx_headerpad_ldflags(os.environ.get("LDFLAGS", "")).split()
                cmd = [
                    arm_fc,
                    source,
                    wrapper_name,
                    *fflags,
                    *ldflags,
                    "-shared",
                    "-O3",
                    "-o",
                    str(lib_name),
                ]
                subprocess.run(cmd, check=True, cwd=lib_path)

        # if cross:
        #     # create arm_64 library to be shipped with package
        #     lib_path = os.path.join(fortran_dir, "PK_force")
        #     lib_name = os.path.join(lib_path, "PK_force.arm64.so")
        #     fflags  = os.environ.get("FFLAGS", "").split()
        #     ldflags = os.environ.get("LDFLAGS", "").split()
        #     cmd = [arm_fc, "PK_force.f90", "PK_force_c_wrapper.f90"] + fflags + ldflags + ["-shared", "-O3", "-o", lib_name]
        #     print(f"[BuildFortran-CrossCompiling]: building arm64 library: {cmd}")
        #     subprocess.run(cmd, check=True, cwd=lib_path)
        # Check if PK_force folder exists
        pk_dir = fortran_dir / "PK_force"
        if pk_dir.exists():
            print(f"[BuildFortran] PK_force directory created: {pk_dir}")
        else:
            print("[BuildFortran] WARNING: PK_force directory not found!")
        pk_dir = fortran_dir / "mod_gbdd"
        if pk_dir.exists():
            print(f"[BuildFortran] mod_gbdd directory created: {pk_dir}")
        else:
            print("[BuildFortran] WARNING: mod_gbdd directory not found!")

        super().run()
