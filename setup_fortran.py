from setuptools import Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import shutil
from pathlib import Path


# Dummy Extension, damit setuptools garantiert build_ext ausführt
dummy_extension = Extension("pylabdd.PK_force", sources=[])


class BuildFortran(build_ext):
    def run(self):
        print("=" * 80)
        print("[BuildFortran] Starte Fortran-Kompilierung mit fmodpy")
        print("=" * 80)

        # Prüfen ob gfortran installiert ist
        try:
            subprocess.run(["gfortran", "--version"], check=True)
            print("[BuildFortran] gfortran gefunden.")
        except Exception as e:
            print("[BuildFortran] gfortran nicht gefunden! Bitte installieren: conda install -c conda-forge gfortran")
            raise e

        # fmodpy importieren oder installieren
        try:
            import fmodpy
            print("[BuildFortran] fmodpy erfolgreich importiert.")
        except ImportError:
            print("[BuildFortran] fmodpy nicht gefunden – Installation wird versucht.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fmodpy"])
            import fmodpy

        # Pfad zur Fortran-Datei
        fortran_dir = Path(__file__).parent / "src" / "pylabdd"
        ffile = fortran_dir / "PK_force.f90"

        if not ffile.exists():
            raise FileNotFoundError(f"[BuildFortran] Fortran-Datei nicht gefunden: {ffile}")

        print(f"[BuildFortran] Kompiliere {ffile}")

        try:
            fmodpy.fimport(
                str(ffile),
                output_dir=str(fortran_dir),
                rebuild=True,
                verbose=True
            )
        except Exception as e:
            print("[BuildFortran] Fehler bei der Fortran-Kompilierung!")
            raise e
            
        # Nach dem Build den Unterordner PK_force auflösen
        subdir = fortran_dir / "PK_force"
        if subdir.exists():
            for file in subdir.iterdir():
                target = fortran_dir / file.name
                print(f"[BuildFortran] Verschiebe {file} -> {target}")
                if target.exists():
                    target.unlink()
                shutil.move(str(file), str(target))
            shutil.rmtree(subdir)

        so_files = list(fortran_dir.glob("PK_force*.so"))
        if so_files:
            print(f"[BuildFortran] Erfolgreich kompiliert: {so_files}")
        else:
            print("[BuildFortran] Keine .so-Datei gefunden!")

        # Prüfen ob .so-Datei entstanden ist
        so_files = list(fortran_dir.glob("PK_force*.so"))
        if so_files:
            print(f"[BuildFortran] Erfolgreich kompiliert: {so_files}")
        else:
            print("[BuildFortran] Keine .so-Datei gefunden!")

        # Restlicher Build
        super().run()


# Wird von setuptools in pyproject.toml importiert
ext_modules = [dummy_extension]
