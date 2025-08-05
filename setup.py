import sys
from pathlib import Path
from setuptools import setup

# Add project root to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

import setup_fortran

if __name__ == "__main__":
    setup(cmdclass={"build_ext": setup_fortran.BuildFortran})
