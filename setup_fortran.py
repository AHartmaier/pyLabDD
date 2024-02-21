"""Post-installation of Fortron subroutine to calculate the
Peach-Koehler (PK) force on dislocations. If successful, it will make the
code run much faster. If not successful, fall back to slower pure Python code."""

import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
MAIN_DIR = os.getcwd()  # directory in which repository is cloned
try:
    path_path = os.environ['CONDA_PREFIX']  # get path to environment
except Exception as e:
    path_path = os.path.join(os.path.expanduser('~'), '.pylabdd')  # otherwise fall back to user home
    logging.error(f'Possibly installing pyLabDD without conda environment. Exception occurred: {e}')
    logging.error(f'Creating a working directory for Kanapy under: {path_path} to store path information.')
    if not os.path.exists(path_path):
        os.makedirs(path_path)

path_path = os.path.join(path_path, 'PATHS.txt')
with open(path_path, 'w') as f:
    f.write(MAIN_DIR)
try:
    import fmodpy
    pkf = fmodpy.fimport('pylabdd/PK_force.f90')
    logging.info('Installation successful, including Fortran subroutine for PK force.')
except Exception as e:
    logging.error(f'An unexpected exception occurred: {e}')
    logging.error('Installation of Fortran subroutine failed.')
    logging.error('Using basic installation with Python subroutine for PK force.')
