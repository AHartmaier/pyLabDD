"""Post-installation of Fortron subroutine to calculate the
Peach-Koehler (PK) force on dislocations. If successful, it will make the
code run much faster. If not successful, fall back to slower pure Python code."""

import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
MAIN_DIR = os.getcwd()  # directory in which repository is cloned
WORK_DIR = os.path.expanduser('~') + '/.pylabdd'  # working directory for temporary files
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
with open(WORK_DIR + '/PATHS.txt', 'w') as f:
    f.write(MAIN_DIR)
try:
    import fmodpy
    pkf = fmodpy.fimport('pylabdd/PK_force.f90')
    logging.info('Installation successful, including Fortran subroutine for PK force.')
except Exception as e:
    logging.error(f'An unexpected exception occurred: {e}')
    logging.info('Basic installation with Python subroutine for PK force.')