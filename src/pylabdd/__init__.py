# -*- coding: utf-8 -*-

"""Top-level package for pyLabDD"""

import os
import logging
from pylabdd.dislocations import Dislocations
from importlib.metadata import version
from importlib.resources import files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = """Alexander Hartmaier"""
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('pylabdd')

try:
    import fmodpy
    from fmodpy import fimport
except ImportError:
    logger.warning("fmodpy is not available. Fortran acceleration will be disabled.")
    fimport = None

# Try to compile Fortran code
def try_build_fortran():
    if fimport is None:
        logger.info("Skipping Fortran build: fmodpy not available.")
        return None

    try:
        fortran_file = files("pylabdd").joinpath("PK_force.f90")
        logger.info(f"Compiling Fortran module: {fortran_file}")
        mod = fimport(str(fortran_file), autocompile=True)
        logger.info("Fortran module compiled successfully.")
        return mod

    except Exception as e:
        logger.warning(f"Failed to compile Fortran module: {e}")
        return None

# Build upon import
fortran = try_build_fortran()
if fortran:
    calc_fpk_pbc = fortran.calc_fpk_pbc
    calc_fpk = fortran.calc_fpk
    print('Using Fortran subroutine for PK force.')
else:
    from pylabdd.pkforce import calc_fpk_pbc, calc_fpk
    print('Using Python subroutines for PK force.')
