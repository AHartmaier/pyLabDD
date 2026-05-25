# src/pylabdd/__init__.py
# -*- coding: utf-8 -*-

"""Top-level package for pyLabDD"""

import logging
from importlib.metadata import version
from .dislocations import Dislocations
from .gbdd import GB_dislocations

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .PK_force import calc_fpk, calc_fpk_pbc
    FORT_AVAIL = True
except Exception as e:
    logging.warning(f'Compilation of F90 subroutine calc_fpk failed: {e}')
    logging.warning('Using slower Python versions.')
    from .PK_force_py import calc_fpk, calc_fpk_pbc
    FORT_AVAIL = False

try:
    from .mod_gbdd import calc_gbdd
    GBDD_AVAIL = True
except Exception as e:
    logging.warning(f'Compilation of F90 subroutine calc_gbdd failed: {e}')
    logging.warning('GBDD module cannot be used.')
    GBDD_AVAIL = False


__author__ = """Alexander Hartmaier"""
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('pylabdd')
__all__ = ["Dislocations", "GB_dislocations", "calc_fpk", "calc_fpk_pbc", "calc_gbdd",
           "FORT_AVAIL", "GBDD_AVAIL"]
