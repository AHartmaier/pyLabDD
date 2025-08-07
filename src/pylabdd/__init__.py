# src/pylabdd/__init__.py
# -*- coding: utf-8 -*-

"""Top-level package for pyLabDD"""

import sys
import logging
from pathlib import Path
from importlib.metadata import version
from .dislocations import Dislocations

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .PK_force import calc_fpk, calc_fpk_pbc
except Exception as e:
    logging.warn(f'Compilation of F90 subroutine failed: {e}')
    logging.warn('Using slower Python versions.')
    from .PK_force_py import calc_fpk, calc_fpk_pbc

__author__ = """Alexander Hartmaier"""
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('pylabdd')
__all__ = ["Dislocations", "calc_fpk", "calc_fpk_pbc"]
