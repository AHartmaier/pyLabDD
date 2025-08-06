# src/pylabdd/__init__.py
# -*- coding: utf-8 -*-

"""Top-level package for pyLabDD"""

import os
import logging
from importlib.metadata import version
from pylabdd.dislocations import Dislocations

# get absolute path to complied F90 paket
#package_path = os.path.abspath(__file__)
#if package_path not in sys.path:
#    sys.path.insert(0, package_path)
from pylabdd.PK_force import calc_fpk, calc_fpk_pbc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = """Alexander Hartmaier"""
__email__ = 'alexander.hartmaier@rub.de'
__version__ = version('pylabdd')
__all__ = ["calc_fpk", "calc_fpk_pbc", "Dislocations"]
