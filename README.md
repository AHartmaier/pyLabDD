[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AHartmaier/pyLabDD.git/main)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# pyLabDD

### Python Laboratory for Dislocation Dynamics

  - Author: Alexander Hartmaier
  - Organization: ICAMS, Ruhr University Bochum, Germany
  - Contact: <alexander.hartmaier@rub.de>

Dislocation Dynamics (DD) is a numerical method for studying
the evolution of a population of discrete dislocations in an elastic medium under mechanical loads. The pyLabDD package
introduces a simple version of Dislocation Dynamics in 2-dimensional space to study 
fundamental aspects of plastic deformation associated with the motion and mutual interaction of dislocations. Dislocations are considered as pure edge dislocations where the line direction is normal to the considered plane.

## Installation

The pyLabDD package requires an [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment with a recent Python version. 

The pyLabDD package can be installed directly from PyPI with the following command

```
$ pip install pylabdd
```

Alternatively, the complete repository can be cloned and installed locally. It is recommended to create a conda environment before installation. This can be done by the following the command line instructions

```
$ git clone https://github.com/AHartmaier/pyLabDD.git ./pyLabDD
$ cd pyLabDD
$ conda env create -f environment.yml  
$ conda activate pylabdd
$ python -m pip install .
```

For this installation method, the correct implementation of the package can be tested with

```
$ pytest tests
```

After this, the package can be used within python, e.g. be importing the entire package with

```python
import pylabdd as dd
```

## Speedup with Fortran subroutines
The subroutines to calculate the Peach-Koehler (PK) force on dislocations are rather time consuming. A Fortran implementation of these subroutines can bring a considerable seepdup of the simulation. To install these faster subroutines, a Fortran compiler is required, e.g. gfortran. On MacOS, this can be achived by installing the command line tools with `xcode-select --install`. The embedding of the Fortran subroutines into Python is accomplished with the leightweight Fortran wrapper [fmodpy](https://pypi.org/project/fmodpy/).

During the installation process it will be automatically tried to implement the faster Fortran subroutines. On import of the pylabdd package into your Python code, you will be informed if the standard Python or the faster Fortran subroutines to calculate the PK force are being used. You will also be notifeid about any error messages during compiliation.

## Jupyter notebooks

pyLabDD is conveniently used with Jupyter notebooks. 
Available notebooks with tutorials on the dislocation dynamics method and the Taylor hardening model are contained in the subfolder `notebooks`. 

The Jupyter notebooks of the pyLabDD tutorials are also available on Binder 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AHartmaier/pyLabDD.git/main)


## Contributions

Contributions to the pyLabDD package are highly welcome, either in form of new 
notebooks with application examples or tutorials, or in form of new functionalities 
to the Python code. Furthermore, bug reports or any comments on possible improvements of 
the code or its documentation are greatly appreciated.

## Dependencies

pyLabDD requires the following packages as imports:

 - [NumPy](http://numpy.scipy.org) for array handling
 - [MatPlotLib](https://matplotlib.org/) for graphical output
 - [fmodpy](https://pypi.org/project/fmodpy/) for embedding of faster Fortran subroutines for PK force calculation (optional)

## Versions

 - v1.0: Initial version (with F90 subroutine)
 - v1.1: Pure Python version (with optional F90 subroutines)
 - V1.2: Automatic compilation of F90 subroutines with fallback on Python version

## License

The pyLabDD package comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of the examples and notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/))
