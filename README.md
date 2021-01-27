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

After download and changing the current working directory to the 'trunk' folder of the repository, the pyLabDD package is installed with the following command 

```
$ python -m pip install . --user
```

After this, the package can by imported with

```python
import pylabdd as dd
```

The correct implementation can be tested with

```
$ pytest tests
```

## Documentation

Online documentation for pyLabDD can be found under https://ahartmaier.github.io/pyLabDD/.
For offline use, open pyLabDD/docs/index.html to browse through the contents.
The documentation is generated using [Sphinx](http://www.sphinx-doc.org/en/main/).

## Jupyter notebooks

pyLabDD is conveniently used with Jupyter notebooks. 
Available notebooks with tutorials on the dislocation dynamics method and the Taylor hardening model are contained in the subfolder `notebooks`. An
overview on the contents of the notebooks is available [here](https://ahartmaier.github.io/pyLabDD/examples.html).

The Jupyter notebooks of the pyLabDD tutorials are also available on Binder 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AHartmaier/pyLabDD.git/master)


## Contributions

Contributions to the pyLabDD package are highly welcome, either in form of new 
notebooks with application examples or tutorials, or in form of new functionalities 
to the Python code. Furthermore, bug reports or any comments on possible improvements of 
the code or its documentation are greatly appreciated.

## Dependencies

pyLabDD requires the following packages as imports:

 - [NumPy](http://numpy.scipy.org) for array handling
 - [MatPlotLib](https://matplotlib.org/) for graphical output

## License

The pyLabDD package comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of the examples and notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/))
