# Gaussian Process Interpolator

author : Reed Essick (reed.essick@gmail.com)

This repository houses a few simple scripts showing how to construct differentiable interpolators equipped with uncertainty quantification based on Gaussian Processes.
The main use case is to construct an interpolator over multi-dimensional tabulated equation of state data.

The code is scoped to handle an arbitrary number of dimensions.
While it is expected that the performance may be slow depending on the number of data used to construct the interpolator, the number of points needed to construct an accurate interpolator may scale poorly with the dimensionality.
Therefore, care is advised before using this on large or high-dimensional data.

## Quick Start Guide

### Installation

The code can be installed in the usual way.
After cloning the repo, do something like
```
python setup.py install --prefix path/to/install
```
After this, be sure to update your environmental variables to point to your installation.

### Dependencies

  * numpy [version?]()

### Examples

**WRITE ME**

## References

  * [Rasmussen and Williams (2006)](http://gaussianprocess.org/gpml/chapters/)
