# fastprof: *A fast profiling tool for likelihood-based statistical analyses*

Presentation
============

The tool defines binned likelihood models with an arbitrary number of bins and parameters. Parameters are either parameters of interest (POIs) or nuisance parameters (NPs).

The likelihood is defined under the assumption that the impact of the NPs on the expected event yield in each bin is a *linear* function of the parameters. This allows to obtain maximum-likelihood estimates of the NPs, for fixed values of the POIs, using simple linear algebra.
The model is intended to be used to perform statistical inference in the context of high-energy physics (HEP), and in particular set limits on new phenomena. Along with the likelihood itself, it includes a set of frequentist statistical tools, based on the techniques described in <https://arxiv.org/abs/1007.1727>

The tool is written in python, with most of the work done by numpy.


Setup
=====

The package can be set up as follows
```
git clone ssh://git@gitlab.cern.ch:7999/nberger/fastprof.git
cd fastprof
source ./setup-env.sh
````
The last command sets up a python3 `pyenv` working environment. The numpy, pandas and matplotlib packages should also be installed within the environment using pip install if they are not already available.

Goals
=====

The main purpose of the tool is to set upper limits on the signal strength of new phenomena, using either the asymptotic formulas of <https://arxiv.org/abs/1007.1727> or tehcniques based on pseudo-experiments ("toys"). The latter are more widely applicable, and work in cases where the likelihood is not approximately Gaussian, but require more computing effort. The linear approximation to the expected yields implemented in fastprof allows this technique to be performed more quickly that in non-linear cases.

Documentation
=============

Detailed documentation can be found in the package itself in `build/sphinx/html`, or on the [documentation website](https://fastprof.web.cern.ch).
