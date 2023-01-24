# fastprof: *A fast profiling tool for likelihood-based statistical analyses*

Presentation
============

The tool is intended to provide a simple way to reuse, reinterpret and combine public likelihoods, in particular the ones used in
high-energy physics experiments. It implements a simplified likelihood format, *Simplified Likelihoods with Linearized Systematics (SLLS), which provides good approximations to full likelihoods, but is faster to evaluate.

The SLLS format is similar to the `HistFactory` format implemented for instance in [pyhf](https://github.com/scikit-hep/pyhf): it is defined in terms of measurement regions, each containing one of more bins, with yields expressed in terms of model parameters. These parameters are either parameters of interest (POIs) or nuisance parameters (NPs). The simplification occurs in the treatment of the NPs, which are considered at linear order only. This in turns provides a simple way to maximize the likelihood with respect to the NPs, using matrix algebra, which is typically much faster than maximizing the full likelihood.

The model is intended to be used to perform statistical inference using classical frequentist techniques: computing p-values, discovery significances and confidence intervals, and setting limits on new phenomena. Along with the likelihood itself, the package includes a set of frequentist statistical tools, based on the techniques described in [arXiv:1007.1727](https://arxiv.org/abs/1007.1727)

The tool is written in python, with most of the work done by numpy, and models are stored in JSON or YAML markup files. Models implemented within `pyhf` or `roofit` can be automatically converted to the SLLS format using tools provided in the package.


Setup
=====

The package can be set up as follows:
```
git clone ssh://git@gitlab.cern.ch:7999/nberger/fastprof.git
cd fastprof
git checkout v0.4.1 -b v0.4.1
source ./setup-env.sh
````

This sets up the latest stable version,`v0.4.1`. (skipping this line sets up the latest `master` version instead, but this is not recommended)

The last command sets up a python3 `pyenv` working environment. The numpy, pandas and matplotlib packages should also be installed within the environment using pip install if they are not already available.


Documentation
=============

Detailed documentation can be found in the package itself in `build/sphinx/html`, or on the [documentation website](https://fastprof.web.cern.ch).
SLLS likelihoods are described in detail [arXiv:2301.05676](https://arxiv.org/abs/2301.05676).
