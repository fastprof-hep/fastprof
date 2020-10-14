.. _introduction:

Introduction
============

The fastprof tool is intended to provide an extremely fast impementation of a class of likelihood models for CPU-intensive applications.

The supported models are simplified versions of the HistFactory standard, implementing event-counting measurements with systematic uncertainties. These models are widely used in High-energy physics (HEP) but also applicable in other contexts.

The models differ from the HistFactory baseline by implementing only Gaussian systematic uncertainties at linear order. This simplication leads to large performance improvements when performing parameter minimization. This leads in particular to large gains in the computation of profile likelihoods and related test statistics, which are widely used in HEP [Asimov]_.

A particular application of fast profiling methods is the generation of sampling distributions for test statistics using pseudo-data generation. This technique provides a robust determination of sampling distributions, in cases where the Gaussian assumptions presented in Ref. [Asimov]_ are invalid. It is however CPU-intensive, requiring a number of pseudo-experiments ranging from :math:`O(10^5)` to :math:`O(10^7)` or more. The faster computation times from fast profiling allows to perform computation of this type within a reasonable running time using typical computing resources.

The code provides the following components:
  * a general implementation of linear likelihood models described in detail in Section [models].
  * tools to import models from other formats such as the ROOT binary format in wide use in HEP
  * utilities for the generation and storage of sampling distributions
  * tools to validate the linear model against the full model
  * a framework for the computation of upper limits on model parameters, following the general prescriptions in HEP.

These components are described in detail in the next sections.

.. [Asimov] G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, Eur. Phys. J. C **71**:1554, 2011, `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_

