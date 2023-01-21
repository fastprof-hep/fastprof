.. _introduction:

Introduction
============

The fastprof tool is implements a class of simplified likelihood models for CPU-intensive applications, denoted as Simplified Likelihoods with Linearized Systematics (SLLS) and described in detail in Ref. [SLLS]_.

SLLS models are simplified versions of the HistFactory standard, implementing event-counting measurements with systematic uncertainties. These models are widely used in High-energy physics (HEP) but also applicable in other contexts.
The models differ from the HistFactory baseline by modeling systematics uncertainties only at linear order and with Gaussian constraints. This simplication leads to large performance improvements when performing parameter minimization. This translates in particular to large gains in the computation of profile likelihoods and related test statistics, which are widely used in HEP [Asimov]_. The paramerers of interest (POIs) of the measurement are treated exactly.

These simplified likelihoods can be used as approximations to the exact measurement likelihoods, in particular for two important applications:

  - *The reuse, reinterpretation and combination of experimental measurements*: these operations are typically CPU-intensive for full likelihoods, especially when many configurations need to be considered -- e.g. when scanning over a large model parameter space. The SLLS formalism preserves the POIs and the nuisance parameters (NPs) of the full likelihood. This allows operations such as reinterpretations and combinations to be performed in the same way as in the original model.
  
  - *To sample test statistic distributions*, in cases where the asymptotic formulas presented in Ref. [Asimov]_ are invalid. This is a CPU-intensive task, since the number of pseudo-experiments is typically :math:`O(10^5)` to :math:`O(10^7)` or more. Simplified likelihoods allow a much faster processing of pseudo-experiments, and provide a better approximation of the full model than the asymptotic formulas.

The code provides the following components:
  * A general implementation of SLLS models described in detail in Section [models].
  * Tools to import models from other formats such as the `ROOT` binary format in wide use in HEP, and the `pyhf` implementation of `HistFactory` models.
  * Tools to validate the linear model against the full model.
  * A framework for the computation of upper limits on model parameters, following the general prescriptions in HEP.
  * Tools to operate on models, e.g. by combining multiple models, pruning NPs and measurement regions, or reinterpreting the measurement using a different set of POIs.
  * Utilities for the generation and storage of sampling distributions

These components are described in detail in the next sections.

.. [SLLS] N\. Berger, *Simplified likelihoods using linearized systematic uncertainties*, `arXiv:2301.05676 <https://arxiv.org/abs/2301.05676>`_

.. [Asimov] G\. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, Eur. Phys. J. C **71**:1554, 2011, `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_

