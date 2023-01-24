.. _stat computations:

Statistical computations
========================

The main statistical results produced by the package are

* p-values for specified model hypotheses;

* Discovery significance relative to the background-only hypothesis

* Confidence intervals for model parameters (one-dimensional intervals for a single parameter, or confidence contours for two parameters)

* Upper limit on a signal yield

These computations are based on the profile likelihood ratio (PLR) test statistic described in the :ref:`profiling` section. Two main modes of operation are provided:

* Asymptotic computations: in this case, the PLR is assumed to follow a Gaussian distribution, as is the case for large expected event yields and Gaussian systematics.

* Pseudo-experiment sampling: in this case, the PLR distributions are sampled using a large number of pseudo-experiments ("toys") to obtain results that do not assume the asymptotic approximation. These results are typically longer to compute, although computation times are typically reasonable for simplified likelihoods.


Asymptotic computations
-----------------------

The computations follow the prescriptions presented in Ref. [Asimov]_. The :math:`t_{\mu}`, :math:`q_{\mu}` and :math:`\tilde{q}_{\mu}` test statistics are implemented. Upper limits can be computed using the :math:`\text{CL}_s` [CLs]_ modified frequentist prescription. 

Tools to perform the computations are listed in the :ref:`utilities-comp` section.


Setting upper limits from sampling distributions
------------------------------------------------

When the observed data is in good agreement with the model, it can be used to set bounds on model parameters. A common use-case is to set an upper limit on the normalization of a model component, that is considered as the signal. Following Ref. [Asimov]_, these limits can be set using the :math:`\tilde{q}_{\mu}` and :math:`q_{\mu}` test statistics defined in terms of the likelihood ratio.

In both cases, :math:`mu` refers to the signal normalization parameter on which the limit is to be set. These test statistics are derived from the profile-likelihood ratio of Eq. :eq:`PLR`, which can be computed as described in Section :numref:`model`.

Large values of the test statistics :math:`\tilde{q}_{\mu}` or :math:`q_{\mu}` correspond to values of :math:`\mu` that are disfavored by the data, while null values correspond to perfect agreement. Setting an upper limit at a given confidence level (CL) involves adjusting :math:`\mu` until a test statistic value is reached which corresponds to a p-value of :math:`p = 1 - \text{CL}`. 

For approximately Gaussian likelihoods, corresponding to the *asymptotic approximation* of large expected event yields, these p-values can be computed in closed form using the formulas in Ref. [Asimov]_. In HEP, limits are usually set at a CL of 95%. For upper limits on a signal normalization, the :math:`CL_s` procedure [CLs]_ is also usually applied. In this case, the quantity used to set the upper limit is not the p-value computed as above, but its ratio to the p-value computed in the hypothesis :math:`\mu = 0`.

Even in the asymptotic approximation, there is usually no practical way to compute the test statistic value corresponding to a given p-value -- instead, one can only go in the direction of the test statistic value to the p-value. Setting a limit at a given CL therefore involves iteratively computing the p-value for different :math:`\mu` hypotheses, until the desired p-value is reached. The corresponding :math:`\mu` hypothesis then provides the upper limit.

In cases where the asymptotic approximation is not valid, p-values can be computed from sampling distributions as follows:

1. for a given hypothesis, a number of pseudo-datasets are randomly generated

2. for each dataset, the test statistic (:math:`\tilde{q}_{\mu}` or :math:`q_{\mu}` above) is computed, and the value stored

3. for a given value :math:`q_{\mu}^{\text{obs}}` of the test statistic, the corresponding p-value is computed as the quantile of this value in the distribution obtained at step 2.

Given the iterative nature of the limit search, the full procedure can be summarized as follows:

1. Define a set of hypotheses to be tested: the hypothesis points should span a region that contains the limit value, with a spacing that allows to reliably interpolate it.

2. For each hypothesis, generate a set of pseudo-experiments and compute the associated test statistic values to produce the sampling distribution.

3. Compute the observed value of the test statistic :math:`q_{\mu}^{\text{obs}}` at each hypothesis :math:`\mu`

4. Compute the p-value at each hypothesis as the quantile of :math:`q_{\mu}^{\text{obs}}` within the corresponding sampling distribution. This uses the *mid-p-value* technique to be robust against narrow peaks in the sampling distributions.

5. Interpolate between the tested hypothesis to determine the upper limit on :math:`\mu` at the desired CL value.

For a  :math:`CL_s` limit, steps 2. and 4. need to be performed twice, one for the nominal hypothesis, and once for :math:`\mu = 0` to compute the :math:`CL_b` term.


.. [CLs] A. L. Read, *Modified frequentist analysis of search results (the* :math:`CL_s` *method)*, `CERN-OPEN-2000-005 <http://cdsweb.cern.ch/record/451614>`_

