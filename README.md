# fastprof: *A fast profiling tool for likelihood-based statistical analyses*

Presentation
============

The tool defines binned likelihood models with an arbitrary number of bins and parameters. Parameters are either parameters of interest (POIs) or nuisance parameters (NPs).

The likelihood is defined under the assumption that the impact of the NPs on the expected event yield in each bin is a *linear* function of the parameters. This allows to obtain maximum-likelihood estimates of the NPs, for fixed values of the POIs, using simple linear algebra.

The tool is written in python, with most of the work done by numpy.

In the current implementation, only a single signal-strength parameter of interest is supported

The model is intended to be used to perform statistical inference in the context of high-energy physics (HEP), and in particular set limits on new phenomena. Along with the likelihood itself, it includes a set of frequentist statistical tools, based on the techniques described in <https://arxiv.org/abs/1007.1727>

Setup
=====

The package can be set up as follows
    mkdir fastprof-work
    git clone <url>
   source fastprof/setup-enf.sh
The last command sets up a python3 pyenv working environment. The numpy, pandas and matplotlib packages should also be installed within the environment using pip install if they are not already available.

Setting upper limits
====================

The main purpose of the tool is to set upper limits on the signal strength of new phenomena, using either the asymptotic formulas of <https://arxiv.org/abs/1007.1727> or tehcniques based on pseudo-experiments ("toys"). The latter are more widely applicable, and work in cases where the likelihood is not approximately Gaussian, but require more computing effort. The linear approximation to the expected yields implemented in fastprof allows this technique to be performed more quickly that in non-linear cases.

The procedure for setting an limit at a given confidence level (CL) in a particular model is as follows:

   - Create a fastprof model for your likelihood. This model is defined in a JSON file. An example can be found here.
   - Create a dataset, again defined in a JSON file.
   - Define a test statistic to use for the limit-setting. This can be one of the definitions introduced in <https://arxiv.org/abs/1007.1727>, based on the profile-likelihood ratio
   - Define a set of signal strength hypotheses at which to compute the exclusion p-value (pv). These should span the range of values where the limit can lie, so that it can be determined by interpolation.
   - For each hypothesis, compute the observed value of the test statistic and the best-fit parameters of the model to the dataset under this signal hypothesis.
   - For each hypothesis, generate pseudo-experiments drawn from the model corresponding to the best-fit value of the parameters obained above. Compute the test statistic value for each 
   - Compute the p-value of the test statistic from the quantile of the observed test statistics (computed 2 steps above) in the distributions obtained from the pseudo-experiments (computed 1 step above)
   - Interpolate between the hypotheses to obtain the hypothesis value corresponding to the desired upper limit.
