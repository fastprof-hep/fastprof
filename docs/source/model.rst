.. _model:

Model
=====

Likelihood model
----------------

The class of models supported by fastprof follows closely the HistFactory definition, describing counting experiment in a number of measurements bins in the presnce of systematic uncertainties. The main components of the model are as follows:

* Each model consists of a number of channels, corresponding to individual measurements, and a set of model parameters, separated into parameters of interest (POIs) and nuisance parameters (NPs)

* Each channel is defined by a number of measurement bins, each corresponding to a separate counting experiment, and a set of samples, corresponding to separate contributions to the expected event yield in each bin.

* Each sample defines an expected event yield for each of the channel bins, variations in these yields due to the effect of NPs, and an overall normalization factor that is a function of the POIs.

* The NPs are each associated with an auxiliary observable, which are interpreted as the observables of a separate Gaussian measurements. The information from these measurements feeds into the model Lagrangian as Gaussian constraint on the NP which is included in the expression of the likelihood.

In the rest of this document, we use the following notations, with bold text indicating vectors :
  * :math:`c` is an index running over model channels, from 1 to :math:`N\chan`.
  * :math:`i` is an index running over measurement bins, from 1 to :math:`N\bins`.
  * :math:`s` is an index running over samples, from 1 to :math:`N\samp`.
  * :math:`\alpha` is an index running over NPs, from 1 to :math:`N\nps`.
  * :math:`\mu` and \theta are the vectors of model POIs and NPs respectively of size :math:`N\pois` and :math:`N\nps`.
  * :math:`\vt\aux` is the vector of auxiliary observables, with each component corresponding to the NP with the same index as in :math:`\vt`.
  * :math:`\Gamma` is the inverse of the covariance matrix of the auxiliary measurement corresponding to the :math:`\vt\aux`.
  * :math:`n\obs_i` is the observed number of events.
  * :math:`N_{s,i}(\vm, \vt)` is the expected events yield for sample :math:`s` in bin :math:`i`, and :math:`N_i(\vm, \vt)` the total yield summed over samples.
  * :math:`\vD^+_{s,i}` and :math:`\vD^-_{s,i}` provide the impact of the NPs on the expected yield of sample :math:`s` in bin :math:`i`, respectively for positive and negative NP values. Each one is a vector of size :math:`N\nps` where each component corresponds to the NP with the same index in :math:`\vt`.
  * :math:`\nu_s(\vm)` is the overall normalization factor for sample :math:`s`.

With the definitions above, the model likelihood can be written as

.. math:: \mathcal{L}(\vm, \vt) = \prod\limits_{i=1}^{N\bins} e^{- N_i(\vm, \vt) } \frac{N_i(\vm, \vt)^{n\obs_i}}{n\obs_i!} \exp\left[-\frac{1}{2} (\vt - \vt\aux)^T \Gamma (\vt - \vt\aux)\right]
   :label: llh

and the expected event yields as

.. math:: N_{s,i}(\vm, \vt) = \nu_k(\vm) N\nom_{s,i} e^{\delta_{s,i}(\vt)}
   :label: nexp

where :math:`N\nom_{s,i}` is the nominal event yield for sample :math:`s` in bin :math:`i` and

.. math:: \delta_{s,i}(\vt) = \sum\limits_{\alpha=1}^{N\nps}\log\left(1 + \Delta^+_{s,i,\alpha}\right) \max(\theta_{\alpha}, 0) + \log\left(1 + \Delta^-_{s,i,\alpha}\right) \min(\theta_{\alpha}, 0)

The impact of the NPs is implemented in exponential form, as this ensures that the expected event yields remain positive for all NP values.

Maximum likelihood estimates
----------------------------

Maximization with respect to the NPs is performed under the assumption that their impact on the expected event yields is linear. This assumption is applies around a reference value :math:`\vt=\vt\ref`, so that for the purpose of the minimization the expected event yields are expressed as

.. math:: \begin{align} 
  N_{s,i}(\vm, \vt) &= \nu_k(\vm) N\nom_{s,i} e^{\delta_{s,i}(\vt\ref)} \left[1 + \vD_{s,i} (\vt - \vt\ref) \right] \\
                    &= N_{s,i}(\vm, \vt\ref)\left[1 + \vD_{s,i} (\vt - \vt\ref) \right] \end{align}

where the symmetric impact values :math:`\vD_{s,i}` are computed as

.. math:: \Delta_{s,i,\alpha} = \sqrt{\frac{1 + \Delta_{s,i,\alpha}^+}{1 + \Delta_{s,i,\alpha}^-}} - 1.

The NP values that maximize the likelihood of :eq:`llh`, expressed using the event yields above, can then be computed as

.. math:: \hat{\hat{\vt}}(\vm) = \vt\ref + \left[ P(\vm) + \Gamma \right]^{-1} \left[ \boldsymbol{Q}(\mu) + \Gamma(\vt\aux - \vt\ref) \right]

where

.. math:: P(\vm) = \sum\limits_{i=1}^{N\bins} n\obs_i \sum_{s,s'=1}^{N\samp} \frac{N_{s,i}(\vm, \vt\ref)}{N_i(\vm, \vt\ref)}\frac{N_{s',i}(\vm, \vt\ref)}{N_i(\vm, \vt\ref)} \vD_{s,i} \otimes \vD_{s',i}

is a matrix of size :math:`N\nps \times N\nps` and

.. math:: \boldsymbol{Q}(\vm) = \sum\limits_{i=1}^{N\bins} \sum_{s=1}^{N\samp} \left(n\obs_i - N_i(\vm, \vt\ref) \right) \frac{N_{s,i}(\vm, \vt\ref)}{N_i(\vm, \vt\ref)} \vD_{s,i}

is a vector of size :math:`N\nps`.

This expression provides the conditional maximum likelihood estimates (MLEs) for :math:`\vt` for a fixed value of :math:`\vm`. The global MLEs :math:`\hat{\vm}` and :math:`\hat{\vt}` can then be obtained by maximizing over :math:`\vm`. Since no approximations are applied on the dependence of the likelihood on the :math:`\vm`, this step must be performed using non-linear minimization routines. This is however quicker than performing non-linear minimization over all parameters, especially if the number of NPs is large.

These expressions can then be used to evaluate the profile-likelihood ratio (PLR) 

.. math:: t(\vm) = -2 \log \frac{\mathcal{L}(\mu, \hat{\hat{\vt}}(\vm))}{\mathcal{L}(\hat{\vm}, \hat{\vt})}.
   :label: PLR

Conversion from other model formats
-----------------------------------

Linear models can be constructed directly, or by approximating an existing, non-linear model. For the latter, the conversion to a linear model proceeds as follows:
* The structure of the model (POIs, NPs, channels, samples and bins) is directly taken from the full model. In case the full model follows the HistFactory specification, the structure can be imported directly. For ROOT workspaces, the POIs, NPs and model PDF are extracted from the ModelConfig of the workspace. The channels are then obtained from the categories implemented in the model PDF; the samples are extracted from the PDFs for each category. The bins are defined as either one bin per channel, in case of a counting experiment, or from the binning in the observable for this channel, in case of a shape analysis.

* The NPs are normalized by considering their difference with respect to a reference value, scaled by their uncertainties. The reference and uncertainties are typically obtained as the best-fit value and parabolic uncertainty estimate in a fit of the full model to a provided dataset. The :math:`\Gamma` matrix is assumed to be diagonal, with diagonal elements equal to 0 for free NPs, and 1 constrained NPs.

* The reference yields for each sample in each measurement bin is computed for reference values of the POIS, and the NPs (scaled as described above) set to 0. For counting experiments, the yield is obtained directly; for a shape analysis, the integral of the channel PDF over the corresponding bin is computed.

* The impacts of each NP are computed by successively setting their scaled values to :math`\pm 1` and, evaluating the per-sample yields as described above. The positive and negative impact values are them computed as :math:`\Delta^+ = N^+/N^0 - 1` and :math:`\Delta^- = N^0/N^- - 1` respectively, where :math:`N^+`, :math:`N^-` and :math:`N^0` are respectively the yields for positive and negative variations and the nominal yield.

Datasets are converted to the linear format in a similar way. For a counting experiment, the observed bin yields are the same as those of the full model; for a shape analysis, they are obtained by counting events within the bins of the channel observable defined above. The auxiliary observable values for constrained NPs are scaled in the same way as the NPs themselves; for free NPs, the auxiliary observables are taken to be 0 by convention.


Regularization procedures
-------------------------

The level of approximation entailed by the assumption of linear NP impact depends on the form of the original model. The linear model is generally valid for a limited range of NP values around the reference point used in the conversion described in the previous section, which depends on the level of tolerance on discrepancies between the full and linear models.

The behavior of the linear model can be improved by using additional constraints that avoid unphysical behavior:

* *Adding constraint on free NP parameters*: in some cases, the linear approximation does not constrain the free NPs as strongly as the full model. This can be the case for instance in case if the parameters are strongly correlated at linear order, so that one linear combination of the parameters is only weakly constrained. In this coniguration, the constraining power of the full model can be dominated by effects beyond linear order, which are not included by definition in the linear model. This can be mitigated by adding an loose artificial constraint to the free NPs, to ensure that their values do not deviate too far from 0. These constraints are applied at several times the nominal uncertainty of the parameter (typically with a factor :math:`K \approx` 3--10), in order to limit their impact on the results. They are implemented by replacing the null diagonal term in :math:`\Gamma` by :math:`1/K^2`.

* *NP impact saturation*: the exponential impact :math:`\exp\left[\delta(\vt)\right]` of NPs implemented as in :eq:`nexp` can lead to large modifications of the event yields, which can in turn lead to unphysical results. These impacts can be reduced by using the replacement
  
  .. math:: \exp\left[\delta(\vm, \vt)\right] \rightarrow 1 + C \tanh\left[ \left(\exp\left[\delta(\vm, \vt)\right] - 1\right)/C \right]
  
  where :math:`C` is a cutoff corresponding approximately to the value above which relative variations are taken to saturate. For instance for :math:`C=2`, relative variations are capped at :math:`\pm 200\%`. Typical values are :math:`C \approx` 1--3.

* *NP bounds in sampling* : in case the procedures above are insufficient to obtain tolerable agreement between the full and the linear model over all NP values, the NPs can be restricted to a specific range. These bounds cannot be enforced during the minimization itself, since it is performed through a linear algebra computation and not an iterative procedure. The bounds are applied within the sampling procedure itself, by rejecting pseudo-datasets for which the \hat values fall outside the specified ranges.

* *Hypothesis reset* :  the random generation of pseudo-datasets is performed for values of the NPs which may in some cases deviate from 0 (see Section). If these values fall outside the region of linear behavior, this can lead to generally unphysical behavior for the pseudo-experiments generated for these values. The generation values can therefore be modified as described in Section, in order to avoid this issue.
