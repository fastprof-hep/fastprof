.. _utilities:

Utilities
=========

.. toctree::
   :maxdepth: 1

Model creation and import
-------------------------

.. _utilities creation:

Simplified models can be created from scratch from a markup file, or (more commonly) converted from other formats.
The conversion involves a linear approximation of the nuisance parameter impacts, as described in the :ref:`model` section.
Conversion from two inputs formats is currently supported:

* Models in the `HistFactory` format implemented using the `pyhf` tool. Conversion is handled by the ``convert_pyhf_json.py`` script in this case.

* Unbinned models implemented in a `ROOT` using the `roofit` toolkit. Conversion is handled by the ``convert_ws.py`` script in this case.


.. toctree::
   :maxdepth: 1

   scripts/convert_ws.rst
   scripts/convert_pyhf_json.rst
   scripts/fit_ws.rst

Model inspection
----------------


.. toctree::
   :maxdepth: 1

   scripts/check_model.rst
   scripts/print_model.rst

   
Computations
------------

.. _utilities-comp:

The following statistical computations are supported (see the :ref:`stat` section for more details on the computations themselves) :

* P-value and significance for a model hypothesis, implemented in the ``fit_model.py`` tool.

* Confidence intervals and confidence contours for one or two parameters of interest, implemented in the ``poi_scan.py`` tool.

* Upper limits in the asymptotic approximation, implemented in the ``compute_fast_limits.py`` tool.

* Upper limits using pseudo-experiments, implemented in the ``compute_limits.py`` tool.


.. toctree::
   :maxdepth: 1

   scripts/fit_model.rst
   scripts/poi_scan.rst
   scripts/compute_fast_limits.rst
   scripts/compute_limits.rst


Plotting
--------

.. _utilities-plot:


The following plotting utilities are provided

* Plotting the expected and observed yields in the measurement bins, implemented in the ``plot.py`` tool.

* Plotting the impact values of nuisance parameters in each bin, implemented in ``plot_impacts.py``.

* Plotting the impact values of the nuisance parameters as a function of their value, to check the linearity of the impacts; implemented in the ``plot_valid.py`` tool.

.. toctree::
   :maxdepth: 1

   scripts/plot.rst
   scripts/plot_impacts.rst
   scripts/plot_valid.rst

Model modification
------------------

Models can be modified after creation in several ways. The main functionalities can be access through the command line tools, listed in the :ref:`utilities` section. The full API is also described in the :ref:`code reference` section. Usual operations include

* Merging multiple models into a single combined model, implemented in the ``merge_models.py`` tool.

* Simplifying ("pruning") model by removing nuisance parameters with small impacts, or bins with low measurement information, implemented in the ``prune_model.py`` tool.

* Performing the measurement in terms of a different physics model, by re-expressing the POIs in terms of an alternative set of parameters ("reparameterization"), implemented in the ``reparam_model.py`` tool.


.. toctree::
   :maxdepth: 1

   scripts/merge_models.rst
   scripts/merge_channels.rst
   scripts/prune_model.rst
   scripts/reparam_model.rst


Other tools
-----------

.. toctree::
   :maxdepth: 1

   scripts/dump_samples.rst
   scripts/dump_debug.rst
   scripts/collect_results.rst
