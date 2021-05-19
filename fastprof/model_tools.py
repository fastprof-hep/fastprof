"""Module defining utility classes

"""

import json
import math
import scipy
import numpy as np
from abc import abstractmethod
import re

from .base import Serializable
from .core import Model, Data, Parameters, ModelPOI
from .minimizers import NPMinimizer, POIMinimizer, OptiMinimizer
from .test_statistics import TMu, QMu, QMuTilda


class ParBound :
  """Class to define and enforce parameter bounds

  Defines upper and lower bounds on a model parameter,
  and implements a test method applied to :class:`Parameters`
  objects.

  Atttributes:
    par    (str)   : parameter name
    minval (float) : parameter lower bound (`None` if no bound)
    maxval (float) : parameter upper bound (`None` if no bound)
  """

  def __init__(self, par : str, minval : float = None, maxval : float = None) :
    """Initialize the `QMuTildaCalculator` object

    Defines a selection minval <= par <= maxval.
    Both bounds are optional and can be omitted by passing `None` as
    the corresponding argument (also default).

    Args:
      par    : parameter name
      minval : parameter lower bound (`None` for no bound, default)
      maxval : parameter upper bound (`None` for no bound, default)
    """
    self.par = par
    self.minval = minval
    self.maxval = maxval
  def test(self, pars : Parameters) -> bool :
    """Applies the selection to a :class:`Parameters` object

    Args:
      pars : a set of model parameter
    Returns:
     `True` if the parameters pass the selection, `False` if they fail.
    """
    try :
      return (pars[self.par] >= self.minval if self.minval != None else True) and (pars[self.par] <= self.maxval if self.maxval != None else True)
    except KeyError :
      return True
  def __str__(self) -> str :
    """Provides a description string for the object

    Returns:
      a description string
    """
    smin = '%s >= %g' % (self.par, self.minval) if self.minval != None else ''
    smax = '%s <= %g' % (self.par, self.maxval) if self.maxval != None else ''
    if smin == '' : return smax
    if smax == '' : return smin
    return smin + ' and ' + smax
  def __repr__(self) -> str:
    """Provides a description string for the object

    Needed in addition to :meth:`__str__` to print out correctly
    lists of :class:`ParBound` objects.

    Returns:
      a description string
    """
    return self.__str__()
