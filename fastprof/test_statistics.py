import math
import scipy.stats
from abc import abstractmethod

# -------------------------------------------------------------------------
class TestStatistic :
  def __init__(self) : 
    pass
  def __float__(self) :
    return value()
  @abstractmethod
  def value(self) :
    pass  
  @abstractmethod
  def asymptotic_cl(self) :
    pass
    

class TMu(TestStatistic) :
  def __init__(self, twice_dll) :
    self.value = twice_dll
  def value(self) : 
    return self.value
  def asymptotic_cl(self) :
    return scipy.stats.norm.sf(math.sqrt(self.value))


class QMu(TestStatistic) :
  def __init__(self, twice_dll, test_mu, best_mu) :
    self.value = twice_dll if best_mu < test_mu else None
    self.test_mu = test_mu
  def value(self) :
    return self.value
  def asymptotic_cl(self) :
    return scipy.stats.norm.sf(math.sqrt(self.value)) if (self.value != None and self.value > 0) else 0.5
  def asymptotic_cls(self, sigma) :
    clsb = self.asymptotic_cl()
    cl_b = scipy.stats.norm.sf(math.sqrt(self.value) - self.test_mu/sigma) if (self.value != None and self.value > 0) else 0.5
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b

