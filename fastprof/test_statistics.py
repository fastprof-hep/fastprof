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
    self.value = twice_dll if best_mu < test_mu else -twice_dll
    self.test_mu = test_mu

  def value(self) :
    return self.value

  def asymptotic_cl(self, offset = 0) :
    if self.value >= 0 :
      return scipy.stats.norm.sf(+math.sqrt(+self.value) - offset)
    else :
      return scipy.stats.norm.sf(-math.sqrt(-self.value) - offset)

  def asymptotic_cls(self, qA = None, sigma = None) :
    if qA != None :
      offset = math.sqrt(qA)
    elif sigma != None :
      offset = self.test_mu/sigma
    else :
      raise ValueError('Should supply either qA or sigma for the asymptotic CL_s computation')
    clsb = self.asymptotic_cl()
    cl_b = self.asymptotic_cl(offset)
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1

