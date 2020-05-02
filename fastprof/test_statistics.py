import math
import scipy.stats
from abc import abstractmethod

# -------------------------------------------------------------------------
class TestStatistic :
  def __init__(self, test_mu) : 
    self.test_mu = test_mu
  def __float__(self) :
    return value()
  @abstractmethod
  def value(self) :
    pass  
  @abstractmethod
  def asymptotic_cl(self) :
    pass
    

class TMu(TestStatistic) :
  def __init__(self, test_mu, twice_dll) :
    super().__init__(test_mu)
    self.value = twice_dll
  def value(self) : 
    return self.value
  def asymptotic_cl(self) :
    return scipy.stats.norm.sf(math.sqrt(self.value))


class QMu(TestStatistic) :
  def __init__(self, test_mu, tmu, best_mu, comp_mu = None, tmu_A = None, sigma = None) :
    super().__init__(test_mu)
    self.tmu = tmu
    self.best_mu = best_mu
    self.comp_mu = comp_mu if comp_mu != None else self.test_mu
    self.tmu_A = tmu_A
    self.sigma = sigma

  def value(self) :
    return self.tmu if self.best_mu < self.test_mu else -self.tmu

  def non_centrality_parameter(self) :
    if self.comp_mu == self.test_mu : return 0
    if self.tmu_A != None :
      return self.tmu_A
    elif self.sigma != None :
      return ((self.comp_mu - self.test_mu)/self.sigma)**2
    else :
      raise ValueError('Should supply either tmu_A or sigma for the asymptotic CL_s computation')

  def asymptotic_cl(self) :
    if self.value() >= 0 :
      return scipy.stats.norm.sf(+math.sqrt(+self.value()) - math.sqrt(self.non_centrality_parameter()))
    else :
      return scipy.stats.norm.sf(-math.sqrt(-self.value()) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_cls(self) :
    clsb = self.asymptotic_cl()
    cl_b = self.asymptotic_clb()
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1

  def asymptotic_clb(self) :
    return QMu(0, self.tmu, self.best_mu, self.comp_mu, self.tmu_A, self.sigma).asymptotic_cl()
