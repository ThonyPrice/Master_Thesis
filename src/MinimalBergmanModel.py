import numpy as np

from HvorkaInsulinModel import HvorkaInsulinModel
from HvorkaGlucoseModel import HvorkaGlucoseModel


class BergmanModel(object):
  """Two compartment insulin model
     Source and parameters: https://www.physiology.org/doi/pdf/
                            10.1152/ajpendo.1979.236.6.E667
  """


  def __init__(self):

    """Sub-Models with remote compartments"""
    self.insulin_system = HvorkaInsulinModel()
    self.glucose_system = HvorkaGlucoseModel()

    """Variables"""
    self.Gt = 81 # Differential plasma glucose relative to the basal glucose value
    self.Xt = 2.7e-3 # Insulin in the remote compartment [unitless]
    self.It = -1.13 # Differential plasma insulin relative to the basal insulin value [mU/liter]

    """Parameters"""
    self.p1 = 1.0e-2 # min-1 # some param values found in: https://journals.sagepub.com/doi/pdf/10.1177/193229680700100605
    self.p2 = 3.33e-2 # mU liter-1 min-2
    self.p3 = 1.33e-5 # min-1

    self.Gb = 55 # Basal glucose value [mg/dl]
    #self.Ib = 108 # pmol/L (current value). Basal insulin value should be in units of [mU/liter]
    #self.Ib = 108 * 0.144 # Convert to mU/L
    self.Ib = 4.25 # From paper... unsure if it's assigned to correct variable
    self.vG = 0.16 # Volume of gut distribution [dl/kg]
    self.Vi = 0.12 # Volume of gut distribution [L/kg]
    self.n = 1


  def get_variables(self):
    """Return vector with compartment values"""
    return [self.It, self.Xt, self.Gt]


  def set_variables(self, It, Xt, Gt):
    """Given vector with compartment values - Set model variables"""
    self.It, self.Xt, self.Gt = It, Xt, Gt
    return


  def calc_dGdt(self, p1, Gt, Xt, Gb, mt, vG):
    """Calc. differential plasma glucose RoC relative to the basal glucose value

    Keyword arguments:
    p1 -- Rate parameter
    Gt -- Differential plasma glucose
    Xt -- Insulin in remote compartment
    Gb -- Basal Glucose value
    mt -- Plasma glucose appearance
    vG -- Volume of gut distribution
    """
    return -p1*Gt-(Xt*(Gt-Gb))+(mt/vG)


  def calc_dXdt(self, p2, Xt, p3, It):
    """Calc. Insulin in remote compartment RoC

    Keyword arguments:
    p2 -- Rate parameter
    Xt -- Insulin in remote compartment
    p3 -- Rate parameter
    It -- Differential plasma insulin relative to the basal insulin value
    """
    return -p2*Xt+p3*It


  def calc_dIdt(self, n, It, Ib, ut, Vi):
    """Calc. RoC of differential plasma insulin relative to the basal insulin value

    Keyword arguments:
    n -- I have no idea what this is...
    It -- Differential plasma insulin relative to the basal insulin value
    Ib -- Basal insulin value
    ut -- Plasma insulin appearance
    Vi -- Volume of gut distibution
    """
    return -n*(It+Ib)+(ut/Vi)


  def update_remote_compartments(self, bolus, food):
    """Given bolus and glucose intake, update the insulin- and glucose in remote compartments"""
    self.insulin_system.update_compartments(bolus)
    self.glucose_system.update_compartments(food)
    u_i = self.insulin_system.bergman_input()
    m_t = self.glucose_system.bergman_input()
    return [u_i, m_t]


  def update_compartments(self, bolus, cgm):
    """Update all compartments - Both remote and within this Model. Return current Glucose estimate"""
    u_i, m_t = self.update_remote_compartments(bolus, cgm)
    self.update_params(m_t, u_i)
    return self.Gt


  def update_params(self, m_t, u_i):
    """Update the parameters of Model given values in remote compartments"""
    self.It = self.It + self.calc_dIdt(self.n, self.It, self.Ib, u_i, self.Vi)
    self.Xt = self.Xt + self.calc_dXdt(self.p2, self.Xt, self.p3, self.It)
    self.Gt = self.Gt + self.calc_dGdt(self.p1, self.Gt, self.Xt, self.Gb, m_t, self.vG)


  def predict_n(self, n, bolus, food):
    """Predict Glucose (self.Gt) n timesteps ahead given bolus and food vectors n steps ahead

    Keyword arguments:
    bolus -- Array of length n with boluses
    food -- float representing food at initial time step
    """

    # Prep prediction vector
    pred = np.zeros(n)

    # Save current state of compartments
    InsulinModel_variables = self.insulin_system.get_variables()
    GlucoseModel_variables = self.glucose_system.get_variables()
    BergmanModel_variables = self.get_variables()

    for it in range(n):
      self.update_compartments(bolus[it], food)
      food = 0
      pred[it] = self.Gt

    # prediction = self.Gt

    # Reset states of compartment models
    self.insulin_system.set_variables(*InsulinModel_variables)
    self.glucose_system.set_variables(*GlucoseModel_variables)
    self.set_variables(*BergmanModel_variables)

    return pred
