class HvorkaInsulinModel(object):


  """Two compartment insulin model
     Source and parameters: https://iopscience-iop-org.focus.lib.kth.se/
                            article/10.1088/0967-3334/25/4/010/meta
  """

  def __init__(self):
    """Model parameters"""
    self.ti_max = 55 # [min] Time-to-max absorption of subcutaneously injected short-acting insulin

    """Variables - Changes each time model is updated"""
    self.s1_t = 0 # Insulin in compartment 1 - subcutaneous tissue
    self.s2_t = 0 # Insulin in compartment 2 - Blood plasma
    self.U_i = 0 # Insulin absorption rate (appearance of insulin in plasma)


  def get_variables(self):
    """Return vector with compartment values"""
    return [self.s1_t, self.s2_t]


  def set_variables(self, s1_t, s2_t):
    """Given vector with compartment values - Set model variables"""
    self.s1_t, self.s2_t = s1_t, s2_t
    return


  def calc_s1_roc(self, u_t, s1_t, ti_max):
    """Calculate Insulin RoC in C.1. [U/min] (Subcutaneous tissue)

    Keyword arguments:
    s1_t -- insulin in C.1 [U]
    u_t -- insulin input (bolus or infusion) [U]
    ti_max -- time to maximal insulin absorption [minutes]
    """
    return u_t - (s1_t/ti_max)


  def calc_s2_roc(self, s1_t, s2_t, ti_max):
    """Calculate Insulin RoC in C.1. [U/min] (Subcutaneous tissue)

    Keyword arguments:
    s1_t -- insulin in C.1 [Units]
    s2_t -- insulin in C.2 [Units]
    ti_max -- time to maximal insulin absorption [minutes]
    """
    return (s1_t/ti_max) - (s2_t/ti_max)


  def calc_Ui(self, s2_t, ti_max):
    """The insulin absorption rate [U/min] (appearance of insulin in plasma)

    Keyword arguments:
    s2_t -- insulin in C.2 [Units]
    ti_max -- time to maximal insulin absorption [minutes]
    """
    return s2_t/ti_max


  def update_compartments(self, bolus):
    """Given a bolus at time t, update model's compartment values

    Keyword arguments:
    bolus -- Administered bolus [Units]
    """
    self.s1_t, self.s2_t, self.U_i = self.new_values(bolus,
            self.get_variables())


  def new_values(self, bolus, old_variables):
    """Prepare to update compartments by calc. and returning new values

    Keyword arguments:
    bolus -- Administered bolus [Units]
    """

    s1_t_old, s2_t_old = old_variables

    # Update Compartments
    s1_t = s1_t_old + self.calc_s1_roc(bolus, s1_t_old, self.ti_max)
    s2_t = s2_t_old + self.calc_s2_roc(s1_t, s2_t_old, self.ti_max)

    # Estimate appearance of insulin in plasma
    U_i = self.calc_Ui(s2_t, self.ti_max)
    return [s1_t, s2_t, U_i]


  def model_predict(self, n=1):
    """Iteratively update model n times, return array of n predicted U_i's

    Keyword arguments:
    n -- integer, how many predictions ahead should be made
    """
    preds = np.arange(n)
    old_variables = self.get_variables()

    for it in range(n):
      new_variables = self.new_values(bolus=0, old_variables=old_variables)
      U_i_n = new_variables[-1]
      preds[it] = U_i_n
      old_variables = new_variables[:-1]

    return preds


  def bergman_input(self):
    """Return the input for the Bergman Minimal Model in mU (milli-Units)"""

    return self.U_i*1000
