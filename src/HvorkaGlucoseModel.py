class HvorkaGlucoseModel(object):


  """Two compartment insulin model
     Source and parameters: https://iopscience-iop-org.focus.lib.kth.se/
                            article/10.1088/0967-3334/25/4/010/meta
  """

  def __init__(self):
    """Model params"""
    self.t_G = 40 # Time of maximum glucose rate of appearance (minutes)
    self.a_G = 0.8 # Carbohydrate bioavailability (unitless)

    """Variables - Changes each time model is updated"""
    self.g_t = 0
    self.m_t = 0


  def get_variables(self):
    """Return vector with compartment values"""
    return [self.g_t, self.m_t]


  def set_variables(self, g_t, m_t):
    """Given vector with compartment values - Set model variables"""
    self.g_t, self.m_t = g_t, m_t
    return


  def glucose_c1(self, g_t, t_G, a_G, d_g_t=0):
    """Calculate RoC in Glucose C.2. (Gut)

    Keyword arguments:
    g_t -- glucose in compartment 1 already [mg]
    t_G -- time of maximum glucose rate of appearance [minutes]
    a_G -- carbohydrate bioavailability [minute]
    d_g_t -- carbohydrate intake [minute]
    """
    return -(1/t_G)*g_t+(a_G/t_G)*d_g_t


  def glucose_c2(self, m_t, g_t, t_G):
    """Calculate RoC in Glucose C.2. (Plasma)

    Keyword arguments:
    m_t -- glucose in plasma (use cgm value) [mg]
    g_t -- glucose in cmopartment 1, the gut [mg]
    t_G -- time of maximum glucose rate of appearance [minutes]
    """
    return -(1/t_G)*m_t+(1/t_G)*g_t


  def update_compartments(self, food_glucose):
    """Update model's compartment values

    Keyword arguments:
    cgm -- Measured glucose value [mg/dl]
    """
    self.g_t, self.m_t = self.new_values(food_glucose, self.get_variables())


  def new_values(self, food_glucose, old_variables):
    """Calculate new compartment values

    Keyword arguments:
    cgm -- Measured glucose value [mg/dl]
    """
    g_t_old, m_t_old = old_variables
    #m_t_old = cgm

    # Update Compartments
    g_t = g_t_old + self.glucose_c1(g_t_old, self.t_G, self.a_G, food_glucose)
    m_t = m_t_old + self.glucose_c2(m_t_old, g_t, self.t_G)

    # Estimate appearance of insulin in plasma
    return [g_t, m_t]


  def bergman_input(self):
    """Return the input for the Bergman Minimal Model"""
    return self.m_t
