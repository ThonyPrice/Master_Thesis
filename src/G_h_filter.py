class G_h_filter(object):
    """Alpha beta (aka g-h) filter implementation

    A-b filter is a simplified form of observer for estimation,
    data smoothing and control applications.

    Implementation inspiration:
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/
        master/01-g-h-filter.ipynb
    """

    def __init__(self, x0, dx=0.1, g=.001, h=.01, dt=1.):
        self.x = x0
        self.x0 = x0
        self.x_est = x0
        self.dx = dx
        self.g = g
        self.h = h
        self.dt = dt
        self.pred = []
        self.results = []
        self.residual = []


    def predict(self):
        # Prediction step
        self.x_est = self.x + (self.dx*self.dt)
        self.x_est = max(0, self.x_est)
        self.dx = self.dx
        self.pred.append(self.x_est)
        return self.x_est


    def update(self, y, y_pred):
        # Update step
        self.residual = y[-1]-y_pred[-1]
        self.dx = self.dx + self.h * self.residual / self.dt
        self.x = self.x_est + self.g * self.residual
        self.x = max(0, self.x)
        self.results.append(self.x)
        return self.x
