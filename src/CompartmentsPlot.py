class CompartmentsPlot(object):
    """Plot substance development in Compartment model."""
    def __init__(self, Model):
        self.Model = Model
        self.pop_model_params(Model)
        self.plot()


    def pop_model_params(self):
        self.s1[i] = self.Model.s1
        self.s2[i] = self.Model.s2
        self.gt[i] = self.Model.gt
        self.mt[i] = self.Model.mt
        self.It[i] = self.Model.It
        self.Xt[i] = self.Model.Xt
        self.Gt[i] = self.Model.Gt

    def plot(self):

        N = self.Gt.shape
        xx = np.arange(N)

        # Plot compartments and Glucose
        sns.set(rc={'figure.figsize':(30,10)})
        fig, axs = plt.subplots(4, 1, sharex=True)

        # Clean up bolus data
        non_zero_bolus_idxs = np.where(self.Model.bolus_series_ != 0)[0]

        # Plot INSULIN prevalence in compartments
        axs[0].plot(xx, self.s1, color='orange',
                    linewidth=2, label='Insulin - Subcutaneous tissue')
        axs[0].plot(xx, self.s2, color='red',
                    linewidth=2, label='Insulin - Plasma')
        axs[0].plot(non_zero_bolus_idxs,
                    self.Model.bolus_series_[non_zero_bolus_idxs], 'D',
                    color='blue', markersize=8, markerfacecolor='None',
                    label='Bolus')
        axs[0].set_ylabel('Insulin')
        axs[0].legend()

        # Plot GLUCOSE prevalence in compartments
        axs[1].plot(xx, self.gt, color='orange',
                    linewidth=2, label='Glucose - Gut')
        axs[1].plot(xx, self.mt, color='red',
                    linewidth=2, label='Glucose - Plasma')
        axs[1].set_ylabel('Glucose')
        axs[1].legend()

        # Plot CGM and Model Predictions
        axs[2].plot(xx, self.Model.cgm_series_, color='black',
                    linewidth=2, label='CGM')
        axs[2].plot(xx, self.Gt, color='black', linewidth=2,
                    linestyle='dashed',
                    label='%d Min ahead prediction'%(self.Model.horizon))
        axs[2].set_ylabel('CGM')
        axs[2].legend()

        # Plot Variables in Bergman Model
        axs[3].plot(xx, self.Gt, color='blue', linewidth=2,
                    linestyle='dashed', label='Gt')
        axs[3].plot(xx, self.Xt, color='orange', linewidth=2, label='Xt')
        axs[3].plot(xx, self.It, color='red', linewidth=2, label='It')
        axs[3].set_ylabel('Bergman Models')

        accelerating_glucose = np.gradient(np.gradient(gc1))
        acc_g_idxs = np.where(accelerating_glucose > 0)[0]
        acc_g_vals = accelerating_glucose[acc_g_idxs]

        acc_rss = np.gradient((X[:,1]-Gt)**2)
        acc_rss_idxs = np.where(acc_rss > 5)[0]
        acc_rss_vals = acc_rss[acc_rss_idxs]

        # Add MEALS
        meals = np.where(self.Model.y_meals==1)[0]
        for m in meals:
          axs[0].axvline(x=m, color='green', alpha=.5, label='Logged Meal')
          axs[1].axvline(x=m, color='green', alpha=.5, label='Logged Meal')
          axs[2].axvline(x=m, color='green', alpha=.5, label='Logged Meal')
          axs[3].axvline(x=m, color='green', alpha=.5, label='Logged Meal')
