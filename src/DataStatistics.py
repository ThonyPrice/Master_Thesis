import pandas as pd

class DataStatistics(object):
    """Calculate basic data overview for Pandas DataFrame containing:
        - Multiple users, and for each user the following data points:
            - bloodGlucoseSampleEvent: CGM measurements at frequent intervals.
            - insulinDeliverySampleEvent: Logs of administered insulin in units.
            - meal: Binary labels vector weather a meal was logged or not.
    """

    def __init__(self, df):
        self.df = df
        self.stats = None


    def summary(self):
        """Construct summary statistics for dataFrame. Included stats are
        similar to pandas describe() function but adds separate stats for
        each user
        """

        df = self.df.copy(deep=True)
        stats = df.copy(deep=True)

        categories = stats[[
            'bloodGlucoseSampleEvent',
            'insulinDeliverySampleEvent',
            'meal']]
        stats['type'] = categories.idxmax(axis=1)
        stats = stats.groupby(['user_id']).describe()

        # Select the relevant stats for each data type
        stats_cgm = stats.xs('quantity', axis=1)[[
                'count', 'mean', 'std', 'min', 'max']]
        stats_cgm = pd.concat([stats_cgm], axis=1, keys=['CGM'])

        stats_units = stats.xs('units', axis=1)[['count', 'mean', 'std']]
        stats_units = pd.concat([stats_units], axis=1, keys=['Bolus'])

        stats_meals = df.groupby('user_id').sum()[['meal']]
        stats_days = df[['user_id']].copy(deep=True)
        stats_days.loc[:, 'date'] = stats_days.index.date
        stats_days = stats_days.groupby('user_id')[['date']].nunique()
        stats_duration = pd.concat([stats_days.join(stats_meals)],
                                    axis=1,
                                    keys=['#'])

        # Merge all stats to Summary DataFrame
        stats = pd.concat([stats_duration, stats_cgm, stats_units], axis=1)
        self.stats = stats.round(1)
        return stats


    def to_tex(self, export_path='./data_summary.tex'):
        """Export statistics summary as LaTex table"""
        self.stats.to_latex(buf=path)
