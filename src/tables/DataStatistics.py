import numpy as np
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

        # Set dateTimetimeIndex
        df['date'] = df['date'].apply(lambda x: x[:19])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        stats = df.copy(deep=True)

        categories = stats[[
            'units',
            'quantity',
            'meal']]

        stats['type'] = categories.idxmax(axis=1)
        stats = stats.groupby(['user_id']).describe()

        # Select the relevant stats for each data type
        stats_cgm = stats.xs('quantity', axis=1)[[
                'count', 'mean', 'std', 'min', 'max']]
        stats_cgm = pd.concat([stats_cgm], axis=1, keys=['CGM'])


        z = df[df['units'] != 0]
        z = z.groupby('user_id').count()[['units']]\
             .rename(columns={'units':'count'})
        stats_units = stats.xs('units', axis=1)[['mean', 'std']]
        stats_units = z.join(stats_units)
        stats_units = pd.concat([stats_units], axis=1, keys=['Bolus'])

        stats_meals = df.groupby('user_id').sum()[['meal']]

        stats_days = df[['user_id']].copy(deep=True)
        stats_days.loc[:, 'date'] = stats_days.index.date
        stats_days = stats_days.groupby('user_id')[['date']].nunique()
        stats_duration = pd.concat([stats_days.join(stats_meals)],
                                    axis=1,
                                    keys=['#'])

        stats = pd.concat([stats_duration, stats_cgm, stats_units], axis=1)

        # Anonymize users
        stats.sort_values(('#', 'date'), ascending=False, inplace=True)
        stats['id'] = np.arange(stats.shape[0])
        stats.set_index('id', inplace=True)
        self.stats = stats.round(1)

        return self.stats


    def to_tex(self, file_name='./'):
        """Export statistics summary as LaTex table"""
        self.stats.to_latex(buf=file_name)
