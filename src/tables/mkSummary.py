import argparse

import pandas as pd

from DataStatistics import DataStatistics



"""
Chapter: Methods

Description:
Read CGM tracking data and mk data summary LaTeX table.

Script:
python3 mkSummary.py --save tables/latex/data_summary.tex
"""


# File path to read CGM data from
F_NAME = 'data/minimal_tracking_data.csv'


def main(args):

    # Fetch data
    df = pd.read_csv(F_NAME)

    # Print data stats
    DataStats = DataStatistics(df)
    print(DataStats.summary())

    # If requested, save
    if args.save:
        DataStats.to_tex(args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
