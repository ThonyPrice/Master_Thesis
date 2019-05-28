import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import pandas as pd

def main(args):

    df = pd.read_pickle(path='./results/m-test.pkl')

    x = np.arange(1440)

    pu.figure_setup()

    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=(fig_size))

    # Insulin compartments...
    ax = fig.add_subplot(411)
    ax.plot(x, df['s1'].values[:1440], c='b', lw=pu.plot_lw())
    ax.plot(x, df['s2'].values[:1440], c='r', lw=pu.plot_lw())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\sigma(x)$')
    ax.set_axisbelow(True)

    # Glucose compartments...
    ax = fig.add_subplot(412)
    ax.plot(x, df['gt'].values[:1440], c='b', lw=pu.plot_lw())
    ax.plot(x, df['mt'].values[:1440], c='r', lw=pu.plot_lw())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\sigma(x)$')
    ax.set_axisbelow(True)

    # CGM and Gt...
    ax = fig.add_subplot(413)
    ax.plot(x, df['cgm_inputs_'].values[:1440], c='b', lw=pu.plot_lw())
    ax.plot(x, df['Gt'].values[:1440], c='r', lw=pu.plot_lw())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\sigma(x)$')
    ax.set_axisbelow(True)

    # Glucose, meals and detections...
    ax = fig.add_subplot(414)
    ax.plot(x, df['carb_ests_'].values[:1440], c='b', lw=pu.plot_lw())

    meals = df['y_meals'].values[:1440]
    meals[ meals==0 ] = np.nan
    ax.scatter(x, meals, c='r', lw=pu.plot_lw())

    meal_preds = df['meal_preds_'].values[:1440]
    meal_preds[ meal_preds==0 ] = np.nan
    ax.scatter(x, meal_preds, c='g', lw=pu.plot_lw())
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\sigma(x)$')
    ax.set_axisbelow(True)

    plt.grid()
    plt.tight_layout()

    if args.save:
        pu.save_fig(fig, args.save)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
