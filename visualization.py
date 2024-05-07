import matplotlib.pyplot as plt
import numpy as np

def correlation_plot(df, cmap="coolwarm"):
    return df.corr().style.background_gradient(cmap=cmap)

def plot(x, y, title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y, marker='o', color='b')

    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

def scatter(x, y, title, xlabel, ylabel, xmin=None, xmax=None, ymin=None, ymax=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y, marker='o', color='#2373a1', alpha=0.25)

    coeffs = np.polyfit(x, y, 1)
    trendline = np.polyval(coeffs, x)
    ax.plot(x, trendline, color='b', linewidth=1.5, label=f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')

    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.show()

def hist(series, title, xlabel, bins=20):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(series, bins=bins,  color='#2373a1')
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    plt.show()