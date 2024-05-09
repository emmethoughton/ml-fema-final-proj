import matplotlib.pyplot as plt
import numpy as np

''' 
Produces a correlation table of the columns of a pandas data frame
    df = the pandas dataframe of interest
    cmap = colormap for table
'''
def correlation_plot(df, cmap="coolwarm"):
    return df.corr().style.background_gradient(cmap=cmap)

'''
Draw a line plot of y against x 
    x = the set for the horizontal axis
    y = the set for the vertical axis
    title = the plot title
    xlabel = label for x variable
    ylabel = label for y variable
'''
def plot(x, y, title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y, marker='o', color='b')

    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

'''
Create a scatterplot of y against x
    x = the set for the horizontal axis
    y = the set for the vertical axis
    title = the plot title
    xlabel = label for x variable
    ylabel = label for y variable 
    xmin = lower bound for x-axis
    xmax = upper bound for x-axis
    ymin = lower bound for y-axis
    ymax = upper bound for y-axis
'''
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

'''
Create a histogram for a series
    series = the vector to plot
    title = the title of the histogram
    xlabel = name of the variable
    bins = number of bins to bucket the elements into
'''
def hist(series, title, xlabel, bins=20):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(series, bins=bins,  color='#2373a1')
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    plt.show()