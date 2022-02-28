import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_treemap(df, path_vars, values_vars, color_vals, save_path, auto_open=True):
    """
        Plot a plotly treemap from a pandas dataframe.

    :param df: pandas dataframe object
    :param path_vars:
    :param values_vars:
    :param color_vals:
    :param auto_open: a boolean indicating whether figure should pop in the browser
                      automatically or not.
    :return:
    """
    fig = px.treemap(df,
                     path=path_vars,
                     values=values_vars,
                     color=color_vals
                    )

    plotly.offline.plot(fig, filename=save_path, auto_open=auto_open)

def plot_hist(series, bins=range(0, 60, 5), save_path=None, auto_open=True):
    """
    Plot a plotly histogram from a pandas series.
    :param series: pandas series object with numeric values
    :param bins: number of bins to plot or a range defining the bins
    :param save_path: a path for saving the html file with the figure
    :param auto_open: a boolean indicating whether figure should pop in the browser
                      automatically or not.
    """

    # create the bins
    counts, bins = np.histogram(series, bins=bins)
    bins = 0.5 * (bins[:-1] + bins[1:])

    fig = px.bar(x=bins, y=counts,
                 labels={'x': series.name if series.name is not None else '',
                         'y': 'count'})
    plotly.offline.plot(fig, filename=save_path, auto_open=auto_open)
    return fig

# from plotly.colors import n_colors
# redVSblue = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', 20, colortype='rgb') # color map with 20 color from blue to red