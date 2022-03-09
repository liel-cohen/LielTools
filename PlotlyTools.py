import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys

if 'LielTools' in sys.modules:
    from LielTools import DataTools
else:
    import DataTools

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

def plot_scatter(df, x_col, y_col, x_title=None, y_title=None, hover_data=None, marker_size=8,
                 line_color='black', line_width=1, ylim=None, xlim=None,
                 save_to_path_html=None, save_to_path_jpg=None, auto_open=True,
                 add_line=False, line_x0=0, line_y0=0, line_x1=1, line_y1=1):
    """
    Create Plotly scatter plot.
    @param df: pd.DataFrame holding the data.
    @param x_col: str. Name of column in df to plot in x axis
    @param y_col: str. Name of column in df to plot in y axis
    @param x_title: str. x axis title text to add. Default None, then uses x_col.
    @param y_title: str. y axis title text to add. Default None, then uses y_col.
    @param hover_data: list of strings. Names of columns in df to add to hover. Default None
    @param marker_size: int. size of plot marker. Default 8
    @param line_color: color of marker line. Default 'black'
    @param line_width: width of marker line. Default 1
    @param ylim: [number, number]. limits of y axis limits. Default None (
    @param xlim:
    @param save_to_path_html:
    @param save_to_path_jpg:
    @param auto_open:
    @param add_line:
    @param line_x0:
    @param line_y0:
    @param line_x1:
    @param line_y1:
    @return:
    """
    if x_title is None:
        x_title = x_col
    if y_title is None:
        y_title = y_col

    fig = px.scatter(df, x=x_col, y=y_col,
                     labels={x_col: x_title,
                             y_col: y_title},
                     hover_data=hover_data)
    fig.update_traces(marker=dict(size=marker_size, line=dict(width=line_width,
                                                     color=line_color)))
    if add_line:
        fig.add_shape(type="line", x0=line_x0, y0=line_y0, x1=line_x1, y1=line_y1)

    if ylim is not None:
        fig.update_layout(yaxis_range=ylim)

    if xlim is not None:
        fig.update_layout(xaxis_range=xlim)

    if save_to_path_html is None:
        plotly.offline.plot(fig, auto_open=auto_open)
    else:
        plotly.offline.plot(fig, filename=save_to_path_html, auto_open=auto_open)

    if save_to_path_jpg is not None:
        fig.write_image(save_to_path_jpg, width=800, height=600)


# from plotly.colors import n_colors
# redVSblue = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', 20, colortype='rgb') # color map with 20 color from blue to red