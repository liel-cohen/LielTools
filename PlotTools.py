import sys

if 'LielTools' in sys.modules:
    from LielTools import DataTools
    from LielTools import StatsTools
else:
    import DataTools
    import StatsTools

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import palettable
import itertools
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.stats import spearmanr
# from pandas.tools.plotting import parallel_coordinates
from pandas.plotting import parallel_coordinates
import scipy.stats as stats

#### ----------------------------------- Figure Drawing ------------------------------------- ###

# still needs fixing the hue in str
# former plotBoxplot
def plot_boxplot(seriesX, seriesY, seriesHue=None,
                 stripplot=True, boxplot=True,
                 ax=None, order=None, palette=None,
                 figsize=(7, 6), showfliers=False, plotTitle='', xTitle='', yTitle='',
                 xRotation=45, titleFontSize=18, titleColor='maroon', legendTitle='',
                 font_scale=1, snsStyle='ticks', boxTransparency=0.6, jitter=0.15,
                 stripplot_alpha=0.7, stripplot_size=4, stripplot_color=None,
                 linewidth=0, stripplot_palette=None,
                 xy_title_fontsize=None,
                 boxplot_color=None,

                 add_mean=False,
                 mean_marker='_', mean_color='red',
                 mean_size=100, mean_linewidth=3, mean_alpha=1,

                 add_gmean=False,
                 gmean_marker='_', gmean_color='blue',
                 gmean_size=100, gmean_linewidth=3, gmean_alpha=1,

                 color_indices=None, color_indices_color='red', color_indices_size=20,
                 color_indices_linewidth=0, color_indices_alpha=1, color_indices_marker='o',

                 hide_indices_in_stripplot=None,
                 horizontal=False,

                 saveFolder=None, save_path=None, dpi=300):
    """
    Function for plotting a boxplot and a stripplot over it (or only one of them).
    Can also add the mean or geometric values for each box - this option currently
    works only if seriesHue is None.
    Can color the stripplot markers for specific given indices in a different color.
    seriesX, seriesY and seriesHue (if given) must have the same indices.

    @param seriesX: pd.Series. The values to be plotted on the x axis - must be categorical/dicrete.
    @param seriesY: pd.Series. The values to be plotted on the y axis - must be numeric.
    @param seriesHue: pd.Series. Optional. Categories by which the x levels should be split. Default None
    @param stripplot: boolean. Whether a stripplot should be plotted.
    @param boxplot: boolean. Whether a boxplot should be plotted.
    @param ax: matplotlib axes object on which plot should be plotted.
               Default None (then a new object will be created).
    @param order: list of x axis (unique) values by the order they should be displayed (from left to right).
    @param palette: str. color palette to be used for the boxplot.
                    Will also be used for the stripplot if stripplot_palette and stripplot_color are None.
    @param figsize: tuple of 2 numbers representing the figure size to be plotted (width, height). Default (7, 6)
    @param showfliers: boolean. Whether the fliers should be plotted on the boxplot. Default False.
    @param plotTitle:
    @param xTitle:
    @param yTitle:
    @param xRotation:
    @param titleFontSize:
    @param titleColor:
    @param legendTitle:
    @param font_scale:
    @param snsStyle:
    @param boxTransparency:
    @param jitter:
    @param stripplot_alpha:
    @param stripplot_size:
    @param stripplot_color:
    @param linewidth:
    @param stripplot_palette:
    @param xy_title_fontsize:
    @param boxplot_color: str. Color to be used for all the boxplot elements
                          (if you don't want to use boxplot_palette).
                          Will also be used for the stripplot if stripplot_palette, stripplot_color and palette are all None.
    @param add_mean: boolean. Whether to add a marker showing the mean value for each boxplot.
                     Currently works only if hue isn't used. Default False
    @param mean_marker: str. Marker symbol for the mean. Default '_'
    @param mean_color: str. Color for the mean. Default 'red'
    @param mean_size:
    @param mean_linewidth:
    @param mean_alpha:
    @param add_gmean: boolean. Whether to add a marker showing the geometric mean value for each boxplot.
                     Currently works only if hue isn't used. Default False
    @param gmean_marker: str. Marker symbol for the gmean. Default '_'
    @param gmean_color: str. Color for the mean. Default 'blue'
    @param gmean_size:
    @param gmean_linewidth:
    @param gmean_alpha:
    @param color_indices: list of indices (from the given data series).
                          These indices stripplot markers will be colored
                          with a different color. Default None.
    @param color_indices_color: str. Color for the color_indices. Default 'red'
    @param color_indices_size: int. Marker size for the color_indices. Default 20
    @param color_indices_linewidth: float. Marker linewidth for the color_indices. Default 0
    @param color_indices_alpha: float. Marker alpha for the color_indices. Default 1
    @param color_indices_marker: str. Marker symbol for the color_indices. Default 'o'
    @param hide_indices_in_stripplot: list of indices (from the given data series).
                                      Specific indices that should not be plotted in the stripplot.
                                      Default None.
    @param horizontal:
    @param saveFolder: str. folder in which to save the plot in jpg file.
                       File name will be automatically formatted using the names
                       of x, y and hue titles. Default None.
    @param save_path: str. full file path for saving the plot to file system. Default None.
    @param dpi: int. The dpi value for file saving. Default 300.
    @return: matplotlib axes object
    """
    # * hide_indices_in_stripplot - don't plot specific values in stripplot
    # (not all given indices must be contained in df.index)

    sns.set(font_scale=font_scale)
    sns.set_context(font_scale=font_scale)
    sns.set_style(snsStyle)

    if color_indices is not None:
        if hide_indices_in_stripplot is None:
            hide_indices_in_stripplot = color_indices
        else:
            for ind in color_indices:
                if ind not in hide_indices_in_stripplot:
                    hide_indices_in_stripplot.append(ind)

    if stripplot:
        if stripplot_color is None and stripplot_palette is None:
            if palette is not None:
                stripplot_palette = palette
            elif boxplot_color is not None:
                stripplot_color = boxplot_color
            else:
                stripplot_color = 'black'

    data = DataTools.join_non_empty_series_f_list([seriesX, seriesY, seriesHue])

    if xTitle=='':
        xTitle = DataTools.get_col_name(seriesX)
    if yTitle=='':
        yTitle = DataTools.get_col_name(seriesY)

    if seriesHue is not None:
        if legendTitle == '': legendTitle = DataTools.get_col_name(seriesHue)

    fontTitle = {'size': titleFontSize, 'color': titleColor, 'weight': 'bold'}
    if ax is None:
        fix, ax = plt.subplots(figsize=figsize)

    if order is None:
        if horizontal is False:
            order = list(seriesX.unique())
        else:
            order = list(seriesY.unique())

    if stripplot:
        stripplot_data = data.copy()
        if hide_indices_in_stripplot is not None:
            stripplot_data = DataTools.get_df_without_indices(stripplot_data,
                                                              hide_indices_in_stripplot)

    if seriesHue is not None: # hue exists
        if boxplot:
            sns.boxplot(data=data, x=DataTools.get_col_name(seriesX),
                        y=DataTools.get_col_name(seriesY), ax=ax,
                        showfliers=showfliers,
                        hue=DataTools.get_col_name(seriesHue),
                        boxprops=dict(alpha=boxTransparency),
                        palette=palette, order=order, color=boxplot_color)
        if stripplot:
            sns.stripplot(data=stripplot_data, x=seriesX.name, y=seriesY.name,
                          hue=DataTools.get_col_name(seriesHue), ax=ax,
                          jitter=jitter, alpha=stripplot_alpha,
                          edgecolor='black', linewidth=linewidth,
                          size=stripplot_size, color=stripplot_color,
                          palette=stripplot_palette, order=order, split=True)
    else:                     # no hue
        if boxplot:
            sns.boxplot(data=data, x=DataTools.get_col_name(seriesX), ax=ax,
                        y=DataTools.get_col_name(seriesY), showfliers=showfliers,
                        boxprops=dict(alpha=boxTransparency),
                        palette=palette, order=order, color=boxplot_color)
        if stripplot:
            sns.stripplot(data=stripplot_data, x=DataTools.get_col_name(seriesX),
                          y=DataTools.get_col_name(seriesY), ax=ax,
                          jitter=jitter, alpha=stripplot_alpha,
                          edgecolor='black', linewidth=linewidth,
                          size=stripplot_size, color=stripplot_color,
                          palette=stripplot_palette, order=order)

        if add_mean: # currently only work if Hue=None # TODO
            for i, label in enumerate(order):
                mean = data.loc[data[DataTools.get_col_name(seriesX)] == label,
                                DataTools.get_col_name(seriesY)].mean()

                ax.scatter(i, mean, s=mean_size, linewidth=mean_linewidth,
                           c=mean_color, marker=mean_marker, alpha=mean_alpha,
                           zorder=10)

        if add_gmean: # currently only work if Hue=None # TODO
            for i, label in enumerate(order):
                gmean = data.loc[data[DataTools.get_col_name(seriesX)] == label,
                                DataTools.get_col_name(seriesY)].apply(stats.gmean)

                ax.scatter(i, gmean, s=gmean_size, linewidth=gmean_linewidth,
                           c=gmean_color, marker=gmean_marker, alpha=gmean_alpha,
                           zorder=10)

    if stripplot:
        if color_indices is not None:
            for ind_to_color in color_indices:
                xticklabel_found = False
                for i, xticklabel in enumerate(ax.get_xticklabels()):
                    xticklabel_text = xticklabel.get_text()

                    # color marker
                    if str(data.loc[ind_to_color, DataTools.get_col_name(seriesX)]) == xticklabel_text:
                        xticklabel_found = True
                        ax.scatter(i+(2*jitter*(random.random()-0.5)), data.loc[ind_to_color, DataTools.get_col_name(seriesY)],
                                   s=color_indices_size, linewidth=color_indices_linewidth,
                                   c=color_indices_color, marker=color_indices_marker, alpha=color_indices_alpha,
                                   zorder=10)

                if xticklabel_found is False:
                    raise Exception(f'Did not find an xticklabel_text that matches index {ind_to_color}! Please check why.')

    ax.set_title(plotTitle, fontdict=fontTitle)
    for tick in ax.get_xticklabels():
        tick.set_rotation(xRotation)
    if (xTitle != None):
        ax.set_xlabel(xTitle)
    if (yTitle != None):
        ax.set_ylabel(yTitle)

    if xy_title_fontsize is not None:
        ax.yaxis.label.set_size(fontsize=xy_title_fontsize)
        ax.xaxis.label.set_size(fontsize=xy_title_fontsize)

    if seriesHue is not None:
        # Get the handles and labels.
        handles, labels = ax.get_legend_handles_labels()
        num_wanted_handels = len(handles)

        ax.legend_.remove()

        if stripplot:
            # When creating the legend, only use the first half of the elements
            # in order to remove the last half - that is the stripplot dots.
            num_wanted_handels = int(num_wanted_handels/2)

        plt.legend(handles[0:num_wanted_handels],
                   labels[0:num_wanted_handels],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   frameon=False, title=legendTitle)

        # if legend is binary, change 0,1 to no,yes
        leg = ax.get_legend()
        if leg is not None:
            legText = leg.get_texts()
            legText = bin_text_to_yes_no(legText)

    # if X is binary, change 0,1 to no,yes
    xticksText = ax.get_xticklabels()
    xticksText = bin_text_to_yes_no(xticksText)
    ax.set_xticklabels(xticksText)

    plt.tight_layout()

    if saveFolder is not None:
        fileName = 'Boxplot - ' + xTitle + ' VS ' + yTitle
        if seriesHue is not None:
            fileName = fileName + ' BY ' + legendTitle + '.jpg'
        else:
            fileName = fileName + '.jpg'
        save_plt(save_path=saveFolder + fileName, dpi=dpi)

    if save_path is not None:
        save_plt(save_path=save_path, dpi=dpi)

    return ax

# former plotBoxplotDF
def plot_boxplot_df(df, stripplot=True, saveFolder=None, figsize=(7, 6),
                    ax=None, showfliers=False,
                    plotTitle='', xTitle='', yTitle='',
                    xRotation=45, titleFontSize=18, titleColor='maroon',
                    font_scale=1, snsStyle='ticks', boxTransparency=0.6,
                    savePath=None, ylim=None, jitter=0.15,
                    stripplot_linewidth=1.5, stripplot_transparency=0.6,
                    stripplot_marker_size=4, plt_show=True,
                    color_boxplot=None, color_stripplot=None,
                    add_mean=False,
                    mean_marker='_', mean_color='red',
                    mean_size=100, mean_linewidth=3, mean_alpha=1,
                    hide_stripplot_mask=None):
    # hide_stripplot_mask: a mask for df. will not plot a stripplot marker for a cell with True value.

    sns.set(font_scale=font_scale)
    sns.set_style(snsStyle)


    fontTitle = {'size': titleFontSize, 'color': titleColor, 'weight': 'bold'}
    plt.figure(figsize=figsize)

    if ax is None:
        ax = sns.boxplot(data=df, showfliers=showfliers,
                         boxprops=dict(alpha=boxTransparency),
                         color=color_boxplot)
    else:
        sns.boxplot(data=df, showfliers=showfliers,
                    boxprops=dict(alpha=boxTransparency), ax=ax,
                    color=color_boxplot)

    if stripplot:
        stripplot_df = df.copy()

        if hide_stripplot_mask is not None:
            stripplot_df[hide_stripplot_mask] = np.nan

        sns.stripplot(data=stripplot_df, ax=ax, jitter=jitter, size=stripplot_marker_size,
                        alpha=stripplot_transparency,
                        edgecolor='black', linewidth=stripplot_linewidth,
                        color=color_stripplot)

    ax.set_title(plotTitle, fontdict=fontTitle)
    for tick in ax.get_xticklabels():
        tick.set_rotation(xRotation)
    if xTitle is not None:
        ax.set_xlabel(xTitle)
    if yTitle is not None:
        ax.set_ylabel(yTitle)
    if ylim is not None:
        ax.set_ylim(ylim)

    # if X is binary, change 0,1 to no,yes
    xticksText = ax.get_xticklabels()
    xticksText = bin_text_to_yes_no(xticksText)
    ax.set_xticklabels(xticksText)

    if add_mean:
        for i, col in enumerate(df.columns):
            ax.scatter(i, df[col].mean(), s=mean_size, linewidth=mean_linewidth,
                       c=mean_color, marker=mean_marker, alpha=mean_alpha,
                       zorder=10)

    plt.tight_layout()
    if saveFolder is not None:
        fileName = 'Boxplot - ' + xTitle + ' VS ' + yTitle + '.jpg'
        save_plt(save_path=saveFolder + fileName, show_if_none=plt_show)
    if savePath is not None:
        save_plt(save_path=savePath, show_if_none=plt_show)

    return ax

def plot_clustermap(numbersTable, cmap='YlGnBu', figsize=(8, 8),
                    title='', title_fontsize=13, title_y_padding=0,
                    adjRight=0.8, adjBottom=0.3, adjLeft=None, adjTop=None,
                    row_clustering=True, col_clustering=True,
                    font_scale=1, snsStyle='ticks', vmin=None, vmax=None,
                    xlabel='', ylabel='', xRotation=0, yRotation=0,
                    xy_labels_fontsize=None,

                    mask=None,

                    cbar_title='', cbar_orient='vertical',
                    cbar_pos=None,
                    cbar_vertical_left=False, cbar_vertical_left_x_factor=3/5,
                    cbar_title_fontsize=None, cbar_ticks_fontsize=None,
                    hide_cbar=False,

                    linewidths=0, linecolor='white',

                    row_color_vals=None, row_cmap='Blues',
                    row_vmin=None, row_vmax=None,
                    row_color_labels=None,
                    row_color_lab_legend=True, row_color_lab_legend_loc='lower center',
                    row_color_lab_legend_ncol=4, row_color_labels_cmap='Set1',
                    row_color_labels_cmap_dict=None, row_color_labels_order=None,
                    row_color_legend_frameon=True,

                    col_color_vals=None, col_cmap='Blues',
                    col_vmin=None, col_vmax=None,
                    col_color_labels=None,
                    col_color_lab_legend=True, col_color_lab_legend_loc='lower center',
                    col_color_lab_legend_ncol=4, col_color_labels_cmap='Set1',
                    col_color_labels_cmap_dict=None, col_color_labels_order=None,
                    col_color_legend_frameon=True,

                    rowcol_color_legend_fontsize=10, rowcol_color_legend_title='',
                    rowcol_color_legend_title_fontsize=11,

                    col_names_to_frame=None, row_names_to_frame=None,
                    names_frame_color='black', names_frame_width=4,

                    xticklabels='auto', yticklabels='auto',
                    hide_ticks=False,

                    fix_smaller_rows_at_y_edges_bug=False):
    """

    :param numbersTable:
    :param cmap:
    :param figsize:
    :param title:
    :param title_fontsize:
    :param title_y_padding: defult 0. y position padding of title. Try enlarging with jumps of 10
                            to get the title up.
                            If left as 0, and col_color_lab_legend is True or row_color_lab_legend is True,
                            will autimatically set title_y_padding=80.
    :param adjRight:
    :param adjBottom:
    :param adjLeft:
    :param adjTop:
    :param row_clustering:
    :param col_clustering:
    :param font_scale:
    :param snsStyle:
    :param vmin:
    :param vmax:
    :param xlabel:
    :param ylabel:
    :param xRotation:
    :param yRotation:
    :param xy_labels_fontsize: x and y axis title font size
    :param cbar_title:
    :param cbar_orient: 'vertical' or 'horizontal'
    :param cbar_title_fontsize: colormap title font size
    :param cbar_ticks_fontsize: colormap tick labels font size
    :param cbar_pos: colorbar position
    :param cbar_vertical_left: bool. If True, sets cbar vertically to the left
                               of the entire heatmap (ignoring cbar_pos)
    :param cbar_vertical_left_x_factor: float between 0 and 1. A factor that determines the
            x axis right limit of the cbar. The smaller the value is, the thinner the cbar will be.
    :param hide_cbar: hide colormap (True / False)
    :param linewidths: heatmap grid width
    :param linecolor: heatmap grid color

    :param row_color_vals: pd.Series of values by which to color the rows.
                           or, a list of pd.Series, each will create a different
                           color strip for coloring the rows. If given a list,
                           row_cmap, row_vmin, and row_vmax should also be lists!
    :param row_cmap: either a colormap name (string), or a list of colors,
                     for example from PlotTools.getColorsList(n)['colors']
    :param row_vmin: row_color_vals - colorman vmin
    :param row_vmax: row_color_vals - colorman vmax

    :param row_color_labels: Series with (categorical) labels for rows.
    :param mask: which cells not to show (will show empty cell - not colored)
    :param row_color_lab_legend: boolean - show legend of row colors
    :param row_color_lab_legend_loc: row color legend location (in heatmap axes).
                                     string or pair of floats.
                                     strings: 'center', 'best', 'upper left', 'upper right',
                                     'lower left', 'lower right'
                                     'upper center', 'lower center', 'center left', 'center right'
    :param row_color_lab_legend_ncol: row color legend number of columns
    :param row_color_labels_cmap: row color seaborn colormap
    :param row_color_labels_cmap_dict: row color dictionary of colors per labels
    :param row_color_labels_order: row color legend display order (list of labels)
    :param row_color_legend_frameon: show legend frame (True / False)
    @ Same params for col exist as well!
    :param rowcol_color_legend_fontsize: legend font size for row/col colors
    :param rowcol_color_legend_title: legend title for row/col colors

    :param col_names_to_frame: list of names of columns to draw frame over.
    :param row_names_to_frame: list of names of rows to draw frame over.
    :param names_frame_color: color of frame to draw over cols/rows
    :param names_frame_width: width of frame to draw over cols/rows

    :param xticklabels: labels for x ticks. either 'auto' or a list of tick-labels (strings) or an empty list for no tick-labels at all.
    :param yticklabels: labels for y ticks. either 'auto' or a list of tick-labels (strings) or an empty list for no tick-labels at all.
    :param hide_ticks: boolean. hide the little tickmarks next to the tick labels.

    :param fix_smaller_rows_at_y_edges_bug: some matplotlib versions have a bug where
                    they cut the first and last rows of the matrix in the heatmap.
                    This flag set to True should fix it.
    :return: grid object
    """
    sns.set(font_scale=font_scale)
    sns.set_style(snsStyle)

    def row_color_vals_to_colors(row_color_vals, row_cmap, row_vmin, row_vmax):
        if type(row_cmap) is str:  # row_cmap is a colormap name (string)
            row_cmap = cm.get_cmap(row_cmap)
            row_colors_norm = matplotlib.colors.Normalize(vmin=row_vmin, vmax=row_vmax)
            row_colors = row_cmap(row_colors_norm(row_color_vals.values))
        else:  # row_cmap is a list of colors
            row_cmap = categorCmapFromList(row_cmap)
            row_colors = row_cmap(row_color_vals.values)

        return pd.Series(row_colors.tolist(), index=row_color_vals.index,
                                              name=row_color_vals.name)

    # row side colors
    if row_color_vals is not None:
        if type(row_color_vals) is not list:
            row_colors = row_color_vals_to_colors(row_color_vals, row_cmap, row_vmin, row_vmax)
        else:
            row_colors = []
            for row_color_vals_k, row_cmap_k, row_vmin_k, row_vmax_k in zip(row_color_vals, row_cmap, row_vmin, row_vmax):
                row_colors_k = row_color_vals_to_colors(row_color_vals_k, row_cmap_k, row_vmin_k, row_vmax_k)
                row_colors.append(row_colors_k)
            row_colors = pd.concat(row_colors, axis=1)

    elif row_color_labels is not None:
        if row_color_labels_cmap_dict is None:
            row_color_labels_cmap_dict = dict(zip(set(row_color_labels),
                                    sns.mpl_palette(row_color_labels_cmap,
                                                    n_colors=len(set(row_color_labels)))))
            if len(row_color_labels_cmap_dict) != len(set(row_color_labels)):
                raise Exception('row_color_labels_cmap - cmap doesnt have enough colors for all values.\nPlease use another cmap.')

        row_colors = pd.DataFrame(row_color_labels).iloc[:,0].map(row_color_labels_cmap_dict)
    else:
        row_colors = None

    # col side colors
    if col_color_vals is not None:
        col_cmap = cm.get_cmap(col_cmap)
        col_colors_norm = matplotlib.colors.Normalize(vmin=col_vmin, vmax=col_vmax)
        col_colors = col_cmap(col_colors_norm(col_color_vals.values))
    elif col_color_labels is not None:
        if col_color_labels_cmap_dict is None:
            col_color_labels_cmap_dict = dict(zip(set(col_color_labels),
                                                  sns.mpl_palette(col_color_labels_cmap,
                                                                  len(set(col_color_labels)))))
            if len(col_color_labels_cmap_dict) != len(set(col_color_labels)):
                raise Exception('col_color_labels_cmap - cmap doesnt have enough colors for all values.\nPlease use another cmap.')
        col_colors = pd.DataFrame(col_color_labels).iloc[:,0].map(col_color_labels_cmap_dict)
    else:
        col_colors = None

    grid = sns.clustermap(numbersTable, cmap=cmap, figsize=figsize,
                          row_cluster=row_clustering, col_cluster=col_clustering,
                          cbar_kws={'label': cbar_title,
                                    'orientation': cbar_orient},
                          row_colors=row_colors, col_colors=col_colors,
                          mask=mask, vmin=vmin, vmax=vmax,
                          linewidths=linewidths, linecolor=linecolor,
                          yticklabels=yticklabels, xticklabels=xticklabels,
                          )

    if fix_smaller_rows_at_y_edges_bug:
        grid.ax_heatmap.set_ylim(len(numbersTable)+0.5, -1)
        grid.ax_row_colors.set_ylim(len(numbersTable)+0.5, -1)

    # Add plot title
    if title != '':
        if col_color_lab_legend or row_color_lab_legend and title_y_padding==0:
            title_y_padding = 80

        if col_clustering or (col_color_labels is not None):
            grid.ax_col_dendrogram.set_title(title, fontdict={'fontsize': title_fontsize,
                                                       'fontweight': 'bold'}, pad=title_y_padding)
        else:
            grid.ax_heatmap.set_title(title, fontdict={'fontsize': title_fontsize,
                                                       'fontweight': 'bold'}, pad=title_y_padding)

    grid.fig.subplots_adjust(right=adjRight, bottom=adjBottom, left=adjLeft, top=adjTop)
    grid.ax_heatmap.set_xlabel(xlabel)
    grid.ax_heatmap.set_ylabel(ylabel)
    for tick in grid.ax_heatmap.get_xticklabels():
        tick.set_rotation(xRotation)
    for tick in grid.ax_heatmap.get_yticklabels():
        tick.set_rotation(yRotation)

    if cbar_vertical_left is True:
        heatmap_start_x = grid.ax_heatmap.get_position().get_points()[0,0]
        heatmap_end_x = grid.ax_heatmap.get_position().get_points()[1,0]
        heatmap_start_y = grid.ax_heatmap.get_position().get_points()[0,1]
        heatmap_end_y = grid.ax_heatmap.get_position().get_points()[1,1]
        cbar_pos = [heatmap_start_x/4, heatmap_start_y, # [left, bottom, width, height]
                    heatmap_start_x*cbar_vertical_left_x_factor , heatmap_end_y-heatmap_start_y]

    if cbar_pos is not None:
        grid.cax.set_position(cbar_pos)
        print('Warning: if using command plt.tight_layout or subplots_adjust, cax position \n'
              'may be distorted and has to be redefined after the command.')

    if cbar_title_fontsize is not None:
        grid.cax.yaxis.label.set_size(cbar_title_fontsize)

    if cbar_ticks_fontsize is not None:
        grid.cax.yaxis.axes.tick_params(axis='both', which='major',
                                        labelsize=cbar_ticks_fontsize)

    if hide_cbar:
        grid.cax.set_visible(False)

    if xy_labels_fontsize is not None:
        grid.ax_heatmap.yaxis.label.set_size(fontsize=xy_labels_fontsize)
        grid.ax_heatmap.xaxis.label.set_size(fontsize=xy_labels_fontsize)

    # row side colors legend
    if row_color_labels is not None and row_color_lab_legend:
        if row_color_labels_order is None:
            row_color_labels_order = row_color_labels.unique()
        for label in row_color_labels_order:
            grid.ax_col_dendrogram.bar(0, 0, color=row_color_labels_cmap_dict[label],
                                    label=label, linewidth=0)
        grid.ax_col_dendrogram.legend(loc=row_color_lab_legend_loc,
                                      ncol=row_color_lab_legend_ncol,
                                      prop={"size": rowcol_color_legend_fontsize},
                                      title=rowcol_color_legend_title,
                                      title_fontsize=rowcol_color_legend_title_fontsize,
                                      frameon=row_color_legend_frameon)

    # col side colors legend
    if col_color_labels is not None and col_color_lab_legend:
        if col_color_labels_order is None:
            col_color_labels_order = col_color_labels.unique()
        for label in col_color_labels_order:
            grid.ax_col_dendrogram.bar(0, 0, color=col_color_labels_cmap_dict[label],
                                    label=label, linewidth=0)
        grid.ax_col_dendrogram.legend(loc=col_color_lab_legend_loc,
                                   ncol=col_color_lab_legend_ncol,
                                   frameon=col_color_legend_frameon)

    # Source: https://stackoverflow.com/questions/62533046/how-to-add-color-border-or-similar-highlight-to-specifc-element-of-heatmap-in-py
    # Add frames to columns
    if col_names_to_frame is not None:
        for col_name_to_frame in col_names_to_frame:
            wanted_col_index = list(numbersTable.columns).index(col_name_to_frame)
            if col_clustering == True:
                wanted_col_index = grid.dendrogram_col.reordered_ind.index(wanted_col_index)

            x, y, w, h = (wanted_col_index, 0, 1, numbersTable.shape[0])
            grid.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False,
                                                edgecolor=names_frame_color, lw=names_frame_width, clip_on=False))
            grid.ax_heatmap.tick_params(length=0)

    # Add frames to rows
    if row_names_to_frame is not None:
        for row_name_to_frame in row_names_to_frame:
            wanted_row_index = list(numbersTable.index).index(row_name_to_frame)
            if row_clustering == True:
                wanted_row_index = grid.dendrogram_row.reordered_ind.index(wanted_row_index)

            x, y, w, h = (0, wanted_row_index, numbersTable.shape[1], 1)
            grid.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False,
                                                edgecolor=names_frame_color, lw=names_frame_width, clip_on=False))
            grid.ax_heatmap.tick_params(length=0)

    if hide_ticks:
        grid.ax_heatmap.tick_params(left=False, bottom=False, top=False, right=False)

    if row_colors is not None:
        grid.ax_row_colors.tick_params(left=False, bottom=False, top=False, right=False)

    if col_colors is not None:
        grid.ax_col_colors.tick_params(left=False, bottom=False, top=False, right=False)


    # get colormap bounds (x pos, y pos, x size, y size)
    # grid.ax_heatmap.get_position().bounds

    return grid

# former plotHeatmap_real
def plot_heatmap(numbersTable, cmap='YlGnBu', figsize=(8, 8),
                 title='', title_fontsize=13, ax=None,
                 font_scale=1, snsStyle='ticks', xRotation=0,
                 yRotation=90,
                 xlabel='', ylabel='', colormap_label='',
                 vmin=None, vmax=None, supress_ticks=True,
                 annotate_text=False, annotate_fontsize=8,
                 annotation_format=".2f",
                 mask=None, colorbar_ticks=None,
                 hide_colorbar=False,
                 xy_labels_fontsize=None,
                 grid_linewidths=0, grid_linecolor='white',
                 color_specific_xticklabels=None,
                 color_specific_yticklabels=None,
                 color_specific_color='red',
                 fix_smaller_rows_at_y_edges_bug=False):
    sns.set(font_scale=font_scale)
    sns.set_context(font_scale=font_scale)
    sns.set_style(snsStyle)

    if ax is None:
        plt.figure(figsize=figsize, dpi=300)

    ax = sns.heatmap(numbersTable, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                annot=annotate_text, annot_kws={"size": annotate_fontsize},
                fmt=annotation_format, mask=mask, cbar=not hide_colorbar,
                cbar_kws={"ticks": colorbar_ticks},
                linewidths=grid_linewidths, linecolor=grid_linecolor)
    ax.set_title(title, fontdict={'fontsize': title_fontsize,
                                  'fontweight': 'bold'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not hide_colorbar:
        ax.collections[0].colorbar.set_label(colormap_label)

    for tick in ax.get_xticklabels():
        tick.set_rotation(xRotation)

    for tick in ax.get_yticklabels():
        tick.set_rotation(yRotation)

    if xy_labels_fontsize is not None:
        ax.yaxis.label.set_size(fontsize=xy_labels_fontsize)
        ax.xaxis.label.set_size(fontsize=xy_labels_fontsize)

    if supress_ticks:
        ax.tick_params(axis=u'both', which=u'both', length=0)
        try:
            if not hide_colorbar:
                ax.collections[0].colorbar.tick_params(axis=u'both', which=u'both', length=0)
        except Exception:
            print("Could not perform line: \nax.collections[0].colorbar.tick_params(axis=u'both', which=u'both', length=0) \nin LielTools_v3\PlotTools.py")

    if color_specific_xticklabels is not None:
        for i, xticklabel in enumerate(ax.get_xticklabels()):
            xticklabel_text = str(xticklabel.get_text())
            for label_from_list in color_specific_xticklabels:
                if xticklabel_text == str(label_from_list):
                    xticklabel.set_color(color_specific_color)

    if color_specific_yticklabels is not None:
        for i, yticklabel in enumerate(ax.get_yticklabels()):
            yticklabel_text = str(yticklabel.get_text())
            for label_from_list in color_specific_yticklabels:
                if yticklabel_text == str(label_from_list):
                    yticklabel.set_color(color_specific_color)

    plt.tight_layout()
    if fix_smaller_rows_at_y_edges_bug:
        ax.set_ylim(len(numbersTable)+0.5, -1)

    return ax

def plot_violinplot(series_x, series_y, series_hue=None,
                    ax=None, figsize=(7, 6),
                    cut=0, scale="count", inner=None,
                    split=True, orient='v',
                    plot_title='', x_title='', y_title='', legend_title='',
                    x_rotation=90, title_fontsize=18, title_color='maroon',
                    font_scale=1, sns_style='ticks',
                    color=None, palette=None, x_order=None,
                    xy_title_fontsize=14):
    """

    :param series_x: pd.Series of values for x axis (categories)
    :param series_y: pd.Series of values for y axis (numeric)
    :param series_hue: pd.Series of values for hue, i.e., sub-categories (2 categories only!)
    :param ax: matplotlib axes object. optional. If None is supplied, will create the object
    :param figsize: tuple (width, height) for the figure size
    :param cut: sns.violinplot parameter: Distance, in units of bandwidth size, to extend the
                                          density past the extreme datapoints.
                                          Set to 0 to limit the violin range within the range
                                          of the observed data
    :param scale: sns.violinplot parameter: {???area???, ???count???, ???width???}
                                            The method used to scale the width of each violin.
                                            If area, each violin will have the same area.
                                            If count, the width of the violins will be scaled by
                                            the number of observations in that bin.
                                            If width, each violin will have the same width.
    :param inner: sns.violinplot parameter: {???box???, ???quartile???, ???point???, ???stick???, None}
                                            Representation of the datapoints in the violin interior.
                                            If box, draw a miniature boxplot. If quartiles, draw the
                                            quartiles of the distribution. If point or stick, show
                                            each underlying datapoint.
                                            Using None will draw unadorned violins.
    :param split: sns.violinplot parameter: when using hue, setting split to True will draw half
                                            of a violin for each level.
    :param orient: Orientation of the plot (vertical 'v' or horizontal 'h').
    :param plot_title: title of the plot (default '')
    :param x_title: x axis title. Default is None, then uses x column name
    :param y_title: y axis title. Default is None, then uses y column name
    :param legend_title: legend title. Default is None, then uses hue column name
    :param x_rotation: xticklabels rotation. Default 90 (degrees)
    :param title_fontsize: plot title text fontsize
    :param title_color: plot title text color
    :param font_scale: seaborn fontscale. Default 1
    :param sns_style: seaborn style (dict, None, or one of {darkgrid, whitegrid, dark, white, ticks})
    :param color: violins color
    :param palette: violins color palette
    :param x_order: list of x axis categories by the requested display order.
                    Default None - automatic order
    :param xy_title_fontsize: fontsize of x and y axis titles
    :return: matplotlib axes object.
    """
    sns.set(font_scale=font_scale)
    sns.set_style(sns_style)

    data = DataTools.join_non_empty_series_f_list([series_x, series_y, series_hue])

    if x_title== '':
        x_title = DataTools.get_col_name(series_x)
    if y_title== '':
        y_title = DataTools.get_col_name(series_y)

    if series_hue is not None:
        if legend_title == '':
            legend_title = DataTools.get_col_name(series_hue)

    if ax is None:
        fix, ax = plt.subplots(figsize=figsize)

    sns.violinplot(data=data, x=DataTools.get_col_name(series_x),
                   y=DataTools.get_col_name(series_y), ax=ax,
                   hue=DataTools.get_col_name(series_hue),
                   scale=scale, inner=inner, split=split,
                   cut=cut, orient=orient, color=color,
                   palette=palette, order=x_order)

    font_title = {'size': title_fontsize, 'color': title_color, 'weight': 'bold'}
    ax.set_title(plot_title, fontdict=font_title)

    for tick in ax.get_xticklabels():
        tick.set_rotation(x_rotation)

    if (x_title != None):
        ax.set_xlabel(x_title)

    if (y_title != None):
        ax.set_ylabel(y_title)

    ax.yaxis.label.set_size(fontsize=xy_title_fontsize)
    ax.xaxis.label.set_size(fontsize=xy_title_fontsize)

    if series_hue is not None:
        ax.legend_.remove()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   frameon=False, title=legend_title)

        # if legend is binary, change 0,1 to no,yes
        legText = ax.get_legend().get_texts()
        legText = bin_text_to_yes_no(legText)

    # if X is binary, change 0,1 to no,yes
    xticksText = ax.get_xticklabels()
    xticksText = bin_text_to_yes_no(xticksText)
    ax.set_xticklabels(xticksText)

    return ax


def plot_violin_boxplot(df, x, y, cut_in_half=True, stripplot=True,
                        figsize=(6,5), xtitle=None, ytitle=None,
                        palette='Set1', jitter=0.05, dot_size=4,
                        dot_color='grey', violin_alpha=0.8,
                        stripplot_alpha=0.3, boxplot_width=0.3,
                        dots_x_offset=0.002, order=None, x_rotation=0,
                        xy_title_fontsize=12, font_scale=1, violin_cut=2):
    """
    Plot a violin plot with a boxplot and stripplot on top.

    :param df: pandas Dataframe from to plot data from
    :param x: string. x variable - column name from df
    :param y: string. y variable (numeric) - column name from df
    :param cut_in_half: boolean. cut violin plot in half such that stripplot
                        dots will be visible
    :param stripplot: boolean. add stripplot
    :param figsize: tuple of 2 numbers, default (6, 5)
    :param xtitle: x axis title. Default is None, then uses x column name
    :param ytitle: y axis title. Default is None, then uses x column name
    :param palette: violin plot color palette (name or dictionary with x
                    values as keys and colors as values)
    :param jitter: stripplot jitter size
    :param dot_size: stripplot dot size
    :param dot_color: stripplot dot color
    :param violin_alpha: violin alpha (transparency)
    :param stripplot_alpha: stripplot dots alpha (transparency)
    :param boxplot_width: width of boxplot
    :param dots_x_offset: offset of stripplot dots from the center of violin
                          plot (only when it's cut in half)
    :param order: list. order of x values
    :param x_rotation: x labels rotation
    :param xy_title_fontsize: x and y axis titles fontsize (default is None,
                              then uses seaborn automaticaly chosen size)
    :param font_scale: seaborn fontscale
    :param violin_cut: sns.violinplot parameter: Distance, in units of bandwidth size, to extend the
                                          density past the extreme datapoints.
                                          Set to 0 to limit the violin range within the range
                                          of the observed data
    :return: axes object
    """
    plt.close('all')
    sns.set(font_scale=font_scale)
    sns.set_style('white')

    plt.figure(figsize=figsize)

    # violin
    ax = sns.violinplot(y=y, x=x, data=df,
                        palette=palette,
                        scale="width", inner=None,
                        order=order, cut=violin_cut)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        # set alpha
        violin.set_alpha(violin_alpha)

        # cut violin in half
        if cut_in_half:
            bbox = violin.get_paths()[0].get_extents()
            x0, y0, width, height = bbox.bounds
            violin.set_clip_path(plt.Rectangle((x0-(width/10), y0), (width/10)+(width / 2), height,
                                               transform=ax.transData))

    # boxplot
    sns.boxplot(y=y, x=x, data=df, saturation=1, showfliers=False,
                width=boxplot_width, boxprops={'zorder': 3, 'facecolor': 'none'},
                ax=ax, order=order)
    old_len_collections = len(ax.collections)

    # stripplot
    if stripplot:
        sns.stripplot(y=y, x=x, data=df, color=dot_color, ax=ax, order=order,
                      alpha=stripplot_alpha, jitter=jitter, size=dot_size)
        if cut_in_half:
            for dots in ax.collections[old_len_collections:]: # set offset - only in the boxplot half
                dots.set_offsets(dots.get_offsets() +
                                 np.array([jitter + dots_x_offset + dot_size/200, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # axis labels and titles
    if xtitle is not None:
        ax.set_xlabel(xtitle)

    if ytitle is not None:
        ax.set_ylabel(ytitle)

    for tick in ax.get_xticklabels():
        tick.set_rotation(x_rotation)

    ax.yaxis.label.set_size(fontsize=xy_title_fontsize)
    ax.xaxis.label.set_size(fontsize=xy_title_fontsize)

    return ax

def plot_boxplot_hue_stats_text(df, x_col_name, y_col_name, hue_col_name,
                                test='Mann-Whitney', comparisons_correction=None,
                                stats_loc='inside', stats_line_offset=None, stats_line_height=0.02,
                                stats_text_offset=1, stats_linewidth=1.5,
                                stats_fontsize='medium', stats_width=0.8,
                                **boxplot_kwargs,
                                ):
    """
    Draws a boxplot using the plot_boxplot function. Then, performs statistical test
    between "hue" col values (must be 2-values only), and adds statistical annotation on top
    of the boxes.

    @param df: pandas dataframe
    @param x_col_name: str. name of df column to be plotted on x axis (categorical)
    @param y_col_name: str. name of df column to be plotted on y axis (numeric)
    @param hue_col_name: str. name of df column to be plotted as hue. can only contain exactly 2 unique values (categories).
                              the statistical test will be performed between these two categories, against the y values.
                              (do the y values differ between the two categories?)

    @params **boxplot_kwargs - keyword arguments passed to the plot_boxplot function.

    statistical annotation parameters:
    @param test: str. default 'Mann-Whitney'
        Statistical test to run. Must be one of: `Levene`, `Mann-Whitney`, `Mann-Whitney-gt`, `Mann-Whitney-ls`,
                                                 `t-test_ind`, `t-test_welch`, `t-test_paired`, `Wilcoxon`, `Kruskal`
                                                 * The Mann-Whitney U test is the nonparametric equivalent of the two sample t-test
                                                 * the Mann-Whitney U-test tests two independent samples, whereas the Wilcox sign test tests two dependent samples.

    @param comparisons_correction: Method for multiple comparisons correction. `bonferroni` or None. default None
    @param stats_loc: 'inside' or 'outside'. Defalt 'inside'
    @param stats_line_height: in axes fraction coordinates
    @param stats_text_offset: in points
    @return: matplotlib axes object
    """
    import statannot # pip install git+https://github.com/webermarcolivier/statannot.git

    hue_vals = df[hue_col_name].unique()
    if len(hue_vals) != 2:
        raise ValueError(f'column {hue_col_name} can only contain exactly 2 unique values. Please revise.')

    ax = plot_boxplot(df[x_col_name], df[y_col_name], seriesHue=df[hue_col_name],
                      **boxplot_kwargs)

    statannot.add_stat_annotation(ax, data=df, x=x_col_name, y=y_col_name, hue=hue_col_name,
                                  box_pairs=[((x_val, hue_vals[0]), (x_val, hue_vals[1])) for x_val in df[x_col_name].unique()],
                                  test=test, text_format='star', loc=stats_loc, verbose=1,
                                  comparisons_correction=comparisons_correction,
                                  width=stats_width, line_offset=stats_line_offset, line_height=stats_line_height,
                                  text_offset=stats_text_offset, linewidth=stats_linewidth, fontsize=stats_fontsize,
                                  )

    return ax



''' gets counts data column/s and creates a bar plot '''
def DFbarPlot(data, columns=None,
              figsize=(6, 4),
              plotTitle='',
              plotOnaxes=None,
              xTitle=None, yTitle=None,
              ylim=None, xRotation=45,
              width=0.8,
              legendLabels=None,
              legendTitle=None,
              grid=False,
              showLegend=True,
              titleFontSize=22,
              axesTitleFontSize=18, axesTicksFontSize=16,
              legendFontSize=16, legendTitleFontSize=17,
              stacked=False,
              add_value_labels=False, float_num_digits=2,
              value_labels_fontsize=12, value_labels_rotation=0,
              savePath=None, color_list=None
              ):
    if type(data) is pd.Series:
        data = pd.DataFrame(data)
    if columns is None:
        columns = data.columns

    figNew = plt.figure()
    if plotOnaxes is not None: # if an axes was provided, use axes
        data[columns].plot.bar(stacked=stacked, grid=grid,
                               figsize=figsize, ax=plotOnaxes,
                               width=width)
        plotOnaxes.set_title(plotTitle, fontsize=titleFontSize)
        for tick in plotOnaxes.get_xticklabels():
            tick.set_rotation(xRotation)
        if (xTitle is not None):
            plotOnaxes.set_xlabel(xTitle)
        if (yTitle is not None):
            plotOnaxes.set_ylabel(yTitle)
        if (ylim is not None):
            plotOnaxes.set_ylim(ylim)
        if (legendLabels is not None):
            plotOnaxes.legend(legendLabels)
        if (showLegend==False):
            plotOnaxes.get_legend().remove()
        if add_value_labels:
            bar_plot_add_value_labels(plotOnaxes)
    else:                       # if not, use plt
        data[columns].plot.bar(stacked=stacked, grid=grid,
                               figsize=figsize, legend=None,
                               width=width,
                               color=color_list)
        plt.title(plotTitle, fontsize=titleFontSize)
        plt.xticks(rotation=xRotation)
        if (xTitle is not None):
            plt.xlabel(xTitle)
        if (yTitle is not None):
            plt.ylabel(yTitle)
        if (ylim is not None):
            plt.ylim(ylim)
        if (showLegend & (legendLabels is not None)):
            plt.legend(legendLabels)
        if (showLegend & (legendLabels is None)):
            plt.legend()
        if (showLegend):
            plt.axes().legend_.remove()
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)

            legText = plt.axes().get_legend().get_texts()
            legText = bin_text_to_yes_no(legText)
            if (legendTitle is not None):
                plt.axes().get_legend().set_title(legendTitle)
                plt.setp(plt.axes().get_legend().get_title(),
                         fontsize=legendTitleFontSize)

        xticksText = plt.axes().get_xticklabels()
        xticksText = bin_text_to_yes_no(xticksText)
        plt.axes().set_xticklabels(xticksText)

        plt.rc('xtick', labelsize=axesTicksFontSize)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=axesTicksFontSize)  # fontsize of the tick labels
        plt.rc('legend', fontsize=legendFontSize)  # legend fontsize
        # plt.rc('figure', titlesize=titleFontSize)  # fontsize of the figure title
        plt.axes().xaxis.label.set_size(axesTitleFontSize)
        plt.axes().yaxis.label.set_size(axesTitleFontSize)
        if add_value_labels:
            bar_plot_add_value_labels(plt.axes(),
                             float_num_digits=float_num_digits,
                             fontsize=value_labels_fontsize,
                             value_labels_rotation=value_labels_rotation)
        plt.tight_layout()
        save_plt(save_path=savePath)

    return(figNew)


def plotSeriesHistogram(numericSeries, useAxes=None, color='green', grid=False):
    if useAxes is None:
        useAxes = numericSeries.hist(color=color, grid=grid)
    else:
        numericSeries.hist(ax=useAxes, color=color, grid=grid)

    useAxes.set_title(numericSeries)

    return(useAxes)


def plot_barplot_from_series(counts, figsize=(18, 8), title='',  ylabel='',
                             xrotation=90, annot=True, annot_format="{:.1f}",
                             annot_fontsize=8):
    """
    Gets a series, plots the values as bars with value annotation.

    :param counts: pandas series
    :param figsize: figsize tuple
    :param title: plot title
    :param ylabel: ylabel text
    :param xrotation: x tick text rotation
    :return:
    """

    plt.figure(figsize=figsize)
    plt.bar(counts.index, counts)

    if annot:
        x_vals = list(counts.index)
        y_vals = list(counts.values)

        for i in range(len(y_vals)):
            plt.annotate(annot_format.format(y_vals[i]), xy=(x_vals[i], y_vals[i]),
                         ha='center', va='bottom', fontsize=annot_fontsize)

    plt.ylabel(ylabel)
    plt.xticks(rotation=xrotation)
    plt.title(title)


def spaghetti_patients(patientDf, classColumn=None, figsize=(6, 8),
                       linewidth=1, alpha=0.9, rotationXlabels=90,
                       plotTitle=None, xTitle=None, yTitle=None,
                       ylim=None, showLegend=True, saveFullPath=None,
                       marker='', markersize=2, legend_title='',
                       shuffle_colors=True, cmap=None):
    ''' Gets a df. Creates a spaghetti plot with each row getting a line,
    and each column getting an x-axis point.

    @ patientDf: This is the only obligatory variable.
            Must contain all numeric columns (except for "classColumn")
    @ classColumn: string. Not obligatory. The name of a column of "class"
        values (can be strings / numeric column).
        Each row will be colored according to classColumn's values.
        (This column will not be used for an x-axis point)
        If classColumn=None, each row will get a different color.
    @ showLegend -  will work only if "classColumn" is given.
    @ alpha - transparency.
    @ showFig = run plt.show()
    @ cmap - If cmap is given, colors will be using the cmap, instead of random.
             if cmap doesn't have enough colors, will use random colors (shuffled or not).
    @ shuffle_colors - boolean. if uses random colors, shuffle them or not
    returns a figure object. '''

    figNew = plt.figure(figsize=figsize)

    # define colors iterator
    if classColumn is None: # each row gets a different color
        colors_rows = get_colors_list(patientDf.shape[0],
                                      shuffle=shuffle_colors)
        color = colors_rows['colors']
        colorIter = colors_rows['iter']
    else:                     # color by "classColumn"
        colorsCat = get_colors_4_categorical_series(patientDf[classColumn],
                                                    shuffle=shuffle_colors,
                                                    cmap=cmap)
        colorIter = colorsCat['iter']
        patientDf = DataTools.get_df_without_cols(patientDf, [classColumn]) # exclude classColumn from df to plot

    # draw a line for each row
    for row in range(patientDf.shape[0]):
        plt.plot(list(patientDf.columns.values), patientDf.iloc[row,:],
                 marker=marker, color=next(colorIter), linewidth=linewidth,
                 alpha=alpha, markersize=markersize, markeredgewidth=0)

    # extra plotting options
    plt.xticks(rotation=rotationXlabels)
    plt.tight_layout()
    if (plotTitle is not None): plt.title(plotTitle)
    if (xTitle is not None): plt.xlabel(xTitle)
    if (yTitle is not None): plt.ylabel(yTitle)
    if (ylim is not None): plt.ylim(ylim)
    if ((classColumn is not None) & (showLegend == True)):
        legendLabels = list(colorsCat['mapper'].keys())
        legendLabels.sort()
        legendColors = []
        for i in range(len(legendLabels)):
            legendColors.append(Line2D([0], [0],
                                       color=colorsCat['mapper'].get(legendLabels[i]),
                                       lw=linewidth, label='Line'))
        plt.legend(legendColors, legendLabels, bbox_to_anchor=(1.05, 1),
                   loc=2, borderaxespad=0., frameon=False, title=legend_title)

    plt.tight_layout()
    save_plt(save_path=saveFullPath, show_if_none=False)
    return figNew

# former plotScatterHue
def plot_scatter_hue(series_x, series_y, series_hue=None,
                     save_folder=None, save_full_path=None,
                     aspect_ratio=1.2, size=7,
                     show_reg_line=False, plot_title='',
                     x_title='', y_title='', x_rotation=45,
                     titleFontSize=18, title_color='maroon',
                     hue_legend_title='', xticks=None, font_scale=1,
                     sns_style="ticks", legend_frame=False,
                     hue_colorscale=False, hue_palette='Set2',
                     marker_size=5,
                     marker_linewidth=0, marker_edgecolor='black',
                     marker_alpha=1,
                     x_log_scale=False, y_log_scale=False,
                     ylim=None, xlim=None):
    sns.set(font_scale=font_scale)
    sns.set_style(sns_style)

    data = DataTools.join_non_empty_series_f_list([series_x, series_y, series_hue])

    if x_title== '':
        x_title = DataTools.get_col_name(series_x)
    if y_title== '':
        y_title = DataTools.get_col_name(series_y)
    if hue_legend_title== '' and series_hue is not None:
        hue_legend_title = DataTools.get_col_name(series_hue)

    if series_hue is not None: # hue exists
        fig11 = sns.lmplot(DataTools.get_col_name(series_x),
                           DataTools.get_col_name(series_y), data,
                           hue=DataTools.get_col_name(series_hue), fit_reg=show_reg_line,
                           legend=False, aspect=aspect_ratio, size=size, palette=hue_palette,
                           scatter_kws={'linewidths': marker_linewidth,
                                        'edgecolor': marker_edgecolor,
                                        'alpha': marker_alpha,
                                        's': marker_size},
                           )
    else:                     # no hue
        fig11 = sns.lmplot(DataTools.get_col_name(series_x), DataTools.get_col_name(series_y),
                           data, fit_reg=show_reg_line, legend=False, aspect=aspect_ratio, size=size,
                           scatter_kws={'linewidths': marker_linewidth,
                                        'edgecolor': marker_edgecolor,
                                        'alpha': marker_alpha})

    fig11.fig.suptitle(plot_title, fontsize=titleFontSize, color=title_color, weight='bold')

    plt.xticks(rotation=x_rotation)
    if xticks is not None:
        fig11.set(xticks=xticks)

    if series_hue is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0., frameon=legend_frame,
                   title=hue_legend_title)

    if (series_hue is not None) and hue_colorscale:
        norm = plt.Normalize(series_hue.min(), series_hue.max())
        sm = plt.cm.ScalarMappable(cmap=hue_palette, norm=norm)
        sm.set_array([])

        fig11.axes[0,0].get_legend().remove()
        cbar = fig11.axes[0,0].figure.colorbar(sm)
        cbar.set_label(hue_legend_title)

    if x_log_scale:
        fig11.set(xscale="log")
    if y_log_scale:
        fig11.set(yscale="log")

    if ylim is not None:
        fig11.set(ylim=ylim)
    if xlim is not None:
        fig11.set(xlim=xlim)

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.tight_layout()

    if save_folder is not None:
        fileName = 'Scatter - ' + x_title + ' VS ' + y_title
        if series_hue is not None: fileName = fileName + ' BY ' + hue_legend_title + '.jpg'
        else: fileName = fileName + '.jpg'
        save_plt(save_path=save_folder + fileName)

    if save_full_path is not None:
        save_plt(save_path=save_full_path)

    return fig11

# former plotScatter
def plot_scatter(x_series, y_series,
                 ax=None, figsize=(6, 5),
                 show_reg_line=True,
                 plt_corr_txt=True, plot_pearson=True, plot_spearman=True,
                 plot_title='', x_title='', y_title='',
                 font_scale=1, sns_style='ticks',
                 markers_color='teal', markers_alpha=0.6,
                 x_rotation=45, title_font_size=18, corr_font_size=14,
                 title_color='maroon', xticks=None,
                 axes_title_font_size=14,
                 x_jitter=None, y_jitter=None,
                 ylim=None, xlim=None,
                 correl_text_x_loc=0.2, correl_text_y_loc=0.96,
                 save_folder=None, save_full_path=None,
                 x_log_scale=False, y_log_scale=False):
    """

    @param x_series: pd.Series. x values series.
    @param y_series: pd.Series. y values series.
    @param ax: matplotlib axes object to plot over. Default None
    @param figsize: tuple (length 2) with numbers indicating figure size. Default (6, 5)
    @param show_reg_line: boolean. Whether to add a regression line. Default True
    @param plt_corr_txt: boolean. Whether to add correlation text. Default True
    @param plot_pearson: boolean. Whether to add the Pearson correlation text. Default True
    @param plot_spearman: boolean. Whether to add the Pearson correlation text. Default True
    @param plot_title: str. Title to be added. Default ''
    @param x_title: str. x axis label to use instead of the x series name.
    @param y_title: str. y axis label to use instead of the y series name.
    @param font_scale: number. Seaborn fontscale. Default 1
    @param sns_style: str. Seaborn style. Default 'ticks'
    @param markers_color: markers color. Default 'teal'
    @param markers_alpha: markers alpha (opacity, between 0-1). Default 0.6
    @param x_rotation: x tick labels rotation (0-360). Default 45
    @param title_font_size: title font size. Default 18
    @param corr_font_size: correlation text font size. Default 18
    @param title_color: title color. Default 'maroon'
    @param xticks: xticks values to use instead of the automatically set ones. Default None
    @param axes_title_font_size: the axes labels (titles) font size. Default 14
    @param x_jitter: boolean. Whether to add jitter to x values (relevant if they are integers). Default False
    @param y_jitter: boolean. Whether to add jitter to y values (relevant if they are integers). Default False
    @param ylim: tuple (length 2) with numbers indicating y axis limits, instead of the automatically set ones. Default None
    @param xlim: tuple (length 2) with numbers indicating x axis limits, instead of the automatically set ones. Default None
    @param correl_text_x_loc: Float. Starting position of the correlation text in the x axis. Default 0.2
    @param correl_text_y_loc: Float. Starting position of the correlation text in the y axis. Default 0.96
    @param save_folder: str. Folder to save the figure (file name determined automatically). Default None
    @param save_full_path: str. Full path to save the figure. Default None
    @return:
    """


    sns.set(font_scale=font_scale)
    sns.set_style(sns_style)

    data = DataTools.join_non_empty_series_f_list([x_series, y_series])

    if x_title== '': x_title = DataTools.get_col_name(x_series)
    if y_title== '': y_title = DataTools.get_col_name(y_series)

    fontTitle = {'size': title_font_size, 'color': title_color, 'weight': 'bold'}
    fig11 = sns.regplot(x=data[DataTools.get_col_name(x_series)],
                        y=data[DataTools.get_col_name(y_series)],
                        ax=ax, color=markers_color, fit_reg=show_reg_line,
                        x_jitter=x_jitter, y_jitter=y_jitter,
                        scatter_kws={'alpha': markers_alpha})

    if plt_corr_txt:
        add_correls_to_fig(fig11, data[DataTools.get_col_name(x_series)],
                           data[DataTools.get_col_name(y_series)],
                           font_size=corr_font_size, plotPearson=plot_pearson,
                           plotSpearman=plot_spearman,
                           text_x_loc=correl_text_x_loc,
                           text_y_loc=correl_text_y_loc)
    fig11.figure.set_size_inches(figsize)
    if xticks is not None: fig11.set(xticks=xticks)

    if x_log_scale:
        fig11.set(xscale="log")
    if y_log_scale:
        fig11.set(yscale="log")

    fig11.set_xlabel(x_title, fontdict={'size': axes_title_font_size})
    fig11.set_ylabel(y_title, fontdict={'size': axes_title_font_size})
    fig11.set_title(plot_title, fontdict=fontTitle)
    plt.xticks(rotation=x_rotation)

    # fig11.figure.subplots_adjust(right=0.2, bottom=0.2)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    # plt.rc('xtick', labelsize=axesTicksFontSize)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=axesTicksFontSize)  # fontsize of the tick labels
    # plt.rc('figure', titlesize=titleFontSize)  # fontsize of the figure title
    # plt.axes().xaxis.label.set_size(axesTitleFontSize)
    # plt.axes().yaxis.label.set_size(axesTitleFontSize)

    plt.tight_layout()

    if save_folder is not None:
        fileName = 'Scatter - ' + x_title + ' VS ' + y_title + '.jpg'
        save_plt(save_path=save_folder + fileName)

    if save_full_path is not None:
        save_plt(save_path=save_full_path)


def plotScatterLine(seriesX, seriesY, ax=None, saveFolder=None,
                saveFullPath=None, figsize=(6, 5),
                plotTitle='',xTitle='', yTitle='',
                dotsColor='teal', ylim=None, xlim=None,
                xRotation=45, titleFontSize=18,
                titleColor='maroon', xticks=None, font_scale=1,
                snsStyle='ticks', grid=False):
    sns.set(font_scale=font_scale)
    sns.set_style(snsStyle)

    data = DataTools.join_non_empty_series_f_list([seriesX, seriesY])

    if xTitle=='': xTitle = DataTools.get_col_name(seriesX)
    if yTitle=='': yTitle = DataTools.get_col_name(seriesY)

    fontTitle = {'size': titleFontSize, 'color': titleColor, 'weight': 'bold'}
    fig11 = sns.lineplot(x=data[DataTools.get_col_name(seriesX)],
                         y=data[DataTools.get_col_name(seriesY)],
                         marker='o', ax=ax, color=dotsColor)

    fig11.figure.set_size_inches(figsize)
    if xticks is not None: fig11.set(xticks=xticks)
    fig11.set_xlabel(xTitle)
    fig11.set_ylabel(yTitle)
    fig11.set_title(plotTitle, fontdict=fontTitle)
    plt.xticks(rotation=xRotation)

    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    if grid:
        plt.rc('grid', linestyle="--", color='grey')
        plt.grid(True, which='both')

    plt.tight_layout()

    if saveFolder is not None:
        fileName = 'Scatter - ' + xTitle + ' VS ' + yTitle + '.jpg'
        save_plt(save_path=saveFolder + fileName)

    if saveFullPath is not None:
        save_plt(save_path=saveFullPath)


def plotLinePlot(data=None, seriesX=None, seriesY=None, ax=None,
                 saveFolder=None, figsize=(6, 5), plotTitle='',
                 xTitle='', yTitle='', dotsColor='teal', xRotation=45,
                 titleFontSize=18, corrFontSize=14, titleColor='maroon',
                 xticks=None, font_scale=1, snsStyle='ticks', legendTitle=''):
    sns.set(font_scale=font_scale)
    sns.set_style(snsStyle)

    if data is not None:
        if xTitle=='': xTitle='Columns'
        if yTitle=='': yTitle='Values'
    else:
        if xTitle == '': xTitle = DataTools.get_col_name(seriesX)
        if yTitle == '': yTitle = DataTools.get_col_name(seriesY)

    [data, seriesX, seriesY] = mergeDF4Plotting(data, seriesX, seriesY, xTitle, yTitle)

    if data is not None:
        fig11 = sns.factorplot(x=DataTools.get_col_name(data.iloc[:, 0]), y=yTitle,
                               hue=xTitle, data=data, ax=ax, color=dotsColor)
    else:
        seriesData = DataTools.join_non_empty_series_f_list([seriesX, seriesY])
        fig11 = sns.factorplot(x=seriesData[DataTools.get_col_name(seriesX)],
                               y=seriesData[DataTools.get_col_name(seriesY)],
                               data=seriesData, ax=ax, color=dotsColor)

    fontTitle = {'size': titleFontSize, 'color': titleColor, 'weight': 'bold'}

    fig11.figure.set_size_inches(figsize)
    if xticks is not None: fig11.set(xticks=xticks)
    fig11.set_xlabel(xTitle)
    fig11.set_ylabel(yTitle)
    fig11.set_title(plotTitle, fontdict=fontTitle)
    plt.xticks(rotation=xRotation)

    # ax1.legend_.remove()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
               frameon=False, title=legendTitle)

    # if legend is binary, change 0,1 to no,yes
    # legText = ax1.get_legend().get_texts()
    # legText = binText2YN(legText)

    # fig11.figure.subplots_adjust(right=0.2, bottom=0.2)

    if saveFolder is not None:
        fileName = 'Scatter - ' + xTitle + ' VS ' + yTitle + '.jpg'
        save_plt(save_path=saveFolder + fileName)


def catergorical_y_with_error(y_categories_series, x_nums_series, x_error_series=None, figsize=(8, 5),
                          shape_size=650, shape_color='#99bbff', shape_marker='h',
                          spines_alpha=0.2, x_title='', x_title_text_size=17, xlim=None,
                          plot_values_text=True, plot_grid=True,
                          xtick_fontsize=15, ytick_fontsize=15):
    # shape marker options: https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers

    # Draw plot
    plt.figure(figsize=figsize, dpi=300)
    plt.scatter(x_nums_series, y_categories_series, s=shape_size,
                color=shape_color, marker=shape_marker)
    ax = plt.gca()

    # Decorations
    # Lighten borders
    plt.gca().spines["top"].set_alpha(spines_alpha)
    plt.gca().spines["bottom"].set_alpha(spines_alpha)
    plt.gca().spines["right"].set_alpha(spines_alpha)
    plt.gca().spines["left"].set_alpha(spines_alpha)

    plt.yticks(y_categories_series, y_categories_series)
    plt.xlabel(x_title, fontdict={'size': x_title_text_size})
    # ax.grid(axis='x', color='#e6e6e6')
    # plt.grid(which='x', linestyle='--', alpha=0.5)
    if xlim is not None:
        plt.xlim(xlim)

    if x_error_series is not None:
        for y, ylabel in zip(ax.get_yticks(), ax.get_yticklabels()):
            f = y_categories_series == ylabel.get_text()
            ax.errorbar(x_nums_series[f].values, np.ones_like(x_nums_series[f].values) * y,
                        xerr=x_error_series[f].values, ls='none',
                        color=shape_color, linewidth=3.5, barsabove=False)

    if plot_values_text:
        for x, y, tex in zip(x_nums_series, y_categories_series, x_nums_series):
            t = plt.text(x, y, round(tex, 2), horizontalalignment='center',
                         verticalalignment='center', fontdict={'color': 'black', 'size': 10})

    if plot_grid:
        plt.grid(axis='x', color='#e6e6e6')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.rc('xtick', labelsize=xtick_fontsize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=ytick_fontsize)  # fontsize of the tick labels
    plt.tight_layout()


def plot_columns_dist(df, output_file_path=None, fig_rows=4, fig_cols=5, figsize=(30, 20),
                      kde_color='black', hist_color='g', hist_alpha=0.3,
                      title='', title_fontsize=18, title_y=1.03, bins=30,
                      rug=False, rug_color='black', rug_alpha=0.3,
                      rug_linewidth=1, rug_height=0.03, font_scale=1,
                      sns_style='ticks', x_rotation=0):
    """
    Plot a grid of distribution plots - one for each column of a given pandas.Dataframe.

    @param df: pd.Dataframes with columns to be plotted (columns must be numeric)
    @param output_file_path: str. full file path for saving the plot to file system. Default None.
    @param fig_rows: int. Number of rows for the distplots grid. Default 4.
    @param fig_cols: int. Number of columns for the distplots grid. Default 5.
    @param figsize:
    @param kde_color:
    @param hist_color:
    @param hist_alpha:
    @param title:
    @param title_fontsize:
    @param title_y:
    @param bins:
    @param rug: boolean. Whether to add a rug at the bottom or not. Default False.
    @param rug_color:
    @param rug_alpha:
    @param rug_linewidth:
    @param rug_height:
    @param font_scale:
    @param sns_style:
    @param x_rotation:
    @return: matplotlib figure object
    """
    sns.set(font_scale=font_scale)
    sns.set_style(sns_style)


    num_columns = df.shape[1]
    if fig_cols * fig_rows < num_columns:
        print('plot_columns_dist: number of columns', num_columns, 'is smaller than fig_cols*fig_rows')

    i = 0
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)
    for row in range(fig_rows):
        for col in range(fig_cols):
            if (i < num_columns):
                sns.distplot(df.iloc[:, i], ax=axes[row, col], rug=rug, bins=bins,
                             kde_kws={"color": kde_color},
                             rug_kws={"color": rug_color, "alpha": rug_alpha,
                                      "linewidth": rug_linewidth, "height": rug_height},
                             hist_kws={"color": hist_color, "alpha": hist_alpha},
                             )
                for tick in axes[row, col].get_xticklabels():
                    tick.set_rotation(x_rotation)
                i = i + 1

    fig.suptitle(title, fontsize=title_fontsize, y=title_y)
    fig.tight_layout()

    if output_file_path is not None:
        plt.savefig(output_file_path, bbox_inches='tight', dpi=500)

    return fig

def plot_columns_dist_hue(df, hue_col, output_file_path=None, fig_cols=5,
                          hist_alpha=0.5, shade=False, palette="Set1",
                          sharex=False, sharey=False, font_scale=1.5,
                          rug=False, rug_color='black', rug_alpha=0.3,
                          rug_linewidth=1, rug_height=0.03,
                          fig_height=3, fig_aspect=1, sns_style='ticks'):
    """
    Plot a grid of distribution plots with hue - one for each column of a given pandas.Dataframe.

    @param df: pd.Dataframes. All columns except for hue_col will be plotted.
                              Columns (other than hue_col) must be numeric.
    @param hue_col: str. The name of the column (from df) to be used as hue.
                         Must be a categorical/discrete values column.
    @param output_file_path: str. full file path for saving the plot to file system. Default None.
    @param fig_cols: int. Number of columns for the distplots grid. Default 5.
    @param hist_alpha:
    @param shade:
    @param palette:
    @param sharex: boolean.
    @param sharey: boolean.
    @param font_scale:
    @param rug: boolean. Whether to add a rug at the bottom or not. Default False.
    @param rug_color:
    @param rug_alpha:
    @param rug_linewidth:
    @param rug_height:
    @param fig_height: numeric - the height of the figure to be created. Default 3
    @param fig_aspect: aspect - the aspect ratio, determining the figure width.
                       Width is determined by fig_aspect * fig_height in inches. Default 1
    @param sns_style:
    @return: seaborn facetgrid object
    """
    sns.set(font_scale=font_scale)
    sns.set_style(sns_style)

    df_melted = df.melt(id_vars=hue_col, var_name='cols', value_name='vals')

    g = sns.FacetGrid(df_melted, col='cols', hue=hue_col, palette=palette, col_wrap=fig_cols,
                      sharex=sharex, sharey=sharey, height=fig_height, aspect=fig_aspect)
    g = (g.map(sns.distplot, "vals", hist=False, rug=rug, kde_kws={"shade": shade},
               hist_kws={"alpha": hist_alpha},
               rug_kws={"color": rug_color, "alpha": rug_alpha,
                        "linewidth": rug_linewidth, "height": rug_height}))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)

    if output_file_path is not None:
        plt.savefig(output_file_path, dpi=500, bbox_inches='tight')

    return g

def plot_boxplot_subplots(df, x_col, y_cols, hue_col=None, output_file_path=None, fig_rows=4, fig_cols=5, figsize=(30, 20),
                          title='', title_fontsize=18, title_y=1.03, font_scale=1, sns_style='ticks',
                          **boxplot_kwargs):
    """

    @param df: pd.DataFrame containing the data for the figures.
    @param x_col: str. The name of the df column to plot in the x axis of all subplots.
    @param y_cols: list of strings. A list that contains the names of all columns to be plotted from df.
                   Each column will be plotted in a boxplot of its own (a single subplot).
    @param hue_col: str. The name of the df column to plot as the hue variable (i.e., sub-categories) of all subplots. Default None.
    @param output_file_path: str. path for the figure to be saved to file system. Default None.
    @param fig_rows: int. Number of subplots rows.
    @param fig_cols: int. Number of subplots columns.
    @param figsize: tuple with 2 numbers. figure size (of entire figure containing all subplots).
    @param title: str. figure title
    @param title_fontsize: int. title text font size.
    @param title_y:
    @param font_scale: float. seaborn fontscale.
    @param sns_style: str. seaborn style.
    @param boxplot_kwargs: keyword parameters to be passed to plot_boxplot for each of the subplots.
    @return: matplotlib axes object
    """
    plt.close('all')
    sns.set(font_scale=font_scale)
    sns.set_style(sns_style)


    num_figs = len(y_cols)
    if fig_cols * fig_rows < num_figs:
        print('plot_boxplot_subplots: number of columns', num_figs, 'is smaller than fig_cols*fig_rows')

    i = 0
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)
    for row in range(fig_rows):
        for col in range(fig_cols):
            if (i < num_figs):
                plot_boxplot(df[x_col], df[y_cols[i]],
                             seriesHue=None if hue_col is None else df[hue_col],
                             ax=axes[row, col],
                             **boxplot_kwargs)
                i = i + 1

    fig.suptitle(title, fontsize=title_fontsize, y=title_y)
    fig.tight_layout()

    if output_file_path is not None:
        plt.savefig(output_file_path, bbox_inches='tight', dpi=500)

    return axes


def pairplot_with_spearman(df, font_scale=1, spearman_fontsize=10):
    """ Draw pairplot with annotation of spearman correlation in each subplot"""
    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        r, _ = spearmanr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'?? = {r:.2f}', xy=(.1, 0.9), xycoords=ax.transAxes, fontsize=spearman_fontsize, zorder=10)

    sns.set(font_scale=font_scale)
    sns.set_style("white")
    g = sns.pairplot(df)
    g.map(corrfunc)

def parallele_coordinates_plot(df, class_column=None, cmap='Set1', figsize=(8, 5),
                               x_title='', y_title='', axis_title_fontsize=13,
                               plot_markers=True, marker_size=25, marker_type='o',
                               marker_alpha=0.7, marker_linewidth=0,
                               line_alpha=0.7, legend_title='', legend_frameon=False,
                               xticks_rotation=0, y_gridlines=True):
    '''
    Plots a parallele coordinates plot from a pandas.DataFrame columns.
    Class column, if given, will be used to color the lines according to the class.
    Can also add markers - can be important when there are many missing data points.

    :param df: pandas.DataFrame. Each column will get an x axis value in the plot.
    :param class_column: string. df column name, of a column with categorical/discrete values.
                         If given, will be used to color the lines according to the class
    :param cmap: string. matplotlib cmap name. Will be used for the class column coloring
    :param figsize: tuple.
    :param x_title: string. x axis title to add
    :param y_title: string. y axis title to add
    :param axis_title_fontsize: x/y axis title fontsize
    :param plot_markers: boolean. True = plot markers
    :param marker_size: int. marker size
    :param marker_type: string. marker type ('o', 'x', etc.)
    :param marker_alpha: float between 0-1. markers opacity
    :param marker_linewidth: numeric. markers edge width
    :param line_alpha: float between 0-1. lines opacity
    :param legend_title: string. legend title to add
    :param legend_frameon: boolean. add frame to legend or not
    :param xticks_rotation: int. angle to rotate xticks
    :param y_gridlines: boolean. If False, gridlines on the y axis are removed.
    :return: matplotlib axes object
    '''

    plt.figure(figsize=figsize)
    cluster_colors = get_colors_4_categorical_series(df[class_column], cmap=cmap)
    class_order = DataTools.get_ordered_unique_vals_from_list(list(df[class_column]))
    ax = parallel_coordinates(df, class_column,
                              colormap=categorCmapFromList([cluster_colors['mapper'][c] for c in class_order]),
                              alpha=line_alpha)
    if class_column is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   frameon=legend_frameon, title=legend_title)

    if plot_markers:
        cluster_color_list = cluster_colors['colorSeries']

        for i, col in enumerate(DataTools.get_df_col_names_without_cols(df, class_column)):
            plt.scatter([i]*len(df[col]), df[col], s=marker_size, linewidth=marker_linewidth,
                        c=cluster_color_list, marker=marker_type, alpha=marker_alpha)

    plt.xticks(rotation=xticks_rotation)
    plt.xlabel(x_title, fontdict={'size': axis_title_fontsize})
    plt.ylabel(y_title, fontdict={'size': axis_title_fontsize})
    plt.tight_layout()
    if y_gridlines is False:
        ax.grid(False)

    return ax

def plot_table(df, row_colors_list=None, col_widths=None, font_size=None):
    """
    col_widths: The column widths in units of the axes.
    If not given, all columns will have a width of 1 / ncols.
    """
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center', cellLoc='center',
                     colWidths=col_widths)

    fig.tight_layout()

    if row_colors_list is not None:
        for i in range(len(row_colors_list)):
            for j in range(len(df.columns)):
                table[(i + 1, j)].get_text().set_color(row_colors_list[i])

    from matplotlib.font_manager import FontProperties
    for (row, col), cell in table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',
                                                              size=font_size))


#### ----------------------------------- Helpers ------------------------------------- ###

def add_horizontal_line_to_ax(height, ax, start_x, end_x,
                              start_xticklabel=None, end_xticklabel=None,
                              find_closest_labels=False,
                              text='', color='red', fontsize=16,
                              text_offset_x=4, text_offset_y=0.15,
                              line_left_offset=0, line_right_offset=0,
                              horizontalalignment='left', verticalalignment='baseline',
                              ):
    """
    Draw a horizontal line on ax, using x positions or x tick labels,
    and a given height (y).
    If start_x and end_x are None, will look for them
    using the x axis xticklabels, using parameters start_xticklabel (left label)
    and end_xticklabel (right label).

    :param height: height of the line (y)
    :param ax: ax on which to draw the line
    :param start_x: line x start.
    :param end_x: line x end
    :param start_xticklabel: Default None. Will be used only if start_x and end_x are None
    :param end_xticklabel: Default None. Will be used only if start_x and end_x are None
    :param find_closest_labels: Default False. If true, and start or end tick labels aren't found,
                 will look for the closest (numeric) tick label. For example, if start_xticklabel=14
                 and 14 doesn't exist, will look for the closest larger number.
                 for end_xticklabel, will look for the closest smaller number.
                 Will not work for tick labels that cannot be converted into a number.
    :param text: text to draw under the line in center. x: (start_x+end_x / 2) - text_offset_x
                                                        y: height - text_offset_y
    :param color: text and line color
    :param fontsize: text fontsize
    :param text_offset_x: x offset of text
    :param text_offset_y: y offset of text (define as negative, for text above the line)
    :param line_left_offset: offset of line left edge (x=start_x-line_left_offset)
    :param line_right_offset: offset of line right edge (x=end_x+line_right_offset)
    :param horizontalalignment: [ 'center' | 'right' | 'left' ]
    :param verticalalignment: [ 'center' | 'top' | 'bottom' | 'baseline' ]
    :return:
    """

    final_start_xticklabel = start_xticklabel
    final_end_xticklabel = end_xticklabel

    if start_x is None and end_x is None:
        if start_xticklabel is None and end_xticklabel is None:
            raise Exception('if start_x is None and end_x is None, must define start_xticklabel and end_xticklabel.')
        else:
            for tick, ticklabel in zip(ax.get_xticks(), ax.get_xticklabels()):
                if ticklabel.get_text() == str(start_xticklabel):
                    start_x = tick
                    final_start_xticklabel = ticklabel
                if ticklabel.get_text() == str(end_xticklabel):
                    end_x = tick
                    final_end_xticklabel = ticklabel

    # if couldn't find start_xticklabel
    if start_x is None:
        if find_closest_labels: # look for the first ticklabel with number larger than start_xticklabel
            for tick, ticklabel in zip(ax.get_xticks(), ax.get_xticklabels()):
                if float(ticklabel.get_text()) >= float(start_xticklabel):
                    start_x = tick
                    final_start_xticklabel = ticklabel
                    print(f'Couldnt find start_xticklabel in labels: {start_xticklabel}'
                          f'\nUsing {final_start_xticklabel.get_text()} instead.')
                    break

        else:
            raise Exception(f'Couldnt find start_xticklabel in labels: {start_xticklabel}')

    if end_x is None:
        if find_closest_labels: # look for the closest ticklabel with number smaller than start_xticklabel
            for tick, ticklabel in zip(ax.get_xticks()[::-1], ax.get_xticklabels()[::-1]):
                if float(ticklabel.get_text()) <= float(end_xticklabel):
                    end_x = tick
                    final_end_xticklabel = ticklabel
                    print(f'Couldnt find end_xticklabel in labels: {end_xticklabel}'
                          f'\nUsing {final_end_xticklabel.get_text()} instead.')
                    break
        else:
            raise Exception('Couldnt find end_xticklabel in labels:', end_xticklabel)

    if start_x is None:
        raise Exception('Could not find proper start_x, please check why')

    if end_x is None:
        raise Exception('Could not find proper end_x, please check why')

    if start_x >= end_x:
        raise Exception('Could not find proper start_x and end_x. Got:\n'
                        'start_x:', start_x, 'and end_x:', end_x, '\n',
                        'final_start_xticklabel:', final_start_xticklabel.get_text(),
                        'and final_end_xticklabel:', final_end_xticklabel.get_text())

    sns.lineplot(y=[height, height], x=[start_x-line_left_offset, end_x+line_right_offset],
                 ax=ax, color=color, linewidth=2)
    ax.text(((start_x+end_x) / 2) - text_offset_x,
            height - text_offset_y,
            text, color=color, fontsize=fontsize,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment)


def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_facecolor('white')

def mapColors2Labels(labels, setStr='Set3', cmap=None):
    """Return pd.Series of colors based on labels"""
    if cmap is None:
        N = max(3,min(12,len(np.unique(labels))))
        cmap = palettable.colorbrewer.get_map(setStr,'Qualitative',N).mpl_colors
    cmapLookup = {k:col for k,col in zip(sorted(np.unique(labels)),itertools.cycle(cmap))}
    return labels.map(cmapLookup.get)


def bar_plot_add_value_labels(ax, spacing=2, float_num_digits=2,
                              fontsize=12, value_labels_rotation=0,
                              float_to_percentage=False, color='black',
                              x_move_left=False):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
        float_num_digits (int): if numbers to present are float - number
                                of digits to show
        float_to_percentage (bool): show float as percentage
        x_move_left (bool): if true, value will be shown at the left side of the bar top
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        if x_move_left:
            x_value = x_value - (rect.get_width() / 4)

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if type(y_value) == np.float64 or type(y_value) == float:
            if float_to_percentage:
                label = format(100 * y_value, "." + str(float_num_digits) + "f") + '%'
            else:
                label = format(y_value, "." + str(float_num_digits) + "f")
        else:
            label = str(y_value)

        # Create annotation
        text = ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',  # Horizontally center label
            va=va,  # Vertically align label differently for
            fontsize=fontsize,
            color=color)  # positive and negative values.

        text.set_rotation(value_labels_rotation)

    ylim = ax.get_ylim()
    ylim_new = [ylim[0], ylim[1] * 1.1]
    ax.set_ylim(ylim_new)


''' former getColors4categoricalSeries '''
def get_colors_4_categorical_series(categ_series, shuffle=False, cmap=None):
    '''
    Get color series, mapper and iterator for categorical series.
    If cmap is None, or doesn't have enough colors, will create random colors (shuffled or not).
    :param categ_series: A series with categorical / discrete values.
    :param shuffle: Only relevant if creates random colors.
    :param cmap: string - matplotlib colormap name.
    :return: A dictionary:
        'colorSeries': series of colors corresponding to the original series,
        'mapper': mapper,
        'iter': iterator over colors series
    '''
    n_classes = len(categ_series.unique())

    # if given cmap name, check if it has a sufficient amount of colors.
    # If not, don't use it.
    if cmap is not None:
        cmap_colors = plt.get_cmap(cmap).colors
        if n_classes > len(cmap_colors):
            cmap = None

    if cmap is None:
        class_color_list = get_colors_list(n_classes, shuffle=shuffle)['colors']
    else:
        class_color_list = cmap_colors

    mapper = {i: c for i, c in zip(set(categ_series.astype('category')), class_color_list)}
    colors = categ_series.map(mapper)

    return({'colorSeries': colors, 'mapper': mapper, 'iter': iter(colors)})


# former getColorsList
def get_colors_list(n_colors, shuffle=False):
    if type(n_colors) != int:
        raise Exception('nClasses must be of type int')
    colors = cm.rainbow(np.linspace(0, 1, n_colors))
    if shuffle:
        np.random.shuffle(colors)
        np.random.shuffle(colors)
        np.random.shuffle(colors)
    return({'colors': colors, 'iter': iter(colors)})

''' Gets a series with categorical data. (can be strings / numeric column)
returns a dict:
'colorSeries': series of colors corresponding to each data point's class.
'mapper': value-color dict '''
def getColors4categoricalSeries_old(catSeries):
    nClasses = len(set(catSeries.get_values()))
    classColors = cm.rainbow(np.linspace(0, 1, nClasses))
    mapper = {i: c for i, c in zip(range(nClasses), classColors)}
    colors = catSeries.astype('category').cat.codes.map(mapper)
    return ({'colorSeries': colors, 'mapper': mapper})


def categorCmapFromList(colorlist):
    cmap = plt.cm.jet
    cmap = cmap.from_list('Custom cmap', colorlist, len(colorlist))
    return(cmap)

# former addFigCorrelations
def add_correls_to_fig(figure, col1, col2, font_size=16,
                       plotPearson=True, plotSpearman=True,
                       text_x_loc=0.2, text_y_loc=0.96):
    font = {'size': font_size}

    pearR = StatsTools.get_df_cols_correl(col1, col2, method='pearson')
    spearR = StatsTools.get_df_cols_correl(col1, col2, method='spearman')
    if (pearR[1] < 0.001):
        text1 = 'Pearson r = %2.2f      p-value = <0.001' % (pearR[0])
    else:
        text1 = 'Pearson r = %2.2f      p-value = %2.3f' % (pearR[0], pearR[1])

    if (spearR[1] < 0.001):
        text2 = 'Spearman r = %2.2f   p-value = <0.001' % (spearR[0])
    else:
        text2 = 'Spearman r = %2.2f   p-value = %2.3f' % (spearR[0], spearR[1])

    if plotPearson:
        figure.text(text_x_loc, text_y_loc, text1, transform = figure.axes.transAxes,
                    fontdict=font)
    elif plotSpearman:
        figure.text(text_x_loc, text_y_loc, text2, transform = figure.axes.transAxes,
                    fontdict=font)

    if plotPearson and plotSpearman:
        figure.text(text_x_loc, text_y_loc-0.06, text2, transform = figure.axes.transAxes,
                    fontdict=font)

# former savePlt
def save_plt(save_path=None, dpi=300, show_if_none=False):
    if save_path is None and show_if_none:
        plt.show()
    elif save_path is not None:
        plt.savefig(save_path, dpi=dpi)

# former binText2YN
def bin_text_to_yes_no(textList):
    if (len(textList) == 2):
        if (textList[0].get_text() == '0' and textList[1].get_text() == '1'):
            textList[0].set_text('No')
            textList[1].set_text('Yes')
        elif (textList[0].get_text() == '1' and textList[1].get_text() == '0'):
            textList[0].set_text('Yes')
            textList[1].set_text('No')
    return(textList)

def diverging_cmap_from_list(color_list):
    cmap = LinearSegmentedColormap.from_list(
        name='from_color_list',
        colors=color_list
        )
    return cmap

def view_color_bar(cmap):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np

    ax = plt.subplot(111)
    im = ax.imshow(np.arange(100).reshape((10, 10)), cmap=cmap)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    #???Accent???, ???Accent_r???, ???Blues???, ???Blues_r???, ???BrBG???, ???BrBG_r???, ???BuGn???, ???BuGn_r???, ???BuPu???, ???BuPu_r???, ???CMRmap???, ???CMRmap_r???, ???Dark2???, ???Dark2_r???, ???GnBu???, ???GnBu_r???, ???Greens???, ???Greens_r???, ???Greys???, ???Greys_r???, ???OrRd???, ???OrRd_r???, ???Oranges???, ???Oranges_r???, ???PRGn???, ???PRGn_r???, ???Paired???, ???Paired_r???, ???Pastel1???, ???Pastel1_r???, ???Pastel2???, ???Pastel2_r???, ???PiYG???, ???PiYG_r???, ???PuBu???, ???PuBuGn???, ???PuBuGn_r???, ???PuBu_r???, ???PuOr???, ???PuOr_r???, ???PuRd???, ???PuRd_r???, ???Purples???, ???Purples_r???, ???RdBu???, ???RdBu_r???, ???RdGy???, ???RdGy_r???, ???RdPu???, ???RdPu_r???, ???RdYlBu???, ???RdYlBu_r???, ???RdYlGn???, ???RdYlGn_r???, ???Reds???, ???Reds_r???, ???Set1???, ???Set1_r???, ???Set2???, ???Set2_r???, ???Set3???, ???Set3_r???, ???Spectral???, ???Spectral_r???, ???Wistia???, ???Wistia_r???, ???YlGn???, ???YlGnBu???, ???YlGnBu_r???, ???YlGn_r???, ???YlOrBr???, ???YlOrBr_r???, ???YlOrRd???, ???YlOrRd_r???, ???afmhot???, ???afmhot_r???, ???autumn???, ???autumn_r???, ???binary???, ???binary_r???, ???bone???, ???bone_r???, ???brg???, ???brg_r???, ???bwr???, ???bwr_r???, ???cividis???, ???cividis_r???, ???cool???, ???cool_r???, ???coolwarm???, ???coolwarm_r???, ???copper???, ???copper_r???, ???cubehelix???, ???cubehelix_r???, ???flag???, ???flag_r???, ???gist_earth???, ???gist_earth_r???, ???gist_gray???, ???gist_gray_r???, ???gist_heat???, ???gist_heat_r???, ???gist_ncar???, ???gist_ncar_r???, ???gist_rainbow???, ???gist_rainbow_r???, ???gist_stern???, ???gist_stern_r???, ???gist_yarg???, ???gist_yarg_r???, ???gnuplot???, ???gnuplot2???, ???gnuplot2_r???, ???gnuplot_r???, ???gray???, ???gray_r???, ???hot???, ???hot_r???, ???hsv???, ???hsv_r???, ???icefire???, ???icefire_r???, ???inferno???, ???inferno_r???, ???jet???, ???jet_r???, ???magma???, ???magma_r???, ???mako???, ???mako_r???, ???nipy_spectral???, ???nipy_spectral_r???, ???ocean???, ???ocean_r???, ???pink???, ???pink_r???, ???plasma???, ???plasma_r???, ???prism???, ???prism_r???, ???rainbow???, ???rainbow_r???, ???rocket???, ???rocket_r???, ???seismic???, ???seismic_r???, ???spring???, ???spring_r???, ???summer???, ???summer_r???, ???tab10???, ???tab10_r???, ???tab20???, ???tab20_r???, ???tab20b???, ???tab20b_r???, ???tab20c???, ???tab20c_r???, ???terrain???, ???terrain_r???, ???twilight???, ???twilight_r???, ???twilight_shifted???, ???twilight_shifted_r???, ???viridis???, ???viridis_r???, ???vlag???, ???vlag_r???, ???winter???, ???winter_r???


def plot_all_cmaps():
    fig, axes = plt.subplots(36, 6, figsize=(10, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        hspace=0.1, wspace=0.1)

    im = np.outer(np.ones(10), np.arange(100))

    cmaps = [m for m in plt.cm.datad if not m.endswith("_r")]
    cmaps.sort()

    axes = axes.T.ravel()
    for ax in axes:
        ax.axis('off')

    for cmap, color_ax, gray_ax, null_ax in \
            zip(cmaps, axes[1::3], axes[2::3], axes[::3]):
        del null_ax
        color_ax.set_title(cmap, fontsize=10)
        color_ax.imshow(im, cmap=cmap)


def hide_every_k_in_axis_labels(axes, x_or_y='y', k=2):
    if x_or_y == 'y':
        get_func = axes.get_yticklabels
        set_func = axes.set_yticklabels
    elif x_or_y == 'x':
        get_func = axes.get_xticklabels
        set_func = axes.set_xticklabels
    else:
        raise Exception('hide_every_k_in_axis_labels function: x_or_y must be "x" or "y"')

    ticks = [item.get_text() for item in get_func()]
    for i, tick in enumerate(ticks):
        if i % k != 0 or k == 1:
            ticks[i] = ''
    set_func(ticks)

def mergeDF4Plotting(data, seriesX, seriesY, xTitle, yTitle, name_index=None):
    '''
    Must provide data or seriesX and seriesY.
    If provided data, it will be melted such that:
    1. the columns will be made into a single variable
    with the column names as values, and column name
    as xTitle.
    2. The values of all columns will be a single
    variable named yValues, with the corresponding column
    name in the xTitle column.
    3. The index will be the third column.
    If name_index is not None, it will be the column name.
    else, the column name will be the index name.
    If there isn't one, it will be "index".

    :param data:
    :param seriesX:
    :param seriesY:
    :param xTitle:
    :param yTitle:
    :param name_index:
    :return:
    '''
    if data is not None:
        if (seriesX is not None) or (seriesY is not None):
            raise Exception('Cant use both data and seriesX+seriesY. Can only get one of them.')
    else:
        if (seriesX is None) or (seriesY is None):
            raise Exception('You must provide either data or seriesX+seriesY.')

    if data is not None:
        indexName = data.index.name
        if indexName is None:
            if name_index is not None:
                indexName = name_index
            else:
                indexName = 'index'

        newData = data.copy()
        newData[indexName] = newData.index
        newData = newData.melt(indexName, var_name='cols', value_name='vals')
        newData = DataTools.renameDFcolumns(newData, {'cols': xTitle, 'vals': yTitle})
    else:
        newData = None

    return([newData, seriesX, seriesY])


# error bars example
#
# plt.errorbar(avg_results_table['exp'].values,
#              avg_results_table['mean '+args.summary_metric_show].values,
#              yerr=std_results_table['std '+args.summary_metric_show].values,fmt='o', markersize=4)
# plt.show()

def get_value_color_from_cmap(value, cmap_name='RdBu_r', vmin=0, vmax=1):
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(value))

# former shiftedColorMap
def shifted_colormap(cmap_name, minval, maxval, cmap_midpoint, name='shiftedcmap'):
    '''
    Source:
    https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. For example, useful for
    data with a negative min and positive max if you want the
    middle of the colormap's dynamic range to be at zero (even if it's not the middle).

    Input
    -----
      cmap_name : The name of matplotlib colormap to be altered
      minval : Minimum cmap val.
      midpoint : The new center of the colormap.
          Must be between minval and maxval.
      stop : Maximum cmap val.
    '''

    def get_point_ratio(minval, maxval, point):
        '''
        Normalize minval to 0, maxval to 1, and return
        the relative 'point' value between 0-1.
        '''
        new_max = maxval - minval
        new_point = point - minval

        return new_point / new_max

    midpoint = get_point_ratio(minval, maxval, cmap_midpoint)

    cmap = plt.cm.get_cmap(cmap_name)

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(0, 1, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def compare_tick_labels(axes1, axes2, type='x'):
    '''
    Compare tick labels of axes objects axes1 and axes2.
    If the number of labels or their text (comparing one by one) is unequal,
    function will throw an assertion error.
    :param axes1, axes2: matplotlib axes objects
    :param type: 'x' (use function get_xticklabels) or 'y' (use function get_yticklabels)
    :return: None.
    '''
    if type == 'x':
        labels1 = axes1.get_xticklabels()
        labels2 = axes2.get_xticklabels()
    elif type == 'y':
        labels1 = axes1.get_yticklabels()
        labels2 = axes2.get_yticklabels()

    assert len(labels1) == len(labels2), \
        '{}ticklabel lists have a different length'.format(type)
    for i in range(len(labels1)):
        assert labels1[i].get_text() == labels2[i].get_text(), \
            '{}ticklabels number {} are different!'.format(type, i)


def changePltProperties(plotTitle='', titleFontSize=18, xRotation=90,
                        xTitle=None, yTitle=None, ylim=None,
                        showLegend=False, legendLabels=None, legendTitle=None,
                        legendTitleFontSize=16, axesTicksFontSize=14,
                        legendFontSize=14, axesTitleFontSize=16):
    plt.title(plotTitle, fontsize=titleFontSize)
    plt.xticks(rotation=xRotation)
    if (xTitle is not None):
        plt.xlabel(xTitle)
    if (yTitle is not None):
        plt.ylabel(yTitle)
    if (ylim is not None):
        plt.ylim(ylim)
    if (showLegend & (legendLabels is not None)):
        plt.legend(legendLabels)
    if (showLegend & (legendLabels is None)):
        plt.legend()
    if (showLegend):
        plt.axes().legend_.remove()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)

        legText = plt.axes().get_legend().get_texts()
        legText = bin_text_to_yes_no(legText)
        if (legendTitle is not None):
            plt.axes().get_legend().set_title(legendTitle)
            plt.setp(plt.axes().get_legend().get_title(), fontsize=legendTitleFontSize)

    plt.rc('xtick', labelsize=axesTicksFontSize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=axesTicksFontSize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=legendFontSize)  # legend fontsize
    # plt.rc('figure', titlesize=titleFontSize)  # fontsize of the figure title
    plt.axes().xaxis.label.set_size(axesTitleFontSize)
    plt.axes().yaxis.label.set_size(axesTitleFontSize)

def color_rgb_to_hexa_string(rgb_tuple, vals_between_0_1=False):
    '''
    Get a tuple (size 3) representing a color with RGB.
    Returns the hexa string representation of the color.

    :param rgb_tuple: tuple (size 3)
    :param vals_between_0_1: False: each color is a number between 0-255.
                             True: each color is a number between 0-1. Function will then multiple each by 255.
    :return: hexa string
    '''
    if vals_between_0_1:
        rgb_tuple = (int(rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255))

    return '#%02x%02x%02x' % rgb_tuple

def jitter_dots_pathcollection(dots, offset=0.3):
    """
    Gets a matplotlib.collections.PathCollection object,
    which is the output of the plt.scatter function.
    Jitters them in the x axis with a uniform distribution range of (-offset, offset).
    Changes the dots object inplace.
    :param dots: a matplotlib.collections.PathCollection object
    :param offset: float used to randomly jitter in the x axis
    """
    offsets = dots.get_offsets()
    jittered_offsets = offsets
    # only jitter in the x-direction
    jittered_offsets[:, 0] += np.random.uniform(-offset, offset, offsets.shape[0])
    dots.set_offsets(jittered_offsets)