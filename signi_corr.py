import pandas as pd
import statsmodels.api as sm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import palettable
import itertools


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


def multipAdjustPvalsMat(pvals, method='FDR', is_corr_matrix=False):
    '''
    :param pvals: pvalues matrix (pd.DataFrame)
    :param method: 'FDR' or 'FWER'
    :param is_corr_matrix: if True, will only calculate adjustment for upper diagonal
           matrix and copy to lower. Also, diagonal will not be ignored.
    :return: adjusted pvalues matrix (pd.DataFrame)
    '''
    pvals = pvals.copy()
    if method == 'FDR':
        methodSM = 'fdr_bh'
    elif method == 'FWER':
        methodSM = 'holm'

    # if it's a correlations matrix, calc only for upper diagonal
    if is_corr_matrix:
        if (pvals.shape[0] != pvals.shape[1]):
            print('Warning! corrMat=True but matrix has unequal dimensions!')

        for i in range(pvals.shape[0]):
            pvals.iloc[i, i] = np.nan
            for j in range(i):
                pvals.iloc[i, j] = np.nan

    # flatten matrix
    pvalsFlattened = pvals.unstack().copy()
    mask = ~pvalsFlattened.isna()  # non-NA values mask

    pval_adj = np.empty(pvalsFlattened.shape)
    pval_adj.fill(np.nan)
    pval_adj[mask] = sm.stats.multipletests(pvalsFlattened[mask],
                                            method=methodSM)[1]  # calc using mask

    # flattened vals - back to matrix
    pval_adj_series = pd.Series(pval_adj, index=pvalsFlattened.index)
    final = pval_adj_series.unstack(0)

    # if is_corr_matrix, copy upper diagonal to lower diagonal
    if is_corr_matrix:
        for i in range(final.shape[0]):
            final.iloc[i, i] = 0
            for j in range(i):
                final.iloc[i, j] = final.iloc[j, i]

    return final


def getCorrelationForDFColumns(col1, col2, method='pearson'): # or 'spearman'
    '''
    Gets 2 numerical pd.Series and returns the correlation between them
    and its pvalue. Drops NA values before calculation.
    :param col1: pd.Series
    :param col2: pd.Series
    :param method: 'pearson' or 'spearman'
    :return: tuple: (correlation, pvalue)
    '''

    tmpDFa = pd.DataFrame(col1)
    tmpDFa.columns = ['col1']

    tmpDFb = pd.DataFrame(col2)
    tmpDFb.columns = ['col2']

    tmpDF1 = tmpDFa.join(tmpDFb).dropna()

    if (method == 'spearman'):
        corr = scipy.stats.stats.spearmanr(tmpDF1.loc[:,'col1'], tmpDF1.loc[:,'col2'])
    elif (method == 'pearson'):
        corr = scipy.stats.stats.pearsonr(tmpDF1.loc[:,'col1'], tmpDF1.loc[:,'col2'])
    else:
        raise ValueError('Unknown method for correlation calculation')

    return corr


def get_value_color_from_cmap(value, cmap_name='RdBu_r', vmin=0, vmax=1):
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(value))


def get_correlation_matrix(data, method='pearson'): # or 'spearman'
    ''' Gets a data pandas dataframe, returns pairwise
    correlations matrix dataframes: correlation coefficient matrix, p-values matrix.
    Drops NA values before each 2 column correlation calculation. '''
    coeffMat = pd.DataFrame(index=data.columns, columns=data.columns)
    pvalMat = pd.DataFrame(index=data.columns, columns=data.columns)

    if ( ((method=='spearman') | (method=='pearson')) == False ): # if method is not 'pearson' or 'spearman'
        raise ValueError('getCorrPvalMat: unknown method for correlation! Please fix!')
    else:
        for var1 in data.columns:
            for var2 in data.columns:
                corrtest = getCorrelationForDFColumns(data[var1], data[var2], method=method)
                coeffMat.loc[var1,var2] = corrtest[0]
                pvalMat.loc[var1,var2] = corrtest[1]

    # check that diagonal == 1
    for _ in range(coeffMat.shape[0]):
        if(coeffMat.iloc[_,_] != 1):
            print('Warning! a diag value is not 1. check for double indexes!')

    return coeffMat, pvalMat


def plot_signi_corr(df, method='spearman', show_significance=True,
                    signif_by_method='FWER', censor_type='FDR',
                    censor_thresh=1,
                    color_only_lower_triangle=False,
                    numbers_upper=True,
                    asterisks_upper=False,
                    figsize=(10, 8),
                    title='',
                    title_fontsize=20,
                    title_color='black',
                    colorbar_title='Correlation',
                    colorscale_fontsize_title=16,
                    colorscale_fontsize_ticks=14,
                    ticks_fontsize=14,
                    asterisks_x=0.6, asterisks_y=0.44,
                    numbers_x=0.55, numbers_y=0.48,
                    aster_size=10,
                    numbers_size=10,
                    grid_colors='white', grid_width=0,
                    label_category=None, label_cmap=None,
                    main_left=0.25, main_bottom=0.17,
                    main_right=0.98, main_top=0.99,
                    colorbar_left=0.01, colorbar_bottom=0.66,
                    colorbar_right=0.04, colorbar_top=0.95,
                    ):
    '''
    Gets a dataframe with numeric columns, calculate the pairwise correlations between
    the columns and portray them in a heatmap. Heatmap colors show the correlation coefficient.
    Correlation coefficients can be annotated with numbers (numbers_upper=True).
    Significance is annotated with asterisks (show_significance=True).
    pvalues can be adjusted for multiple testing using FDR or FWER (signif_by_method).
    * When show_significance=True, correlation coefficient number are only annotated
    if significant.
    Asterisks indicate p-values (or FDR or FWER) with ***, **, and *
    indicating p ≤ 0.0005, 0.005, and 0.05, respectively

    :param df: pandas dataframe
    :param method: for calculating correlations - 'spearman' or 'pearson'
    :param show_significance: show significance asterisks.
                              If True, and numbers_upper is True,
                              will only annotate numbers when significant
                              (according to asterisks_by_method)
    :param signif_by_method: adjustment method to use for significance
                             annotation - 'FWER', 'FDR' or 'pvals'
                             (i.e., without multiplicity adjustment)
    :param censor_type: adjustment method to use for censoring cells -
                        'FWER', 'FDR' or 'pvals' (i.e., without multiplicity adjustment)
    :param censor_thresh: threshold over which cells should be censored based on
                          censor_type: adjustment method.
                          Defaule 1 - no censoring
    :param color_only_lower_triangle: Show only lower triangle heatmap (with or without
                                annotating the correlation coefficient in the
                                upper triangle - depending on parameter numbers_upper
    :param numbers_upper: annotate the correlation coefficient in the
                          upper triangle, only for significant values (unless show_significance=False)
    :param asterisks_upper: annotate the pvals in the upper triangle
    :param figsize: figure size
    :param title: figure title. Default ''
    :param title_fontsize: figure title fontsize. Default 20
    :param title_color: figure title fontsize. Default 'black'
    :param colorbar_title: colorbar label string
    :param colorscale_fontsize_title:
    :param colorscale_fontsize_ticks:
    :param ticks_fontsize: fontsize of the ticks (df column names)

    :param asterisks_x: shift the astersiks text on x axis
    :param asterisks_y: shift the astersiks text on y axis
    :param numbers_x: shift the numbers text on x axis
    :param numbers_y: shift the numbers text on y axis
    :param aster_size: fontsize of asterisks to annotate
    :param numbers_size: fontsize of numbers to annotate
    :param grid_colors: heatmap "grid" color
    :param grid_width: heatmap "grid" line width
    :param label_category: pd.Series - add color to each row by these categories.
    :param label_cmap: color map for label categories
    :param main_left: shift the heatmap edges
    :param main_bottom: shift the heatmap edges
    :param main_right: shift the heatmap edges
    :param main_top: shift the heatmap edges
    :param colorbar_left: shift the colorbar edges
    :param colorbar_bottom: shift the colorbar edges
    :param colorbar_right: shift the colorbar edges
    :param colorbar_top: shift the colorbar edges

    :return: matplotlib figure object
    '''

    if asterisks_upper and numbers_upper:
        raise ValueError('asterisks_upper and numbers_upper - only one of them can be True')

    corrs, pvals = get_correlation_matrix(df, method=method)

    # Calculate p-vals FDR and FWER adjustment
    # if showSig:
    adjPvals = {'FWER': multipAdjustPvalsMat(pvals, method='FWER', is_corr_matrix=True),
                'FDR': multipAdjustPvalsMat(pvals, method='FDR', is_corr_matrix=True),
                'pvals': pvals}

    # fill NA values in the data / pvals
    adjPvals['FWER'] = adjPvals['FWER'].fillna(1)
    adjPvals['FDR'] = adjPvals['FDR'].fillna(1)
    adjPvals['pvals'] = adjPvals['pvals'].fillna(1)
    corrs = corrs.fillna(0)

    # Censor values
    censorInd = adjPvals[censor_type] > censor_thresh  # cells to censor
    adjPvals['FWER'].values[censorInd] = 1.
    adjPvals['FDR'].values[censorInd] = 1.
    adjPvals['pvals'].values[censorInd] = 1.
    corrs[censorInd] = 0.

    # pvals to plot
    pValsToStar = adjPvals[signif_by_method]

    # corrs to plot
    valsToPlot = corrs
    valsToPlot_orig = valsToPlot.copy()

    # if only_lower_triangle - change upper vals to color & pvals to np.nan
    if color_only_lower_triangle:
        for i in range(valsToPlot.shape[0]):
            for j in range(i):
                valsToPlot.iloc[j, i] = np.nan
                # if showSig:
                #     pValsToStar.iloc[j, i] = np.nan

    # plot parameters
    cmap = cm.get_cmap('RdBu_r')
    heatmap_params = dict(vmin=-1.0, vmax=1.0, cmap=cmap, edgecolors=grid_colors,
                    linewidths=grid_width, shading='flat')
    colorbar_title = colorbar_title
    ytl = np.array([-1.0, -0.5, 0, 0.5, 1.0])
    yt = np.array([-1.0, -0.5, 0, 0.5, 1.0])

    # main plot - heamtap using valsToPlot
    plt.figure(figsize=figsize)
    figh = plt.gcf()
    plt.clf()
    ax_heatmap = figh.add_subplot(plt.GridSpec(1, 1, left=main_left, bottom=main_bottom,
                                        right=main_right, top=main_top)[0, 0])
    ax_heatmap.set_title(title, fontdict={'fontsize': title_fontsize,
                                          'color': title_color})
    ax_heatmap.grid(None)
    pcolOut = plt.pcolormesh(valsToPlot, **heatmap_params)
    plt.yticks(())  # empty y tick labels (rows)
    plt.xticks(())  # empty x tick labels (columns)
    ax_heatmap.xaxis.set_ticks_position('top')
    plt.xlim((0, valsToPlot.shape[1]))
    plt.ylim((0, valsToPlot.shape[0]))
    ax_heatmap.invert_yaxis()
    spineColor = 'black'

    if color_only_lower_triangle:
        # Remove right and top spines
        ax_heatmap.spines['right'].set_visible(False)
        ax_heatmap.spines['top'].set_visible(False)
    else:
        # Change right and top spines color and width
        ax_heatmap.spines['right'].set_color(spineColor)
        ax_heatmap.spines['right'].set_linewidth('1.5')
        ax_heatmap.spines['top'].set_color(spineColor)
        ax_heatmap.spines['top'].set_linewidth('1.5')

    # Change left and bottom spines color and width
    ax_heatmap.spines['left'].set_color(spineColor)
    ax_heatmap.spines['left'].set_linewidth('1.5')
    ax_heatmap.spines['bottom'].set_color(spineColor)
    ax_heatmap.spines['bottom'].set_linewidth('1.5')

    # lines
    lineColor = 'black'

    # add significance asterisks
    if show_significance:
        for cyi, cy in enumerate(valsToPlot.index):
            for outi, out in enumerate(valsToPlot.columns):
                # if lower triangle - add significance
                # if upper triangle - add significance only if asterisks_upper
                if cyi > outi or (cyi < outi and asterisks_upper):
                    if pValsToStar.loc[cy, out] < 0.0005:
                        ann = '***'
                    elif pValsToStar.loc[cy, out] < 0.005:
                        ann = '**'
                    elif pValsToStar.loc[cy, out] < 0.05:
                        ann = '*'
                    else:
                        ann = ''
                    if not ann == '':
                        # if color_only_lower_triangle, color asterisks by corr value. else - black
                        if color_only_lower_triangle and (cyi < outi and asterisks_upper):
                            text_color = get_value_color_from_cmap(valsToPlot_orig.loc[cy, out],
                                                                   cmap_name='RdBu_r', vmin=-1, vmax=1)
                        else:
                            text_color = 'black'

                        plt.annotate(ann, xy=(outi + asterisks_x, cyi + asterisks_y),
                                     weight='bold', size=aster_size, ha='center',
                                     va='center', rotation=90, color=text_color)

    # if numbers_upper=True, annotate numbers of corr value in upper triangle
    if numbers_upper:
        for cyi, cy in enumerate(valsToPlot.index):
            for outi, out in enumerate(valsToPlot.columns):
                if (cyi != outi and cyi < outi):
                    # if color_only_lower_triangle=True, color numbers by corr value. else - black
                    if color_only_lower_triangle:
                        text_color = get_value_color_from_cmap(valsToPlot_orig.loc[cy, out],
                                                        cmap_name='RdBu_r', vmin=-1, vmax=1)
                        do_annotate = True
                    else:
                        text_color = 'black'
                        if show_significance:
                            # annotate only if pValsToStar is significant
                            do_annotate = pValsToStar.loc[cy, out] < 0.05
                        else:
                            do_annotate = True

                    if do_annotate:
                        plt.annotate(np.round(valsToPlot_orig.loc[cy, out], 2),
                                     xy=(outi + 0.9 * numbers_x, cyi + 1.1 * numbers_y),
                                     size=numbers_size, ha='center',
                                     va='center', rotation=0,
                                     color=text_color)

    # add labels over the rows
    if label_category is not None:
        row_color_width = 0.035
    else:
        row_color_width = 0

    # add labels over the rows
    ax_row_labels = figh.add_subplot(plt.GridSpec(1, 1, left=main_left-0.012-row_color_width,
                                          bottom=main_bottom,
                                          right=main_left-0.011-row_color_width,
                                          top=main_top)[0, 0])
    ax_row_labels.grid(None)
    plt.ylim((0, valsToPlot.shape[0]))
    plt.yticks(np.arange(valsToPlot.shape[0]), valsToPlot.index, size=ticks_fontsize)
    plt.xlim((0, 0.5))
    plt.ylim((-0.5, valsToPlot.shape[0] - 0.5))
    plt.xticks(())
    ax_row_labels.invert_yaxis()
    plt.box(on=None)  # remove the frame border
    ax_row_labels.tick_params(axis=u'both', which=u'both', length=0)  # remove the little tick marks

    # add labels over the columns
    ax_col_labels = figh.add_subplot(plt.GridSpec(1, 1, left=main_left, bottom=main_bottom-0.01,
                                          right=0.98, top=main_bottom-0.00889)[0, 0])
    ax_col_labels.grid(None)
    plt.xlim((0, 4 * valsToPlot.shape[0]))
    plt.xticks(4 * np.arange(valsToPlot.shape[0]), valsToPlot.index, size=ticks_fontsize)
    plt.ylim((0, 0.5))
    plt.xlim((-2.2, 4 * valsToPlot.shape[0] - 2.0))
    plt.yticks(())
    plt.box(on=None)  # remove the frame border
    ax_col_labels.tick_params(axis=u'both', which=u'both', length=0)  # remove the little tick marks
    plt.xticks(rotation=90)

    # add colorbar
    ax_colorbar = figh.add_subplot(plt.GridSpec(1, 1, left=colorbar_left, bottom=colorbar_bottom,
                                             right=colorbar_right, top=colorbar_top)[0, 0])
    cb = figh.colorbar(pcolOut, cax=ax_colorbar, ticks=yt)
    cb.set_label(colorbar_title, size=colorscale_fontsize_title)
    cb.ax.set_yticklabels(ytl, fontsize=colorscale_fontsize_ticks)
    plt.tick_params(axis=u'both', which=u'both', length=0)  # remove the little tick marks
    cb.outline.set_edgecolor(lineColor)

    # add row colors by categories
    if label_category is not None:
        ax_row_colors = figh.add_subplot(GridSpec(1, 1, left=main_left - 0.04, bottom=main_bottom,
                                                 right=main_left - 0.005, top=main_top)[0, 0])
        row_colors_map = mapColors2Labels(label_category, cmap=label_cmap)
        ax_row_colors.imshow([[x] for x in row_colors_map.values], interpolation='nearest',
                            aspect='auto', origin='upper')
        clean_axis(ax_row_colors)

    plt.tight_layout()

    return figh


