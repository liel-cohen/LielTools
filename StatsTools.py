import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
import math
from matplotlib.patches import Rectangle
from itertools import cycle

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import log_loss
import sklearn.metrics
from scipy import stats
from scipy.stats import wilcoxon
from scipy import linalg                  # for linear regression
from statsmodels.base.elastic_net import RegularizedResultsWrapper

if 'LielTools' in sys.modules:
    from LielTools import DataTools
    from LielTools import PlotTools
    from LielTools import FileTools
else:
    import DataTools
    import PlotTools
    import FileTools


def perform_mann_whitney_u_wilcoxon_rank_sum(series1, series2, alternative='two-sided', print_res=True, alpha=0.05):
    """
    The Mann-Whitney U test is a nonparametric statistical significance test for determining whether two independent samples
    were drawn from a population with the same distribution.

    The test was named for Henry Mann and Donald Whitney, although it is sometimes called the Wilcoxon-Mann-Whitney test,
    also named for Frank Wilcoxon, who also developed a variation of the test.

    The two samples are combined and rank ordered together. The strategy is to determine if the values from the two samples are
    randomly mixed in the rank ordering or if they are clustered at opposite ends when combined.
    A random rank order would mean that the two samples are not different,
    while a cluster of one sample values would indicate a difference between them.

    The default assumption or null hypothesis is that there is no difference between the distributions of the data samples.
    Rejection of this hypothesis suggests that there is likely some difference between the samples.
    More specifically, the test determines whether it is equally likely that any randomly selected observation
    from one sample will be greater or less than a sample in the other distribution.
    If violated, it suggests differing distributions.

    Fail to Reject H0: Sample distributions are equal.
    Reject H0: Sample distributions are not equal.

    Info from: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
    @param: series1: data series 1. Can be pd.Series or N-d arrays of samples
    @param: series2: data series 2. Can be pd.Series or N-d arrays of samples
    @param: alternative: ‘two-sided’, ‘less’, ‘greater’. Default 'two-sided'
    @param: print_res: boolean. Whether to print the test statistic, pvalue and conclusion. Default True
    @return: alpha: Alpha to determine whether p-value is significant or not. default 0.05
    return: stat, pval
    """
    stat, pval = stats.mannwhitneyu(x=series1, y=series2, alternative=alternative)

    if print_res:
        print(f'Mann-whitney U / Wilcoxon rank-sum test (alternative: {alternative}): \nStatistic={stat}, p-value={pval}.')

        if pval > alpha:
            print('Data drawn from same distribution (failed to reject H0)')
        else:
            print('Data drawn from different distributions (can reject H0)')

    return pval

def perform_wilcoxon_signed_rank(series1, series2, alternative='two-sided', print_res=True, alpha=0.05):
    """
    A nonparametric statistical significance test for determining whether two dependent samples (paired)
    were drawn from a population with the same distribution.
    For the test to be effective, it requires at least 20 observations in each data sample.

    Fail to Reject H0: Sample distributions are equal.
    Reject H0: Sample distributions are not equal.

    Info from: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/

    @param: series1: data series 1. Can be pd.Series or N-d arrays of samples
    @param: series2: data series 2. Can be pd.Series or N-d arrays of samples
    @param: alternative: ‘two-sided’, ‘less’, ‘greater’. Default 'two-sided'
    @param: print_res: boolean. Whether to print the test statistic, pvalue and conclusion. Default True
    @return: alpha: Alpha to determine whether p-value is significant or not. default 0.05
    return: stat, pval
    """
    stat, pval = wilcoxon(x=series1, y=series2, alternative=alternative)

    if print_res:
        print(f'Wilcoxon signed-rank test (alternative: {alternative}): \nStatistic={stat}, p-value={pval}.')

        if pval > alpha:
            print('Data drawn from same distribution (failed to reject H0)')
        else:
            print('Data drawn from different distributions (can reject H0)')

    return pval

def perform_shapiro_wilk(data_series, print_res=True, alpha=0.05):
    """
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

    If the p-value obtained from the Shapiro-Wilk test is significant (p < 0.05),
    we conclude that the data is not normally distributed.

    @param: data_series: data series to test. Can be pd.Series or N-d arrays of samples
    @param: print_res: boolean. Whether to print the test statistic, pvalue and conclusion. Default True
    @return: alpha: Alpha to determine whether p-value is significant or not. default 0.05
    return: stat, pval
    """
    stat, pval = stats.shapiro(data_series)

    if print_res:
        print(f'Shapiro-Wilk test - Statistic={stat}, p-value={pval}.')

        if pval > alpha:
            print('Data is normally distributed (failed to reject H0)')
        else:
            print('Data is not normally distributed (rejected H0)')


def chi_square_test_independence(df=None, var1_name=None, var2_name=None, table_observed=None, alpha=0.05,
                                 print_res=True, plot_fig=True, figsize=(12, 5), annotate_fontsize=8,
                                 font_scale=1, fix_smaller_rows_at_y_edges_bug=True):
    """
    Perform a Chi-square test of independence over a table of observed values between
    two categories.
    Function can either get the table of observed values itself (as a pd.Dataframe),
    for example:
                   Yes  No
            Red     50  20
            Blue    20  90
            Green   30  90

    ( table_observed = pd.DataFrame({'Yes': {'Red': 50, 'Blue': 20, 'Green': 30}, 'No': {'Red': 20, 'Blue': 90, 'Green': 90}})  )
    or, can get a pd.Dataframe and 2 variables (columns) names and create such table.

    @param df: pd.Dataframe.
    @param var1_name: str, the name of a column in df with categorical data.
                        If table_observed is given, string will be used as title in plot. (optional)
    @param var2_name: str, the name of a column in df with categorical data.
                        If table_observed is given, string will be used as title in plot. (optional)
    @param table_observed: pd.Dataframe of an observed values table
    @param alpha: Alpha to determine whether p-value is significant or not. default 0.05
    @param print_res: boolean. Whether to print the test statistic, pvalue and conclusion. Default True
    @param plot_fig: boolean. Whether to plot a figure with the tables of
                     expected, observed and (O-E)^2 / E values
    @return: chi_square_statistic, p_value, fig (if plot_fig is True)
    """
    if table_observed is None:
        if (df is not None) and (var1_name is not None) and (var2_name is not None):
            table_observed = df.groupby(by=[var1_name, var2_name]).size().unstack().fillna(0)
        else:
            raise ValueError('Must get either table_observed or df, var1 and var2.')

    n = table_observed.sum().sum()
    prob_var1 = table_observed.sum(axis=1) / n
    prob_var2 = table_observed.sum() / n

    table_expected = pd.DataFrame(index=table_observed.index, columns=table_observed.columns)
    for val1 in table_expected.index:
        for val2 in table_expected.columns:
            table_expected.loc[val1, val2] = prob_var1[val1] * prob_var2[val2] * n

    table_X2_per_cell = pd.DataFrame(index=table_observed.index, columns=table_observed.columns)
    for val1 in table_X2_per_cell.index:
        for val2 in table_X2_per_cell.columns:
            X2_cell = ((table_observed.loc[val1, val2] - table_expected.loc[val1, val2]) ** 2) / table_expected.loc[
                val1, val2]
            table_X2_per_cell.loc[val1, val2] = X2_cell

    X2 = table_X2_per_cell.sum().sum()
    deg_freedom = (len(prob_var1)-1) * (len(prob_var2)-1)
    p_value = 1 - stats.chi2.cdf(X2, deg_freedom)

    if print_res:
        conclusion = f"Null Hypothesis - variables are independent. Failed to reject it with alpha={alpha}."
        if p_value <= alpha:
            conclusion = f"Null Hypothesis - variables are independent. It is rejected with alpha={alpha}."

        print(f'Chi-square score is {X2:.3f}, p-value is {p_value:.7f}, degrees of freedom: {deg_freedom}')
        print(conclusion)

    if plot_fig:
        sns.set(font_scale=font_scale)
        sns.set_context(font_scale=font_scale)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        PlotTools.plot_heatmap(table_observed.astype(float), cmap='YlGnBu',
                               title=f'Observed', title_fontsize=13, ax=axes[0],
                               font_scale=font_scale, snsStyle='ticks', xRotation=0,
                               yRotation=90,
                               xlabel=var2_name, ylabel=var1_name, colormap_label='',
                               vmin=None, vmax=None, supress_ticks=True,
                               annotate_text=True, annotate_fontsize=annotate_fontsize,
                               annotation_format=".0f",
                               mask=None, colorbar_ticks=None,
                               hide_colorbar=False,
                               xy_labels_fontsize=None,
                               grid_linewidths=0, grid_linecolor='white',
                               fix_smaller_rows_at_y_edges_bug=fix_smaller_rows_at_y_edges_bug)
        PlotTools.plot_heatmap(table_expected.astype(float), cmap='YlGnBu',
                               title=f'Expected', title_fontsize=13, ax=axes[1],
                               font_scale=font_scale, snsStyle='ticks', xRotation=0,
                               yRotation=90,
                               xlabel=var2_name, ylabel=var1_name, colormap_label='',
                               vmin=None, vmax=None, supress_ticks=True,
                               annotate_text=True, annotate_fontsize=annotate_fontsize,
                               annotation_format=".1f",
                               mask=None, colorbar_ticks=None,
                               hide_colorbar=False,
                               xy_labels_fontsize=None,
                               grid_linewidths=0, grid_linecolor='white',
                               fix_smaller_rows_at_y_edges_bug=fix_smaller_rows_at_y_edges_bug)
        PlotTools.plot_heatmap(table_X2_per_cell.astype(float), cmap='YlGnBu',
                               title='(O-E)^2 / E', title_fontsize=13, ax=axes[2],
                               font_scale=font_scale, snsStyle='ticks', xRotation=0,
                               yRotation=90,
                               xlabel=var2_name, ylabel=var1_name, colormap_label='',
                               vmin=None, vmax=None, supress_ticks=True,
                               annotate_text=True, annotate_fontsize=annotate_fontsize,
                               annotation_format=".1f",
                               mask=None, colorbar_ticks=None,
                               hide_colorbar=False,
                               xy_labels_fontsize=None,
                               grid_linewidths=0, grid_linecolor='white',
                               fix_smaller_rows_at_y_edges_bug=fix_smaller_rows_at_y_edges_bug)

        if p_value is not np.nan:
            if p_value <= alpha:
                title = f"P-value is {p_value:.7f}. H0 is rejected with alpha={alpha} - variables are dependent"
            else:
                title = f"P-value is {p_value:.7f}. Failed to reject H0 with alpha={alpha} - variables are independent"
        else:
            title = 'Cannot determine test result. P-value is nan. Please check'

        fig.suptitle(title)
        fig.subplots_adjust(top=0.88)

        sns.set(font_scale=1)
        sns.set_context(font_scale=1)

        return X2, p_value, fig
    else:
        return X2, p_value


# former getCorrelationForDFColumns
def get_df_cols_correl(col1, col2, method='pearson', conf_interval=False): # or 'spearman'
    '''
    Gets 2 numerical pd.Series and returns the correlation between them
    and its pvalue. Drops NA values before calculation.
    (Will only calculate correlation for values with matching index
    in both series)
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
        if conf_interval:
            corr = list(corr)
            corr.append(spearman_confidence_interval(corr[0], len(tmpDF1.loc[:,'col1'])))
    elif (method == 'pearson'):
        corr = scipy.stats.stats.pearsonr(tmpDF1.loc[:,'col1'], tmpDF1.loc[:,'col2'])
        if conf_interval:
            corr = list(corr)
            corr.append(pearson_confidence_interval(corr[0], len(tmpDF1.loc[:, 'col1'])))
    else:
        raise ValueError('get_df_cols_correl: unknown method for correlation! Please fix!')
        corr = np.nan

    return corr

def spearman_confidence_interval(r, n):
    # https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    stderr = 1.0 / math.sqrt(n - 3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    return {'lower': lower, 'upper': upper, 'std': stderr}

def pearson_confidence_interval(r, n, alpha=0.05):
    # https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    ''' calculate Pearson correlation confidence interval using scipy and numpy
    Parameters
    ----------
    r : pearson r
    n : sample size
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r_z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return {'lower': lo, 'upper': hi, 'std': se}

# former getCorrelationMat
def getCorrelationMat(data, method='pearson'): # or 'spearman'
    ''' Gets a data DF, returns a dict of all column
    correlations matrix DFs: {'coeffMat': .., 'pvalMat': ..}
    Drops NA values before each 2 column correlation calculation. '''
    coeffMat = pd.DataFrame(index=data.columns, columns=data.columns)
    pvalMat = pd.DataFrame(index=data.columns, columns=data.columns)

    if ( ((method=='spearman') | (method=='pearson')) == False ): # if method is not 'pearson' or 'spearman'
        sys.exit('getCorrPvalMat: unknown method for correlation! Please fix!')
    else:
        for var1 in data.columns:
            for var2 in data.columns:
                corrtest = get_df_cols_correl(data[var1], data[var2], method=method)
                coeffMat.loc[var1,var2] = corrtest[0]
                pvalMat.loc[var1,var2] = corrtest[1]

    # check that diagonal == 1
    for _ in range(coeffMat.shape[0]):
        if(coeffMat.iloc[_,_] != 1):
            print('Warning! a diag value is not 1. check for double indexes!')

    return({'coeffMat': coeffMat, 'pvalMat': pvalMat})


def getCorrelationsWithVars(Xdata, Ydata, method='pearson'): # or 'spearman'
    ''' Gets numeric multiple X and multiple Y DFs, with the same indexes.
    returns a dict of correlations matrix DFs: {'coeffMat': .., 'pvalMat': ..}
    Drops NA values before each 2 column correlation calculation. '''
    if (Xdata.shape[0] != Ydata.shape[0]):
        sys.exit('getCorrelationsWithVars: Xdata and Ydata have different sample size! please fix.')
    if (len(Xdata.index.intersection(Ydata.index)) != Xdata.shape[0]):
        sys.exit('getCorrelationsWithVars: Xdata and Ydata indexes are not identical! please fix.')

    coeffMat = pd.DataFrame(index=Xdata.columns, columns=Ydata.columns)
    pvalMat = pd.DataFrame(index=Xdata.columns, columns=Ydata.columns)

    if ( ((method=='spearman') | (method=='pearson')) == False ): # if method is not 'pearson' or 'spearman'
        sys.exit('getCorrPvalMat: unknown method for correlation! Please fix!')
    else:
        for x in Xdata.columns:
            for y in Ydata.columns:
                corrtest = get_df_cols_correl(Xdata[x], Ydata[y], method=method)
                coeffMat.loc[x,y] = corrtest[0]
                pvalMat.loc[x,y] = corrtest[1]

    return({'coeffMat': coeffMat, 'pvalMat': pvalMat})


def correctedPvalsFromMatrixDF(pvalsDF):
    ''' gets a matrix DF of Pvalues
    returns a dict with 2 matrix DFs of multiplicity adjusted Pvalues:
    {'FWER': .., 'FDR': ..} '''
    pvals = pvalsDF.copy()

    pvalsFlattened = pvals.unstack().copy()    # flatten matrix
    mask = ~pvalsFlattened.isna()              # mask - only not NA values

    pval_FDR = np.empty(pvalsFlattened.shape)  # new np array
    pval_FDR.fill(np.nan)                      # fill it with nan
    pval_FDR[mask] = sm.stats.multipletests(pvalsFlattened[mask], method='fdr_bh')[1] # FDR correct values different that nan
    pval_FDR_series = pd.Series(pval_FDR, index=pvalsFlattened.index) # turn into series with original indexes

    # same process, with FWER
    pval_FWER = np.empty(pvalsFlattened.shape)
    pval_FWER.fill(np.nan)
    pval_FWER[mask] = sm.stats.multipletests(pvalsFlattened[mask], method='holm')[1]
    pval_FWER_series = pd.Series(pval_FWER, index=pvalsFlattened.index)

    # turn back into DF
    adjPvals = {'FWER': pval_FWER_series.unstack(0),
                'FDR': pval_FDR_series.unstack(0)}

    return(adjPvals)

# unfinished!!
def runPCA(df, n_components):
    standardizeFunc = lambda col: (col - np.nanmean(col)) / np.nanstd(col)
    tmpDF = df.apply(standardizeFunc, raw=True)
    tmpDF = tmpDF.dropna()
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(tmpDF) # ?
    pcaCompo = pd.DataFrame(pca.components_, columns=tmpDF.columns)
    pd.DataFrame(pcaCompo.loc[0, :]).sort_values(by=0).plot(kind='bar', fontsize='9'), plt.show()


def multipAdjustPvalsMat(pvals, method='FDR', corrMat=False):
    '''
    :param pvals: pvalues matrix (pd.DataFrame)
    :param method: 'FDR' or 'FWER'
    :param corrMat: if True, will only calculate adjustment for upper diagonal
           matrix and copy to lower. Also, diagonal will not be ignored.
    :return: adjusted pvalues matrix (pd.DataFrame)
    '''
    pvals = pvals.copy()
    if method == 'FDR':
        methodSM = 'fdr_bh'
    elif method == 'FWER':
        methodSM = 'holm'

    # if it's a corrMat, calc only for upper diagonal
    if corrMat:
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
    pval_adj[mask] = sm.stats.multipletests(pvalsFlattened[mask], method=methodSM)[1]  # calc using mask

    # flattened vals - back to matrix
    pval_adj_series = pd.Series(pval_adj, index=pvalsFlattened.index)
    final = pval_adj_series.unstack(0)

    # if it's a corrMat, copy upper diagonal to lower diagonal
    if corrMat:
        for i in range(final.shape[0]):
            final.iloc[i, i] = 0
            for j in range(i):
                final.iloc[i, j] = final.iloc[j, i]

    return (final)


def adjustY4vars(data, yColName, xColsNamesList, ResidualsVsFitted=True):
    DataTools.check_series_index_equal([data[xColsNamesList].iloc[:, 0], data[yColName]])  # will raise exception if not
    model = sm.GLM(endog=data[yColName], exog=sm.add_constant(data[xColsNamesList]),
                   missing='drop')  # compute GLM. x=1s+muVec, y=colVec

    try:
        result = model.fit()
    except sm.tools.sm_exceptions.PerfectSeparationError:
        print('PerfectSeparationError!')
        result = None

    if result is None:
        controlled = data[yColName]
    else:
        controlled = data[yColName] - result.predict(sm.add_constant(data[xColsNamesList]))

    controlled = controlled.rename(yColName + ' - adjusted')

    if ResidualsVsFitted:
        fitted = pd.DataFrame(result.predict(sm.add_constant(data[xColsNamesList])))
        PlotTools.plot_scatter(fitted, pd.DataFrame(controlled), plt_corr_txt=True, show_reg_line=True, plot_title='', x_title='Fitted', y_title='Residuals', title_font_size=18, corr_font_size=8)
        plt.show()

    return (controlled)


def ML_model_test(fitted_model, df, y_col_name, x_cols_list,
                   drop_na, add_constant, logistic=True, calc_roc_auc=True, suppress_stand_message=False):
    ''' Test a GLM model with data in df (after dropping rows with NAs).
        fitted_model - model that was fitted to similar data, that has a .predict(x) function.
        df - data to test the model on. Pandas dataframe
        y_col_name - name of the response (y) column.
        x_cols_list - list of names of the predictor (x) columns.
        drop_na - boolean. Whether to drop rows with NA values (at least one)
        logistic - True if logistic model, False if linear model.

        Returns roc and auc (if logistic), predicted y values, true y values, fold change
    '''
    if type(fitted_model) == RegularizedResultsWrapper and not suppress_stand_message:
        print('Have you standardized data before function call?')

    results_dict = {}

    if drop_na:
        df_final = df[[y_col_name] + x_cols_list].dropna()
    else:
        df_final = df[[y_col_name] + x_cols_list].copy()

    if add_constant:
        x_final = sm.add_constant(df_final[x_cols_list].astype(float), has_constant='add')
    else:
        x_final = df_final[x_cols_list].astype(float)

    y_pred = fitted_model.predict(x_final)
    results_dict['y_pred'] = y_pred
    results_dict['y_true'] = df_final[y_col_name]

    if logistic:
        results_dict['Fold change'] = df_final[x_cols_list].loc[df_final[y_col_name] == 1].mean() / \
                                      df_final[x_cols_list].loc[df_final[y_col_name] == 0].mean()
        if calc_roc_auc:
            results_dict['roc_auc_info'] = roc_auc(y_true=results_dict['y_true'],
                                               y_pred=results_dict['y_pred'])
    else:
        results_dict['RMSE'] = root_mean_squared_error(results_dict['y_true'],
                                                       results_dict['y_pred'])
        results_dict['SE'] = squared_error(results_dict['y_true'],
                                           results_dict['y_pred'])

    return results_dict


def GLM_model(df, y_col_name, x_cols_list, logistic=True,
              penalize=False, alpha=0.5, L1_wt=0.5, calc_roc_auc=False, suppress_stand_message=False):
    ''' Build a GLM model from data in df (after dropping rows with NAs):
        df - pandas dataframe
        y_col_name - name of the response (y) column (in df).
        x_cols_list - list of names of the predictor (x) columns (in df).
        logistic - True if logistic model, False if linear model.

        Returns a dictionary with coefficients pvalues, model summary, model,
        fold change, roc and auc (if logistic), predicted y values, sample size.

        Important!!! Don't forget to standardize data before usage!!
        '''
    if not suppress_stand_message:
        print('Have you standardized data before function call?')

    if logistic:
        family = sm.families.Binomial()
        coef_func = np.exp
        coef_type = 'Odds Ratio'
    else:
        family = sm.families.Gaussian()
        coef_func = lambda x: x
        coef_type = 'Coef'

    results_dict = {}

    df_no_na = df[[y_col_name] + x_cols_list].dropna()

    model = sm.GLM(endog=df_no_na[y_col_name].astype(float),
                   exog=sm.add_constant(df_no_na[x_cols_list].astype(float), has_constant='add'),
                   family=family)
    try:
        if penalize:
            fitted_model = model.fit_regularized(method='elastic_net',
                                                 alpha=alpha,
                                                 L1_wt=L1_wt)
        else:
            fitted_model = model.fit()
            results_dict['summary'] = fitted_model.summary()
            print(results_dict['summary'])

            results_dict[coef_type] = coef_func(fitted_model.params[x_cols_list])
            results_dict[coef_type + ' conf intervals'] = \
                        fitted_model.conf_int().rename(columns={0: 'Lower', 1: 'Upper'})
            results_dict['pvalues'] = fitted_model.pvalues

        results_dict['model'] = fitted_model

        y_pred = fitted_model.predict()
        results_dict['y_pred'] = y_pred
        results_dict['y_true'] = df_no_na[y_col_name]

        if logistic:
            results_dict['Fold change'] = df_no_na[x_cols_list].loc[df_no_na[y_col_name] == 1].mean() / \
                                          df_no_na[x_cols_list].loc[df_no_na[y_col_name] == 0].mean()
            if calc_roc_auc:
                results_dict['roc_auc_info'] = roc_auc(y_true=results_dict['y_true'],
                                                       y_pred=y_pred)
        else:
            results_dict['RMSE'] = root_mean_squared_error(results_dict['y_true'],
                                                           results_dict['y_pred'])
            results_dict['SE'] = squared_error(results_dict['y_true'],
                                               results_dict['y_pred'])

    except sm.tools.sm_exceptions.PerfectSeparationError:
        print('PerfectSeparationError: A complete separation happens when the outcome variable\n'
              'separates a predictor variable or a combination of predictor variables completely.\n'
              'see https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/')

    results_dict['n'] = df_no_na.shape[0]

    return results_dict

def random_forest_model(df, y_col_name, x_cols_list, num_trees,
                        num_vars=None, criterion='mse', calc_roc_auc=False):
    results_dict = {}

    # Instantiate model with num_trees decision trees and num_vars maximum vars in tree
    rf = RandomForestRegressor(n_estimators=num_trees, max_depth=num_vars,
                               criterion=criterion, random_state=1)
    # Train the model on training data
    rf.fit(df[x_cols_list], df[y_col_name])
    results_dict['model'] = rf

    if calc_roc_auc:
        results_dict['y_pred'] = rf.predict(df[x_cols_list])
        results_dict['y_true'] = df[y_col_name]
        results_dict['roc_auc_info'] = roc_auc(y_true=results_dict['y_true'],
                                               y_pred=results_dict['y_pred']
                                               )

    return results_dict


def roc_auc(y_true, y_pred, save_path=None, save_auc_txt=None):
    assert len(y_true) == len(y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plot_roc(roc_auc, fpr, tpr,
                   save_path=save_path, save_auc_txt=save_auc_txt)

    return {'fig': fig, 'auc': roc_auc,
            'rates': {'false positive rate': fpr,
                      'true positive rate': tpr}}


def plot_roc(auc, fpr, tpr,
             save_path=None, save_auc_txt=None, show_if_none=False,
             title='ROC curve'):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='best')

    PlotTools.savePlt(savePath=save_path, showIfNone=show_if_none)
    if save_auc_txt is not None:
        FileTools.write_list_to_txt([auc], save_auc_txt)

    return fig


def standardize_dataset(df):
    '''
    Gets a numeric df and standardizes to N(0,1) every column
    that is not binary (i.e. has more than 2 unique values)
    :param df: Numeric pandas df
    :return: Numeric pandas df, columns standardized
    '''
    df_new = df.copy()
    standardizeFunc = lambda col: (col - np.nanmean(col)) / np.nanstd(col)

    for col in df_new.columns:
        if len(df_new[col].unique()) > 2:
            df_new[[col]] = df_new[[col]].apply(standardizeFunc)

    return df_new


def root_mean_squared_error(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    sum_se = 0
    for i in range(len(y_true)):
        pred_i = y_pred.iloc[i] if type(y_pred) is pd.Series else y_pred[i]
        true_i = y_true.iloc[i] if type(y_true) is pd.Series else y_true[i]

        sum_se += ((pred_i - true_i) ** 2)

    return (sum_se / len(y_true)) ** 0.5

def squared_error(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    sum_se = 0
    for i in range(len(y_true)):
        pred_i = y_pred.iloc[i] if type(y_pred) is pd.Series else y_pred[i]
        true_i = y_true.iloc[i] if type(y_true) is pd.Series else y_true[i]

        sum_se += ((pred_i - true_i) ** 2)

    return sum_se

def pca_2d_plot(features_df, target_val, standardize=True,
                saveFullPath=None,
                plotTitle='', aspectRatio=1.2, xRotation=45,
                titleFontSize=18, titleColor='maroon',
                legendTitle='', xticks=None, font_scale=1,
                snsStyle="ticks", legend_frame=False):
    if standardize:
        features_df = standardize_dataset(features_df)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features_df)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['PC 1', 'PC 2'],
                               index=features_df.index)

    PlotTools.plot_scatter_hue(principalDf['PC 1'], principalDf['PC 2'],
                               series_hue=target_val,
                               save_folder=None, save_full_path=saveFullPath,
                               aspect_ratio=aspectRatio,
                               show_reg_line=False, plot_title=plotTitle,
                               x_rotation=xRotation,
                               titleFontSize=titleFontSize, title_color=titleColor,
                               hue_legend_title=legendTitle, xticks=xticks,
                               legend_frame=legend_frame,
                               font_scale=font_scale,
                               sns_style=snsStyle)

def find_best_roc_auc_cutoff(y_true, y_predicted):
    """ Find the best probability cutoff point for
    predictions of a binary classification model,
    by the Youden's J index:
            max(t) {Sensitivity(t) + Specificity(t) − 1} =
            max(t) {true positive rate (t) - false positive rate (t)}
    parameters: y_true - vector of true binary labels (0/1)
                y_predicted - vector of predicted probabilities (0-1)
    returns: prediction threshold
            (positive labels should be then predicted
            if probability >= threshold)
    """
    fpr, tpr, threshold = roc_curve(y_true, y_predicted)
    calc = pd.DataFrame({'tpr-fpr': tpr - fpr, 'threshold': threshold})
    ind_measure_max = calc['tpr-fpr'].idxmax()
    best_threshold = calc.loc[ind_measure_max, 'threshold']

    return best_threshold

def find_best_threshold_f1(y_true, y_pred):
    """ Find the best probability cutoff point for
        predictions of a binary classification model,
        by the F1 metric.
        parameters: y_true - vector of true binary labels (0/1)
                    y_predicted - vector of predicted probabilities (0-1)
        returns: prediction threshold
                (positive labels should be then predicted
                if probability >= threshold)
        """

    thresh_f1_results = pd.DataFrame(columns=['threshold', 'f1'])
    i = 0
    for t in np.unique(y_pred):
        y_pred_bin = (y_pred >= t).astype(int)

        thresh_f1_results.loc[i, 'threshold'] = t
        thresh_f1_results.loc[i, 'f1'] = sklearn.metrics.f1_score(y_true, y_pred_bin)
        i += 1

    index_of_max = thresh_f1_results['f1'].astype(float).idxmax()
    best_thresh = thresh_f1_results.loc[index_of_max, 'threshold']

    return best_thresh

def measures_from_2by2_conf_mat(conf_mat):
    """
    :param conf_mat: array, confusion matrix
    :return: A dictionary with measures:
    """
    try:
        tn = conf_mat[0,0]
        fp = conf_mat[0,1]
        fn = conf_mat[1,0]
        tp = conf_mat[1,1]
    except Exception:
        print('oh no')
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = recall
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    false_negative_rate = fn / (fn + tp)
    false_positive_rate = fp / (fp + tn)
    false_discovery_rate = 1 - precision

    measures = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'accuracy': accuracy,
                'precision': precision, 'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity, 'f1': f1,
                # 'false_negative_rate': false_negative_rate,
                # 'false_positive_rate': false_positive_rate,
                # 'false_discovery_rate': false_discovery_rate,
               }
    return measures

def plot_confusion_matrix(y_true, y_pred, classes_names, add_border_diag=True, add_class_size=False, annot_fontsize=11):
    """
    Plots confusion matrix plots (counts and normalized).
    @param y_true: true labels vector
    @param y_pred: predicted scores matrix, shape [n_samples, n_classes], from model.predict_proba(X_test)
    @param classes_names: list of class names with the same order of y_pred columns.
                          can be output of model.classes_
    @param add_border_diag: boolean. Whether to add a border around the diagonal values.
    @return: fig object
    """

    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_pred),
                            columns=classes_names, index=classes_names)
    conf_mat_normed = pd.DataFrame(confusion_matrix(y_true, y_pred, normalize='true'),
                                   columns=classes_names, index=classes_names)
    acc = accuracy_score(y_true, y_pred)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19,8))
    sns.set_context(font_scale=1.7)
    sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='', ax=axes[0], square=True, annot_kws={"size": annot_fontsize})
    axes[0].set_xlabel('Predicted', fontsize=15)
    axes[0].set_ylabel('True', fontsize=15)
    axes[0].set_title(f'Confusion matrix (accuracy = {acc:.3f})', fontsize=17)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(0)
    if add_class_size:
        _ = axes[0].set_yticklabels([f'{cla}\n(n={(y_true==cla).sum()})' for cla in classes_names])

    sns.heatmap(conf_mat_normed, annot=True, cmap='Greens', fmt='.2f', ax=axes[1], square=True, annot_kws={"size": annot_fontsize})
    axes[1].set_title('Confusion matrix, normalized', fontsize=17)
    axes[1].set_xlabel('Predicted', fontsize=15)
    axes[1].set_ylabel('True', fontsize=15)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(0)
    if add_class_size:
        _ = axes[1].set_yticklabels([f'{cla}\n(n={(y_true==cla).sum()})' for cla in classes_names])

    if add_border_diag:
        for epi_ind in range(len(classes_names)):
            axes[0].add_patch(Rectangle((epi_ind, epi_ind), 1, 1, ec='black', fc='none', lw=1.5, clip_on=False))
            axes[1].add_patch(Rectangle((epi_ind, epi_ind), 1, 1, ec='black', fc='none', lw=1.5, clip_on=False))

    return fig

def plot_roc_curve_multiclass(classes_names, y_true_series, y_pred, colors=None): # TODO might need to debug
    """
    @param y_true_series: true labels pd.Series
    @param y_pred: predicted scores matrix, shape [n_samples, n_classes], from model.predict_proba(X_test)
    @param classes_names: list of class names with the same order of preds_prob columns.
                          can be output of model.classes_
    @param colors: list of colors to use for the classes. If None, assigns colors from list (n=22):
                  ['red', 'dodgerblue', 'darkviolet', 'lightgreen', 'mediumblue',
                  'darkorange', 'maroon', 'teal', 'purple', 'green', 'deepskyblue',
                  'yellowgreen', 'lightcoral', 'gold', 'aqua', 'slateblue', 'sienna',
                  'magenta', 'darkturquoise', 'lawngreen', 'olive', 'orchid']
    @return: fig object
    """

    Y_matrix = pd.get_dummies(y_true_series)

    if colors is None:
        colors = ['red', 'dodgerblue', 'darkviolet', 'lightgreen', 'mediumblue',
                  'darkorange', 'maroon', 'teal', 'purple', 'green', 'deepskyblue',
                  'yellowgreen', 'lightcoral', 'gold', 'aqua', 'slateblue', 'sienna',
                  'magenta', 'darkturquoise', 'lawngreen', 'olive', 'orchid']
    colors = cycle(colors)

    # Get curve info for each epitope
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, cla in enumerate(classes_names):
        fpr[cla], tpr[cla], _ = roc_curve(Y_matrix.loc[:, cla], y_pred[:, i])
        roc_auc[cla] = auc(fpr[cla], tpr[cla])

    # Plot curves
    fig = plt.figure(figsize=(8, 6))
    for cla, color in zip(classes_names, colors):
        plt.plot(fpr[cla], tpr[cla], color=color, lw=1.5,
                 label=f'{cla} (AUC = {roc_auc[cla]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver operating characteristic curve for each class vs. all')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
               frameon=False, title='Class')
    plt.setp(plt.axes().get_legend().get_title(), fontsize=15)

    return fig

def metrics_multilabel_f_labels(y_true, y_pred, average_type='macro'): # TODO might need to debug
    """
    Gets 2 vectors: y_true (true labels), y_pred (predicted labels).
    Returns a dictionary of different performance metrics.

    Average type: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’, None} default=’macro’
    This parameter is required for multiclass/multilabel targets.
    If None, the scores for each class are returned. Otherwise, this determines
    the type of averaging performed on the data:

    'binary':
    Only report results for the class specified by pos_label.
    This is applicable only if targets (y_{true,pred}) are binary.

    'micro':
    Calculate metrics globally by counting the total true positives,
    false negatives and false positives.

    'macro':
    Calculate metrics for each label, and find their unweighted mean.
    This does not take label imbalance into account.

    'weighted':
    Calculate metrics for each label, and find their average weighted by support
    (the number of true instances for each label). This alters ‘macro’ to account
    for label imbalance; it can result in an F-score that is not between precision and recall.

    'samples':
    Calculate metrics for each instance, and find their average
    (only meaningful for multilabel classification where this differs from accuracy_score).
    """
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['roc_auc_score'] = roc_auc_score(y_true, y_pred)
    res['precision'] = precision_score(y_true, y_pred, average=average_type)
    res['recall'] = recall_score(y_true, y_pred, average=average_type)
    res['f1'] = f1_score(y_true, y_pred, average=average_type)
    res['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred, average=average_type)
    res['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return res

def metrics_multiclass_f_preds(y_true, y_pred): # TODO might need to debug
    """
    Gets
    y_true: true labels vector
    y_pred: predicted scores matrix, shape [n_samples, n_classes], from model.predict_proba(X_test)
    @return: a dictionary with sklearn roc_auc, average_precision_score
    """
    res = {}
    res['roc_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    res['average_precision'] = average_precision_score(y_true, y_pred, average='macro', multi_class='ovr')
    return res

def metrics_binary_predictions(y_true, y_pred):
    """
    :param conf_mat: array, confusion matrix
    :return: A dictionary with measures:
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    if conf_mat.shape == (2,2):
        tn = conf_mat[0,0]
        fp = conf_mat[0,1]
        fn = conf_mat[1,0]
        tp = conf_mat[1,1]
    elif conf_mat.shape == (1, 1): # all true and predicted labels are the same single value: 0 or 1
        if y_true[0] == 0: # all true and predicted labels are 0
            tn = conf_mat[0,0]
            tp, fp, fn = 0, 0, 0
        elif y_true[0] == 1: # all true and predicted labels are 0
            tp = conf_mat[0,0]
            tn, fp, fn = 0, 0, 0
        else:
            raise Exception('Something here isnt right. Please check.')
    else:
        raise Exception('Confusion matrix isnt right. Please check.')

    try:
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0

    metrics = {'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
                'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
                'precision': sklearn.metrics.precision_score(y_true, y_pred),
                'recall': sklearn.metrics.recall_score(y_true, y_pred),
                'sensitivity': sklearn.metrics.recall_score(y_true, y_pred),
                'specificity': sklearn.metrics.recall_score(y_true, y_pred, pos_label=0),
                'f1': sklearn.metrics.f1_score(y_true, y_pred),
                'roc_auc': roc_auc,
               }
    return metrics

def geometric_mean(iterable):
    nums = np.array(iterable)
    return nums.prod()**(1.0/len(nums))

def rooted_mean_sum_squares(iterable):
    nums = np.array(iterable)
    squares = list(map(np.square, nums))
    sum_squares = np.sum(squares)
    mean_sum_squares = sum_squares / len(nums)
    rooted_mean_sum_squ = np.sqrt(mean_sum_squares)
    return rooted_mean_sum_squ

def showColCounts(column):
    ''' Gets a column, shows a figure of value counts '''
    sns.countplot(column,label="Count")
    plt.show()

def binary_model_log_likelihood(y_true, y_pred):
    """
    https://stackoverflow.com/questions/48185090/how-to-get-the-log-likelihood-for-a-logistic-regression-model-in-sklearn

    Example:
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0.1, 0.3, 0.7, 1, 0, 0.1, 0.3, 0.7, 1,])

        log_likelihood_elements:
        [-9.99200722e-16 -1.05360516e-01 -3.56674944e-01 -1.20397280e+00   -3.45395760e+01
	    -3.45387764e+01  -2.30258509e+00  -1.20397280e+00  -3.56674944e-01  -9.99200722e-16]

    This example shows what is the log likelihood result per each element: it estimates the distance
    between the predicted value and the true value.

    The log likelihood of the entire model is the minus of the mean over all elements

    Possible use: McFadden's R^2 - http://thestatsgeek.com/2014/02/08/r-squared-in-logistic-regression/

    @param y_true: np.array of the true labels
    @param y_pred: np.array of the labels predicted by the model
    @return: float. The log likelihood of the model
    """
    # np.log for y_pred[i]=1 and y_pred[i]=0 is undefined (nan and -inf).
    # So changing it to 1.e-15 and 0.99999999999 using clip_func
    clip_func = lambda pred: max(1e-15, min(1 - 1e-15, pred))
    y_pred_clipped = np.array(list(map(clip_func, y_pred)))

    log_likelihood_elements = y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    log_likelihood = -np.sum(log_likelihood_elements) / len(y_true)

    assert log_likelihood == log_loss(y_true, y_pred)

    return log_likelihood

def standardize_series(series):
    """
    Standardize pd.Series values to standard normal distribution (Z) - Normal distribution with mean 0 and variance 1.
    @param series: pd.Series numeric
    @return: pd.Series with standardized values
    """
    std = np.nanstd(series)
    mean = np.nanmean(series)
    standardizeFunc = lambda x: (x - mean) / std
    return series.apply(standardizeFunc)

#Partial Correlation in Python - from https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/SubsurfaceDataAnalytics_Feature_Ranking.ipynb

#This uses the linear regression approach to compute the partial correlation
#(might be slow for a huge number of variables). The algorithm is detailed here:

# http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression

#Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
#the algorithm can be summarized as
#    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
#    2) calculate the residuals in Step #1
#    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
#    4) calculate the residuals in Step #3
#    5) calculate the correlation coefficient between the residuals from Steps #2 and #4;
#    The result is the partial correlation between X and Y while controlling for the effect of Z

#Date: Nov 2014
#Author: Fabian Pedregosa-Izquierdo, f@bianp.net
#Testing: Valentina Borghesani, valentinaborghesani@gmail.com

def partial_corr(C):
    #    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    #    for the remaining variables in C.

    #    Parameters
    #    C : array-like, shape (n, p)
    #        Array with the different variables. Each column of C is taken as a variable
    #    Returns
    #    P : array-like, shape (p, p)
    #        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
    #        for the remaining variables in C.

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

def semipartial_corr(C): # Michael Pyrcz modified the function above by Fabian Pedregosa-Izquierdo, f@bianp.net for semipartial correlation
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] # just use the value, not a residual
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

def get_class_preds_sklearn_model(model, preds_proba, class_name, raise_exception=True):
    """
    Gets a sklearn model, a matrix of predicted probabilities (output from function model.predict_proba)
    and the class identifier (string/int. according to the model y_train vector).
    Returns the vector of predictions for the specified class, from preds_proba matrix.
    If the class is not in the model classes list, will return none or raise an error.
    @param model: sklearn model object
    @param preds_proba: predicted class probabilities, output from function model.predict_proba
    @param class_name: target class identifier (string/int. according to the model y_train vector)
    @param raise_exception: boolean. Whether to raise an error if the requested class is
                            not in the model classes list.
    @return: the vector of predictions for the specified class, from preds_proba
    """
    model_classes = model.classes_

    class_ind = None
    for i, cl in enumerate(model_classes):
        if cl==class_name:
            class_ind = i

    if class_ind is not None:
        return preds_proba[:, class_ind]
    else:
        if raise_exception:
            raise Exception('class_name was not found in the given model classes.')
        else:
            return None

def binary_vectors_similarity(vec1, vec2):
    """
    Gets two binary vectors (numpy arrays dim1) with equal lengths,
    and returns similarity metrics calculated between them:
        # a - number of positions where the values of vec1 and vec2 are both 1, meaning 'positive matches'
        # b - number of positions where the value of vec1=1 and vec2=0 ('vec1 only')
        # c - number of positions where the value of vec1=0 and vec2=1 ('vec2 only')
        # d - number of positions where the values of vec1 and vec2 are both 0, meaning 'negative matches'

        # jaccard (a / a+b+c)
        # hamming distance (b+c)
        # hamming percentile (b+c / n)
        # shared_f_vec1 - percentile of shared ones out of vec1 ones (a / a+b)
        # shared_f_vec2 - percentile of shared ones out of vec2 ones (a / a+c)

    Paper with many more similarity metrics definitions:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.352.6123&rep=rep1&type=pdf (notice that b and c are swapped in the paper!)

    :param vec1: <class 'numpy.ndarray'>
    :param vec2: <class 'numpy.ndarray'>
    :return: metrics (dict) with values:
             'jaccard', 'hamming', 'hamming_per', 'shared_f_vec1', 'shared_f_vec2', 'a', 'b', 'c', 'd'
    """
    assert len(vec1) == len(vec2)

    n = len(vec1)
    a = 0 # a is the number of positions where the values of vec1 and vec2 are both 1, meaning 'positive matches',
    b = 0 # b is the number of positions where the value of vec1=1 and vec2=0
    c = 0 # c is the number of positions where the value of vec1=0 and vec2=1
    d = 0 # d is the number of positions where the values of vec1 and vec2 are both 0, meaning 'negative matches'
    for i in range(len(vec1)):
        if vec1[i] == vec2[i] == 1:
            a += 1
        if vec1[i] == 1 and vec2[i] == 0:
            b += 1
        if vec1[i] == 0 and vec2[i] == 1:
            c += 1
        if vec1[i] == vec2[i] == 0:
            d += 1

    metrics = {}
    metrics['a'] = a
    metrics['b'] = b
    metrics['c'] = c
    metrics['d'] = d
    metrics['hamming'] = b + c
    metrics['hamming_per'] = (b + c) / n

    if a+b+c == 0:
        metrics['jaccard'] = 0
    else:
        metrics['jaccard'] = a / (a + b + c)

    if a+b == 0:
        metrics['shared_f_vec1'] = 0
    else:
        metrics['shared_f_vec1'] = a / (a+b)

    if a+c == 0:
        metrics['shared_f_vec2'] = 0
    else:
        metrics['shared_f_vec2'] = a / (a+c)

    return metrics


def calc_f1_score(recall, precision):
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1