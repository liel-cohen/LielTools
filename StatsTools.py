import pandas as pd
import  scipy.signal.signaltools
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
scipy.signal.signaltools._centered = _centered
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
from statsmodels.base.elastic_net import RegularizedResultsWrapper
import math
from scipy import stats
from sklearn.metrics import confusion_matrix

if 'LielTools' in sys.modules:
    from LielTools import DataTools
    from LielTools import PlotTools
    from LielTools import FileTools
else:
    import DataTools
    import PlotTools
    import FileTools


def showColCounts(column):
    ''' Gets a column, shows a figure of value counts '''
    sns.countplot(column,label="Count")
    plt.show()

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

    return(corr)

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

''' gets a matrix DF of Pvalues
returns a dict with 2 matrix DFs of multiplicity adjusted Pvalues:
{'FWER': .., 'FDR': ..} '''
def correctedPvalsFromMatrixDF(pvalsDF):
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
        PlotTools.plot_scatter(fitted, pd.DataFrame(controlled), pltCorr=True, showRegLine=True, plotTitle='', xTitle='Fitted', yTitle='Residuals', titleFontSize=18, corrFontSize=8)
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
            if calc_roc_auc: results_dict['roc_auc_info'] = roc_auc(y_true=results_dict['y_true'],
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
