import sys

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

remove_values_from_list(sys.path, 'C:/Users/liel-/Dropbox/PyCharm/PycharmProjectsNew/TCR_dataset_v2/LielTools_v2/')
remove_values_from_list(sys.path, 'C:\\Users\\liel-\\Dropbox\\PyCharm\\PycharmProjectsNew\\TCR_dataset_v2\\LielTools_v2')

sys.path.append(r'C:\Users\liel-\Dropbox\PyCharm\PycharmProjectsNew\TCR_dataset_v3')
sys.path.append(r'C:\Users\lielc\Dropbox\PyCharm\PycharmProjectsNew\TCR_dataset_v3')

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LielTools_v3 import StatsTools
from LielTools_v3 import DataTools
from LielTools_v3 import PlotTools
from LielTools_v3 import FileTools
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold as sk_KFold
import statsmodels.api as sm

# Must define variable: fig_path_liel

# elastic net L1_wt should be between 0-1

def glm_cv(fig_path_liel, model_df, y_col_name, x_cols_list, cv_folds, model_name, figsize=(12, 8),
           alpha=0.001, L1_wt=0.01):
    FileTools.createFolder(fig_path_liel + '/GLM/{}/'.format(model_name))

    random.seed(42)
    np.random.seed(42)
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    obese_GLM_res, obese_GLM_test_res = [], []
    for ind_train, ind_test in kfold.split(model_df[x_cols_list], model_df[y_col_name]):
        obese_GLM_res_fold = StatsTools.GLM_model(model_df.iloc[ind_train],
                                                  y_col_name, x_cols_list,
                                                  logistic=True,
                                                  penalize=True,
                                                  alpha=alpha, L1_wt=L1_wt,
                                                  suppress_stand_message=True)
        obese_GLM_res.append(obese_GLM_res_fold)

        obese_GLM_test_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
                                                           model_df.iloc[ind_test],
                                                           y_col_name, x_cols_list,
                                                           drop_na=True, add_constant=True,
                                                           logistic=True,
                                                           calc_roc_auc=False)
        obese_GLM_test_res.append(obese_GLM_test_res_fold)

    roc_auc_info = calc_cv_auc(obese_GLM_test_res)
    final_auc = roc_auc_info['auc']

    print('{} CV-{} test AUC: {}'.format(model_name, cv_folds, final_auc))

    FileTools.write2Excel(fig_path_liel + '/GLM/{}/AUC____{}.xlsx'.format(model_name,
                                                                          model_name),
                          pd.DataFrame([final_auc], index=['AUC']))

    title_text = 'y={}, AUC={}, n={}, CV-{} '.format(y_col_name, np.round(final_auc, 3),
                                                             model_df.shape[0],
                                                             cv_folds)

    plot_params(fig_path_liel, obese_GLM_res, title_text, model_name, figsize)

    plot_predictions(fig_path_liel, obese_GLM_test_res, model_name, title_text)

    FileTools.copy_all_folder_files_to_folder(fig_path_liel + '/GLM/{}/'.format(model_name),
                                              fig_path_liel + '/GLM/All_models/')


def glm_cv_linear(fig_path_liel, model_df, y_col_name, x_cols_list, cv_folds, model_name, figsize=(12, 8),
                  alpha=0.001, L1_wt=0.01):
    FileTools.createFolder(fig_path_liel + '/GLM/{}/'.format(model_name))

    random.seed(42)
    np.random.seed(42)

    n = model_df.shape[0]

    kfold = sk_KFold(n_splits=cv_folds, shuffle=True)
    obese_GLM_res, obese_GLM_test_res = [], []
    for ind_train, ind_test in kfold.split(model_df[x_cols_list]):
        obese_GLM_res_fold = StatsTools.GLM_model(model_df.iloc[ind_train],
                                                  y_col_name, x_cols_list,
                                                  logistic=False,
                                                  penalize=True,
                                                  alpha=alpha, L1_wt=L1_wt,
                                                  suppress_stand_message=True)
        obese_GLM_res.append(obese_GLM_res_fold)

        obese_GLM_test_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
                                                           model_df.iloc[ind_test],
                                                           y_col_name, x_cols_list,
                                                           drop_na=True, add_constant=True,
                                                           logistic=False, calc_roc_auc=False)
        obese_GLM_test_res.append(obese_GLM_test_res_fold)

    sum_SE = 0
    for i in list(range(cv_folds)):
        sum_SE += obese_GLM_test_res[i]['SE']
    final_RMSE = (sum_SE / n) ** 0.5

    print('{} CV-{} test RMSE: {}'.format(model_name, cv_folds, final_RMSE))
    # FileTools.writeList2txt([final_RMSE],
    #                         fig_path_liel + '/GLM/{}/RMSE____{}.txt'.format(model_name,
    #                                                                         model_name))

    FileTools.write2Excel(fig_path_liel + '/GLM/{}/RMSE____{}.xlsx'.format(model_name,
                                                                           model_name),
                            pd.DataFrame([final_RMSE], index=['RMSE']))

    title_text = 'y={}, RMSE={}, n={}, CV-{} '.format(y_col_name, np.round(final_RMSE, 3),
                                                      n, cv_folds)

    plot_params(fig_path_liel, obese_GLM_res, title_text, model_name, figsize)

    plot_predictions(fig_path_liel, obese_GLM_test_res, model_name, title_text)

    FileTools.copy_all_folder_files_to_folder(fig_path_liel + '/GLM/{}/'.format(model_name),
                                              fig_path_liel + '/GLM/All_models/')


### Before unifying with glm_LOO_Linear
# def glm_LOO(fig_path_liel, model_df, y_col_name, x_cols_list, model_name, heatmap_figsize=(12, 8), bar_figsize=(10,8),
#             alpha=0.001, L1_wt=0.01, heatmap_annotate_text=True):
#     res = {}
#
#     FileTools.createFolder(fig_path_liel + '/GLM/{}/'.format(model_name))
#     cv_folds = 'LOO'
#     random.seed(42)
#     np.random.seed(42)
#     obese_GLM_res, obese_GLM_test_res = [], []
#     for i_index_test in range(len(model_df.index)):
#         ind_train = list(range(len(model_df.index)))
#         ind_train.remove(i_index_test)
#         ind_train = np.array(ind_train)
#         ind_test = np.array([i_index_test])
#         obese_GLM_res_fold = StatsTools.GLM_model(model_df.iloc[ind_train],
#                                                   y_col_name, x_cols_list,
#                                                   logistic=True,
#                                                   penalize=True,
#                                                   alpha=alpha, L1_wt=L1_wt,
#                                                   suppress_stand_message=True)
#         obese_GLM_res.append(obese_GLM_res_fold)
#
#         obese_GLM_test_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
#                                                            model_df.iloc[ind_test],
#                                                            y_col_name, x_cols_list,
#                                                            drop_na=True, add_constant=True,
#                                                            logistic=True, calc_roc_auc=False)
#         obese_GLM_test_res.append(obese_GLM_test_res_fold)
#
#     roc_auc_info = calc_cv_auc(obese_GLM_test_res)
#     final_auc = roc_auc_info['auc']
#
#     print('{} CV-{} test AUC: {}'.format(model_name, cv_folds, final_auc))
#     res['roc_auc_info'] = roc_auc_info
#
#     FileTools.write2Excel(fig_path_liel + '/GLM/{}/AUC____{}.xlsx'.format(model_name,
#                                                                          model_name),
#                           pd.DataFrame([final_auc], index=['AUC']))
#
#     title_text = 'y={}, AUC={}, n={}, CV-{} '.format(y_col_name, np.round(final_auc, 3),
#                                                              model_df.shape[0],
#                                                              cv_folds)
#
#     res['GLM_params'] = plot_params(fig_path_liel, obese_GLM_res, title_text, model_name,
#                       heatmap_figsize, bar_figsize=bar_figsize,
#                       heatmap_annotate_text=heatmap_annotate_text)
#
#     plot_predictions(fig_path_liel, obese_GLM_test_res, model_name, title_text)
#
#     FileTools.copy_all_folder_files_to_folder(fig_path_liel + '/GLM/{}/'.format(model_name),
#                                               fig_path_liel + '/GLM/All_models/')
#
#     return res


def glm_LOO(fig_path_liel, model_df, y_col_name, x_cols_list, model_name, heatmap_figsize=(12, 8), bar_figsize=(10,8),
            alpha=0.001, L1_wt=0.01, heatmap_annotate_text=True, logistic=True):
    res = {}

    FileTools.createFolder(fig_path_liel + '/GLM/{}/'.format(model_name))
    cv_folds = 'LOO'

    random.seed(42)
    np.random.seed(42)

    obese_GLM_res, obese_GLM_test_res = [], []
    for i_index_test in range(len(model_df.index)):
        ind_train = list(range(len(model_df.index)))
        ind_train.remove(i_index_test)
        ind_train = np.array(ind_train)
        ind_test = np.array([i_index_test])

        obese_GLM_res_fold = StatsTools.GLM_model(model_df.iloc[ind_train],
                                                  y_col_name, x_cols_list,
                                                  logistic=logistic,
                                                  penalize=True,
                                                  alpha=alpha, L1_wt=L1_wt,
                                                  suppress_stand_message=True)
        obese_GLM_res.append(obese_GLM_res_fold)

        obese_GLM_test_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
                                                           model_df.iloc[ind_test],
                                                           y_col_name, x_cols_list,
                                                           drop_na=True, add_constant=True,
                                                           logistic=logistic, calc_roc_auc=False)
        obese_GLM_test_res.append(obese_GLM_test_res_fold)

    if logistic:
        res['roc_auc_info'] = calc_cv_auc(obese_GLM_test_res)
        final_measure = res['roc_auc_info']['auc']
        measure_name = 'AUC'
    else:
        sum_SE = 0
        for i in list(range(len(obese_GLM_test_res))):
            sum_SE += obese_GLM_test_res[i]['SE']
        final_measure = (sum_SE / len(obese_GLM_test_res)) ** 0.5
        res['RMSE'] = final_measure
        measure_name = 'RMSE'


    print('{} CV-{} test {}: {}'.format(model_name, cv_folds, measure_name, final_measure))

    FileTools.write2Excel(fig_path_liel + '/GLM/{}/{}____{}.xlsx'.format(model_name, measure_name,
                                                                         model_name),
                          pd.DataFrame([final_measure], index=[measure_name]))

    title_text = 'y={}, {}={}, n={}, CV-{} '.format(y_col_name, measure_name, np.round(final_measure, 3),
                                                             model_df.shape[0],
                                                             cv_folds)

    res['GLM_params'] = plot_params(fig_path_liel, obese_GLM_res, title_text, model_name,
                      heatmap_figsize, bar_figsize=bar_figsize,
                      heatmap_annotate_text=heatmap_annotate_text)

    plot_predictions(fig_path_liel, obese_GLM_test_res, model_name, title_text)

    FileTools.copy_all_folder_files_to_folder(fig_path_liel + '/GLM/{}/'.format(model_name),
                                              fig_path_liel + '/GLM/All_models/')

    return res

# def glm_LOO_linear(fig_path_liel, model_df, y_col_name, x_cols_list, model_name, figsize=(12, 8),
#                   alpha=0.001, L1_wt=0.01):
#     FileTools.createFolder(fig_path_liel + '/GLM/{}/'.format(model_name))
#     cv_folds = 'LOO'
#     random.seed(42)
#     np.random.seed(42)
#
#     n = model_df.shape[0]
#
#     obese_GLM_res, obese_GLM_test_res = [], []
#     for i_index_test in range(len(model_df.index)):
#         ind_train = list(range(len(model_df.index)))
#         ind_train.remove(i_index_test)
#         ind_train = np.array(ind_train)
#         ind_test = np.array([i_index_test])
#         obese_GLM_res_fold = StatsTools.GLM_model(model_df.iloc[ind_train],
#                                                   y_col_name, x_cols_list,
#                                                   logistic=False,
#                                                   penalize=True,
#                                                   alpha=alpha, L1_wt=L1_wt,
#                                                   suppress_stand_message=True)
#         obese_GLM_res.append(obese_GLM_res_fold)
#
#         obese_GLM_test_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
#                                                            model_df.iloc[ind_test],
#                                                            y_col_name, x_cols_list,
#                                                            drop_na=True, add_constant=True,
#                                                            logistic=False, calc_roc_auc=False)
#         obese_GLM_test_res.append(obese_GLM_test_res_fold)
#
#     sum_SE = 0
#     for i in list(range(len(obese_GLM_test_res))):
#         sum_SE += obese_GLM_test_res[i]['SE']
#     final_RMSE = (sum_SE / n) ** 0.5
#
#     print('{} CV-{} test RMSE: {}'.format(model_name, cv_folds, final_RMSE))
#
#     FileTools.write2Excel(fig_path_liel + '/GLM/{}/RMSE____{}.xlsx'.format(model_name,
#                                                                           model_name),
#                           pd.DataFrame([final_RMSE], index=['RMSE']))
#
#     title_text = 'y={}, RMSE={}, n={}, CV-{} '.format(y_col_name, np.round(final_RMSE, 3),
#                                                       n, cv_folds)
#
#     params_df = plot_params(fig_path_liel, obese_GLM_res, title_text, model_name, figsize)
#
#     plot_predictions(fig_path_liel, obese_GLM_test_res, model_name, title_text)
#
#     FileTools.copy_all_folder_files_to_folder(fig_path_liel + '/GLM/{}/'.format(model_name),
#                                               fig_path_liel + '/GLM/All_models/')
#
#     return params_df, title_text

def calc_cv_auc(folds_test_res):
    print('Number of CV folds: ', str(len(folds_test_res)))
    y_pred_list = []
    y_true_list = []
    y_ind_list = []
    for i in range(len(folds_test_res)):
        y_pred_list += list(folds_test_res[i]['y_pred'])
        y_true_list += list(folds_test_res[i]['y_true'])
        y_ind_list += list(folds_test_res[i]['y_true'].index)

    roc_auc_info = StatsTools.roc_auc(y_true=y_true_list, y_pred =y_pred_list)
    return roc_auc_info

def plot_predictions(fig_path_liel, folds_test_res, model_name, title_text):
    y_pred_list = []
    y_true_list = []
    y_ind_list = []
    for i in range(len(folds_test_res)):
        y_pred_list += list(folds_test_res[i]['y_pred'])
        y_true_list += list(folds_test_res[i]['y_true'])
        y_ind_list += list(folds_test_res[i]['y_true'].index)

    predictions = pd.DataFrame({'y_true': y_true_list, 'y_pred': y_pred_list}, index=y_ind_list)
    # predictions['BMI'] = vac_df_pre['BMI']
    # predictions.sort_values(by='BMI', inplace=True)

    spearman = StatsTools.getCorrelationForDFColumns(predictions['y_true'], predictions['y_pred'],
                                                    method='spearman', conf_interval=True)
    spearman_text = 'y_true vs. y_predicted: Spearman r={}, pval={}'.format(np.round(spearman[0], 3), np.round(spearman[1], 7))
    spearman_df = pd.DataFrame([spearman[0], spearman[1],
                                spearman[2]['lower'], spearman[2]['upper'], spearman[2]['std']],
                               index=['r', 'p-value', 'CI lower', 'CI upper', 'std'])
    FileTools.write2Excel(fig_path_liel + '/GLM/{}/y_true vs y_predicted Spearman _____{}.xlsx'.format(model_name,
                                                                                                       model_name),
                          spearman_df)

    # fig, ax = plt.subplots(figsize=(12, 6))
    # PlotTools.plotScatter(predictions['BMI'], predictions['y_true'], pltCorr=False, ax=ax, dotsColor='Orange', figsize=(14, 8))
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # fig.text(0.2, 0.9, pearson_text, fontdict={'size': 18})
    # PlotTools.plotScatter(predictions['BMI'], predictions['y_pred'], pltCorr=False, ax=ax,
    #                       figsize=(12, 6), axesTitleFontSize=18,
    #                          saveFullPath=fig_path_liel + '/GLM/{}/true vs preds by BMI _____{}.jpg'.format(model_name,
    #                                                                                                         model_name))

    logistic = len(predictions['y_true'].unique()) == 2
    if logistic:
        x_jitter = 0.07
    else:
        x_jitter = None

    fig, ax = plt.subplots(figsize=(10, 7))
    PlotTools.plotScatter(predictions['y_true'], predictions['y_pred'], pltCorr=False,
                          figsize=(10, 7), axesTitleFontSize=24, x_jitter=x_jitter,
                          plotTitle=title_text)
    if logistic:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = [item if (item=='0.0' or item=='1.0' or item=='0' or item=='1') else '' for item in labels]
        ax.set_xticklabels(labels)

    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.text(0.2, 0.9, spearman_text, fontdict={'size': 18})
    plt.savefig(fig_path_liel + '/GLM/{}/y_true vs y_predicted _____{}.jpg'.format(model_name,
                                                                                   model_name))

def plot_params(fig_path_liel, folds_res, title_text, model_name,
                heatmap_figsize, bar_figsize=(10,8), params_GLM=None,
                heatmap_annotate_text=True):
    if params_GLM is None:
        params_GLM = pd.DataFrame(folds_res[0]['model'].params, columns=['Model 1'])
        for i in list(range(1, len(folds_res))):
            params_GLM = params_GLM.join(pd.DataFrame(folds_res[i]['model'].params,
                                                      columns=['Model {}'.format(i + 1)]))

    PlotTools.plotHeatmap_real(params_GLM.astype(float), cmap='RdBu_r', figsize=heatmap_figsize,
                               title=title_text, title_fontsize=21,
                               font_scale=1.5, snsStyle='ticks',
                               xlabel='CV Model', ylabel='Variable', colormap_label='',
                               vmin=-1, vmax=1,
                               annotate_text=heatmap_annotate_text, annotate_fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_path_liel + '/GLM/{}/params_heatmap____{}.jpg'.format(model_name,
                                                                          model_name))

    params_GLM_stats = pd.DataFrame(params_GLM.mean(axis=1).sort_values(), columns=['Mean'])
    params_GLM_stats = params_GLM_stats.join(pd.DataFrame(params_GLM.std(axis=1), columns=['Std']))
    FileTools.write2Excel(fig_path_liel + '/GLM/{}/params_stats____{}.xlsx'.format(model_name,
                                                                                   model_name),
                          params_GLM_stats)

    plt.figure(figsize=bar_figsize)
    sns.barplot(x=params_GLM_stats.index, y=params_GLM_stats.Mean, yerr=params_GLM_stats.Std.values)
    plt.xticks(rotation=90)
    plt.ylim((-1, 1))
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.title(title_text)
    plt.ylabel('Mean Variable Weight in CV Models')
    plt.xlabel('Variable')
    plt.tight_layout()
    plt.savefig(fig_path_liel + '/GLM/{}/params_bar____{}.jpg'.format(model_name,
                                                                      model_name))
    return params_GLM

def tune_GLM_elastic_net(fig_path_liel, model_df, y_col_name, x_cols_list, cv_folds, model_name, test_size=0.25,
                         logistic=True,
                         alphas=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 5],
                         l1s=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 5]):

    elastic = pd.DataFrame(index=alphas, columns=l1s)
    for alp in alphas:
        for l1 in l1s:
            print('alpha: ', alp, '   l1: ', l1)
            random.seed(1)
            np.random.seed(1)

            if logistic:
                kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True)
                k_splits = kfold.split(model_df[x_cols_list], model_df[y_col_name])
            else:
                kfold = sk_KFold(n_splits=cv_folds, shuffle=True)
                k_splits = kfold.split(model_df[x_cols_list])

            obese_GLM_valid_res = []
            for ind_train, ind_test in k_splits:
                if logistic:
                    stratify_val = model_df.iloc[ind_train][y_col_name]
                else:
                    stratify_val = None

                ind_train_train, ind_train_valid = train_test_split(model_df.iloc[ind_train].index,
                                                                    test_size=test_size,
                                                                    stratify=stratify_val,
                                                                    shuffle=True)
                try:
                    obese_GLM_res_fold = StatsTools.GLM_model(model_df.loc[ind_train_train],
                                                              y_col_name, x_cols_list,
                                                              logistic=logistic,
                                                              penalize=True,
                                                              alpha=alp, L1_wt=l1)

                    # print('train.shape', model_df.loc[ind_train_train].shape)
                    # print('params.shape', obese_GLM_res_fold['model'].params.shape)
                    # print('valid.shape', model_df.loc[ind_train_valid].shape)
                    obese_GLM_valid_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
                                                                        model_df.loc[ind_train_valid],
                                                                        y_col_name, x_cols_list,
                                                                        drop_na=False, add_constant=True,
                                                                        logistic=logistic)
                    obese_GLM_valid_res.append(obese_GLM_valid_res_fold)

                    del obese_GLM_res_fold
                except ValueError:
                    print('grid point has a ValueError in 1 fold')

            if len(obese_GLM_valid_res) == 1:
                print('!!!!!@@@@ warning, number of valid folds for grid point is 1!')

            if logistic:
                # sum_auc = 0
                # for i in list(range(len(obese_GLM_valid_res))):
                #     sum_auc += obese_GLM_valid_res[i]['roc_auc_info']['auc']
                # elastic.loc[alp, l1] = sum_auc / len(obese_GLM_valid_res)

                measure_name = 'AUC'
                roc_auc_info = calc_cv_auc(obese_GLM_valid_res)
                elastic.loc[alp, l1] = roc_auc_info['auc']
            else:
                measure_name = 'RMSE'
                sum_SE = 0
                for i in list(range(len(obese_GLM_valid_res))):
                    sum_SE += obese_GLM_valid_res[i]['SE']
                elastic.loc[alp, l1] = (sum_SE / len(obese_GLM_valid_res)) ** 0.5

            del obese_GLM_valid_res, obese_GLM_valid_res_fold

    plt.figure(figsize=(13, 10))
    sns.heatmap(elastic.astype(float), annot=True)
    plt.title(measure_name)
    plt.xlabel('L1_weight')
    plt.ylabel('Alpha')
    plt.savefig(fig_path_liel + '/Tuning__{}.jpg'.format(model_name))

    return elastic

def doGLM(df, outcomedf, outcomeVar,predictors, covariates=[], logistic=True):
    '''
    A function which performs glm analysis on the given input dataframe and outcome df on the list of predictor varaibles 
    given. The funciton also takes into account the any covaraites given for buidling the model.
    The functions does a univariate analysis on each of the given input variables against the outcome.
    A dataframe with Odd ratio,Lower limit , Upper limit ,pvalue ,difference , number of observation, const and beta for prec and each covaraiate is 
    constructed.
    :param df:dataframe with all the variables to be used for analysis. 
    :param outcomedf :dataframe with all the outcome variables against which input variable have to analysed
    :param outcomeVar :string. name of the outcome var to be analysed 
    :param predictors :list.all the predictors or variable to be analysed.
    :param covariates :list. variables to be used as covariates along with input variable.
    :param logistic: Boolean. to check whether logistic or linear analyis to perform. default is logistic 
    :return dataframe with ['OR', 'LL', 'UL', 'pvalue', 'Diff', 'N',"ParamConst","ParamBeta"] and with all the const and beta values of covaraites.
    '''
    cols = []
    index = []
  
    if logistic:
        family = sm.families.Binomial()
        coefFunc = np.exp
        cols = ['OR', 'LL', 'UL', 'pvalue', 'Diff', 'N',"ParamConst","ParamBeta"]
    else:
        family = sm.families.Gaussian()
        coefFunc = lambda x: x
        cols = ['Coef', 'LL', 'UL', 'pvalue', 'Diff', 'N',"ParamConst","ParamBeta"]
    
    cols = cols+covariates

    #ignore if the predictors are unit8(categorical)
    for col in predictors:
        if df[col].dtype.name =='uint8':
            continue
        index.append(col)
        
    outDf = pd.DataFrame(index=index,columns=cols)
    params = []
    pvalues = []
    resObj = []
    for i, predc in enumerate(predictors):
        if df[predc].dtype.name =='uint8':
            continue
        #adjust may be called as covariate or control var
        exogVars = list(set([predc] + covariates))
        tmp = df[exogVars].join(outcomedf).dropna()
        model = sm.GLM(endog=tmp[outcomeVar].astype(float), exog=sm.add_constant(tmp[exogVars].astype(float)),
                       family=family)
        try:
            res = model.fit()
            outDf.OR[predc] = coefFunc(res.params[predc])
            outDf.pvalue[predc] = res.pvalues[predc]
            coefficent=  coefFunc(res.conf_int().loc[predc])
            outDf.LL[predc] = coefficent[0]
            outDf.UL[predc] = coefficent[1]
            outDf.Diff[predc] = tmp[predc].loc[tmp[outcomeVar] == 1].mean() - tmp[predc].loc[tmp[outcomeVar] == 0].mean()
            outDf.ParamConst[predc] = res.params.get(key="const")
            outDf.ParamBeta[predc] = res.params.get(key=predc)
            for c in covariates:
                outDf.at[predc,c]=  res.params.get(key=c)
           # params.append(res.params.to_dict())
            pvalues.append(res.pvalues.to_dict())
            resObj.append(res)
        except sm.tools.sm_exceptions.PerfectSeparationError:
            outDf.OR[predc] = np.nan
            outDf.pvalue[predc] = 0.0
            outDf.LL[predc] = np.nan
            outDf.UL[predc] = np.nan
            outDf.Diff[predc] = tmp[predc].loc[tmp[outcomeVar] == 1].mean() - tmp[predc].loc[tmp[outcomeVar] == 0].mean()
            outDf.ParamConst[predc] = np.nan
            outDf.ParamBeta[predc] = np.nan
            for c in covariates:
                outDf.at[predc,c]=  np.nan
            #params.append({k: np.nan for k in [predc] + adj})
            pvalues.append({k: np.nan for k in [predc] + covariates})
            resObj.append(None)
            print('PerfectSeparationError: %s with %s' % (predc, outcomeVar))
        outDf.N[predc] = tmp.shape[0]
    
    #outDf['params'] = params
    outDf['pvalues'] = pvalues
    outDf['res'] = resObj
    
    return outDf

def GLMAnalysis(dataDf,predictors,outcomeVars=[],covariateVars=[],standardize=True,logistic=True,univariate=True):
    """
    A functions which prepares the data for GLM analysis and performs GLM Analysis on the input dataframe
    The function applies normalisation on the input the variables. It splits the df into output and input df.
    :param df:dataframe with all the variables to be used for analysis. 
    :param dataDf :dataframe
    :param outcomeVar :list. all the outcome vairables to be used for analysis 
    :param predictors :list.all the predictors or variable to be analysed.
    :param covariateVars :list. variables to be used as covariates along with input variable.default is to apply 
    :param standardize: boolean. To apply normalisation or not the input variables.
    :param logistic: Boolean. to check whether logistic or linear analyis to perform. default is logistic regression
    :param univariate: Boolean perform univariate or multivariate. default is univariate 
    :return dataframe with ['OR', 'LL', 'UL', 'pvalue', 'Diff', 'N',"ParamConst","ParamBeta"] and with all the const and beta values of covaraites.
    """
    resL = []
    standardizeFunc = lambda col: (col - np.nanmean(col)) / np.nanstd(col)
   
    df = dataDf.copy()
    #drop the column names which are not required for standardisation and also glm analysis 
    #e.g ptid 
    for outcome in outcomeVars:
        """Logistic regression on outcome"""
        if standardize:  # standardize the values
            for prec in predictors:
                if len(df[prec].unique()) > 2:
                    df[[prec]] = df[[prec]].apply(standardizeFunc)
        
                if not logistic: # if continuous outcomes, standardize to normal distribution Z
                    df[[outcome]] = df[[outcome]].apply(standardizeFunc)
                    #outcomeseries = outcomeseries.apply(standardizeFunc)
        # remove outcome variable from the df and create a seperate outcome df 
        outcomeDf = df[[outcome]]
        df = df.drop(outcome,axis=1)
        if outcome in predictors:
            predictors.remove(outcome)
        #resDf = performGLM(df, outcome, predictors, adj=adjustmentVars, logistic=logistic)
        resDf = doGLM(df, outcomeDf, outcome,predictors, covariates=covariateVars, logistic=logistic)

    #resDf = pd.concat(resL, axis=0, ignore_index=True)
    return resDf

