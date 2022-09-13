from email.policy import default
import sys

from firthlogist import FirthLogisticRegression

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

remove_values_from_list(sys.path, 'C:/Users/liel-/Dropbox/PyCharm/PycharmProjectsNew/TCR_dataset_v2/LielTools_v2/')
remove_values_from_list(sys.path, 'C:\\Users\\liel-\\Dropbox\\PyCharm\\PycharmProjectsNew\\TCR_dataset_v2\\LielTools_v2')

sys.path.append(r'C:\Users\liel-\Dropbox\PyCharm\PycharmProjectsNew\TCR_dataset_v3')
sys.path.append(r'C:\Users\lielc\Dropbox\PyCharm\PycharmProjectsNew\TCR_dataset_v3')
sys.path.append(r'D:\PhD-Tomer\Tools_and_Libraries\LielTools')

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from LielTools_v3 import StatsTools
#from LielTools_v3 import DataTools
#from LielTools_v3 import PlotTools
#from LielTools_v3 import FileTools
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold as sk_KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
if 'LielTools' in sys.modules:
    from LielTools import StatsTools
    from LielTools import DataTools
    from LielTools import PlotTools
    from LielTools import FileTools
else:
    import StatsTools
    import DataTools
    import PlotTools
    import FileTools
import warnings
warnings.filterwarnings('ignore', '.*did not.*', )
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
    FileTools.create_folder(fig_path_liel + '/GLM/{}/'.format(model_name))

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

def get_important_feature(xtrain,ytrain,num_features):
    fs = SelectFromModel(RandomForestClassifier(n_estimators=1000),threshold=-np.inf,max_features=num_features)
    fs = fs.fit(xtrain,ytrain)
    feature_idx = fs.get_support()
    feature_name = xtrain.columns[feature_idx].tolist()
    return feature_name

def glm_LOO(fig_path_liel, model_df, y_col_name, x_cols_list, model_name, heatmap_figsize=(12, 8), bar_figsize=(10,8),
            alpha=0.001, L1_wt=0.01, heatmap_annotate_text=True, logistic=True,featureSelection=False,num_feature = None):
    res = {}

    FileTools.create_folder(fig_path_liel + '/GLM/{}/'.format(model_name))
    cv_folds = 'LOO'

    random.seed(42)
    np.random.seed(42)

    obese_GLM_res, obese_GLM_test_res = [], []
    for i_index_test in range(len(model_df.index)):
        ind_train = list(range(len(model_df.index)))
        ind_train.remove(i_index_test)
        ind_train = np.array(ind_train)
        ind_test = np.array([i_index_test])

        if featureSelection:
            xtrain = model_df.loc[ind_train,x_cols_list]
            ytrain = model_df.loc[ind_train,y_col_name]
            x_cols_list = get_important_feature(xtrain,ytrain,num_feature)
            #use feature selection model to choose features 
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

    FileTools.copy_all_folder_files_to_folder(fig_path_liel + r'\GLM\{}\\'.format(model_name),
                                              fig_path_liel + r'\GLM\All_models')

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

def calc_cv_auc(folds_test_res,auc_plot=False,save_path_plot=None):
    #print('Number of CV folds: ', str(len(folds_test_res)))
    y_pred_list = []
    y_true_list = []
    y_ind_list = []
    for i in range(len(folds_test_res)):
        y_pred_list += list(folds_test_res[i]['y_pred'])
        y_true_list += list(folds_test_res[i]['y_true'])
        y_ind_list += list(folds_test_res[i]['y_true'].index)

    roc_auc_info = StatsTools.roc_auc(y_true=y_true_list, y_pred =y_pred_list,plotroc = auc_plot,save_path=save_path_plot)
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

    spearman = StatsTools.get_df_cols_correl(predictions['y_true'], predictions['y_pred'],
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
    PlotTools.plot_scatter(predictions['y_true'], predictions['y_pred'], plt_corr_txt=False,
                          figsize=(10, 7), axes_title_font_size=24, x_jitter=x_jitter,
                          plot_title=title_text)
    if logistic:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = [item if (item=='0.0' or item=='1.0' or item=='0' or item=='1') else '' for item in labels]
        ax.set_xticklabels(labels)

    ax.tick_params(axis='both', which='major', labelsize=5)
    fig.text(0.2, 0.9, spearman_text, fontdict={'size': 10})
    fig.savefig(fig_path_liel + '/GLM/{}/y_true vs y_predicted _____{}.jpg'.format(model_name,
                                                                                   model_name))

def plot_params(fig_path_liel, folds_res, title_text, model_name,
                heatmap_figsize, bar_figsize=(10,8), params_GLM=None,
                heatmap_annotate_text=True,show_only_nonzero=True,color_specific_yticklabels=None):
    if params_GLM is None:
        params_GLM = pd.DataFrame(folds_res[0]['model'].params, columns=['Model 1'])
        for i in list(range(1, len(folds_res))):
            params_GLM = params_GLM.join(pd.DataFrame(folds_res[i]['model'].params,
                                                      columns=['Model {}'.format(i + 1)]))

    params_GLM = params_GLM.drop(index="const")
    params_GLM_copy = params_GLM.copy()
    if show_only_nonzero:
        params_GLM_copy["zerocount_percentage"] = params_GLM_copy.apply(lambda row: row.value_counts().get(key=0,default=0)/row.shape[0],axis=1)
        params_GLM_copy = params_GLM_copy[params_GLM_copy["zerocount_percentage"]<=0.5].drop("zerocount_percentage",axis=1)

    FileTools.write2Excel(fig_path_liel + '/GLM/{}/params_beta____{}.xlsx'.format(model_name,
                                                                                   model_name),params_GLM)
    #params_GLM_cpy = params_GLM[params_GLM.iloc[:,0:params_GLM.shape[1]]>0.001]
    PlotTools.plot_heatmap(params_GLM_copy.astype(float), cmap='RdBu_r', figsize=heatmap_figsize,
                               title=title_text, title_fontsize=21,
                               font_scale=1, snsStyle='ticks',
                               xlabel='CV Model', ylabel='Variable', colormap_label='',
                               vmin=-1, vmax=1,
                               annotate_text=heatmap_annotate_text, annotate_fontsize=6,xy_labels_fontsize=20,yRotation=0,xRotation=90,color_specific_yticklabels =color_specific_yticklabels
                               , color_specific_color = "green")
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
    plt.yticks(rotation=90)
    plt.ylim((-1, 1))
    plt.rc('xtick',labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)
    plt.title(title_text,fontsize = 21)
    plt.ylabel('Mean Variable Weight in CV Models',fontsize=20)
    plt.xlabel('Variable',fontsize=20)
    ax = plt.gca()
    if color_specific_yticklabels is not None:
        for xticklabel in ax.get_xticklabels():
            xticklabel_text = str(xticklabel.get_text())
            for label_from_list in color_specific_yticklabels:
                if xticklabel_text == str(label_from_list):
                    xticklabel.set_color("green")
                    xticklabel.set_weight("bold")
    plt.tight_layout()
    plt.savefig(fig_path_liel + '/GLM/{}/params_bar____{}.jpg'.format(model_name,
                                                                      model_name))
    plt.close('all')
    return params_GLM

def tune_GLM_elastic_net(fig_path_liel, model_df, y_col_name, x_cols_list, cv_folds, model_name, test_size=0.25,
                         logistic=True,
                         alphas=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 5],
                         l1s=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 5],plotalphal1 = False):

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
                                                              alpha=alp, L1_wt=l1,suppress_stand_message= True)

                    # print('train.shape', model_df.loc[ind_train_train].shape)
                    # print('params.shape', obese_GLM_res_fold['model'].params.shape)
                    # print('valid.shape', model_df.loc[ind_train_valid].shape)
                    obese_GLM_valid_res_fold = StatsTools.ML_model_test(obese_GLM_res_fold['model'],
                                                                        model_df.loc[ind_train_valid],
                                                                        y_col_name, x_cols_list,
                                                                        drop_na=False, add_constant=True,
                                                                        logistic=logistic,suppress_stand_message= True)
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

    if plotalphal1:
        plt.figure(figsize=(13, 10))
        sns.heatmap(elastic.astype(float), annot=True)
        plt.title(measure_name)
        plt.xlabel('L1_weight')
        plt.ylabel('Alpha')
        plt.savefig(fig_path_liel + '/Tuning__{}.jpg'.format(model_name))

    return elastic

def doGLM(df, outcomedf, outcomeVar,predictors, interaction,covariates=[], logistic=True,frithlogistic = True):
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

    intreactionVar = []
    if interaction:
        for c in covariates:
            intreactionVar.append("intract_"+c)
            cols.append(c+"_pvalue")
            cols.append("intract_"+c+"_pvalue")
    
    
    cols = cols+covariates+intreactionVar

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
        #adding interaction to the df
        if interaction:
            for intvar in intreactionVar:
                tmp[intvar] = tmp[predc]*df[c]
                exogVars.append(intvar)
        if frithlogistic:
            fl = FirthLogisticRegression(fit_intercept = True,max_iter = 100,tol=0.001)
        else:
            model = sm.GLM(endog=tmp[outcomeVar].astype(float), exog=sm.add_constant(tmp[exogVars].astype(float)),
                       family=family)
        try:
            if not  frithlogistic:
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
                    outDf.loc[predc,c]=  res.params[c]
                    outDf.loc[predc,c+"_pvalue"] = res.pvalues[c]
                for i in intreactionVar:
                    outDf.loc[predc,i]=  res.params[i]
                    outDf.loc[predc,i+"_pvalue"] = res.pvalues[i]
            else:
                res = fl.fit(tmp[exogVars],tmp[outcomeVar])
                outDf.OR[predc] = coefFunc(res.coef_[0])
                outDf.pvalue[predc] = res.pvals_[0]
                coefficent=  coefFunc(res.ci_[0])
                outDf.LL[predc] = coefficent[0]
                outDf.UL[predc] = coefficent[1]
                outDf.ParamConst[predc] = res.intercept_
                outDf.ParamBeta[predc] = res.coef_[0]
                outDf.Diff[predc] = tmp[predc].loc[tmp[outcomeVar] == 1].mean() - tmp[predc].loc[tmp[outcomeVar] == 0].mean()
           # params.append(res.params.to_dict())
            #pvalues.append(res.pvalues.to_dict())
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
                outDf.loc[predc,c]=  np.nan
                outDf.loc[predc,c+"_pvalue"] = np.nan
            for i in intreactionVar:
                outDf.loc[predc,i]=  np.nan
                outDf.loc[predc,i+"_pvalue"] = np.nan
            #params.append({k: np.nan for k in [predc] + adj})
            #pvalues.append({k: np.nan for k in [predc] + covariates})
            resObj.append(None)
            print('PerfectSeparationError: %s with %s' % (predc, outcomeVar))
        outDf.N[predc] = tmp.shape[0]
    
    #outDf['params'] = params
    #outDf['pvalues'] = pvalues
    outDf['res'] = resObj
    
    return outDf

def GLMAnalysis(dataDf,predictors,outcomeVars=[],covariateVars=[],standardize=True,logistic=True,univariate=True,interaction=False):
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
        resDf = doGLM(df, outcomeDf, outcome,predictors, interaction,covariates=covariateVars, logistic=logistic)

    #resDf = pd.concat(resL, axis=0, ignore_index=True)
    return resDf



def glm_LOO_LR(model_df, y_col_name, x_cols_list,ind_train,ind_test,alpha=0.001, L1_wt=0.01):
  

    random.seed(42)
    np.random.seed(42)

    GLM_res_fold = StatsTools.GLM_model(model_df.iloc[ind_train],
                                                  y_col_name, x_cols_list,
                                                  logistic=True,
                                                  penalize=True,
                                                  alpha=alpha, L1_wt=L1_wt,
                                                  suppress_stand_message=True)
    #obese_GLM_res.append(obese_GLM_res_fold)

    GLM_test_res_fold = StatsTools.ML_model_test(GLM_res_fold['model'],
                                                           model_df.iloc[ind_test],
                                                           y_col_name, x_cols_list,
                                                           drop_na=True, add_constant=True,
                                                           logistic=True, calc_roc_auc=False)
   # obese_GLM_test_res.append(obese_GLM_test_res_fold)



    return (GLM_res_fold,GLM_test_res_fold)
    


def glm_loo_plot(fig_path,model_test_res_list,model_name,y_col_name,no_data_pts,model_train_res_list,
                    heatmap_figsize=(12, 8), bar_figsize=(10,8),heatmap_annotate_text=True,plot_auc = False,color_specific_yticklabels = None):

    res = {}

    FileTools.create_folder(fig_path + '/GLM/{}/'.format(model_name))
    cv_folds = 'LOO'

    auc_roc_fig = fig_path + '/GLM/{}/ROC_AUC____{}.jpg'.format(model_name,model_name)
    res['roc_auc_info'] = calc_cv_auc(model_test_res_list,auc_plot=plot_auc,save_path_plot=auc_roc_fig)
    final_measure = res['roc_auc_info']['auc']
    measure_name = 'AUC'
 
    print('{} CV-{} test {}: {}'.format(model_name, cv_folds, measure_name, final_measure))

    FileTools.write2Excel(fig_path + '/GLM/{}/{}____{}.xlsx'.format(model_name, measure_name,
                                                                         model_name),
                          pd.DataFrame([final_measure], index=[measure_name]))

    title_text = 'y={}, {}={}, n={}, CV-{} '.format(y_col_name, measure_name, np.round(final_measure, 3),
                                                             no_data_pts,
                                                             cv_folds)

    res['GLM_params'] = plot_params(fig_path, model_train_res_list, title_text, model_name,
                      heatmap_figsize, bar_figsize=bar_figsize,
                      heatmap_annotate_text=heatmap_annotate_text,color_specific_yticklabels = color_specific_yticklabels)

    plot_predictions(fig_path, model_test_res_list, model_name, title_text)

    FileTools.copy_all_folder_files_to_folder(fig_path + r'\GLM\{}\\'.format(model_name),
                                              fig_path + r'\GLM\All_models')

    return res
    

