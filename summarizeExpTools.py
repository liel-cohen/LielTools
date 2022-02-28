import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

if 'LielTools_4' in sys.modules:
    from LielTools_4 import PlotTools
    from LielTools_4 import FileTools
    from LielTools_4 import DataTools
else:
    import PlotTools
    import FileTools
    import DataTools


def get_exp_eval_results(exp_name,
                  folder_exps,
                  exp_display_name='',
                  summary_metric='auc_metric',
                  summary_metric_show='AUC',
                  create_figs=True,
                  save_exp_fig=True
                  ):

    folder_exp = folder_exps + exp_name + '/'  # Folder for writing output

    final_results = pd.DataFrame(columns=[exp_name])
    fileList = [file for file in os.listdir(folder_exp + 'Metrics/') if
                'ation_results.xlsx' in file]

    for fileName in fileList:
        resDf = FileTools.read_excel(folder_exp + 'Metrics/' + fileName)
        if resDf.shape[1] == 1:
            final_results.loc[fileName.split(sep='_')[0], :] = resDf.loc[summary_metric][0]
        else:
            final_results.loc[fileName.split(sep='_')[0], :] = resDf.mean()[summary_metric]

    if final_results.shape[0] == 10: final_results = final_results.loc[
                                                     ['pp65', 'm139', 'NP',
                                                      'M45', 'M1', 'F2',
                                                      'PB1', 'BMLF', 'PA',
                                                      'M38'], :]  # Nature paper order, if analyzing all 10 epitopes

    FileTools.write_list_to_txt([final_results.mean()],
                                folder_exp + 'average_' + summary_metric_show + '.txt')

    if create_figs:
        if save_exp_fig:
            save_path = folder_exp + 'final_evals_' + summary_metric_show + '.jpg'
        else:
            save_path = None

        PlotTools.DFbarPlot(final_results, columns=None, figsize=(7, 5),
                            showLegend=False, xTitle='Epitope',
                            yTitle=summary_metric_show, ylim=(0, 1),
                            plotTitle=exp_display_name,
                            legendLabels=None, legendTitle=None,
                            xRotation=90, grid=False,
                            titleFontSize=12, axesTitleFontSize=18,
                            axesTicksFontSize=18, legendFontSize=16,
                            legendTitleFontSize=17,
                            savePath=save_path,
                            add_value_labels=True, float_num_digits=3, value_labels_fontsize=8
                            )

        if save_exp_fig:
            save_path = folder_exp + 'boxplot_final_evals_' + summary_metric_show + '.jpg'
        else:
            save_path = None

        # get CV boxplot
        if os.path.exists(folder_exp + 'Metrics/folds_eval_tables/'):
            cv_results = pd.DataFrame()
            fileList = [file for file in os.listdir(folder_exp + 'Metrics/folds_eval_tables/') if 'eval_results.xlsx' in file]
            for fileName in fileList:
                resDf = FileTools.read_excel(folder_exp + 'Metrics/folds_eval_tables/' + fileName)
                cv_results[fileName.split(sep='_')[0]] = resDf[summary_metric]

        PlotTools.plotBoxplotDF(cv_results, stripplot=True, savePath=save_path, figsize=(5.5, 4.5),
                                showf=False, xTitle='Epitope', plotTitle=exp_display_name,
                                yTitle=summary_metric_show, xRotation=45, titleFontSize=12,
                                titleColor='maroon', font_scale=1,
                                snsStyle='ticks', boxTransparency=0.6, ylim=(0.5,1))

    return final_results


def get_exp_best_validation_info(exp_name,
                  folder_exps,
                  save_results=True
                  ):

    folder = folder_exps + exp_name + '/Metrics/Train_best_validation_epoch/'
    assert os.path.exists(folder)

    fileList = [file for file in os.listdir(folder) if
                'average,std__val_epoch.txt' in file]
    best_epochs = pd.DataFrame(columns=['Average best epoch'],
                               index=[file.split(sep='_')[0] for file in fileList])

    for fileName in fileList:
        epitope = fileName.split(sep='_')[0]
        file_content = FileTools.read_txt_to_strings_list(folder + fileName)
        average_best_epoch = float(file_content[0])
        best_epochs.loc[epitope, 'Average best epoch'] = average_best_epoch

    average_epoch = best_epochs.mean()
    median_epoch = best_epochs.median()

    if save_results:
        folder = folder_exps + exp_name + '/Metrics/Train_best_validation_epoch/'
        FileTools.write_list_to_txt([average_epoch], folder + 'average_all_epitopes.txt')
        FileTools.write2Excel(folder + 'averages.xlsx', best_epochs)

    return [best_epochs, average_epoch, median_epoch]

def get_average_train_tables(exp_name,
                            folder_exps,
                            save_results=True,
                            ):

    folder = folder_exps + exp_name + '/Metrics/folds_training_tables/'
    assert os.path.exists(folder)

    file_list = [file for file in os.listdir(folder) if
                            '_training.xlsx' in file]

    epitopes = ['pp65', 'm139', 'NP', 'M45', 'M1',
                'F2', 'PB1', 'BMLF', 'PA', 'M38']

    epitopes_dfs = {}
    for epitope in epitopes:
        dfs_list = []
        for file_name in file_list:
            if file_name.split(sep='_')[0] == epitope:
                dfs_list.append(FileTools.read_excel(folder + file_name))
        avg_df = DataTools.averageDFs(dfs_list)
        if save_results:
            FileTools.write2Excel(folder + epitope + '_folds_avg_training.xlsx',
                                  avg_df, index=True)
        epitopes_dfs[epitope] = avg_df

    return epitopes_dfs

def get_best_epoch_results(exp_name,
                                folder_exps,
                                epitopes_dfs,
                                save_results=True,
                                epochs_list_for_best=[5, 10, 15, 20]):
    folder = folder_exps + exp_name + '/Metrics/folds_training_tables/'
    assert os.path.exists(folder)

    epitopes = ['pp65', 'm139', 'NP', 'M45', 'M1',
                'F2', 'PB1', 'BMLF', 'PA', 'M38']

    epochs_df = pd.DataFrame(0, index=epitopes, columns=epochs_list_for_best)
    for epoch in epochs_list_for_best:
        for epitope in epitopes:
            epochs_df.loc[epitope, epoch] = epitopes_dfs[epitope].loc[epoch-1, 'val_auc_metric']
    epochs_df.loc['Mean',:] = epochs_df.mean()

    if save_results:
        FileTools.write2Excel(folder + '_validation_auc_summary.xlsx',
                              epochs_df, index=True)

    best_epoch = epochs_df.loc['Mean',:].idxmax(axis=1)
    print(exp_name)
    print(best_epoch)
    # print(epochs_df.loc['Mean',:])

    best_results = epochs_df.loc[epitopes, best_epoch]
    best_results.rename(exp_name, inplace=True)

    return best_results