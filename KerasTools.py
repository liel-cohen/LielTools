import sys
import keras
import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import warnings
import math

if 'LielTools' in sys.modules:
    from LielTools import FileTools
    from LielTools.DataGenerator import data_generator
    from LielTools import seqTools as SeqC
    from LielTools import PlotTools
else:
    import FileTools
    from DataGenerator import data_generator
    from LielTools_4.LielTools.RepertoireTools import seqTools as SeqC
    import PlotTools



warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

''' obtain the output of an intermediate layer '''
def getLayerOutput(keras_model, layer_name, model_input_data):
    intermediate_layer_model = keras.Model(inputs=keras_model.input, outputs=keras_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(model_input_data)
    return(intermediate_output)

def createTensorBoardCallback(tensorboard_dir, batch_size=10,
                              deleteExistingInFolder=True, histogram_freq=1,
                              write_images=True):
    FileTools.create_folder(tensorboard_dir) # create folder in case it doesn't exist
    if (deleteExistingInFolder):  # delete other files in folder (former tensorboard logs)
        FileTools.delete_folder(tensorboard_dir, keepFolderItself=True)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir                = tensorboard_dir, # log file directory
        histogram_freq         = histogram_freq, # change to 1 for write_images to work
        batch_size             = batch_size,
        write_graph            = True,
        write_grads            = False,
        write_images           = write_images, # ** For true - must change histogram_freq to at least 1
        embeddings_freq        = 0,
        embeddings_layer_names = None,
        embeddings_metadata    = None
    )

    return(tensorboard_callback)

def createCheckpointCallback(path, measure='mse', minOrMax='min', verbose=1):
    FileTools.create_folder(path)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        path,
        verbose=verbose,
        period=1,
        save_best_only=True,
        monitor=measure,
        mode=minOrMax
    )

    return(checkpoint_callback)

def plotTrainValResults(modelHistory=None, historyTable=None, metric='auc_metric', pltTitle='', savePath=None):
    valMetric = 'val_' + metric
    if metric == 'auc_metric':
        showMetric = 'AUC' # just nicer writing.
    elif metric == 'balanced_accuracy':
        showMetric = 'BCC'
    elif metric == 'f1_metric':
        showMetric = 'F1'

    else:
        showMetric = metric

    if historyTable is None: historyTable = pd.DataFrame(modelHistory.history)

    historyTable['Epoch'] = list(range(1, 1 + historyTable.shape[0]))

    historyTable = historyTable[['Epoch', metric, valMetric]]

    fig, ax = plt.subplots(figsize=(6,4.5), sharey=True, sharex=True)

    plt.plot('Epoch', metric, data=historyTable, marker='o', markerfacecolor='red', markersize=5, color='red', linewidth=3, label='Training')
    plt.plot('Epoch', valMetric, data=historyTable, marker='o', markerfacecolor='blue', markersize=5, color='blue', linewidth=3, label='Validation')
    if metric=='f1_metric':
        plt.ylim([0,1])
    else:
        plt.ylim([0.5,1])

    # history4plot = history.melt('Epoch', var_name='cols', value_name='vals')
    # ax = sns.lineplot(x='Epoch', y='vals', hue='cols', data=history4plot)
    text1 = 'Final ' + showMetric + ':       %2.4f' % historyTable[metric][historyTable.shape[0] - 1]
    text2 = 'Final val_' + showMetric + ': %2.4f' % historyTable[valMetric][historyTable.shape[0] - 1]
    plt.text(0.62, 0.2, text1, fontdict={'size': 10}, transform=fig.transFigure)
    plt.text(0.62, 0.16, text2, fontdict={'size': 10}, transform=fig.transFigure)

    plt.xlabel('Epoch')
    plt.ylabel(showMetric)
    plt.title(pltTitle)
    plt.legend(loc='upper left')

    PlotTools.savePlt(savePath=savePath)

def getClassWeights(yLabels, returnBalanced=False):
    labels = yLabels.unique()
    class_weight = {}
    for lab in labels:
        if returnBalanced:
            n_lab = 1
        else:
            n_lab = (yLabels == lab).sum()
        class_weight[lab] = n_lab

    if not returnBalanced:
        n_max = max(class_weight.values())
        class_weight.update((x, n_max / y) for x, y in class_weight.items())

    return(class_weight)

def create_kmer_filters(k, groupLetters, groupEncoder, bias=None):
    # Create all k-mers filters
    allKmers = SeqC.seqTools.createAllKmersList(k, groupLetters)
    kmer_filters = SeqC.seqTools.strSeries2_1H(pd.Series(allKmers),
                                               charEncoder=groupEncoder,
                                               padding=False) # we don't want to pad the filters. Each should have length k

    # change axes to fit keras weights shape: (k, len(groupLetters), len(groupLetters) ** k)
    kmer_filters = np.swapaxes(kmer_filters, 0, 2)
    kmer_filters = np.swapaxes(kmer_filters, 0, 1)

    # bias
    if bias is None:
        bias = 0 - k + 0.5

    # Create weights
    kmer_filters_bias = np.full((len(groupLetters) ** k,), bias)
    layer_kmer_weights = [kmer_filters, kmer_filters_bias]
    return(layer_kmer_weights)


def create_kmer_filters_from_existing(TCRdata_obj, k,
                                      group_encoder, aa_to_groups_map,
                                      bias=None, gap=0,
                                      from_indices=None):
    # bias
    if bias is None:
        bias = 0 - k + 0.5

    # Find all k-mers in the sequences
    TCRdata_obj = TCRdata_obj.copy()
    TCRdata_obj.initAndAddOtherLetters(aa_to_groups_map, 'other')

    kmer_hist = TCRdata_obj.get_existing_kmers_hist(k,
                                                    other_letters=True,
                                                    from_indices=from_indices)
    all_kmers = list(kmer_hist.keys())

    if gap > 0: # see onenote -> "How to define gapped k-mer filters - X vector & bias"
        # time_gaps = time.time()
        print('@@@ number of kmers before gaps:', len(all_kmers))
        all_kmers, num_gaps = create_gapped_kmers_from_kmers(all_kmers, gap)
        print('@@@ number of kmers after gaps:', len(all_kmers))
        # print('Time for constructing gapped k-mers:', (time.time() - time_gaps)/60)
        kmer_filters_bias = np.full((len(num_gaps),), bias) + num_gaps
        group_encoder_2 = group_encoder.copy()
        group_encoder_2.addLabel('0')
    else:
        kmer_filters_bias = np.full((len(all_kmers),), bias)
        group_encoder_2 = group_encoder

    kmer_filters = SeqC.seqTools.strSeries2_1H(pd.Series(all_kmers),
                                               charEncoder=group_encoder_2,
                                               padding=False, # we don't want to pad the filters. Each should have length k
                                               gaps=gap > 0)

    # change axes to fit keras weights shape: (k, len(groupLetters), len(all_kmers))
    kmer_filters = np.swapaxes(kmer_filters, 0, 2)
    kmer_filters = np.swapaxes(kmer_filters, 0, 1)

    # Create weights
    return [[kmer_filters, kmer_filters_bias], all_kmers]

def create_gapped_kmers_from_kmers(kmers_list, gap):
    '''
    Get gapped kmers for all kmers in kmers_list,
    and also a list with the number of gaps for each gapped kmer.
    :param kmers_list: a list of kmers (list of strings)
    :param gap: maximum gap (int)
    :return: list of gapped kmers (length l), list of gaps (length l)
    '''
    gapped_kmers_all = []
    for kmer in kmers_list:
        gapped_for_1_kmer = SeqC.seqTools.create_gapped_list_from_kmer(kmer, gap)
        gapped_kmers_all.extend(gapped_for_1_kmer)

    gapped_kmers_all = list(pd.Series(gapped_kmers_all).unique())
    num_gaps = [gapped_kmer.count('0') for gapped_kmer in gapped_kmers_all]

    return gapped_kmers_all, num_gaps

def trans_1Hseqs_2groupSeqs(data1H, transMap, d1AlphabetMap, d2AlphabetMap):
    # create transition matrix for aa -> aaGroups
    mat_aa_2groups = SeqC.seqTools.matrix_1H_to_1H(transMap=transMap, d1AlphabetMap=d1AlphabetMap, d2AlphabetMap=d2AlphabetMap)

    # params
    sample_size = data1H.shape[0]
    sequence_len = data1H.shape[1]
    num_groups = len(d2AlphabetMap.keys())

    # apply matrix multiplication
    data1H_groups = np.zeros((sample_size, sequence_len, num_groups))
    for i in range(sample_size):
        data1H_groups[i] = np.matmul(data1H[i], mat_aa_2groups)

    return(data1H_groups)

def model_evaluate(model, X_test, Y_test, verbose=1):
    eval_res = model.evaluate(X_test, Y_test, verbose=verbose, batch_size=len(Y_test))
    eval_res = {model.metrics_names[i]: eval_res[i] for i in range(len(eval_res))} # instead of unnamed list, return dict with metric:reults
    return(eval_res)

def get_model_predictions(model, X_test, Y_test, batch_size, return_double_x=False):
    data_gen_prediction = data_generator(X_test, Y_test,
                                         batch_size=batch_size,
                                         shuffle=False, stratify=False,
                                         return_only_x=True, return_double_x=return_double_x)
    predicted_probs = model.predict_generator(data_gen_prediction,
                                              steps=math.ceil(len(X_test) / batch_size),
                                              workers=0,
                                              )  # workers=0 - executes on the main thread. workers>1 - pred will be shuffled and thus worthless
    assert len(predicted_probs) == len(Y_test)

    return predicted_probs


def model_predict_evaluate(model, X_test, Y_test, args):
    # Predict
    predicted_probs = get_model_predictions(model, X_test, Y_test,
                                      args.batch_size_validation,
                                      return_double_x=args.multi_k)

    # Calc metrics
    roc_auc = roc_auc_score2(Y_test, predicted_probs, error_if_0=True)
    if balanced_accuracy in args.metrics:
        bcc = balanced_accuracy_score2(Y_test, predicted_probs)
    else:
        bcc = None
    if f1_metric in args.metrics:
        f1 = f1_metric(Y_test, predicted_probs)
    else:
        f1 = None

    eval_res = {'auc_metric': roc_auc, 'balanced_accuracy': bcc,
                'f1_metric': f1, 'predictions': predicted_probs}
    return(eval_res)

def get_time_from(startTime, second_or_minute='minute'):
    if second_or_minute=='minute': units = 60.0
    elif second_or_minute=='second':  units = 1.0
    else: raise Exception('unknown second_or_minute value')

    return (time.time() - startTime)/units

def get_str_time_from(startTime, second_or_minute='minute'):
    time_from = get_time_from(startTime, second_or_minute=second_or_minute)

    return ("%2.4f %ss" % (time_from, second_or_minute))

def get_timedate_str_of_time(time_float, format='%m/%d/%Y %H:%M:%S'):
    str_time = time.strftime(format, time.localtime(time_float))
    return str_time

def getLayerResultImage(kerasModel, layerName,
                        modelInputData,
                        col_size=15, row_size=17,
                        savePath=None):
    intermediate_layer_model = keras.Model(inputs=kerasModel.input, outputs=kerasModel.get_layer(layerName).output)
    activation = intermediate_layer_model.predict(modelInputData)

    def display_activation(activation, col_size, row_size):
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
        for row in range(0, row_size):
            for col in range(0, col_size):
                ax[row][col].imshow(activation[0, :, :], cmap='gray')

    display_activation(activation, col_size, row_size)
    plt.tight_layout()

    PlotTools.savePlt(savePath=savePath, dpi=150)

def getFilterImage(kerasModel, layerName, letters_encoder, filter_weights=None,
                   col_size=15, row_size=15, save_image_path=None, save_filters_path=None, cmap='RdYlGn'):
    filters_and_bias = kerasModel.get_layer(layerName).get_weights()
    filters = filters_and_bias[0]
    shifted_cmap = PlotTools.shiftedColorMap(cmap, filters.min().min(),
                                             filters.max().max(), 0)

    if filter_weights is not None:
        # get maximum weight for filter over all locations
        max_weight_over_locations = pd.DataFrame({'max_weights': filter_weights.max()})
        max_weight_over_locations['index'] = [i for i in range(filter_weights.shape[1])]

        # sort by weight (descending) and get order (by indices)
        max_weight_over_locations_ord = max_weight_over_locations.sort_values(by='max_weights',
                                                                          ascending=False)
        order = list(max_weight_over_locations_ord['index'])

        # get filter name (series row name) and its max weight (series value)
        name_and_weight = max_weight_over_locations['max_weights']
    else:
        order, name_and_weight = None, None

    def display_filters(weights, col_size, row_size, order=None, name_and_weight=None,
                        cmap='RdYlGn', x_ticks=None, y_ticks=None):
        if order is None:
            order = list(range(weights.shape[2]))  # regular order

        vmin = np.min(weights)
        vmax = np.max(weights)

        fig, ax = plt.subplots(row_size, col_size,
                               figsize=(row_size * 1.8, col_size * 2.0))
        row, col = 0, 0
        for i_filter in order:
            im = ax[row][col].imshow(weights[:, :, i_filter], cmap=cmap,
                                     vmin=vmin, vmax=vmax)
            if x_ticks is not None:
                ax[row][col].set_xticks([i for i in range(len(x_ticks))])
                ax[row][col].set_xticklabels(x_ticks)
            if y_ticks is not None:
                ax[row][col].set_yticks([i for i in range(len(y_ticks))])
                ax[row][col].set_yticklabels(y_ticks)
            if name_and_weight is not None:
                ax[row][col].set_title('{}: max weight {:.4f}'.
                                       format(name_and_weight.index[i_filter],
                                              name_and_weight.iloc[i_filter]))

            col += 1
            if col == col_size:
                row += 1
                col = 0
                if row == row_size:
                    break

        # fig.subplots_adjust(hspace=0.9)
        # put colorbar at desire position
        # cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        return im

    colorbar = display_filters(filters, col_size, row_size, order,
                               name_and_weight=name_and_weight,
                               cmap=shifted_cmap if filters.min().min() < 0 else 'RdYlGn',
                               x_ticks=[letters_encoder.reverseMappingDict[i] for i in range(len(letters_encoder.labels))],
                               y_ticks=[i+1 for i in range(filters.shape[0])])
    plt.tight_layout()
    PlotTools.savePlt(savePath=save_image_path, dpi=300)

    # save colorbar separately
    fig = plt.figure()
    fig.colorbar(colorbar)
    PlotTools.savePlt(savePath=save_image_path[:-4] + '_colorbar.jpg', dpi=300)

    # save filters matrix to file
    if save_filters_path is not None:
        FileTools.write_var_to_dill(save_filters_path, filters_and_bias)


def getDenseWeightImage(kerasModel, layerName, savePath=None):
    weights = kerasModel.get_layer(layerName).get_weights()[0]

    # TODO@ tweak to get a nicer figure
    PlotTools.plot_clustermap(weights.transpose(), cmap="gray", figsize=(30,15), title='layerName',
                              row_clustering=False, col_clustering=False, font_scale=1)

    plt.tight_layout

    PlotTools.savePlt(savePath=savePath, dpi=300) #@ TODO add parameter showIfNull=True

def eval_plot_roc(model, X_test, Y_test,
                  save_path=None, save_auc_txt=None,
                  title='ROC curve', args=None):

    if args is None:
        batch_size = 32
    else:
        batch_size = args.batch_size_validation

    print('@@@ eval_plot_roc: start predict. batch size =', batch_size)
    y_pred_keras = model.predict(X_test, batch_size=batch_size).ravel()
    print('@@@ eval_plot_roc: end predict')
    fpr, tpr, thresholds_keras = roc_curve(Y_test, y_pred_keras)
    roc_auc = auc(fpr, tpr)

    fig = plot_roc(roc_auc, fpr, tpr,
                     save_path=save_path, save_auc_txt=save_auc_txt,
                     title=title)

    assert len(y_pred_keras) == len(Y_test)

    # pd.DataFrame(y_pred_keras).to_csv('eval_plot_roc___pred.csv', index=True)
    # pd.DataFrame(Y_test).to_csv('eval_plot_roc___true.csv', index=True)

    return {'fig': fig, 'auc': roc_auc, 'rates': {'fpr': fpr, 'tpr': tpr}, 'predictions': y_pred_keras}

def preds_plot_roc(y_test, y_pred, save_path=None, save_auc_txt_path=None,
                   title='ROC curve', showIfNone=True, plot_fig=True):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    if plot_fig:
        fig = plot_roc(roc_auc, fpr, tpr,
                   save_path=save_path, save_auc_txt=save_auc_txt_path,
                   title=title, showIfNone=showIfNone)
    else:
        fig = None

    assert len(y_pred) == len(y_test)
    return {'fig': fig, 'auc': roc_auc, 'rates': {'fpr': fpr, 'tpr': tpr}}


def plot_roc(auc, fpr, tpr,
             save_path=None, save_auc_txt=None,
             title='ROC curve', showIfNone=True):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.4f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='best')
    PlotTools.savePlt(savePath=save_path, showIfNone=showIfNone)
    if save_auc_txt is not None:
        FileTools.write_list_to_txt([auc], save_auc_txt)
    return fig

def train_test_generator(model, X_train, X_test, Y_train, Y_test,
                         args, epitope, callbacks, class_weights,
                         fold_number='',
                         write_eval_table=True,
                         write_time_path=None,
                         write_history_path=None,
                         write_best_val_epoch=None,
                         write_best_val_metric=None):
    start_time = time.time()

    # data generator for training data
    data_gen_train = data_generator(X_train, Y_train, batch_size=args.batch_size,
                                    shuffle=True, return_double_x=args.multi_k)

    # add validation metric callback
    if 'validation_metrics' not in dir():
        from LielTools_v3.KerasCallbacks import Metrics as validation_metrics
    val_metrics = validation_metrics(X_test, Y_test, args)
    callbacks.append(val_metrics)

    # train
    if args.print_stage: print('@@@ Starting model training')
    modelHistory = model.fit_generator(data_gen_train,
                                       epochs=args.epochs,
                                       steps_per_epoch=math.ceil(len(X_train) / args.batch_size),
                                       callbacks=callbacks, class_weight=class_weights,
                                       verbose=args.verbose,
                                       )

    # Evaluate last model
    if args.print_stage: print('@@@ Starting model evaluation')
    eval_scores = model_predict_evaluate(model, X_test, Y_test, args)

    # Print results # TODO@ turn into function with evaluate (modelHistory=None - not mandatory)
    history_table = pd.DataFrame(modelHistory.history)
    history_table = history_table.join(val_metrics.history_table)

    lastTrainMetric = history_table[args.plot_metric][history_table.shape[0] - 1]
    print("%s # Training: %.4f , Final evaluation: %.4f"
          % (args.plot_metric, lastTrainMetric, eval_scores[args.plot_metric]))

    if fold_number == '':
        fold_text = ''
    else:
        fold_text = 'fold_' + str(fold_number)

    # Write output to files
    if args.write_output:

        folder_write_jpg = args.folder_write + 'Figures/' + epitope + '/'
        FileTools.create_folder(folder_write_jpg)

        plotTrainValResults(historyTable=history_table, metric=args.plot_metric,
                            pltTitle='Single run results',
                            savePath=folder_write_jpg + epitope + '_' + fold_text + '__TrainingMetric.jpg')
        if write_eval_table:
            FileTools.write2Excel(args.folder_write + 'Metrics/' + epitope + '_' + fold_text + '__evaluation_results.xlsx',
                                  eval_scores)

        if write_history_path is not None:
            FileTools.write2Excel(write_history_path, history_table)

    if args.write_output:
        roc_path = args.folder_write + 'Figures/' + epitope + '/' + epitope + '_' + fold_text + '__ROC.jpg'
    else:
        roc_path = None

    if args.multi_k:
        X_test = [X_test, X_test]

    roc_test = eval_plot_roc(model, X_test, Y_test,
                             save_path=roc_path,
                             save_auc_txt=None,
                             title='ROC curve - ' + epitope + ' ' + fold_text,
                             args=args)

    if round(roc_test['auc'], 4) != round(eval_scores['auc_metric'], 4):
        print('!!!!!!! Separate auc evaluation and keras auc evaluation are not the same!')
        print('predict: ' + str(round(roc_test['auc'], 4)))
        print('eval: ' + str(round(eval_scores['auc_metric'], 4)))

    total_time = get_time_from(start_time)
    if args.write_output and write_time_path is not None:
        FileTools.write_list_to_txt([total_time], write_time_path)

    best_val_epoch = get_best_validation_result(history_table, args.best_val_epoch_metric, epoch_or_value='epoch')
    if args.write_output and write_best_val_epoch is not None:
        FileTools.write_list_to_txt([best_val_epoch], write_best_val_epoch)

    best_val_value = get_best_validation_result(history_table, args.best_val_epoch_metric, epoch_or_value='value')
    if args.write_output and write_best_val_metric is not None:
        FileTools.write_list_to_txt([best_val_value], write_best_val_metric)

    print('Epoch with best result for "fit" validation: ' + str(best_val_epoch))

    return [history_table, eval_scores, roc_test, total_time, best_val_epoch, best_val_value]

def get_best_validation_result(historyTable, metric, epoch_or_value='epoch'):
    if epoch_or_value=='epoch':
        return 1 + historyTable['val_' + metric].idxmax()
    elif epoch_or_value=='value':
        return historyTable['val_' + metric].max()

def get_tf_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def auc_metric(y_true, y_pred):
    res = tf.py_func(roc_auc_score2, (y_true, y_pred), tf.double)
    return(res)

def roc_auc_score2(y_true, y_pred, error_if_0=False):
    try:
        res = roc_auc_score(y_true, y_pred)
    except(ValueError):
        res = 0.0
        # print('roc_auc_score2 ValueError. set res = 0.0')
        if error_if_0:
            raise Exception('roc_auc_score2 returns 0!')
    return(res)

def balanced_accuracy(y_true, y_pred):
    res = tf.py_func(balanced_accuracy_score2, (y_true, y_pred), tf.double)
    return(res)

def balanced_accuracy_score2(y_true, y_pred, threshold=0.5):
    y_pred2 = y_pred > threshold
    y_pred2 = y_pred2.astype(float)
    # print(np.unique(y_pred2))
    # print(np.unique(y_true))
    res = balanced_accuracy_score(y_true, y_pred2)
    return(res)

def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_padding_between_cdrs_new(a_max, b_max, args, k, gap):
    if args.global_max_pooling:  # global max-pooling - over entire CDR3
        pad_between_chains = k

    elif args.chain_max_pooling:  # i.e., max pooling over each chain (alpha and beta)
        if args.multi_k == True:
            raise Exception("args.multi_k=True and args.chain_max_pooling=True together, are not supported in code")

        if args.cdrs == 'cdr3':
            pad_between_chains = k  # minimal pad - k
            if (pad_between_chains + a_max + b_max - k + 1) % 2 != 0:  # if L-k+1 is odd, add 1
                pad_between_chains += 1  # so that the split to 2 will be exact

            len_after_convolution = a_max + b_max + pad_between_chains - k + 1
            args.max_pool_size = int(len_after_convolution / 2)

        elif args.cdrs == 'all':
            raise Exception("args.cdrs='all' and args.chain_max_pooling=True together, are not supported in code")

    # else: # see onenote 2019_12 Pad size page
    #     pad_between_chains = k + args.max_pool_size + 1 + gap
    else: # see onenote 2020_12_31 Padding and empty pooling window
        pad_between_chains = max(args.max_pool_size - k + (2*gap) + (3*args.max_pool_stride),
                                 k)

    assert pad_between_chains >= k

    return pad_between_chains

def get_padding_between_chains_v1_very_old(a_max, b_max, args):
    if args.global_max_pooling:  # global max-pooling - over entire CDR3
        pad_between_chains = args.k
    elif args.chain_max_pooling:  # i.e., max pooling over each chain (alpha and beta)
        pad_between_chains = args.k  # minimal pad - k
        if (pad_between_chains + a_max + b_max - args.k + 1) % 2 != 0:  # if L-k+1 is odd, add 1
            pad_between_chains += 1  # so that the split to 2 will be exact

        len_after_convolution = a_max + b_max + pad_between_chains - args.k + 1
        args.max_pool_size = int(len_after_convolution / 2)

    else:
        remainder_a = args.max_pool_size - ((a_max - args.k + 1) % args.max_pool_size)
        remainder_b = args.max_pool_size - ((b_max - args.k + 1) % args.max_pool_size)
        pad_between_chains = remainder_a + remainder_b + args.max_pool_size - args.k + 1

        assert (a_max + b_max + pad_between_chains - args.k + 1) % args.max_pool_size == 0
        assert pad_between_chains >= args.max_pool_size

    assert pad_between_chains >= args.k

    return pad_between_chains

def get_padding_between_chains_v2_old(a_max, b_max, args):
    if args.global_max_pooling:  # global max-pooling - over entire CDR3
        pad_between_chains = args.k
    elif args.chain_max_pooling:  # i.e., max pooling over each chain (alpha and beta)
        pad_between_chains = args.k  # minimal pad - k
        if (pad_between_chains + a_max + b_max - args.k + 1) % 2 != 0:  # if L-k+1 is odd, add 1
            pad_between_chains += 1  # so that the split to 2 will be exact

        len_after_convolution = a_max + b_max + pad_between_chains - args.k + 1
        args.max_pool_size = int(len_after_convolution / 2)

    else:
        min_pad = args.max_pool_size + 3*args.k - 3 # see onenote 09/07 worklog
        extra_pad = args.max_pool_size - ((a_max + min_pad - args.k + 1) % args.max_pool_size)
        pad_between_chains = min_pad + extra_pad

        assert (a_max + pad_between_chains - args.k + 1) % args.max_pool_size == 0
        assert pad_between_chains >= args.max_pool_size

    assert pad_between_chains >= args.k

    return pad_between_chains

def get_padding_after_b_old(args):
    if args.global_max_pooling or args.chain_max_pooling:
        extra_b = 0
    else:
        extra_b = args.max_pool_size * args.max_pool_stride
        # extra_b = args.max_pool_size - ((b_max - args.k + 1) % args.max_pool_size)
        # assert (b_max + extra_b - args.k + 1) % args.max_pool_size == 0
    return extra_b

def analyze_flattened_feature_weights(model, flattened_layer_name, original_layer_name,
                                      kmer_list, write_to_path, cv_num, plot=False):

    # Get final k-mer feature weights matrix
    weights_global_dense = model.get_layer(flattened_layer_name).get_weights()
    flattened_weights = weights_global_dense[0]
    flattened_weights = flattened_weights.reshape((len(flattened_weights), ))
    original_shape = model.get_layer(original_layer_name).output_shape[1:3]
    assert len(flattened_weights) == original_shape[0] * original_shape[1]

    # unflatten weights array
    unflattened_array = np.empty((0, original_shape[1]), float)
    location = 0
    for row in range(original_shape[0]):
        end_location = location + original_shape[1]
        params = flattened_weights[location:end_location]
        unflattened_array = np.append(unflattened_array, params.reshape(1, len(params)), axis=0)
        location = end_location

    # organize to DF with k-mer names
    kmer_weights = pd.DataFrame(unflattened_array, columns=kmer_list)

    # Save heatmap - all weights
    if plot:
        min_val = unflattened_array.min()
        max_val = unflattened_array.max()
        max_abs_val = max(abs(min_val), abs(max_val))

        PlotTools.plot_heatmap(kmer_weights, cmap='RdBu_r', figsize=(35, 12),
                               title='', title_fontsize=13,
                               font_scale=2, snsStyle='ticks',
                               xlabel='k-mer', ylabel='Pooled Position', colormap_label='Weight',
                               vmin=-max_abs_val, vmax=max_abs_val)
        plt.tight_layout
        plt.savefig(write_to_path + 'all_weights__model{}.jpg'.format(cv_num), dpi=300)

    # Save to csv - all weights
    kmer_weights.to_csv(write_to_path + 'all_weights__model{}.csv'.format(cv_num))

    # # Save heatmap and csv - only the "bigger" weights
    # bigger_true = kmer_weights > max_val/2.5
    # bigger_weights = kmer_weights.loc[:, bigger_true.sum(axis=0)>0]
    #
    # if plot:
    #     PlotTools.plotHeatmap_real(bigger_weights, cmap='RdBu_r', figsize=(35, 12),
    #                     title='', title_fontsize=13,
    #                     font_scale=2, snsStyle='ticks',
    #                     xlabel='k-mer', ylabel='Pooled Position', colormap_label='Weight',
    #                     vmin=-max_abs_val, vmax=max_abs_val)
    #     plt.tight_layout
    #     plt.savefig(write_to_path + 'bigger_weights__model{}.jpg'.format(cv_num), dpi=300)
    #
    # bigger_weights.to_csv(write_to_path + 'bigger_weights__model{}.csv'.format(cv_num))

    return kmer_weights

def get_data_flattened_features(x_data, model, flattened_layer_name, original_layer_name,
                           feature_names, data_indices, write_to_path, cv_num):
    features_output = getLayerOutput(model, flattened_layer_name,
                                                x_data)
    # flattened_layer_name = 'flatten'
    # original_layer_name='max_pooling1d'
    # feature_names=feature_names[0]
    # x_data=xData_groups[train_ind]

    # Original shape - filters + locations
    original_shape = model.get_layer(original_layer_name).output_shape[1:3]
    assert features_output.shape[1] == original_shape[0] * original_shape[1]

    # Get feature names list
    num_locations = original_shape[0]
    names_list = []
    for location in range(num_locations):
        for feature_name in feature_names:
            names_list.append(feature_name + '__loc' + str(location+1))

    # save matrix and column names (filter+location)
    FileTools.save_df_as_sparse_csr(write_to_path + 'data_flattened_features_mod{}'.format(cv_num),
                    pd.DataFrame(features_output))
    pd.Series(names_list).to_csv(write_to_path + 'data_flattened_features__column_names__mod{}.csv'.format(cv_num))
    pd.Series(data_indices).to_csv(write_to_path + 'data_flattened_features__row_names__mod{}.csv'.format(cv_num))
