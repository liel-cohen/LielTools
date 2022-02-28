from keras.callbacks import Callback
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import sys

if 'LielTools' in sys.modules:
    from LielTools.KerasTools import model_predict_evaluate
else:
    from KerasTools import model_predict_evaluate



class Metrics(Callback):

    def __init__(self, val_x_data, val_y_data, args):
        super().__init__()
        self.val_x_data = val_x_data
        self.val_y_data = val_y_data
        self.my_args = args

    def on_train_begin(self, logs={}):
        print(self.validation_data)
        self.val_auc = []
        self.val_balanced_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        eval_res = model_predict_evaluate(self.model,
                               self.val_x_data, self.val_y_data,
                               self.my_args)

        self.val_auc.append(eval_res['auc_metric'])
        self.val_balanced_accuracy.append(eval_res['balanced_accuracy'])

        return

    def on_train_end(self, logs={}):
        self.history_table =  pd.DataFrame(
                    {'val_balanced_accuracy': self.val_balanced_accuracy,
                     'val_auc_metric': self.val_auc},
                     index=list(range(0, len(self.val_balanced_accuracy))))
