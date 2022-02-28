import random
import math
import time
from sklearn.model_selection import StratifiedKFold

def batch_indexes_generator(indexes, batch_size=32):
    """Yield successive (approximately) batch_size-sized chunks from indexes list."""
    for i in range(0, len(indexes), batch_size):
        yield indexes[i: i+batch_size]


def batch_indexes_generator_stratified(indexes, y_labels, batch_size=32):
    """Yield (approximately) batch_size-sized chunks from indexes list,
        stratified according to y_labels. """
    if batch_size == len(y_labels):
        yield indexes
    else:
        kfold = StratifiedKFold(n_splits=math.ceil(len(indexes)/batch_size), shuffle=False)
        for train_ind, test_ind in kfold.split(indexes, y_labels.values):
            yield test_ind

def data_generator(x_data, y_labels, batch_size=32, shuffle=True,
                   return_only_x=False, stratify=False, return_double_x=False):
    '''
    A generator function that yields batches of [data, labels].
    Goes over the entire x_data without replacement.
    When called more than number_of_steps=ceil(n/batch_size) times,
    data is shuffled (if shuffled=True and stratify=False)
    and batching starts all over again.
    If stratify set to True, will stratify batches according to y_labels.
    However, can't use both stratify and return_only_x together,
    or shuffle and return_only_x together:
    Even without shuffling, stratify changes the samples order, and then
    its resulting x can only be matched with the resulting y,
    and not with the original y.

    :param x_data: features data, length n.
    :param y_labels: labels matching the data, length n.
    :param shuffle: boolean
    :param stratify: boolean. if True, will not shuffle!
    :param return_double_x: boolean. if True, will return [x_data, x_data] (both identical)
                            instead of x_data, for use when network receives two inputs
                            (two networks which later merge)
    :return: [data, labels] - each element has length batch_size, or
              only data if return_only_x==True
    '''
    if (stratify and return_only_x) or (shuffle and return_only_x):
        raise Exception("Can't use both stratify/shuffle and return_only_x together! See function documentation")

    if batch_size > len(x_data):
        batch_size = len(x_data)

    num_steps = math.ceil(len(x_data) / batch_size)
    indexes = list(range(0, len(x_data)))

    step = 0
    while True:
        # time.sleep(5)
        step += 1
        # print('step ', step)
        if step == 1:
            if shuffle and not stratify: random.shuffle(indexes)
            # print('indexes list now shuffled ', indexes)

            if stratify:
                batch_index_generator = batch_indexes_generator_stratified(indexes, y_labels, batch_size=batch_size)
            else:
                batch_index_generator = batch_indexes_generator(indexes, batch_size=batch_size)

        batch_indexes = next(batch_index_generator)
        # time.sleep(5)
        # print('batch_indexes ', batch_indexes)
        if return_only_x:
            if not return_double_x:
                yield(x_data[batch_indexes])
            else:
                yield([x_data[batch_indexes], x_data[batch_indexes]])

        else:
            if not return_double_x:
                yield([x_data[batch_indexes],
                       y_labels[batch_indexes].values])
            else:
                yield([[x_data[batch_indexes], x_data[batch_indexes]],
                       y_labels[batch_indexes].values])

        if step == num_steps:
            step = 0


# test
# import numpy as np
# import pandas as pd
#
# gen = data_generator(x_data, y_labels, batch_size=batch_size, shuffle=False, return_only_x=False, stratify=True)
#
# x = np.zeros((0,70,6))
# y = pd.Series()
# x_list = []
# y_list = []
# for i in range(math.ceil(len(x_data) / batch_size)):  # TODO@ wrap with a nicer generator
#     batch = next(gen)
#     x = np.append(x, batch[0], 0)
#     y = y.append(batch[1])
#     x_list.append(batch[0])
#     y_list.append(batch[1])
#
# x.sum()==x_data.sum()
# len(y.index.intersection(y_labels.index)) == len(y_labels.index)
#
# summ = 0
# for y_batch in y_list:
#     print(len(y_batch))
#     print(sum(y_batch))
#     summ += sum(y_batch)





# sum = 0
# for fold in y_labels_folds:
#     sum = sum + fold.sum()
#



# import numpy as np
# y_labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])
# x_data = np.array(list(range(101,127)))
# batch_size = 19_07_08 only active kmers first try
# shuffle=True
# gen = data_generator(x_data, y_labels, batch_size=batch_size, shuffle=shuffle)
#
# list1 = []
# list2 = []
#
#
# for i in range(math.ceil(len(x_data) / batch_size)):
#     list1.append(next(gen))
#
# for i in range(math.ceil(len(x_data) / batch_size)):
#     list2.append(next(gen))
#
# x1 = list(list1[0][0]) + list(list1[1][0]) + list(list1[2][0]) + list(list1[3][0])
# y1 =  list(list1[0][1]) +  list(list1[1][1]) +  list(list1[2][1]) +  list(list1[3][1])
#
# sum(x1) == sum(list(range(101,127)))
# sum(y1) == 13
#
# x1.sort()
# x1 == list(range(101,127))
#
#
#
# x2 = list(list2[0][0]) + list(list2[1][0]) + list(list2[2][0]) + list(list2[3][0])
# y2 =  list(list2[0][1]) +  list(list2[1][1]) +  list(list2[2][1]) +  list(list2[3][1])
#
# sum(x2) == sum(list(range(101,127)))
# sum(y2) == 13
#
# x2.sort()
# x2 == list(range(101,127))


#
#
# class data_generator():
#
#     def __init__(self, x_data, y_labels, batch_size=32, shuffle=True):
#         self.batch_size = batch_size
#         self.num_steps = math.ceil(len(x_data) / batch_size)
#         self.indexes = list(range(0, len(x_data)))
#         self.shuffle = shuffle
#
#
#
#     def data_generator(self):
#         step = 0
#         while True:
#             step += 1
#             print('step ', step)
#             if step == 1:
#                 if shuffle: random.shuffle(self.indexes)
#                 print('indexes list ', self.indexes)
#                 batch_index_generator = batch_indexes_generator(self.indexes, self.batch_size)
#
#             batch_indexes = next(batch_index_generator)
#             print('batch_indexes ', batch_indexes)
#             yeald[x_data[batch_indexes], y_labels[batch_indexes]]
#
#             if step == max_step:
#                 step = 0
#
#
#     @staticmethod
#     def batch_indexes_generator(indexes, batch_size):
#         """Yield successive batch_size-sized chunks from indexes list."""
#         for i in range(0, len(indexes), batch_size):
#             yield indexes[i: i +batch_size]







