import pandas as pd
import functools
import math
import numpy as np
import os
import sys
from collections import Counter
import random
from scipy import sparse

if 'LielTools' in sys.modules:
    from LielTools import FileTools
else:
    import FileTools


class Object(object):
    pass

# former getDFwithoutColumns
def get_df_without_cols(df, cols_to_exclude):
    """
    returns a copy of pandas.DataFrame df that contains all
    columns except for cols_to_exclude.
    *Not all items in cols_to_exclude must be in df.cols.

    :param df: pandas.DataFrame
    :param cols_to_exclude: list of columns to exclude from
                            returned df
    :return: a copy of df that contains all columns except
             for cols_to_exclude
    """

    cols = [col for col in df.columns if col not in cols_to_exclude]
    return(df[cols].copy())


def get_df_without_indices(df, ind_to_exclude):
    """
    returns a copy of pandas.DataFrame df that contains all
    indices except for ind_to_exclude.
    *Not all items in ind_to_exclude must be in df.index.

    :param df: pandas.DataFrame
    :param ind_to_exclude: list of indices to exclude from
                           returned df
    :return: a copy of df that contains all indices except
             for ind_to_exclude
    """

    indices = [ind for ind in df.index if ind not in ind_to_exclude]
    return(df.loc[indices].copy())

def get_df_only_with_indices(df, ind_to_include):
    """
    returns a copy of pandas.DataFrame df that contains all
    indices in ind_to_include.
    *Not all items in ind_to_include must be in df.index.

    :param df: pandas.DataFrame
    :param ind_to_include: list of indices to include from in returnes df
    :return: a copy of df that contains only indices in ind_to_include
    """

    indices = [ind for ind in df.index if ind in ind_to_include]
    return(df.loc[indices].copy())

def get_df_col_names_without_cols(df, cols_to_exclude):
    """
    Returns a list of pandas.DataFrame columns except for cols_to_exclude.
    *Not all items in cols_to_exclude must be in df.cols.

    :param df: pandas.DataFrame
    :param cols_to_exclude: list of columns to exclude from
                            returned df
    :return: a list of column names except for cols_to_exclude
    """

    cols = [col for col in df.columns if col not in cols_to_exclude]
    return cols

# former seriesListRemoveEmpty
def series_list_remove_empty(series_list, names=None):
    """
    gets a list of several pd.Series and returns a list of the same
    series except for ones that were 'None'.
    Can also get a list of names corresponding to the series and
    remove names of series that were removed from series_list.
    :param series_list: a list with several pd.Series
    :param names:
    :return:
    """
    if (names is not None):
        if (len(series_list) != len(names)): raise Exception('series list and names list have different lengths!!')

    newList = [series_list[i] for i in range(len(series_list)) if series_list[i] is not None]

    if (names is None):
        return([newList, None])
    else:
        newNames = [names[i] for i in range(len(series_list)) if series_list[i] is not None]
        return([newList, newNames])

# former checkSeriesIndexEqual
def check_series_index_equal(series_list, raise_error=True):
    ''' Gets a list of pd.Series objects, and compares their
        indices for equality.
        (Only checks non 'None' list components)
        :param series_list: a list with pd.Series objects
        :param raise_error: If True, will raise error if indices
                            are not equal. If False, will only print to console.
         '''

    for i in range(len(series_list)):
        for j in range(len(series_list)):
            if (i != j):
                if ((series_list[i] is not None) & (series_list[j] is not None)):
                    # print('Series ' + str(i) + ' and series ' + str(j) + str(seriesList[i].index.equals(seriesList[j].index)))
                    if (series_list[i].index.equals(series_list[j].index) == False):
                        if raise_error:
                            raise Exception('Series ' + str(i) + ' and series ' + str(j) + ' indexes are not equal!')
                        else:
                            print('Series ' + str(i) + ' and series ' + str(j) + ' indexes are not equal!')


# former joinNonEmptySeriesFromList
def join_non_empty_series_f_list(seriesList, names=None):
    check_series_index_equal(seriesList)
    newLists = series_list_remove_empty(seriesList, names=names)
    newDF = pd.concat(newLists[0], axis=1)
    if names is not None:
        newDF.columns = newLists[1]
    return(newDF)

def joinDFsFromList(dfList):
    def join_dfs(ldf, rdf):
        return ldf.join(rdf, how='inner')

    final_df = functools.reduce(join_dfs, dfList)  # that's the magic - do the given function on the list sequentialy
    return (final_df)

def addDFs(dfsList):
    def addDFs(ldf, rdf):
        return ldf.add(rdf)

    final_df = functools.reduce(addDFs, dfsList)  # that's the magic - do the given function on the list sequentially
    return (final_df)

def averageDFs(dfsList):
    sum_df = addDFs(dfsList)
    return (sum_df / len(dfsList))

def div_dfs(df1_numerator, df2_denominator):
    ''' numerator = mone, denominator = mechane '''
    return df1_numerator.div(df2_denominator)

def multDFs(df1_numerator , df2_denominator):
    return df1_numerator.multiply(df2_denominator)

def censorDFvalsBelowThresh(df, valueThresh, newVal):
    dfCopy = df.copy()
    censorIndices = dfCopy < valueThresh
    dfCopy.values[censorIndices] = newVal
    return(dfCopy)

'''Former getUnsharedComponents'''
def get_unshared_components(list1, list2):
    ''' Gets 2 lists/series/sets and returns their unshared components:
        1) in1: in list1 and not in list2
        2) in2: in list2 and not in list1'''
    shared = set(list1).intersection(set(list2))
    in1 = [v for v in set(list1) if v not in shared]
    in2 = [v for v in set(list2) if v not in shared]
    return in1, in2

'''Former getSharedComponents'''
def get_shared_components(list1, list2, order_matters=False):
    ''' Gets 2 lists/series/sets and returns their shared components.
        If gets 2 lists, order_matters can be changed to True.
        If order_matters is false, returns a set.
        If order_matters is True, returns a list. (order will be according to list1 items order)
        '''
    if not order_matters:
        shared = set(list1).intersection(set(list2))
    else:
        if type(list1) is list and type(list2) is list:
            shared = []
            for item1 in list1:
                for item2 in list2:
                    if item1 == item2:
                        shared.append(item1)
        else:
            raise Exception('get_shared_components: order_matters can be True only if list1 and list2 are both of type list')

    return shared


def getDF_ColumnNamesWithPhrase(dfAll, phrase, at_end=False):
    ''' Get list with column names that contain phrase '''

    colList = []
    for colName in dfAll.columns:
        if not at_end:
            if phrase in colName:
                colList.append(colName)
        else:
            if colName[-1*len(phrase):] == phrase:
                colList.append(colName)
    return colList


def getDF_ColumnsWithPhrase(dfAll, phrase, startsWith=False):
    ''' Get new df with columns that contain phrase,
    or start with phrase (if startsWith=True)  '''

    if startsWith:
        dfRes = dfAll.loc[: , dfAll.columns.str.startswith(phrase)].copy()
    else:
        dfRes = dfAll.loc[: , dfAll.columns.str.contains(phrase)].copy()
    return dfRes


def getDF_IndexWithPhrase(dfAll, phrase, startsWith=False):
    ''' Get new df with indices that contains phrase,
    or start with phrase (if startsWith=True)  '''

    if startsWith:
        dfRes = dfAll[dfAll.index.str.startswith(phrase)].copy()
    else:
        dfRes = dfAll[dfAll.index.str.contains(phrase)].copy()
    return dfRes


def logDF(data, base=10):
    suffix = ' (log' + str(base) + ')'
    def logWrapper(number, base=10):
        return(math.log(number, base))

    function = functools.partial(logWrapper, base=base)
    transformed = pd.DataFrame(data).applymap(function)
    transformed = addToColNames(transformed, suffix=suffix)
    return(transformed)

def addToColNames(df, prefix='', suffix=''):
    df.columns = [prefix + str(col) + suffix for col in df.columns]
    return(df)

# former getColName
def get_col_name(col):
    """
    Get Series or single-column-dataframe column name.
    If col is None, returns None.
    :param col: string or None
    :return: string or None
    """
    if col is None:
        return None

    if type(col) is pd.core.frame.DataFrame:
        name = col.columns.values[0]
    elif type(col) is pd.core.series.Series:
        name = col.name
    else:
        raise ValueError('unknown column type')

    return name

def dfIntoPercentages(df, onColumns=True):
    dfCopy = df.copy()

    if not onColumns:
        dfCopy = dfCopy.transpose()

    res = dfCopy.div(dfCopy.sum(axis=0), axis=1).multiply(100)

    if not onColumns:
        res = res.transpose()

    return(res)

def sort_df_cols_by_name(df):
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def splitColBySeperator(col, sep, delFirstSpaces=True):
    splitList = col.str.split(sep).tolist()
    df = pd.DataFrame(splitList, index=col.index)
    df.columns = [get_col_name(col) + ' ' + str(i + 1) for i in range(len(df.columns))]
    if delFirstSpaces: df = df.apply(lambda x: x.str.lstrip())

    return(df)

# dictionary - for substituting each value according to dictionary.
def multiValCol2Binary(col, sep=',', delFirstSpaces=True, dictionary=None):
    dataSeparated = splitColBySeperator(col, sep, delFirstSpaces=delFirstSpaces)

    # first change all vals according to dict
    if dictionary is not None:
        for column in dataSeparated.columns:
            dataSeparated[column] = dataSeparated[column].replace(dictionary)

    # get all unique vals
    vals = getUniqueVals(df=dataSeparated)

    # create new binary DF with vals as columns
    binaryDF = pd.DataFrame(0, index=col.index, columns=list(vals))

    for ind in dataSeparated.index:
        for column in dataSeparated.columns:
            value = dataSeparated.loc[ind, column]
            if value is not None:
                binaryDF.loc[ind, value] = binaryDF.loc[ind, value] + 1

    return binaryDF


def getUniqueVals(col=None, df=None, sep=None, delFirstSpaces=None):
    '''
    # must get a column or a df with multiple columns.
    # if is given a column, should also get a sep and a delFirstSpaces.
    '''
    if (col is None and df is None) or  (col is not None and df is not None):
        raise Exception('Must get a column or a df!')
    elif (col is None):
        dataSeparated = df
    elif (df is None):
        if (sep is None or delFirstSpaces is None):
            raise Exception('I was given a col -> Must get a sep and a delFirstSpaces!')
        dataSeparated = splitColBySeperator(col, sep, delFirstSpaces=delFirstSpaces)

    vals = set(dataSeparated.iloc[:, 0].unique())
    for i in range(1, len(dataSeparated.columns)):
        vals.update(dataSeparated.iloc[:, 1].unique())
    vals.remove(None)

    return(vals)

def binDf2YN(col):
    return(col.replace({0: 'No', 1: 'Yes'}))

def boolDf2Bin(col):
    return(col.replace({False: 0, True: 1}))

def TwoBinCols_2_4categoryStrsCol(data, colName1, colName2):
    data = data.copy()
    newColName = colName1 + ' or ' + colName2
    data[newColName] = 0
    for i in data.index:
        row = data.loc[i]
        if row[colName1] == 1 and row[colName2] == 1: data.loc[i, newColName] = colName1 + ' & ' + colName2
        if row[colName1] == 1 and row[colName2] == 0: data.loc[i, newColName] = colName1
        if row[colName1] == 0 and row[colName2] == 1: data.loc[i, newColName] = colName2
        if row[colName1] == 0 and row[colName2] == 0: data.loc[i, newColName] = 'None'
    return data[newColName]

def pivottable(data, rowsColName, colsColName, valuesColName, naValue=np.nan, aggfunc='count'):
    """
    :param data: pd.DataFrame
    :param rowsColName: name of columns to transform into rows
    :param colsColName: name of columns to transform into columns
    :param valuesColName: name of columns to transform into values
    :param naValue: value to set in cells with NA
    :param aggfunc: aggfunc can be 'count', 'mean'
    :return: new pivot table
    """
    pivot = data.pivot_table(index=rowsColName, columns=colsColName,
                             values=valuesColName, aggfunc=aggfunc)
    pivot[pivot.isna()] = naValue
    return pivot

def renameDFcolumns(df, newNamesDict):
    df = df.copy().rename(columns=newNamesDict)
    return(df)

def list_removeItemIfExists(list1, itemToRemove):
    listCopy = list1.copy()

    try:
        listCopy.remove(itemToRemove)
    except ValueError:
        pass

    return(listCopy)

def list_remove_items_if_exist(list1, items_list):
    list_copy = list1.copy()
    for item in items_list:
        list_copy = list_removeItemIfExists(list_copy, item)

    return(list_copy)

# former do_lists_contain_same
def do_lists_contain_same(list1, list2):
    return set(list1) == set(list2)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def ndarray_add_columns(array, new_column):
    if len(array) == 0:
        new_array = np.array(new_column)
    else:
        new_array = np.column_stack((array, new_column))

    return new_array

def assert_column_exists_in_path(file_path, col_name, sheet=0):
    df = FileTools.read_excel(file_path, sheet=sheet)

    if col_name not in df.columns:
        print('Column', col_name, 'does not exist')
        print('in file', file_path, '.')
        print('Fix column name and re-run script.')
        exit()

def series_to_dict(series):
    """
    Assumes each index is unique!
    """
    series_dict = {}
    for i in series.index:
        series_dict[i] = series[i]
    return series_dict


def unique_in_df1_not_in_df2_by_cols(df1, df2, cols=None):
    '''
    return unique rows (partially - by cols) that are in 'df1' but not in 'df2',
    according to specified columns list 'cols'

    :param df1: pd.DataFrame
    :param df2: pd.DataFrame
    :param cols: list of strings (column names)
    '''
    if cols is None:
        cols = df1.columns

    count_df1 = Counter(map(tuple, df1[cols].values))
    count_df2 = Counter(map(tuple, df2[cols].values))
    diff = count_df1 - count_df2
    new_df = pd.DataFrame(list(diff.keys()), columns=cols)
    return new_df

def drop_duplicate_rows(df, by_cols=None, print_dropped=True):
    if by_cols is None:
        by_cols = df.columns

    new_df = df.drop_duplicates(subset=by_cols)
    if print_dropped:
        print('Dropped ' + str(df.shape[0] - new_df.shape[0]) + ' duplicate rows')

    return new_df

def randomly_partition_list(list_in, k):
    '''
    Randomly partition a given list into k lists of (almost) equal length.
    Returns k lists with the shuffled items of the original list.
    :param list_in: A list
    :param k: Number of partitions to create
    :return: a list of k lists
    '''
    list_in = list_in.copy()
    random.shuffle(list_in)
    return [list_in[i::k] for i in range(k)]

def df_to_csr_matrix(df):
    return sparse.csr_matrix(df.values)

def csr_matrix_to_df(matrix):
    dense = matrix.todense()
    return pd.DataFrame(dense)


def get_df_col_unique_vals_dict(db, col_names=None, print_cols=False):
    if col_names is None:
        col_names = db.columns

    col_unique_vals = dict()
    for col in col_names:
        col_unique_vals[col] = list(db.loc[:, col].unique())

        if print_cols:
            print('##### Column: {} \nUnique values: \n{}\n'.format(col, col_unique_vals[col]))

    return col_unique_vals

def get_col_index_by_name(df, name):
    return df.columns.get_loc(name)

def print_df_col_value_counts(df, max_vals_print=60):
    '''
    Prints the value counts of each column in df.
    From each column, prints up to max_vals_print values
    (descending order by value prevalence)
    :param df: pandas dataframe
    :param max_vals_print: int, maximun amount of values to print from each column
    :return: None
    '''
    for col in df.columns:
        counts = df[col].value_counts()

        print('\n### -------- Column: {}         ({} values)\n'.format(col, len(counts)))

        vals_print = min(max_vals_print, len(counts))
        if vals_print > 60: # if > 60, console doesn't print entire column
            for i in range(vals_print):
                print(counts.index[i], '  ', counts.iloc[i])
        else:
            print(counts[counts.index[:max(1,vals_print)]]) # for numeric indices, counts[:num] uses triggers loc instead of iloc

        # Let user know if not all values were printed
        if len(counts) > max_vals_print:
            print('\n*Column contains', len(counts) - max_vals_print, 'more values not printed.')

def get_rows_by_col_vals(df, col_vals_dict):
    '''
    Get subset from a dataframe based on specific values from multiple columns,
    defined by col_vals_dict. In col_vals_dict, each key is a column name by
    which you'd like to filter the dataframe. The value of the key should be
    a list with values that may be contained in that column, that you'd like
    to be included in the returned dataframe. For example, for a df containing
    column names 'col1', 'col2' and 'col3', and
    col_vals_dict={'col1': [1, 2], 'col2': ['a']}
    the returned dataframe will only have rows that had the values of 1 or 2
    in column 'col1', *and* the value of 'a' in the column 'col2'.

    :param df: pandas dataframe
    :param col_vals_dict: a dictionary.
    :return: pandas dataframe - a (copy) subset of rows from df
    '''
    subset_df = df.copy()
    for col in col_vals_dict:
        subset_df = subset_df.loc[lambda x: x[col].isin(col_vals_dict[col])]

    return subset_df

def get_ordered_unique_vals_from_list(mylist):
    """
    Gets a list, and returns a list with all unique vals in it, by order
    """
    used = set()
    uniq = [x for x in mylist if x not in used and (used.add(x) or True)]
    return uniq