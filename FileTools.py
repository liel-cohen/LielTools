import sys
import numpy as np
import dill
import pandas as pd
import os
import shutil
import urllib.request
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle
import json

if 'LielTools' in sys.modules:
    from LielTools import DataTools
else:
    import DataTools


def write2Excel(path, data, index=True, csv=False):
    ''' "data" can be a dataframe, dict, list or set.
        If index=True, will write index as well.'''
    if type(data)== dict:
        data = pd.Series(data).to_frame()
    elif type(data) == list:
        data = pd.Series(data).to_frame()
    elif type(data)== set:
        data = pd.Series(list(data)).to_frame()

    if csv:
        data.to_csv(path, index=index)
    else:
        data.to_excel(path, index=index)

def write_var_to_dill(path, variable):
    with open(path , 'wb') as d:
        dill.dump(variable, d, protocol=-1)

def dill_to_var(path):
    with open(path, 'rb') as fh:
        return dill.load(fh)

def write_list_to_txt(listToWrite, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as thefile:
        for item in listToWrite:
            thefile.write("%s\n" % str(item))

def read_txt_to_strings_list(path):
    with open(path, 'r') as f:
        str_list = f.read().splitlines()
    return str_list

# former readExcel
def read_excel(path, sheet=0, indexCol=None,engine=None):
    ''' indexCol - name (string) of the column to be defined as index '''
    data = pd.read_excel(path, sheet_name=sheet, index_col=indexCol,engine=engine)
    if len(data.index) != len(data.index.unique()):
        print('Warning: DF index not unique!')

    return data

# former createFolder
def create_folder(path):
    ''' If folder doesn't exist, create it '''
    if not os.path.exists(path):
        os.makedirs(path)

    return str(path)

# former deleteFolder
def delete_folder(path, keepFolderItself=True):
    if not os.path.exists(path):
        raise Exception('Folder does not exist!')

    shutil.rmtree(path)
    if keepFolderItself: create_folder(path)

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def excel2dict(path, sheet=0, colKey=None, colVal=None):
    ''' if colKey and colVal = None:
     first excel column = keys
     second = values '''
    if ( (colKey is not None and colVal is None) or (colVal is not None and colKey is None)):
        raise Exception('colKey and colVal can be both None, or neither')

    info = read_excel(path, sheet=sheet, indexCol=colKey)

    if colKey is None:
        info.index = info.iloc[:,0]
        colVal = info.iloc[:,1].name
    else:
        info.index = info[colKey]

    dict = info[colVal].dropna().to_dict()
    return(dict)

def runPy(filename):
    """Aid in the use of exec as replacement for execfile
    in Python 3

    Parameters
    ----------
    filename : str

    Returns
    -------
    code : code object
        For use with exec

    Usage
    -----

    >>> exec(mycompile(filename))

    Same as:

    >>> execfile(filename)"""

    return compile(open(filename).read(), filename, 'exec')

def download_file_to_path(url, path):
    ''' Download file from url to exact file path.
        path folder must be created before.
    '''
    urllib.request.urlretrieve(url, path)

def copy_all_folder_files_to_folder(source_folder, dest_folder):
    files = os.listdir(source_folder)

    for f in files:
        shutil.copy(source_folder + f, dest_folder)

# Sparse matrices
def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def load_sparse_csr_to_df(filename):
    matrix = load_sparse_csr(filename)
    return DataTools.csr_matrix_to_df(matrix)

def save_df_as_sparse_csr(filename, df):
    save_sparse_csr(filename,
                    DataTools.df_to_csr_matrix(df))

def get_directory_size(folder_path):
    '''
    returns size in megabytes
    :param folder_path: string - path
    '''
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)

    return total_size/(1024**2)

''' former get_paths_of_subfolders '''
def get_subfolders(folder_path, return_full_paths=True):
    '''
    only one level down.
    :param folder_path: root folder path
    :param return_full_paths: if False - will return subfolders names
                                         instead of full paths
    :return: list of folder paths, only one level bellow root path.
    '''
    paths = []
    listdir = os.listdir(folder_path)
    for name in listdir:
        path = os.path.join(folder_path, name)
        if os.path.isdir(path):
            if return_full_paths:
                paths.append(path)
            else:
                paths.append(name)

    return paths

def print_subfolders_sizes(folder_path):
    '''
    only one level down.
    :param folder_path:
    :return:
    '''
    paths = get_subfolders(folder_path)
    for path in paths:
        print(path, '\nSize in megabytes:', get_directory_size(path))

def load_from_pickle(path):
    def open_file():
        return open(path, 'rb')

    with open_file() as f:
        return pickle.load(f)

def dump_to_pickle(path, my_info):
    with open(path, 'wb') as f:
        pickle.dump(my_info, f)


    with open(path, 'rb') as f:
        my_info = pickle.load(f)
    return my_info

def dict_to_json_txt(d, file_path):
    json.dump(d, open(file_path, 'w'))

def json_txt_to_dict(file_path):
    d = json.load(open(file_path))
    return d


