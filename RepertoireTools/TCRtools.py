import pandas as pd

def add_clone_column(tcrs_df, cols_find_duplicates,
                     clone_text='clone_', clone_id_colname='clone_id',
                     add_num_clones=True, num_clones_colname='clone_size'):
    """
    Add column indicating the clone ID, i.e., create an identical ID
    for all duplicate TCRs, based on df columns indicated by cols_find_duplicates.
    :param tcrs_df: TCRs pandas dataframe
    :param cols_find_duplicates: column names to look for duplicated
    :param clone_text: text for clone names: default 'clone_1', 'clone_2' etc.
    :param clone_id_colname: the new column name for the clone column
    :param add_num_clones: add another column with clone size - number of TCRs found for the clone
    :param num_clones_colname: name of new num_clones column
    :return: a copy of tcrs_df with the added column.
    """

    df = tcrs_df.copy()
    df_dict = df[cols_find_duplicates].to_dict(orient='index')

    df[clone_id_colname] = None
    clone_num = 1

    for tcr1 in df.index: # go over all TCRs. using df index since order is important here
        if df.loc[tcr1, clone_id_colname] is None: # if TCR wasn't already identified as part of a clone
            found_dups = False # flag = found duplicates for this TCR

            for tcr2 in df.index: # go over all TCRs - inner loop. using df index since order is important here
                if tcr1 != tcr2 and df.loc[tcr2, clone_id_colname] is None: # if not tcr1 TCR, and tcr2 wasn't already identified as part of a clone
                    identical = True
                    # check if subset columns are identical for tcr1 and tcr2
                    for col in cols_find_duplicates:
                        if df_dict[tcr1][col] != df_dict[tcr2][col]: # check if identical
                            identical = False
                            break

                    if identical: # if didn't enter former "if" - found dup!
                        df.loc[tcr2, clone_id_colname] = clone_text + str(clone_num)
                        found_dups = True

            # if found_dups is True: # if found dup for tcr1
            df.loc[tcr1, clone_id_colname] = clone_text + str(clone_num)
            clone_num += 1

    if add_num_clones:
        dict_clones = df.groupby([clone_id_colname]).size().to_dict()
        df[num_clones_colname] = df[clone_id_colname].map(dict_clones)

    return df
