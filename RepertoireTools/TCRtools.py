import pandas as pd
import sys
import copy as copy_module
import matplotlib.pyplot as plt
import warnings

if 'LielTools' in sys.modules:
    from LielTools import DataTools
    from LielTools import PlotTools
    from LielTools import seqTools
else:
    import DataTools
    import PlotTools
    from LielTools_4.LielTools import seqTools


###### ------------------ General ------------------# # <editor-fold>

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


###### ------------------ TCRdata class ------------------# # <editor-fold>


class TCRdata():
    '''
    TCR sequences object.

    ## params:
        # self.df
        # self.epitopes, self.epitopesNum, self.epitopeIndices
        # self.otherLettersName, self.otherLetters, self.otherLettersDict
        # self.oLettersColA, self.oLettersColB

        *any of the init series components can be none.
    '''

    def __init__(self, cloneID, subjectSeries, epitopeSeries,
                 CDR3aSeries, CDR3bSeries, dataset_name='',
                 CDR1a_Series=None, CDR1b_Series=None,
                 CDR2a_Series=None, CDR2b_Series=None,
                 CDR25a_Series=None, CDR25b_Series=None,
                 drop_duplicates=['CDR3a', 'CDR3b', 'Epitope'],
                 otherLettersDict=None, otherLettersName=None,
                 cdrs='cdr3',  # 'cdr3' or 'all',
                 remove_cell_if_cdr_missing=True,
                 drop_cdr3_first_cysteine=False):
        """
        Create a TCRdata object.

        :param cloneID: If cloneID is None, create sequential fake IDs with format: clone__X
        :param subjectSeries: if subjectSeries is None, 'Subject' column will be filled with "Unknown"
        :param epitopeSeries: if epitopeSeries is None, 'Epitope' column will be filled with "Unknown"
        :param CDR3aSeries:
        :param CDR3bSeries:
        :param dataset_name:
        :param CDR1a_Series:
        :param CDR1b_Series:
        :param CDR2a_Series:
        :param CDR2b_Series:
        :param CDR25a_Series:
        :param CDR25b_Series:
        :param drop_duplicates: list of columns by which duplicates will be considered.
        :param otherLettersDict:
        :param otherLettersName:
        :param cdrs: value must be "cdr3" or "all"
        :param remove_cell_if_cdr_missing:
        :param drop_cdr3_first_cysteine: Remove first letter (C) from cdr3a and cdr3b.
                                        (If any don't start with a C - will raise exception)
        """

        self.dataset_name = dataset_name

        #### init dataset, only with non-none series provided. None series will be auto-filled in df.
        self.df = DataTools.join_non_empty_series_f_list([cloneID, subjectSeries, epitopeSeries,
                                                          CDR1a_Series, CDR2a_Series, CDR25a_Series, CDR3aSeries,
                                                          CDR1b_Series, CDR2b_Series, CDR25b_Series, CDR3bSeries],
                                                         ['CloneID', 'Subject', 'Epitope',
                                                          'CDR1a', 'CDR2a', 'CDR2.5a', 'CDR3a',
                                                          'CDR1b', 'CDR2b', 'CDR2.5b', 'CDR3b'])

        if (cloneID is None):  # if clone IDs were not provided, create sequential fake IDs
            self.df['CloneID'] = ['clone__' + str(num) for num in list(range(1, 1 + self.df.shape[0]))]
        self.df = self.df.set_index('CloneID')

        if subjectSeries is None:
            self.df['Subject'] = 'Unknown'

        if epitopeSeries is None:
            self.df['Epitope'] = 'Unknown'

        self.cdrs = cdrs
        if cdrs == 'cdr3':
            self.cdr_col_names = ['CDR3a', 'CDR3b']
        elif cdrs == 'all':
            self.cdr_col_names = ['CDR1a', 'CDR2a', 'CDR2.5a', 'CDR3a',
                                  'CDR1b', 'CDR2b', 'CDR2.5b', 'CDR3b']
        else:
            raise Exception('Unknown "cdrs" value: must be "cdr3" or "all"')

        # Make sure appropriate CDR columns were provided
        for cdr_col_name in self.cdr_col_names:
            assert cdr_col_name in self.df.columns, 'Missing {} column'.format(cdr_col_name)

        # drop duplicates
        self.droppedDuplicates = 0
        if drop_duplicates is not None:
            self.drop_duplicate_TCRs(drop_duplicates)

        # remove TCRs with missing CDRs
        if remove_cell_if_cdr_missing:
            for cdr_col_name in self.cdr_col_names:
                self.drop_if_cdr_is_missing(cdr_col_name)

        # remove TCRs with unknown characters * or _ or -
        for cdr_col_name in self.cdr_col_names:
            self.drop_if_unknown_char(cdr_col_name)

        self.drop_cdr3_first_cysteine = drop_cdr3_first_cysteine
        if drop_cdr3_first_cysteine:
            self.delete_cdr3_first_c()

        #### create epitope lists and indices
        self.initEpitopeInfo()

        #### other letters
        self.initAndAddOtherLetters(otherLettersDict, otherLettersName)

        self.get_CDRs_lengths()

    def get_CDRs_lengths(self):
        self.CDRs_max_len = dict()
        for cdr_col_name in self.cdr_col_names:
            self.CDRs_max_len[cdr_col_name] = self.df[cdr_col_name].apply(len).max()

    def uniteWithOther_TCRdata(self, other):
        if type(other) != TCRdata:
            raise Exception('other given object is not a TCRdata object!')

        if len(self.df.index.intersection(other.df.index)) > 0:
            raise Exception('other TCRdata has overlapping indices!')

        if self.cdrs != other.cdrs:
            raise Exception('other TCRdata has other CDRs!')

        if self.drop_cdr3_first_cysteine != other.drop_cdr3_first_cysteine:
            raise Exception('other TCRdata has a different drop_cdr3_first_cysteine value!')

        self.df = self.df.append(other.df)
        self.initEpitopeInfo()  #### create epitope lists and indices
        if (self.otherLettersDict is not None):  #### create "otherLetters" sequences in df
            self.createOtherLettersSeqs()

        self.get_CDRs_lengths()

        if 'CDRs' in self.df.columns or 'CDRs' in other.df.columns:
            warnings.warn("One of the objects contains concatenated CDRs. "
                          "Padding might be different! It's recommended to reconcatenate.")

        return (self)

    def copy(self):
        return (copy_module.deepcopy(self))

    def initEpitopeInfo(self):
        self.epitopes = self.df['Epitope'].unique()
        self.epitopesNum = len(self.epitopes)

        self.epitopeIndices = {}
        for ep in self.epitopes:
            self.epitopeIndices[ep] = (self.df['Epitope'] == ep)

    def initAndAddOtherLetters(self, otherLettersDict, otherLettersName):
        # init otherLetters
        self.otherLettersName = otherLettersName
        self.otherLettersDict = otherLettersDict

        if (self.otherLettersDict is not None):
            self.otherLetters = list(set(self.otherLettersDict.values()))  # create otherLetters list
            if (
                    self.otherLettersDict is None): self.otherLettersName = ''  # if they weren't given a name, initialize to an empty string

            # create "otherLetters" sequences in df
            self.createOtherLettersSeqs()

    def createOtherLettersSeqs(self):
        self.cdr_col_names_other = [col_name + '_' + self.otherLettersName for col_name in self.cdr_col_names]
        for col_name in self.cdr_col_names:
            self.df[col_name + '_' + self.otherLettersName] = seqTools.series_seqs_letters_to_other_letters(self.df[col_name],
                                                                                                            letterDict=self.otherLettersDict)

        # self.oLettersColA = 'CDR3a_' + self.otherLettersName  # col name a
        # self.df[self.oLettersColA] = series_seqs_letters_to_other_letters(self.df['CDR3a'], letterDict=self.otherLettersDict)
        #
        # self.oLettersColB = 'CDR3b_' + self.otherLettersName  # col name b
        # self.df[self.oLettersColB] = series_seqs_letters_to_other_letters(self.df['CDR3b'], letterDict=self.otherLettersDict)

    def addOtherLetters(self, otherLettersDict=seqTools.allAA2_6AAG, otherLettersName='6AAG'):
        if (self.otherLettersDict is not None):
            raise Exception('Warning! You tried to add otherLetters, but otherLetters already exist in this object!')

        self.initAndAddOtherLetters(otherLettersDict, otherLettersName)

    def get_max_cdr3_len(self, CDR3type='both', savePath=None, color='green', create_plot=True):  # 'both', 'a', or 'b'
        TCRdsNew = self.df.copy()
        TCRdsNew['cdr3a_length'] = TCRdsNew['CDR3a'].apply(len)
        TCRdsNew['cdr3b_length'] = TCRdsNew['CDR3b'].apply(len)
        if (CDR3type == 'a'):
            maxLen = TCRdsNew['cdr3a_length'].max()
        elif (CDR3type == 'b'):
            maxLen = TCRdsNew['cdr3b_length'].max()
        elif (CDR3type == 'both'):
            maxLen = max(TCRdsNew['cdr3a_length'].max(), TCRdsNew['cdr3b_length'].max())
        else:
            raise Exception('wrong #which# argument')

        if create_plot:
            fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, sharex=True)
            TCRdsNew['cdr3a_length'].hist(ax=axes1[0], grid=False, range=(5, 25), color=color), axes1[0].set_title(
                'CDR3a length')
            TCRdsNew['cdr3b_length'].hist(ax=axes1[1], grid=False, range=(5, 25), color=color), axes1[1].set_title(
                'CDR3b length')

            PlotTools.savePlt(savePath=savePath, dpi=300)

        return maxLen

    def get_len_freq_table(self, suffix=''):
        TCRdsNew = self.df.copy()

        cdr3a_freq = TCRdsNew['CDR3a'].apply(len).value_counts().sort_index(0)
        cdr3a_freq.name = 'CDR3a'

        cdr3b_freq = TCRdsNew['CDR3b'].apply(len).value_counts().sort_index(0)
        cdr3b_freq.name = 'CDR3b'

        lengths = pd.DataFrame(cdr3a_freq).join(cdr3b_freq)
        lengths = DataTools.addToColNames(lengths, suffix=suffix)

        return lengths

    def getCountByEpitopes(self, show=True):  # TODO add show
        TCRcounts = pd.DataFrame(0, index=self.epitopes, columns=['TCR count'])
        for ep in self.epitopes:
            TCRcounts.loc[ep, 'TCR count'] = self.epitopeIndices[ep].sum()
        fig = PlotTools.DFbarPlot(TCRcounts, plotTitle='TCR Counts by Epitope', showLegend=False)
        return ({'counts': TCRcounts, 'figure': fig})

    def concatCDR3chains(self, pad=True, extraPadBetweenCDRs=0, extra_pad_after_b=0):
        if pad:
            CDR3a = seqTools.pad_str_series(self.df.copy()['CDR3a'], extraPad=extraPadBetweenCDRs)
            CDR3b = seqTools.pad_str_series(self.df.copy()['CDR3b'], extraPad=extra_pad_after_b)
        else:
            CDR3a = self.df.copy()['CDR3a']
            CDR3b = self.df.copy()['CDR3b']
        self.df['CDR3'] = pd.DataFrame(CDR3a).join(CDR3b).apply(lambda row: ''.join(row), axis=1)

    def concatCDRs(self, pad=True, extraPadBetweenCDRs=0, other_letters=False):
        cdr_columns = self.cdr_col_names_other if other_letters else self.cdr_col_names
        self.CDRs_padded_max_len = dict()
        new_df = pd.DataFrame(index=self.df.index)

        for col in cdr_columns:
            if pad:
                new_col = seqTools.pad_str_series(self.df.copy()[col], extraPad=extraPadBetweenCDRs)
            else:
                new_col = self.df.copy()[col]
            new_df = new_df.join(new_col)
            self.CDRs_padded_max_len[col] = new_col.apply(len).max()

        self.df['CDRs'] = new_df.apply(lambda row: ''.join(row), axis=1)

    # def add_mirror_CDR3(self, pad=True, pad_between_copies=0):
    #     if pad:
    #         CDR3 = pad_str_series(self.df.copy()['CDR3'], extraPad=pad_between_copies)
    #     else:
    #         CDR3 = self.df.copy()['CDR3']
    #
    #     CDR3_mirror = mirror_string_series(self.df.copy()['CDR3'])
    #     self.df['CDR3'] = pd.DataFrame(CDR3).join(CDR3_mirror).apply(lambda row: ''.join(row), axis=1)

    # letters = 'other' or 'aa' , CDR3type = 'a' or 'b' or 'bothAdded' or 'bothConcatenated'
    #     def getKmerFeaturesAndEpitopeDS(self, k, CDR3type, letterType='other', writeToPath=None):
    #         if (letterType=='other'):
    #             cols = self.cdr_col_names_other
    #             # aCol = self.oLettersColA
    #             # bCol = self.oLettersColB
    #             letters = self.otherLetters
    #         elif (letterType=='aa'):
    #             cols = self.cdr_col_names
    #             # aCol = 'CDR3a'
    #             # bCol = 'CDR3b'
    #             letters = allAA
    #         else:
    #             raise Exception('unknown letter type')
    #
    #         allKmers = get_all_possible_kmers(k, letters)
    #
    #         if (CDR3type == 'a'):
    #             ds = get_kmers_df_for_series(self.df[aCol], k, allKmers)
    #         elif (CDR3type == 'b'):
    #             ds = get_kmers_df_for_series(self.df[bCol], k, allKmers)
    #         elif (CDR3type == 'bothAdded'):
    #             TCRa_features = get_kmers_df_for_series(self.df[aCol], k, allKmers)
    #             TCRb_features = get_kmers_df_for_series(self.df[bCol], k, allKmers)
    #             ds = DataTools.addDFs([TCRa_features, TCRb_features])
    #         elif (CDR3type == 'bothConcatenated'):
    #             TCRa_features = get_kmers_df_for_series(self.df[aCol], k, allKmers).add_suffix('_a')
    #             TCRb_features = get_kmers_df_for_series(self.df[bCol], k, allKmers).add_suffix('_b')
    #             ds = TCRa_features.join(TCRb_features)
    #
    #         dsFinal = ds.join(self.df['Epitope'])
    #         if (writeToPath is not None):
    #             FileTools.write2Excel(writeToPath, dsFinal)
    #         return(dsFinal)

    def getLengthHistogram(self, sequenceColNames):
        singleCol = type(sequenceColNames) == str

        if singleCol:
            sequenceColNames = [sequenceColNames]

        fig1, axes1 = plt.subplots(nrows=1, ncols=len(sequenceColNames), figsize=(3.5 * len(sequenceColNames), 3),
                                   sharey=True, sharex=True)
        colorsIter = PlotTools.getColorsList(len(sequenceColNames))['iter']

        for i, column in enumerate(sequenceColNames):
            if singleCol:
                useAxes = axes1
            else:
                useAxes = axes1[i]

            # strLengths = self.df[column].apply(len)
            # strLengths.hist(ax=useAxes, grid=False, color=next(colorsIter)), useAxes.set_title(column)

            PlotTools.plotSeriesHistogram(self.df[column].apply(len), useAxes=useAxes,
                                          color=next(colorsIter), grid=False)

        plt.show()

    def drop_duplicate_TCRs(self, drop_duplicates):
        """
        drops duplicate TCRs in self.df. keeps the first duplicated line.
        If duplicates were dropped, will print how many.

        :param drop_duplicates: list of columns by which duplicates will be considered.
        :return: Nothing.
        """
        originalN = self.df.shape[0]
        self.df.drop_duplicates(subset=drop_duplicates, inplace=True)
        self.droppedDuplicates = originalN - self.df.shape[0]
        if self.droppedDuplicates > 0:
            print('TCRdata {}: dropped {} TCRs with duplicate CDRs'.format(self.dataset_name,
                                                                           self.droppedDuplicates))

    def drop_if_cdr_is_missing(self, cdr_col_name):
        originalN = self.df.shape[0]
        self.df = self.df.loc[~self.df[cdr_col_name].isna()]
        self.df = self.df.loc[~(self.df[cdr_col_name] == '')]

        dropped = originalN - self.df.shape[0]
        if dropped > 0:
            print('TCRdata {}: dropped {} TCRs with missing {}'.format(self.dataset_name,
                                                                       dropped, cdr_col_name))

    def drop_if_unknown_char(self, cdr_col_name):
        originalN = self.df.shape[0]
        self.df = self.df.loc[~self.df[cdr_col_name].map(lambda x: '*' in x)]
        self.df = self.df.loc[~self.df[cdr_col_name].map(lambda x: '-' in x)]
        self.df = self.df.loc[~self.df[cdr_col_name].map(lambda x: '_' in x)]

        dropped = originalN - self.df.shape[0]
        if dropped > 0:
            print('TCRdata {}: dropped {} TCRs with unknown character *, - or _ in column {}'.format(self.dataset_name,
                                                                                                     dropped,
                                                                                                     cdr_col_name))

    def get_existing_kmers_hist(self, k, other_letters=False, from_indices=None):
        """
        :param k: size of k for analysis
        :param other_letters: False - analyze AA sequences,
                              True - analyze other letter sequences
        :param from_indices: List of indices (integers. for pandas iloc function) of rows
                             from self.df to include in the exitings-kmers check.
                             If None, all df rows will be analyzed.

        returns: dict of kmers that exist in the sequences and their overall count
        """

        if from_indices is None:
            relevant_df = self.df
        else:
            relevant_df = self.df.iloc[from_indices, :]

        if other_letters:
            assert self.otherLettersDict is not None
            cols = self.cdr_col_names_other
            # a_col_name = self.oLettersColA
            # b_col_name = self.oLettersColB
        else:
            cols = self.cdr_col_names
            # a_col_name = 'CDR3a'
            # b_col_name = 'CDR3b'

        cdrs_concat_series = pd.Series()
        for cdr_col in cols:
            cdr_series = relevant_df[cdr_col].copy()
            cdr_series.index = cdr_col + '__' + cdr_series.index.astype(str)
            cdrs_concat_series = cdrs_concat_series.append(cdr_series)

        kmer_dict = seqTools.all_available_kmers_dict(cdrs_concat_series, k)
        return kmer_dict

    def delete_cdr3_first_c(self):
        """
        Check if all cdr3a and cdr3b start with a C.
        If not, will raise exception.
        Else, will drop this first C.
        """
        # Make sure all cdr3a and cdr3b in all cells start with a cysteine.
        for ind in self.df.index:
            if self.df.loc[ind, 'CDR3a'][0] != 'C' or \
                    self.df.loc[ind, 'CDR3b'][0] != 'C':
                raise Exception("drop_cdr3_first_cysteine: "
                                "cell's cdr3 does not start with a cysteine. "
                                "Change to false or delete cell: " + ind)

        # Drop first C in cdr3a and cdr3b
        self.df.loc[:, 'CDR3a'] = self.df.loc[:, 'CDR3a'].apply(lambda x: x[1:])
        self.df.loc[:, 'CDR3b'] = self.df.loc[:, 'CDR3b'].apply(lambda x: x[1:])

# </editor-fold>