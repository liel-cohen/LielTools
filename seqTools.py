import pandas as pd
import numpy as np
from functools import partial
from itertools import product
import sys
import keras
import copy as copy_module
import matplotlib.pyplot as plt
import warnings

if 'LielTools_4' in sys.modules:
    from LielTools_4 import FileTools
    from LielTools_4 import DataTools
    from LielTools_4 import PlotTools
    from LielTools_4 import MyLabelEncoder
else:
    import FileTools
    import DataTools
    import PlotTools
    import MyLabelEncoder



####### ------------------ seqTools ------------------#### <editor-fold>

allAA = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V',
         'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']
all6AAG =  ['O',  # small and polar residues
            'N',  # small and nonpolar
            'D',  # polar or acidic residues
            'B',  # basic
            'H',  # large and hydrophobic
            'A']  # aromatic
allAA2_6AAG =  {'C': 'O', 'S': 'O', 'T': 'O',
                'P': 'N', 'A': 'N', 'G': 'N',
                'N': 'D', 'D': 'D', 'E': 'D', 'Q': 'D',
                'H': 'B', 'R': 'B', 'K': 'B',
                'M': 'H', 'I': 'H', 'L': 'H', 'V': 'H',
                'F': 'A', 'Y': 'A', 'W': 'A'}
allAA2_aa = dict([(aa, aa) for aa in allAA])
aa_three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                   'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                   'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                   'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                   'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
aa_one_to_three = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
                   'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                   'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
                   'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
                   'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}



def aaSeq2Letters(sequence, letterDict=allAA2_6AAG):
    newSeq = ''
    for letter in range(len(sequence)):
        aa = sequence[letter]
        if (not aa == '-'):
            newSeq = newSeq + letterDict[aa]
        else:
            newSeq = newSeq + '-'
    return(newSeq)


def aaSeqSeries2Letters(seriesAA, letterDict=allAA2_6AAG):
    newSeries = seriesAA.apply(partial(aaSeq2Letters, letterDict=letterDict))
    return (newSeries)


def length(sequence):
    return(len(sequence))


def cut_edge_letters(string, start_or_end, num_letters=3):
    if start_or_end == 'start':
        return string[:num_letters]
    elif start_or_end == 'end':
        return string[-num_letters:]
    else:
        raise ValueError('cut_edge_letters: unknown start_or_end')


def createAllKmersList(k, letters):
    kmersList = []

    for i in product(*([letters] * k)):
        kmer = ''.join(i)
        kmersList.append(kmer)

    return(kmersList)


def getKmerFeaturesForSeries(seqSeries, k, kmersList):
    kmersCount = pd.DataFrame(0, index=seqSeries.index, columns=kmersList)

    for row in seqSeries.iteritems():
        seq = row[1]
        for start in range(len(seq) - k + 1):
            kmer = seq[start:(start + k)]
            kmersCount.loc[row[0], kmer] = kmersCount.loc[row[0], kmer] + 1

    return(kmersCount)


def padStringSeries(series, extraPad=0):
    maxLen = series.apply(len).max()

    def ljust0(string):
        return(string.ljust(maxLen+extraPad, '0'))

    seriesPadded = series.apply(ljust0)
    return(seriesPadded)


def mirror_string_series(series):
    def mirror(string):
        return(string[::-1])

    series_mirror = series.apply(mirror)
    series_mirror.rename(series_mirror.name+'_mirror', inplace=True)
    return(series_mirror)


def strSeries2charLists(series):
    ''' Gets a series of strings.
    Returns a list of char lists - each string is transformed into a char list '''

    strsList = list(series)

    newlist = list()
    alphabetSet = set()
    for string in strsList:
        newChars = list(string)
        newlist.append(newChars)
        alphabetSet.update(newChars)

    alphabet = list(alphabetSet)
    alphabet.sort()

    return([newlist, alphabet])


def getChar2intEncoder(alphabetList, add0=True, sortAlphabet=True):
    ''' Get an encoder for the given characters set (sklearn.preprocessing.LabelEncoder) '''

    if '0' in alphabetList: # we want the '0' padder at the end of the encoder map (if amino acids - int 20)
        alphabetList.remove('0')

    le = MyLabelEncoder.MyLabelEncoder(labelsList=alphabetList, sort=sortAlphabet)
    if add0: le.addLabel('0')        # add '0' as the last (if amino acids - 20) int
    le_alphabet_mapping = le.getEncoderMappingDict()
    return([le, le_alphabet_mapping])


def charLists2intEncoding(charlists, charEncoder):
    ''' Gets lists of chars: each list is a sample - characters sequence (string as list of chars).
     Returns each char list as an int list.
     Organized as ndarray: each row is a sample, converted to ints '''
    # encode each charlist to ints
    integer_encoded = list()
    for item in charlists:
        integer_encoded.append(charEncoder.transform(item))

    return (np.array(integer_encoded))


def strSeries2_1H(stringSeries, charEncoder=None, padding=True, gaps=False):
    ''' Gets a pd.Series of strings, with length n. Maximum string length l.
        Can also get an encoder that fits the strings alphabet. Alphabet size d.
        Returns an ndarray with the shape: (n, l, d).
        If padding=True, will add padding at the end of sequences so they all
        have the same length (max length in the series).
        If gaps=True, will consider '0's as gaps and delete the '0' character
        one hot vector.
        If charEncoder is given, makes sure the charEncoder alphabet is identical
        to the the stringSeries alphabet'''

    charEncoderWasGiven = charEncoder is not None

    # keras.utils.to_categorical does not support samples with different lengths! Must pad.
    if padding: # pad sequences
        stringSeries = padStringSeries(stringSeries.copy())

    # stringSeries to char lists
    charlists, alphabet = strSeries2charLists(stringSeries)

    if charEncoder is None:
        # get an encoder (into integers) for the stringSeries alphabet
        charEncoder, encoderAlphabetMap = getChar2intEncoder(alphabet)
    else:
        # make sure the given encoder's alphabet fits the stringSeries alphabet
        encoderAlphabetMap = charEncoder.getEncoderMappingDict()
        alphabetCheck = set(alphabet)
        alphabetFromMap = set(encoderAlphabetMap.keys())
        if padding and not gaps: alphabetCheck.add('0')
        if set(alphabetCheck) != alphabetFromMap:
            raise Exception('Warning: stringSeries letters arent identical to given charEncoder letters!')

    # use encoder to encode chars to ints (for each sample) - getting a 2d matrix
    integer_encoded = charLists2intEncoding(charlists, charEncoder)

    # encode ints to 1-hot vectors (for each sample)
    oneHot = keras.utils.to_categorical(integer_encoded)

    if padding or gaps: # delete the 0 encoding column (if amino acids - column 20)
        paddingCol = len(encoderAlphabetMap)-1
        if encoderAlphabetMap['0'] != paddingCol:  # Check.
            raise Exception('Problem here! Encoder char zero must be mapped to the last int!')
        oneHot = oneHot[:, :, 0:(paddingCol)]

    if charEncoderWasGiven:
        return (oneHot)
    else:
        return ([oneHot, charEncoder])


def oneHot_add0encodingColumn(oneHot):
    # add last column - for '0' encoding,
    # fill it with 1's at positions where there are no 1's at all (i.e., padding position)
    oneHot = oneHot.copy()
    shape = oneHot.shape
    zeros = np.zeros((shape[0], shape[1], 1))
    oneHot = np.append(oneHot, zeros, axis=2)
    for i in range(len(oneHot)):
        whichIsPad = oneHot[i].sum(axis=1) == 0
        oneHot[i][whichIsPad, shape[2]] = 1
    return(oneHot)


def oneHot2strSeries(oneHot, encoder, padded=True):
    if padded:
        # add last column of '0' encoding
        oneHot = oneHot_add0encodingColumn(oneHot)

    stringList = list()
    for i in range(len(oneHot)):
        inverse = np.argmax(oneHot[i], axis=1)
        string = ''.join(encoder.inverse_transform(inverse))
        stringList.append(string)

    return(pd.Series(stringList))


def matrix_1H_to_1H(transMap=None, d1AlphabetMap=None, d2AlphabetMap=None):
    if transMap is None or d1AlphabetMap is None:
        raise Exception('Must provide values for transMap and d1AlphabetMap!')

    d1_letters = list(d1AlphabetMap.keys())

    if d2AlphabetMap is None:
        d2_letters = list(set(transMap.values()))
        d2Encoder = MyLabelEncoder(labelsList=d2_letters, sort=True)
        d2AlphabetMap = d2Encoder.getEncoderMappingDict()
    else:
        d2_letters = list(d2AlphabetMap.keys())

    # Remove 0 values
    d1_letters = DataTools.list_removeItemIfExists(d1_letters, '0')
    d2_letters = DataTools.list_removeItemIfExists(d2_letters, '0')

    # Make sure filterDict and d1AlphabetMap have the same letters (keys)
    if not DataTools.do_lists_contain_same(transMap.keys(), d1_letters):
        raise Exception('filterDict and d1AlphabetMap must have the same keys!!!')

    # Make sure filterDict vals and d2AlphabetMap keys are the same letters
    if not DataTools.do_lists_contain_same(transMap.values(), d2_letters):
        raise Exception('filterDict vals and d2AlphabetMap keys must be the same letters!!!')

    filter = np.zeros((len(d1_letters), len(d2_letters)))

    for letter in d1_letters:
        d1_letter_int = d1AlphabetMap[letter]
        d1_letter_to_d2 = transMap[letter]
        d2_letter_int = d2AlphabetMap[d1_letter_to_d2]
        filter[d1_letter_int, d2_letter_int] = 1

    return(filter)


def kmers_hist_from_seq_series(seq_series, k):
    """
    :param seq_series: pd.Series with sequences to be analyzed
    :param k: size of k for analysis
    returns: dict of kmers that exist in the sequences and their overall count
    """
    assert type(seq_series) is pd.Series

    # initialize ans dfs
    kmers_hist = {}

    # loop over all sequences and extract kmers
    for ind in seq_series.index:
        seq = seq_series[ind]

        for start in range(len(seq) - k + 1):  # go over all positions in seq
            kmer = seq[start:start + k]  # extract k-mer in position
            kmers_hist[kmer] = kmers_hist.get(kmer, 0) + 1

    return kmers_hist


def create_gapped_list_from_kmer(kmer, gap, gap_symbol='0'):
    '''
    Gets a single kmer, and the amount of gapped letters.
    Returns a list with the kmer + all possible gapped k-mers.
    Gapped k-mers will include all possibilities of kmer letters
    being repaced with with gaps. Number of gaps - 0, ... , gap.
    For example, if gap=2, the returned list will include the original k-mers,
    all gapped k-mers with 1 gap and all gapped k-mers with 2 gaps.
    :param kmer: string
    :param gap: int
    :return: list of unique strings
    '''
    new_kmers = [kmer]

    if gap > 0:
        for index in range(len(kmer)):
            if kmer[index] != gap_symbol:
                new_kmer = kmer[0:index] + gap_symbol + kmer[index + 1:]
                new_kmers += create_gapped_list_from_kmer(new_kmer, gap - 1)

    return list(pd.Series(new_kmers).unique())


def get_other_aa_abbrev(aa):
    '''
    The function get_other_aa_abbrev gets a string of amino acid name in one(‘V’) or in three letters(‘VAL’)
    and returns a string of an id of amino acids in three letters(‘VAL’) or in one letter(‘V’), respectively.
    The function should accept only one string argument and check if it is one letter code or three letter code
    and return the amino acid code in the different length.
    '''

    ans = 'X'

    if len(aa) == 3:
        if aa in aa_three_to_one.keys():  # is aa in dict?
            ans = aa_three_to_one[aa]
    elif len(aa) == 1:
        if aa in aa_one_to_three.keys():  # is aa in dict?
            ans = aa_one_to_three[aa]

    return ans

# </editor-fold>


###### ------------------ class TCRdata ------------------# # <editor-fold>

# TCR sequences object.
# any of the init components can be none.
class TCRdata():
## params:
    # self.df
    # self.epitopes, self.epitopesNum, self.epitopeIndices
    # self.otherLettersName, self.otherLetters, self.otherLettersDict
    # self.oLettersColA, self.oLettersColB

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
                                        (If any doesn't start with a C - will raise exception)
        """
        
        self.dataset_name = dataset_name

        #### init dataset, only with non-none series provided. None series will be auto-filled in df.
        self.df = DataTools.join_non_empty_series_f_list([cloneID, subjectSeries, epitopeSeries,
                                                          CDR1a_Series, CDR2a_Series, CDR25a_Series, CDR3aSeries,
                                                          CDR1b_Series, CDR2b_Series, CDR25b_Series, CDR3bSeries],
                                                         ['CloneID', 'Subject', 'Epitope',
                                                        'CDR1a', 'CDR2a', 'CDR2.5a', 'CDR3a',
                                                        'CDR1b', 'CDR2b', 'CDR2.5b', 'CDR3b'])

        if (cloneID is None): # if clone IDs were not provided, create sequential fake IDs
            self.df['CloneID'] = [ 'clone__'+str(num) for num in list(range(1, 1+self.df.shape[0]))]
        self.df = self.df.set_index('CloneID')

        if subjectSeries is None:
            self.df['Subject'] = 'Unknown'

        if epitopeSeries is None:
            self.df['Epitope'] = 'Unknown'

        self.cdrs = cdrs
        if cdrs=='cdr3':
            self.cdr_col_names = ['CDR3a', 'CDR3b']
        elif cdrs=='all':
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

        return(self)

    def copy(self):
        return(copy_module.deepcopy(self))

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
            self.otherLetters = list(set(self.otherLettersDict.values())) # create otherLetters list
            if (self.otherLettersDict is None): self.otherLettersName = '' # if they weren't given a name, initialize to an empty string

            # create "otherLetters" sequences in df
            self.createOtherLettersSeqs()

    def createOtherLettersSeqs(self):
        self.cdr_col_names_other = [col_name + '_' + self.otherLettersName for col_name in self.cdr_col_names]
        for col_name in self.cdr_col_names:
            self.df[col_name + '_' + self.otherLettersName] = aaSeqSeries2Letters(self.df[col_name],
                                                                      letterDict=self.otherLettersDict)

        # self.oLettersColA = 'CDR3a_' + self.otherLettersName  # col name a
        # self.df[self.oLettersColA] = aaSeqSeries2Letters(self.df['CDR3a'], letterDict=self.otherLettersDict)
        #
        # self.oLettersColB = 'CDR3b_' + self.otherLettersName  # col name b
        # self.df[self.oLettersColB] = aaSeqSeries2Letters(self.df['CDR3b'], letterDict=self.otherLettersDict)

    def addOtherLetters(self, otherLettersDict=allAA2_6AAG, otherLettersName='6AAG'):
        if (self.otherLettersDict is not None):
            raise Exception('Warning! You tried to add otherLetters, but otherLetters already exist in this object!')

        self.initAndAddOtherLetters(otherLettersDict, otherLettersName)

    def get_max_cdr3_len(self, CDR3type='both', savePath=None, color='green', create_plot=True): # 'both', 'a', or 'b'
        TCRdsNew = self.df.copy()
        TCRdsNew['cdr3a_length'] = TCRdsNew['CDR3a'].apply(len)
        TCRdsNew['cdr3b_length'] = TCRdsNew['CDR3b'].apply(len)
        if (CDR3type=='a'):
            maxLen = TCRdsNew['cdr3a_length'].max()
        elif (CDR3type=='b'):
            maxLen = TCRdsNew['cdr3b_length'].max()
        elif (CDR3type=='both'):
            maxLen = max(TCRdsNew['cdr3a_length'].max(), TCRdsNew['cdr3b_length'].max())
        else:
            raise Exception('wrong #which# argument')

        if create_plot:
            fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, sharex=True)
            TCRdsNew['cdr3a_length'].hist(ax=axes1[0], grid=False, range = (5,25), color=color), axes1[0].set_title('CDR3a length')
            TCRdsNew['cdr3b_length'].hist(ax=axes1[1], grid=False, range = (5,25), color=color), axes1[1].set_title('CDR3b length')

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

    def getCountByEpitopes(self, show=True): # TODO add show
        TCRcounts = pd.DataFrame(0, index=self.epitopes, columns=['TCR count'])
        for ep in self.epitopes:
            TCRcounts.loc[ep, 'TCR count'] = self.epitopeIndices[ep].sum()
        fig = PlotTools.DFbarPlot(TCRcounts, plotTitle='TCR Counts by Epitope', showLegend=False)
        return({'counts': TCRcounts, 'figure': fig})

    def concatCDR3chains(self, pad=True, extraPadBetweenCDRs=0, extra_pad_after_b=0):
        if pad:
            CDR3a = padStringSeries(self.df.copy()['CDR3a'], extraPad=extraPadBetweenCDRs)
            CDR3b = padStringSeries(self.df.copy()['CDR3b'], extraPad=extra_pad_after_b)
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
                new_col = padStringSeries(self.df.copy()[col], extraPad=extraPadBetweenCDRs)
            else:
                new_col = self.df.copy()[col]
            new_df = new_df.join(new_col)
            self.CDRs_padded_max_len[col] = new_col.apply(len).max()

        self.df['CDRs'] = new_df.apply(lambda row: ''.join(row), axis=1)

    # def add_mirror_CDR3(self, pad=True, pad_between_copies=0):
    #     if pad:
    #         CDR3 = padStringSeries(self.df.copy()['CDR3'], extraPad=pad_between_copies)
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
#         allKmers = createAllKmersList(k, letters)
#
#         if (CDR3type == 'a'):
#             ds = getKmerFeaturesForSeries(self.df[aCol], k, allKmers)
#         elif (CDR3type == 'b'):
#             ds = getKmerFeaturesForSeries(self.df[bCol], k, allKmers)
#         elif (CDR3type == 'bothAdded'):
#             TCRa_features = getKmerFeaturesForSeries(self.df[aCol], k, allKmers)
#             TCRb_features = getKmerFeaturesForSeries(self.df[bCol], k, allKmers)
#             ds = DataTools.addDFs([TCRa_features, TCRb_features])
#         elif (CDR3type == 'bothConcatenated'):
#             TCRa_features = getKmerFeaturesForSeries(self.df[aCol], k, allKmers).add_suffix('_a')
#             TCRb_features = getKmerFeaturesForSeries(self.df[bCol], k, allKmers).add_suffix('_b')
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

        fig1, axes1 = plt.subplots(nrows=1, ncols=len(sequenceColNames), figsize=(3.5*len(sequenceColNames), 3), sharey=True, sharex=True)
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
                                                                                                  dropped, cdr_col_name))

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

        kmer_dict = kmers_hist_from_seq_series(cdrs_concat_series, k)
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
