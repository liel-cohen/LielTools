import pandas as pd
import numpy as np
from functools import partial
from itertools import product
import sys
import keras
import copy as copy_module
import matplotlib.pyplot as plt
import warnings

if 'LielTools' in sys.modules:
    from LielTools import FileTools
    from LielTools import DataTools
    from LielTools import PlotTools
    from LielTools import MyLabelEncoder
else:
    import FileTools
    import DataTools
    import PlotTools
    import MyLabelEncoder


####### ------------------ seqTools - general ------------------#### <editor-fold>

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

# former seq_letters_to_other_letters or aaSeq2groupSeq
def seq_letters_to_other_letters(sequence, letterDict=allAA2_6AAG):
    """
    Convert a certain sequence of letters to other letters by a given dict.
    @param sequence: string
    @param letterDict: dictionary of letters to convert (key: original letter, value: new letter)
    @return: converted string
    """
    newSeq = ''
    for letter in range(len(sequence)):
        aa = sequence[letter]
        if (not aa == '-'):
            newSeq = newSeq + letterDict[aa]
        else:
            newSeq = newSeq + '-'
    return newSeq

# former series_seqs_letters_to_other_letters or aaSeqSeries2groupSeq
def series_seqs_letters_to_other_letters(seriesAA, letterDict=allAA2_6AAG):
    """
        Convert a pd.Series of sequences of letters to sequences of other letters by a given dict.
        @param seriesAA: pd.Series of strings
        @param letterDict: dictionary of letters to convert (key: original letter, value: new letter)
        @return: pd.Series of converted strings
        """
    newSeries = seriesAA.apply(partial(seq_letters_to_other_letters, letterDict=letterDict))
    return newSeries


def cut_edge_letters(string, start_or_end, num_letters=3):
    """
    Return num_letters from the start or end of string.
    @param string: query string
    @param start_or_end: 'start' or 'end'
    @param num_letters: int
    @return:
    """
    if start_or_end == 'start':
        return string[:num_letters]
    elif start_or_end == 'end':
        return string[-num_letters:]
    else:
        raise ValueError('cut_edge_letters: unknown start_or_end')

# former pad_str_series
def pad_str_series(series, extraPad=0):
    """
    Gets a pd.Series of strings.
    Returns the same series with strings padded with zeros, i.e.,
    zeros added to shorter strings to have all strings have the same length.
    Can also add extra padding to all strings (over the max length).
    @param series: pd.Series of strings
    @param extraPad: int. Amount of extra padding to add, such that the
                          new strings length will be maxLen+extraPad. Default 0
    @return: pd.Series of padded strings
    """
    maxLen = series.apply(len).max()

    def ljust0(string):
        return(string.ljust(maxLen+extraPad, '0'))

    seriesPadded = series.apply(ljust0)
    return(seriesPadded)


def mirror_string_series(series):
    """
    Gets a pd.Series of strings. Returns the series with mirrored strings (i.e., from end to start).
    Also adds '_mirror' to the series name.
    @param series: pd.Series of strings
    @return: pd.Series of mirrored strings
    """
    series_mirror = series.apply(lambda string: string[::-1])
    series_mirror.rename(series_mirror.name+'_mirror', inplace=True)
    return series_mirror


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


# former str_series_to_char_lists
def str_series_to_char_lists(series):
    ''' Gets a series of strings.
    Returns: 1. a list of char lists - each string is transformed into a list of chars.
             2. a list with all unique chars included in the series. '''

    strsList = list(series)

    newlist = list()
    alphabetSet = set()
    for string in strsList:
        newChars = list(string)
        newlist.append(newChars)
        alphabetSet.update(newChars)

    alphabet = list(alphabetSet)
    alphabet.sort()

    return [newlist, alphabet]


# </editor-fold>

####### ------------------ seqTools - kmers ------------------#### <editor-fold>

# former get_all_possible_kmers
def get_all_possible_kmers(k, letters):
    """
    Gets a list of letters and a number k, and returns
    all possible kmer (strings, motifs - sized k) combinations
    from all letters.
    @param k: int
    @param letters: a list of letters
    @return: A list of kmers.
    Example: get_all_possible_kmers(2, ['A', 'B', 'C'])
             returns ['AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC']
    """
    kmersList = []

    for i in product(*([letters] * k)):
        kmer = ''.join(i)
        kmersList.append(kmer)

    return(kmersList)

# former get_kmers_df_for_series
def get_kmers_df_for_series(seqSeries, k, kmersList, raise_error_if_no_kmer=True):
    """
    Gets a pd.Series of strings, and a kmers list.
    Creates a pd.DataFrame with counts of each kmer (in columns) in each string (rows).
    @param seqSeries: pd.Series of strings
    @param k: int. the size of kmers.
    @param kmersList: a list of strings. The kmers to count for each string.
    @param raise_error_if_no_kmer: boolean. When True, if encounters a kmer in a
                                   certain string that is not in the kmersList, will
                                   raise an error. If False, will only print a
                                   message to console. Default True.
    @return: a pd.DataFrame with counts of each kmer (in columns) in each string (rows)

    Warning: this is not a very efficient way to count kmers.
    It is advised to use function all_available_kmers_dict for large numbers of strings.
    """
    kmersCount = pd.DataFrame(0, index=seqSeries.index, columns=kmersList)

    for row in seqSeries.iteritems():
        seq = row[1]
        for start in range(len(seq) - k + 1):
            kmer = seq[start:(start + k)]
            try:
                kmersCount.loc[row[0], kmer] = kmersCount.loc[row[0], kmer] + 1
            except KeyError:
                if raise_error_if_no_kmer:
                    raise KeyError(f'kmer {kmer} not contained in the given kmersList')
                else:
                    print(f'kmer {kmer} not contained in the given kmersList')

    return kmersCount

# same as former kmers_hist_from_seq_series
def all_available_kmers_dict(series, k):
    """
    Gets a pd.Series of strings. Returns a dictionary with all
    kmers (motifs) contained in the strings,
    and the amount of appearances for each.
    @param series: pd.Series of strings
    @param k: int. the size of motifs to extract (kmers)
    @return: dict

    Example: all_available_kmers_dict(pd.Series(['ABC', 'ABD']), 2)
             returns {'AB': 2, 'BC': 1, 'BD': 1}
    """
    kmer_dict = {}
    for i in series.index:
        seq = series.loc[i]
        for start in range(len(seq) - k + 1):
            kmer = seq[start:start+k]
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1

    return kmer_dict


def get_kmers_counts_and_positions(series, k, indices=None,
                                   return_pd_objects=False):
    """
    Get the counts of kmer appearances in a pd.Series of strings,
    and also their counts by starting position in the string.

    :param series: pd.Series with sequences to be analyzed.
    :param k: size of k for analysis
    :param indices: the subset of indices (labels/bool series) from series
                    to be analyzed. If None, all indices will be analyzed.
                    Default None
    :param return_pd_objects: boolean. If True, will return kmers_counts as a pd.Series
                                       and kmers_pos_counts as a pd.Dataframe
                                       (rows=kmers, columns=positions).
                                       If False, will return them as a dict and a
                                       dict of dicts. Default False

    :return: 1. kmers_counts (dict or pd.Series):
                        counts of kmer appearances in all given strings.
             2. kmers_pos_counts (dict of dicts or pd.Dataframe):
                        counts of kmer appearances by starting position in the string
    """
    # prep params and data
    max_length = series.apply(len).max()

    if indices is None:
        indices = series.index

    seq_series = series.loc[indices]  # Series to be analyzed.

    # initialize dicts
    kmers_counts = {}
    kmers_pos_counts = {}
    # an empty positions dict (to be used to initialize each kmer entry in kmers_pos_counts)
    empty_pos_dict = {num: 0 for num in list(range(1, max_length - k + 2))}

    # loop over all sequences and extract kmers and their position
    for ind in seq_series.index:
        seq = seq_series.loc[ind]

        for start in range(len(seq) - k + 1):  # go over all positions in seq
            kmer = seq[start:start + k]  # extract k-mer in position

            # if needed, add an empty positions dict to kmers_pos_hist
            if kmer not in kmers_pos_counts:
                kmers_pos_counts[kmer] = empty_pos_dict.copy()

            kmers_counts[kmer] = kmers_counts.get(kmer, 0) + 1
            kmers_pos_counts[kmer][start + 1] += 1

    if return_pd_objects:
        kmers_counts = pd.Series(kmers_counts).sort_index()
        kmers_pos_counts = pd.DataFrame.from_dict(kmers_pos_counts, orient='index').sort_index()

    return kmers_counts, kmers_pos_counts


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


# </editor-fold>

####### ------------------ seqTools - Encoders and one-hot encoding ------------------#### <editor-fold>

def getChar2intEncoder(alphabetList, add0=True, sortAlphabet=True):
    ''' Get an encoder for the given characters set (sklearn.preprocessing.LabelEncoder) '''

    if '0' in alphabetList: # we want the '0' padder at the end of the encoder map (if amino acids - int 20)
        alphabetList.remove('0')

    le = MyLabelEncoder.MyLabelEncoder(labelsList=alphabetList, sort=sortAlphabet)
    if add0:
        le.addLabel('0')        # add '0' as the last (if amino acids - 20) int
    le_alphabet_mapping = le.getEncoderMappingDict()
    return [le, le_alphabet_mapping]


def charLists2intEncoding(charlists, charEncoder):
    ''' Gets lists of chars: each list is a sample - characters sequence (string as list of chars).
     Returns each char list as an int list.
     Organized as ndarray: each row is a sample, converted to ints '''
    # encode each charlist to ints
    integer_encoded = list()
    for item in charlists:
        integer_encoded.append(charEncoder.transform(item))

    return np.array(integer_encoded)


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
        stringSeries = pad_str_series(stringSeries.copy())

    # stringSeries to char lists
    charlists, alphabet = str_series_to_char_lists(stringSeries)

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
        return oneHot
    else:
        return [oneHot, charEncoder]


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
    return oneHot


def oneHot2strSeries(oneHot, encoder, padded=True):
    if padded:
        # add last column of '0' encoding
        oneHot = oneHot_add0encodingColumn(oneHot)

    stringList = list()
    for i in range(len(oneHot)):
        inverse = np.argmax(oneHot[i], axis=1)
        string = ''.join(encoder.inverse_transform(inverse))
        stringList.append(string)

    return pd.Series(stringList)


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

    return filter


# </editor-fold>



