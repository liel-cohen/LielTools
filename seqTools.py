import pandas as pd
import numpy as np
from functools import partial
from itertools import product
import sys
# import keras # inserted to relevant function instead here: strSeries2_1H
import copy as copy_module
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
import math
import matplotlib

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

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import Bio.Data.CodonTable as CodonTable
from Bio import Align
from Bio.Align import substitution_matrices

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

# </editor-fold>

####### ---------- Reduced alphabet dictionaries (Bio.Alphabet.Reduced (RIP)) -------#### <editor-fold>

''' Reduced alphabet dictionaries from the former Bio.Alphabet.Reduced module (RIP) :
https://biopython.org/docs/1.75/api/Bio.Alphabet.Reduced.html

The following Alphabet classes are available:

@ Murphy15: Maps 20 amino acids to 15, use murphy_15_tab for conversion,
ambiguous letters: L: LVIM, F: FY, K: KR

@ Murphy10: Maps 20 amino acids to 10, use murphy_10_tab for conversion,
ambiguous letters: L: LVIM, S: ST, F: FYW, E: EDNQ, K: KR

@ Murphy8: Maps 20 amino acids to 8, use murphy_8_tab for conversion,
ambiguous letters: L: LVIMC, A: AG, S: ST, F: FYW, E: EDNQ, K: KR

@ Murphy4: Maps 20 amino acids to 4, use murphy_4_tab for conversion,
ambiguous letters: L: LVIMC, A: AGSTP, F: FYW, E: EDNQKRH

@ HPModel: Groups amino acids as polar (hydrophilic) or hydrophobic (non-polar), 
use hp_model_tab for conversion, P: AGTSNQDEHRKP, H: CMFILVWY

@ PC5: Amino acids grouped according to 5 physico-chemical properties,
use pc_5_table for conversion, A (Aliphatic): IVL, R (aRomatic): FYWH, C (Charged): KRDE, T (Tiny): GACS, D (Diverse): TMQNP


The Murphy tables are from here:

Murphy L.R., Wallqvist A, Levy RM. (2000) 
Simplified amino acid alphabets for protein fold recognition and implications for folding. 
Protein Eng. 13(3):149-152

*dicts taken from their original code (biopython-1.76):
https://github.com/2021-environmental-bioinformatics/clean_room/blob/a38c8168ccf777aa50dbb6e6bb2ee1b58e82bdb2/code/biopython-1.76-py3.8-linux-x86_64.egg/Bio/Alphabet/Reduced.py

'''

reduced_alphabet = {'murphy_15': {"L": "L",
                                 "V": "L",
                                 "I": "L",
                                 "M": "L",
                                 "C": "C",
                                 "A": "A",
                                 "G": "G",
                                 "S": "S",
                                 "T": "T",
                                 "P": "P",
                                 "F": "F",
                                 "Y": "F",
                                 "W": "W",
                                 "E": "E",
                                 "D": "D",
                                 "N": "N",
                                 "Q": "Q",
                                 "K": "K",
                                 "R": "K",
                                 "H": "H"},

                    'murphy_10': {"L": "L",
                                 "V": "L",
                                 "I": "L",
                                 "M": "L",
                                 "C": "C",
                                 "A": "A",
                                 "G": "G",
                                 "S": "S",
                                 "T": "S",
                                 "P": "P",
                                 "F": "F",
                                 "Y": "F",
                                 "W": "F",
                                 "E": "E",
                                 "D": "E",
                                 "N": "E",
                                 "Q": "E",
                                 "K": "K",
                                 "R": "K",
                                 "H": "H"},

                    'murphy_8': {"L": "L",
                                "V": "L",
                                "I": "L",
                                "M": "L",
                                "C": "L",
                                "A": "A",
                                "G": "A",
                                "S": "S",
                                "T": "S",
                                "P": "P",
                                "F": "F",
                                "Y": "F",
                                "W": "F",
                                "E": "E",
                                "D": "E",
                                "N": "E",
                                "Q": "E",
                                "K": "K",
                                "R": "K",
                                "H": "H"},

                    'murphy_4': {"L": "L",
                                "V": "L",
                                "I": "L",
                                "M": "L",
                                "C": "L",
                                "A": "A",
                                "G": "A",
                                "S": "A",
                                "T": "A",
                                "P": "A",
                                "F": "F",
                                "Y": "F",
                                "W": "F",
                                "E": "E",
                                "D": "E",
                                "N": "E",
                                "Q": "E",
                                "K": "E",
                                "R": "E",
                                "H": "E"},

                    'hp_model': {"A": "P",  # Hydrophilic
                                "G": "P",
                                "T": "P",
                                "S": "P",
                                "N": "P",
                                "Q": "P",
                                "D": "P",
                                "E": "P",
                                "H": "P",
                                "R": "P",
                                "K": "P",
                                "P": "P",
                                "C": "H",  # Hydrophobic
                                "M": "H",
                                "F": "H",
                                "I": "H",
                                "L": "H",
                                "V": "H",
                                "W": "H",
                                "Y": "H"},

                    'pc_5': {"I": "A",  # Aliphatic
                              "V": "A",
                              "L": "A",
                              "F": "R",  # Aromatic
                              "Y": "R",
                              "W": "R",
                              "H": "R",
                              "K": "C",  # Charged
                              "R": "C",
                              "D": "C",
                              "E": "C",
                              "G": "T",  # Tiny
                              "A": "T",
                              "C": "T",
                              "S": "T",
                              "T": "D",  # Diverse
                              "M": "D",
                              "Q": "D",
                              "N": "D",
                              "P": "D"}
                    }


# </editor-fold>

####### ---------- General -------#### <editor-fold>

def seq_series_to_char_df(series):
    """
    Transform a series of strings into a dataframe with a column
    for each character.

    Example:
        seq_series_to_char_df(pd.Series(['ABCDEF-1', 'ABC-D-G2']))

        Out:
           0  1  2  3  4  5  6  7
        0  A  B  C  D  E  F  -  1
        1  A  B  C  -  D  -  G  2

    @param series: pd.Series
    @return: pd.DataFrame

    """
    if len(series.map(len).value_counts()) > 1:
        warnings.warn('Series sequences have different lengths. Missing characters will be stored in dataframe as None')
    return pd.DataFrame(series.apply(list).tolist(), index=series.index)

def concat_df_string_cols_to_single_col(df, cols=None):
    """
    Concatenate multiple column values into a single dataframe (for each row).
    Example:
        df_examp = pd.DataFrame([['AB', 'CD', 'EF'], ['11', '22', '33']], columns=['a', 'b', 'c'])
        df_examp
        Out[1]:
                a   b   c
            0  AB  CD  EF
            1  11  22  33

        concat_df_string_cols_to_single_col(df_examp, ['a', 'b'])
        Out[2]:
            0    ABCD
            1    1122

    @param df: pandas dataframe
    @param cols: a list with the names of columns to concatenate. If None, concatenate all df columns.
    @return: pandas series
    """
    if cols is None:
        cols = df.columns
    return df[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


def remove_gapped_position_from_seqs(series, gap_char='-'):
    """
    Gets a series of sequences of the same length.
    Checks each position for gap_char. If all sequences have the gap_char
    at a certain position, remove that position from all sequences.
    Example:
        series = pd.Series(['ARDFYE------GSFDI', 'ARGEGSP----GNWFDP', 'ARDFYE------DPQWF'])
        series
            Out[1]:
                0    ARDFYE------GSFDI
                1    ARGEGSP----GNWFDP
                2    ARDFYE------DPQWF
                dtype: object
        remove_gapped_position_from_seqs(series, gap_char='-')
            Out[2]:
                0    ARDFYE--GSFDI
                1    ARGEGSPGNWFDP
                2    ARDFYE--DPQWF
                dtype: object

    @param series: pd.Series
    @param gap_char: the character considered as "gap"
    @return: pd.Series
    """
    # separate sequences to chars
    series_split = seq_series_to_char_df(series)
    # drop cols that only contain the gap_char
    cols_to_drop = (series_split == gap_char).sum(axis=0) == series_split.shape[0]
    series_split = series_split.drop(columns=[ind for ind in cols_to_drop.index if cols_to_drop.loc[ind]==True])
    # reconcatenate chars to string
    series_shorter = concat_df_string_cols_to_single_col(series_split)
    return series_shorter

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


def positions_dict_from_seq(seq, starting_pos=None, ending_pos=None):
    """
    Get a dictionary with positions (key) and letters (value) from a string sequence.
    User can specify a start and end position, and only position between them (inclusive)
    will be added.

    Examples:
        In: positions_dict_from_seq('ABCDE')
        Out: {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

        In: positions_dict_from_seq('ABCDE', starting_pos=2, ending_pos=4)
        Out: {2: 'B', 3: 'C', 4: 'D'}

    @param seq: string
    @param starting_pos: first position to add to dictionary. If None, dictionary starts from the first position.
    @param ending_pos: last position to add to dictionary. If None, dictionary ends at the last position.
    @return: dictionary
    """
    if starting_pos is None:
        starting_pos = 1
    if ending_pos is None:
        ending_pos = len(seq)

    assert (starting_pos >= 1) and (starting_pos <= len(seq))
    assert (ending_pos >= 1) and (ending_pos <= len(seq))
    assert ending_pos >= starting_pos

    seq_dict = {}
    for index in range(len(seq)):
        pos = index + 1
        if (pos >= starting_pos) and (pos <= ending_pos):
            seq_dict[pos] = seq[index]

    return seq_dict

def codon_dict_by_pos_from_codon_seq(seq, start_from_number=1):
    """
    Gets a RNA/DNA sequence of length n (i.e., n/3 codons). Returns a dictionary of position
    in the protein (key, from 1 to n/3) and codon (value).
    Example:
        codon_dict_by_pos_from_codon_seq('TTTACGGGG', start_from_number=1)
        output: {1: 'TTT', 2: 'ACG', 3: 'GGG'}

        codon_dict_by_pos_from_codon_seq('TTTACGGGG', start_from_number=10)
        output: {10: 'TTT', 11: 'ACG', 12: 'GGG'}

    @param seq: string. RNA/DNA sequence
    @param start_from_number: int. Number to start from when numbering the codons. Default 1.
    @return: dictionary
    """
    n = len(seq)
    assert n % 3 == 0, 'seq length should divide by 3 with no remainder'

    codon_dict = {}
    for i in range(0, n - n % 3, 3):
        codon = seq[i : i + 3]
        pos = math.ceil(i / 3) + start_from_number
        codon_dict[pos] = codon

    assert len(codon_dict.keys()) == n/3

    return codon_dict


def translate_codon_from_table(codon, codon_table=CodonTable.standard_dna_table, stop_letter='*'):
    """
    Gets a codon (DNA or RNA 3 letter sequence) and returns an amino acid (1 letter).
    @param codon: string. (DNA or RNA 3 letter sequence)
    @param codon_table: a codon table from Bio.Data.CodonTable module. default: CodonTable.standard_dna_table
    @param stop_letter: letter used to represent a stop codon. default: '*'
    @return: string - single letter or stop_letter
    """
    try:
        aa = codon_table.forward_table[codon]
    except:
        if codon in codon_table.stop_codons:
            aa = stop_letter
        else:
            raise Exception(f'Could not find codon {codon} in forward_table or stop_codons list. Please revise.')

    return aa


def get_codon_distances_to_target_aa(orig_codon, target_aa, codon_table=CodonTable.standard_dna_table):
    """
    Gets a codon, and a target amino acid letter. Checks the hamming distances between the
    codon, and the codons translating to the target amino acid. Returns a dictionary
    with hamming distances as keys (between 1 to 3), and codons with this distance as values.
    Example:
        get_codon_distances_to_target_aa('TTT', 'L')
        Output: {1: ['TTA', 'TTG', 'CTT'],
                 2: ['CTC', 'CTA', 'CTG']}

            * 'L' is translated by codons ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG']
            * 'TTT' codon translates to 'F'. So, to mutate 'F' to 'L', the minimal number
              of nucleotide substitutions needed is 1 (to 'TTA', 'TTG' or 'CTT').

    @param orig_codon: string. (DNA or RNA 3 letter sequence)
    @param target_aa: string. single letter representing an amino acid
    @param codon_table: a codon table from Bio.Data.CodonTable module. default: CodonTable.standard_dna_table
    @return: dict
    """
    aa_codon_dict = DataTools.reverse_dict_to_dict_w_value_list(codon_table.forward_table)

    hamming_dict = {}
    for target_codon in aa_codon_dict[target_aa]:
        hamm = hamming_dist(orig_codon, target_codon)
        hamming_dict.setdefault(hamm, []).append(target_codon)

    return hamming_dict

def hamming_dist(str1, str2):
    """ Gets 2 strings and returns their hamming distance.
    If the strings are not the same length, returns None.
    :param str1, str2: strings
    :return: int, or None.
    """
    if len(str1) != len(str2):
        return None

    hamming = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            hamming += 1

    return hamming

import Bio.SubsMat.MatrixInfo as matlist

def get_substitution_matrix(substitut_matrix_name='blosum62'):
    """
    Get a substitution matrix from the Bio.SubsMat.MatrixInfo module.
    @param substitut_matrix_name: The substitution matrix name, out of available matrices in Bio.SubsMat.MatrixInfo.available_matrices:
                    ['benner6', 'benner22', 'benner74', 'blosum100', 'blosum30', 'blosum35', 'blosum40', 'blosum45',
                     'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80',
                     'blosum85', 'blosum90', 'blosum95', 'feng', 'fitch', 'genetic', 'gonnet', 'grant', 'ident',
                     'johnson', 'levin', 'mclach', 'miyata', 'nwsgappep', 'pam120', 'pam180', 'pam250', 'pam30',
                     'pam300', 'pam60', 'pam90', 'rao', 'risler', 'structure']
    @return: dict
    """
    sub_mat = getattr(matlist, substitut_matrix_name)
    return sub_mat

def get_substitution_score_for_2AAs(aa1, aa2, substitut_matrix_name='blosum62'):
    """
    Get the substitution score for 2 amino acids, out of a substitution matrix from the Bio.SubsMat.MatrixInfo module.
    @param aa1: string (upper case single character)
    @param aa2: string (upper case single character)
    @param substitut_matrix_name: The substitution matrix name, out of available matrices in Bio.SubsMat.MatrixInfo.available_matrices:
                    ['benner6', 'benner22', 'benner74', 'blosum100', 'blosum30', 'blosum35', 'blosum40', 'blosum45',
                     'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80',
                     'blosum85', 'blosum90', 'blosum95', 'feng', 'fitch', 'genetic', 'gonnet', 'grant', 'ident',
                     'johnson', 'levin', 'mclach', 'miyata', 'nwsgappep', 'pam120', 'pam180', 'pam250', 'pam30',
                     'pam300', 'pam60', 'pam90', 'rao', 'risler', 'structure']
    @return: numeric score
    """
    sub_mat = get_substitution_matrix(substitut_matrix_name=substitut_matrix_name)
    try:
        score = sub_mat[(aa1, aa2)]
    except KeyError:
        try:
            score = sub_mat[(aa2, aa1)]
        except:
            raise Exception(f'{aa1} and {aa2} might not exist in Bio.SubsMat.MatrixInfo.{substitut_matrix_name}. Please check.')

    return score

def get_new_aa_f_mut_str(mut):
    """
    Gets a mutation string in format original AA, position, new AA.
    For example: E484K (original AA - E, position - 484, new AA - K)
    Returns the new AA (single character string). (In the example: K)
    @param mut: string
    @return: string
    """
    return mut[-1]

def get_orig_aa_f_mut_str(mut):
    """
    Gets a mutation string in format: original AA, position, new AA.
    For example: E484K (original AA - E, position - 484, new AA - K)
    Returns the original AA (single character string). (In the example: E)
    @param mut: string
    @return: string
    """
    return mut[0]

def get_pos_f_mut_str(mut):
    """
    Gets a mutation string in format: original AA, position, new AA.
    For example: E484K (original AA - E, position - 484, new AA - K)
    Returns the position as an int. (In the example: 484)
    @param mut: string
    @return: int
    """
    return int(mut[1:-1])

def seq_insert_single_mutation(seq, mut_position, mut_new_char):
    """
    Mutate a single character in a given sequence, in a specific given position, into a new given character.
    For example:
        Input:  aa_seq_insert_mutation('ABCDEFG', 3, 'X')
        Output: 'ABXDEFG'

    @param seq: str. sequence to mutate
    @param mut_position: int. a position to mutate within the sequence
    @param mut_new_char: str. a single character
    @return: str. mutated seq
    """
    mutated_seq = ''
    mutated_seq += seq[:mut_position-1]
    mutated_seq += mut_new_char
    mutated_seq += seq[mut_position:]
    return mutated_seq

def check_2seqs_diff(seq1, seq2, print_res=True):
    """ Gets 2 strings, compares them, returns and prints their differences.
    If the strings are not the same length, returns None.
    Differences dictionary format:
        {position: (character_from_seq1, character_from_seq2)}

    Example:
        Input:  check_2seqs_diff('AAAAA', 'AaAXB', print_res=True)
        Output: {2: ('A', 'a'), 4: ('A', 'X'), 5: ('A', 'B')}

    :param seq1, seq2: strings
    :return: dict, or None.
    """
    if len(seq1) != len(seq2):
        if print_res:
            print(f'Differences dict: None. Sequences have different lengths!')
        return None

    diff_dict = dict()
    hamming = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            hamming += 1
            diff_dict[i+1] = (seq1[i], seq2[i])

    if print_res:
        if len(diff_dict) == 0:
            print(f'Differences dict: {diff_dict}. Sequences are identical!')
        else:
            print(f'Differences dict: {diff_dict}')

    return diff_dict

# </editor-fold>

####### ---------- Alignment -------#### <editor-fold>

''' for info on biopython's alignment module see:
https://biopython.org/docs/1.76/api/Bio.pairwise2.html?highlight=globalms
https://biopython.org/docs/1.76/api/Bio.Align.html
'''

def align_seqs_series_to_longest_seq(seqs_series=None, seqs_list=None, aligner=None, print_alignment=False):
    """
    Align sequences to the longest sequence, by pairwise aligning each sequence
    to the longest sequence separately, using Biopython's aligner with high
    penalties for opening gaps in the query sequence (longest sequence).
    Can get a pd.Series of sequences *or* a list of sequences. Returnes a pandas
    dataframe with original and gapped sequences.

    @param: seqs_series: a pd.Series object with sequences. Default None (can get seqs_series *or* seqs_list)
    @param: seqs_list: a list of sequences. Default None (can get seqs_series *or* seqs_list)
    @param: aligner: a biopython aligner object to use. If not given, will automotically
                     create an aligner with high penalties for opening gaps in the query sequence.
    return: (1) a pandas dataframe with 2 columns: original sequences, aligned sequences.
            ordered by original sequences order.
            (2) the longest sequence used as a query sequence.
    """
    if seqs_series is None and seqs_list is None:
        raise Exception('Must get seqs_series or seqs_list. None of them were given.')
    if seqs_series is not None and seqs_list is not None:
        raise Exception('Must get seqs_series or seqs_list, not both')
    if seqs_series is not None:
        assert type(seqs_series) is pd.Series, f'Given seqs_series must be type pd.Series. Given seqs_series type: {type(seqs_series)}'
    if seqs_list is not None:
        assert type(seqs_list) is list, f'Given seqs_list must be type list. Given seqs_list type: {type(seqs_list)}'
        seqs_series = pd.Series(seqs_list, name='orig_seq')

    # get col names
    col_name = seqs_series.name
    aligned_col_name = col_name + ' aligned'

    # sort sequences by length and get the longest one
    seqs = seqs_series.copy()
    seqs = seqs.loc[seqs.map(len).sort_values(ascending=False).index]
    longest_seq = seqs.loc[seqs.index[0]]

    # prepare as dataframe to insert aligned sequences
    seqs = pd.DataFrame(seqs)
    seqs[aligned_col_name] = None
    seqs['longest seq aligned'] = None

    # prepare aligner object
    if aligner is None:
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = 0
        aligner.extend_gap_score = 0
        aligner.query_end_gap_score = -30
        aligner.query_internal_open_gap_score = -30
        aligner.query_internal_extend_gap_score = -30
        aligner.query_left_open_gap_score = -30
        aligner.query_left_extend_gap_score = -30
        aligner.query_right_open_gap_score = -30
        aligner.query_right_extend_gap_score = -30

    seqs.loc[seqs.index[0], aligned_col_name] = longest_seq
    seqs.loc[seqs.index[0], 'longest seq aligned'] = longest_seq

    for index in seqs.index[1:]:
        alignments = aligner.align(longest_seq, seqs.loc[index, col_name])
        alignment = list(alignments)[0]
        if print_alignment:
            print(alignment)

        formatted_alignment = alignment._format_pretty()
        seqA_aligned = formatted_alignment.split('\n')[0]
        seqB_aligned = formatted_alignment.split('\n')[-2]
        seqs.loc[index, 'longest seq aligned'] = seqA_aligned
        seqs.loc[index, aligned_col_name] = seqB_aligned

    num_length_as_longest = (seqs[aligned_col_name].map(len) == len(longest_seq)).sum()
    if num_length_as_longest != seqs.shape[0]:
        print(f'Out of {seqs.shape[0]} sequences, {num_length_as_longest} aligned sequences are of length equal to the longest sequence ({len(longest_seq)}).')
    else:
        print('All aligned sequences are of length equal to the longest sequence.')

    return seqs.loc[seqs_series.index], longest_seq

# </editor-fold>

####### ------------------ Logoplot ------------------#### <editor-fold>

def create_PSSM(seqs_series): # Elinor's function (repaired)
    ''' All sequences must have the same length! '''
    seps_len = len(seqs_series.iloc[0])
    amino_acid_list=["A","R","N","D","C","E","Q","G","H","I","L","K","M","P","S","T","W","Y","F","V"]
    counts={}#dictionary for creating df key is the aa and value is the frequency
    for letter in amino_acid_list:
        counts[letter] = [0.0] * seps_len
    for pos in range(seps_len):
        for aa in amino_acid_list:
            counts[aa][pos]=[pep[pos] for pep in seqs_series].count(str(aa))/len(seqs_series)
    df_counts = pd.DataFrame(counts)
    df_counts["position"]=[i for i in range(1,seps_len+1)]
    df_counts.set_index("position", inplace=True)
    return df_counts

def make_logo_plot(seq_series, fig_title='', figsize=(6, 3), ax=None):
    ''' All sequences must have the same length! '''

    if not 'logomaker' in sys.modules:
        import logomaker

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    seq_len = seq_series.map(len).max()
    # counts_df = logomaker.alignment_to_matrix(sequences=seq_series, to_type='counts', characters_to_ignore='-X') # Elinor says it has some bug
    counts_df = create_PSSM(seq_series)
    plt.grid(b=None)
    logo = logomaker.Logo(counts_df, color_scheme='weblogo_protein',
                   font_name='Arial Rounded', stack_order='small_on_top',
                   ax=ax)
    logo.ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(0, seq_len+1)))
    logo.ax.set_xticklabels(range(0, seq_len+1))
    logo.ax.set_xlabel('Position', fontsize=14)
    logo.ax.set_ylabel("Counts", labelpad=-1, fontsize=14)
    logo.ax.set_title(fig_title, fontsize=16)
    plt.grid(False)
    plt.tight_layout()

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

# def get_kmers_from_seqs_by_class(seq_series, class_series):
#     classes = list(class_series.unique())


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
    import keras

    return_charEncoder = charEncoder is None

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
        if padding and not gaps:
            alphabetCheck.add('0')
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

    if return_charEncoder:
        return [oneHot, charEncoder]
    else:
        return oneHot

def strSeries2_1H_sklearn(stringSeries, charEncoder=None, padding=True, gaps=False):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    ''' Gets a pd.Series of strings, with length n. Maximum string length l.
    Can also get an encoder that fits the strings alphabet. Alphabet size d.
    Returns an ndarray with the shape: (n, l, d).
    If padding=True, will add padding at the end of sequences so they all
    have the same length (max length in the series).
    If gaps=True, will consider '0's as gaps.
    If padding=True or gaps=True, delete the '0' character from the one hot matrix.
    If charEncoder is given, makes sure the charEncoder alphabet is identical
    to the the stringSeries alphabet'''

    return_charEncoder = charEncoder is None

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
        if padding and not gaps:
            alphabetCheck.add('0')
        if set(alphabetCheck) != alphabetFromMap:
            raise Exception('Warning: stringSeries letters arent identical to given charEncoder letters!')

    # use encoder to encode chars to ints (for each sample) - getting a 2d matrix
    integer_encoded = charLists2intEncoding(charlists, charEncoder)

    # encode to integers
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(integer_encoded.shape[0], integer_encoded.shape[1], 1)

    # encoding integers to one-hot - fit encoder
    integer_encoded_ints = np.array(list(encoderAlphabetMap.values())) # get list of ints that need to be encoded
    integer_encoded_ints = integer_encoded_ints.reshape(integer_encoded_ints.shape[0], 1)

    onehot_encoder.fit(integer_encoded_ints)

    # apply encoder to sequences (from integer encoded sequences)
    onehot_encoded_list = []
    for i in range(0, integer_encoded.shape[0]):
        onehot_encoded_list.append(onehot_encoder.transform(integer_encoded[i]))

    oneHot = np.array(onehot_encoded_list)

    if padding or gaps: # delete the 0 encoding column (if amino acids - column 20)
        paddingCol = len(encoderAlphabetMap)-1
        if encoderAlphabetMap['0'] != paddingCol:  # Check.
            raise Exception('Problem here! Encoder char zero must be mapped to the last int!')
        oneHot = oneHot[:, :, 0:(paddingCol)]

    if return_charEncoder:
        return [oneHot, charEncoder]
    else:
        return oneHot



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
    d1_letters = DataTools.list_remove_instance_first_match(d1_letters, '0')
    d2_letters = DataTools.list_remove_instance_first_match(d2_letters, '0')

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



