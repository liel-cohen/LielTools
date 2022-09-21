import pandas as pd
import numpy as np
import sys

if 'LielTools' in sys.modules:
    from LielTools import DataTools
else:
    import DataTools

from collections import defaultdict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import Bio

####### ------------------ seqTools - fasta files ------------------#### <editor-fold>

def fasta_count_num_seqs(fasta_path):
    """
    Count the number of sequences in a fasta file using a for loop.
    @param fasta_path: str. Path to fasta file
    @return: int. Number of sequences in the file
    """
    input_handle = open(fasta_path, "r")
    count = 0
    for record in SeqIO.parse(input_handle, "fasta"):
        count += 1
    return count


def fasta_extract_seqs_to_list(fasta_path, k=np.inf):
    """
    Extract the first k sequences from a fasta file and return as a list of records.
    @param fasta_path: str. Path to fasta file
    @param k: int. Number of sequences to be retrieved from the file.
              Default is np.inf (infinity), i.e., extract all sequences
    @return: A list of sequence records.
    """
    seqs = list()
    input_handle = open(fasta_path, "r")
    count = 0
    for record in SeqIO.parse(input_handle, "fasta"):
        seqs.append(record)
        count += 1

        if count >= k:
            break

    return seqs


def write_seqs_to_fasta_file(sequences, fasta_path):
    """
    Writes a list of sequence records to a fasta file.
    @param sequences: List  of sequence records (biopython SeqRecord objects)
    @param fasta_path: Path of fasta file to write the sequences to
    @return: None
    """
    with open(fasta_path, "w") as output_handle:
        SeqIO.write(sequences, output_handle, "fasta")

def write_seqs_series_to_fasta(seq_series, fasta_path):
    """
    Writes a series of sequences to a fasta file. Each sequence in the fasta
    file will be named by its sequence series index.
    @param seq_series: pandas series of sequences
    @param fasta_path: Path of fasta file to write the sequences to
    @return: None
    """
    seq_list = []
    for i in seq_series.index:
        seq = seq_series.loc[i]
        seq_list.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq), id=i))

    write_seqs_to_fasta_file(seq_list, fasta_path)


def fasta_seq_length_hist(fasta_path, output_format='series', print_hist=False):
    """
    Create a histogram of the lengths of sequences from a given fasta file.

    @param fasta_path: str. Path to fasta file
    @param output_format: output format of the histogram - 'series' or 'dict'
    @param print_hist: if True, will print the histogram (as a Series sorted by lengths)
    @return: The histogram
    """
    len_hist = defaultdict(int)
    input_handle = open(fasta_path, "r")
    for record in SeqIO.parse(input_handle, "fasta"):
        len_hist[len(record.seq)] += 1

    len_hist_series = pd.Series(len_hist).sort_index()
    if print_hist:
        print(len_hist_series)

    if output_format=='series':
        return len_hist_series
    elif output_format=='dict':
        return len_hist
    else:
        raise ValueError(f'Unsupported output_format: {output_format}')


def fasta_unique_seqs(fasta_path, fasta_output_path=None, trim_last_char_if_invalid=False, invalid_chars=['*', 'X']):
    """
    Get the uniques set of sequences and their counts from a fasta file.
    Each sequence ID will be seq{i}_n={n}, i being the sequence index in
    the list of sequences (sorted by descending counts), n being the number
    of sequences in the original file in fasta_path, with the same sequence.

    @param fasta_path: str. Path to fasta file
    @param fasta_output_path: Path for saving the new file of unique sequences.
                              If None, sequences won't be saved to file. Default None.
    @param trim_last_char_if_invalid: If true, the last character of each sequence will be trimmed
                                      if it's an invalid character.
    @params invalid_chars: list. list of characters to be considered invalid. default ['*', 'X']
    @return: seq_hist (dict of sequences and their counts),
             seq_series (series of sequences and their counts, ordered by descending counts),
             seq_ordered_list (list of sequences, ordered by descending counts)
    """
    # get a histogram dict of unique sequences and their counts
    seq_hist = defaultdict(int)
    counter = 0
    input_handle = open(fasta_path, "r")
    for record in SeqIO.parse(input_handle, "fasta"):
        seq_hist[record.seq] += 1
        counter += 1

    # just make sure the histogram sum == num of sequences
    n = DataTools.sum_dict_vals(seq_hist)
    assert n == counter

    # print num sequences found out of n
    print(f'Out of {n:,} sequences, number of unique sequences found: {len(seq_hist):,}')

    # create a sorted series (by descending count)
    seq_series = pd.Series(seq_hist).sort_values(ascending=False)
    seq_series.index = seq_series.index.map(lambda x: ''.join(x))

    # make a sorted sequences list
    seq_ordered_list = []
    for i in range(len(seq_series.index)):
        seq = str(seq_series.index[i])
        if trim_last_char_if_invalid:
            if seq[-1] in invalid_chars:
                seq = seq[:-1]
        seq_id = f'seq{i + 1}_n-{seq_series.iloc[i]}'

        seq_ordered_list.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq), id=seq_id))

    # save sequences to a new fasta
    if fasta_output_path is not None:
        write_seqs_to_fasta_file(seq_ordered_list, fasta_output_path)
        print(f'Wrote unique sequences to output file: {fasta_output_path}')

    return seq_series, seq_ordered_list


def fasta_find_character(fasta_path, char='*', print_res=True):
    """
    Check appearances of a certain character in all sequences of a fasta file.
    Useful for checking if there are invalid '*' or 'X' characters.
    @param fasta_path: str. Path to fasta file
    @param char: character to find. Default '*'
    @param print_res: boolean. Whether to print the histograms. Default True
    @return: tuple:
        # int. count of sequences containing char
        # int. total number of sequences
        # dict. Histogram of index of first appearance in each string
        # dict. Histogram of the number of appearances in each string
    """
    first_appearance_hist = defaultdict(int)
    num_appearances_hist = defaultdict(int)
    counter = 0
    count_seqs_with_char = 0
    input_handle = open(fasta_path, "r")
    for record in SeqIO.parse(input_handle, "fasta"):
        seq = str(record.seq)
        if char in seq:
            count_seqs_with_char += 1
            first_appearance_hist[seq.index(char)] += 1
            num_appearances_hist[seq.count(char)] += 1
        counter += 1

    print(f'Found {count_seqs_with_char:,} sequences containing {char} out of {counter:,} total sequences.')
    if print_res:
        print(f'\nHistogram of index of first appearance of {char} in each string:')
        print(first_appearance_hist)
        print(f'\nHistogram of number of appearances of {char} in each string:')
        print(first_appearance_hist)

    return count_seqs_with_char, counter, first_appearance_hist, num_appearances_hist

def fasta_remove_seqs_longer_than_x(fasta_path, x, fasta_output_path=None):
    """
    Gets sequences from fasta file, removes sequences that are of length larger than x.
    (The order of the sequences that are kept is preserved)

    @param fasta_path: str. Path to fasta file
    @param x: int. Maximal length for a sequence to be kept.
    @param fasta_output_path: Path for saving the new file of sequences not longer than x.
                              If None, sequences won't be saved to file. Default None.
    @return: List of sequences not longer than x
    """
    counter = 0
    count_valid_seqs = 0
    sequences = []
    input_handle = open(fasta_path, "r")
    for record in SeqIO.parse(input_handle, "fasta"):
        if len(record.seq) <= x:
            sequences.append(record)
            count_valid_seqs += 1
        counter += 1

    print(f'Found and removed {(counter-count_valid_seqs):,} sequences longer than {x} out of {counter:,} total sequences.')

    if fasta_output_path is not None:
        write_seqs_to_fasta_file(sequences, fasta_output_path)
        print(f'Wrote sequences to output file: {fasta_output_path}')

    return sequences

def trim_seqs_by_positions(start_pos, end_pos, fasta_path=None, seqs=None, including_end_pos=True, fasta_output_path=None):
    """
    Function gets a fasta file (in path) or a list of sequences (Biopython record objects)
    and trims each of them by the given positions.
    @param start_pos: starting position of the subsequence to return (inclusive).
                      For example, for sequence 'abcdef', if you wish to get 'bcd',
                      the start_pos should be 2.
    @param end_pos: the end position of the subsequence to return.
                    if including_end_pos=True, the end position will be included.
                    For example, for sequence 'abcdef', if you wish to get 'bcd',
                    the end_pos should be 4 (if including_end_pos=True.
                    If including_end_pos=False, end_pos should be 5)
    @param fasta_path: path of fasta file with sequences to trim
    @param seqs: list of Biopython record objects with sequences to trim
    @param including_end_pos: boolean. Whether to include end_pos or not.
    @param fasta_output_path: str. Path to save a fasta file with the trimmed
                              sequences. Default None (no file is saved)
    @return: The trimmed sequences list (Biopython record objects)
    """
    start_ind = start_pos - 1
    if including_end_pos:
        end_ind = end_pos
    else:
        end_ind = end_pos - 1

    if (fasta_path is None and seqs is None) or (fasta_path is not None and seqs is not None):
        raise ValueError('Must get either fasta_path or seqs')

    sequences = []

    if fasta_path is not None:
        input_handle = open(fasta_path, "r")
        iterate_over = SeqIO.parse(input_handle, "fasta")
    elif seqs is not None:
        iterate_over = seqs

    for record in iterate_over:
        trimmed_seq = record.seq[start_ind:end_ind]
        new_record = SeqRecord(id=record.id, seq=trimmed_seq, name=record.name, description=record.description, dbxrefs=record.dbxrefs)
        sequences.append(new_record)

    if fasta_output_path is not None:
        write_seqs_to_fasta_file(sequences, fasta_output_path)
        print(f'Wrote sequences to output file: {fasta_output_path}')

    return sequences


def replace_subseq_in_range(start_pos, end_pos, fasta_path=None, seqs=None, fasta_output_path=None, char_replace='X'):
    """
    Function gets a fasta file (in path) or a list of sequences (Biopython record objects).
    For each sequence, the subsequence starting at start_pos and ending in end_pos (inclusive)
    will be replaced by characters defined by char_replace.
    For example, for sequence 'abcdefg', start_pos=2, end_pos=4, char_replace='X'
    the new sequence will be 'aXXXefg'

    @param start_pos: starting position of the subsequence to be replaced (inclusive).
    @param end_pos: the end position of the subsequence to be replaced (inclusive).
    @param fasta_path: path of fasta file with sequences to edit
    @param seqs: list of Biopython record objects with sequences to edit
    @param fasta_output_path: str. Path to save a fasta file with the new
                              sequences. Default None (no file is saved)
    @return: The new sequences list (Biopython record objects)
    """

    if (fasta_path is None and seqs is None) or (fasta_path is not None and seqs is not None):
        raise ValueError('Must get either fasta_path or seqs')

    sequences = []

    if fasta_path is not None:
        input_handle = open(fasta_path, "r")
        iterate_over = SeqIO.parse(input_handle, "fasta")
    elif seqs is not None:
        iterate_over = seqs

    start_ind = start_pos - 1
    seq_replace = char_replace * (end_pos-start_pos+1)

    for record in iterate_over:
        seq_before = record.seq[:start_ind] # original seq *before* the subseq to replace
        seq_after = record.seq[end_pos:] # original seq *after* the subseq to replace
        new_seq = seq_before + seq_replace + seq_after
        new_record = SeqRecord(id=record.id, seq=new_seq, name=record.name, description=record.description, dbxrefs=record.dbxrefs)
        sequences.append(new_record)

    if fasta_output_path is not None:
        write_seqs_to_fasta_file(sequences, fasta_output_path)
        print(f'Wrote unique sequences to output file: {fasta_output_path}')

    return sequences

# </editor-fold>
