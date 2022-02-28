### Base was copied from TCRdist2, then I added stuff.

"""
Contains the RefGeneSet class

Contains the TCRGene class

"""

# from .paths import path_to_db
from collections import OrderedDict
# from . import translation
import pandas as pd
import os.path as op

# path_to_db = 'C:/Users/liel-/Dropbox/PHD/Projects/TCR/Nature data/tcrdist_extras_v2/db/alphabeta_db.tsv'


class RefGeneSet:
    """
    RefGeneSet is a class for holding a set of reference TCR sequences

    Attributes
    ----------
    db_file : string
        specifies the name of the file in path_db with reference TCRS
        (e.g alphabeta_db.tsv)
    all_genes : OrderedDict
        contains library of all reference TCRs represented as a dictionary
        of TCRGene instances

    Methods
    -------
    generate_all_genes()
        generates the library of all reference TCRs represented as a dictionary
        of TCRGene instances

    Notes
    -----

    all_genes = RefGeneSet(db_file = "alphabeta_db.tsv").all_genes

    would be equivalent to previously used:

    from all_genes import all_genes


    Information is now callable as follows:
    attribute = 'cdrs'
    RefGeneSet.all_genes['mouse']['TRAV1*01'].__dict__[attribute]

    """
    def __init__(self, db_file):
        self.db_file = db_file
        self.all_genes = self.generate_all_genes()

    def generate_all_genes(self):
        """
        create the reference dataset that will be used throughout tcrdist
        see the TCRGene class below for details

        Parameters
        ----------
        db_file : string
            "alphabeta_db.tsv"

        """
        db_file = self.db_file
        assert op.exists(db_file)

        all_genes = OrderedDict()

        db_df = pd.read_csv(db_file, delimiter='\t')

        for rowi, row in db_df.iterrows():
            try:
                g = TCRGene( row )
            except:
                print(row)
                raise
            if g.organism not in all_genes:
                all_genes[g.organism] = OrderedDict() # map from id to TCR_Gene objects
            all_genes[g.organism][g.id] = g
        return(all_genes)


class TCRGene:
    """
    TCRGene is a class for holding information about a single TCR sequence representative

    Parameters
    ----------
    l : list


    Attributes
    ----------
    id : string

    organism : string
        human or mouse
    chain : string
        A or B !!!! this is not the same in TCRrep calls
    region : string
        V or J !!!! this is not the same in TCRrep calls
    protseq: string
        raw protein sequence
    nucseq : string

    alseq : string
        aligned protein sequence
    frame : int

    nucseq_offset : int

    cdr_columns_str : strings
        '28-39;57-66;82-88;106-111',
    cdr_columns : list
        [[28, 39], [57, 66], [82, 88], [106, 111]],
    cdrs : list
        list of cdrs present based on string in the input file l['cdrs']
    cdrs_extracted : list
        list of cdrs extracted based on cdr_column index with gaps retained
    cdrs_aligned : list
        list of cdrs with gaps retained
    cdrs_no_gaps: list
        list of cdrs with gaps removed


    Methods
    -------
    extract_aligned_cdrs()
        wraps _extract_aligned_cdrs
    extract_cdrs_without_gaps()
        wraps _extract_aligned_cdrs

    """
    cdrs_sep = ';'
    gap_character = '.'

    def __init__( self, l):
        self.id       = l['id']
        self.organism = l['organism']
        self.chain    = l['chain']
        self.region   = l['region']
        self.nucseq   = l['nucseq']
        self.alseq    = l['aligned_protseq']
        self.cdr_columns_str = l['cdr_columns']
        self.frame    = l['frame']
        self.nucseq_offset = self.frame - 1 ## 0, 1 or 2 (0-indexed for python)
        # self.protseq = translation.get_translation( self.nucseq, self.frame )[0]

        if pd.isnull(l['cdrs']):
            self.cdrs = []
            self.cdr_columns = []
        else:
            self.cdrs = l['cdrs'].split(self.cdrs_sep)
            ## these are still 1-indexed !!!!!!!!!!!!!!
            self.cdr_columns = [ list(map( int, x.split('-'))) for x in l['cdr_columns'].split(self.cdrs_sep) ]


            self.cdrs_extracted = [ self.alseq[ x[0]-1 : x[1] ] for x in self.cdr_columns ]
            self.cdrs_aligned   = self.extract_aligned_cdrs()
            self.cdrs_no_gaps   = self.extract_cdrs_without_gaps()

        # Defensive Assert Statements
        assert self.frame in [1, 2, 3]
        if self.cdrs:
            assert self.cdrs == self.cdrs_extracted # confirm that cdrs in the input file match expectation
            assert self.cdrs_aligned == self.cdrs_extracted

    def extract_aligned_cdrs(self):
        return _extract_aligned_cdrs(self.alseq, self.cdr_columns_str)

    def extract_cdrs_without_gaps(self):
        return _extract_cdrs_without_gaps(self.alseq, self.cdr_columns_str)


def _extract_aligned_cdrs(aligned_protseqs, cdr_columns):
    """

    from db use the aligned_protseq and cdr_column to return the aligned cdrs with gaps preserved


    Parameters
    ----------
    aligned_protseqs : string
        example: 'GQGVEQ.P.AKLMSVEGTFARVNCTYSTSG......FNGLSWYQQREGQAPVFLSYVVL....DGLKDS.....GHFSTFLSRSN.GYSYLLLTELQIKDSASYLCAVR..'
    cdr_columns : string
        example: '28-39;57-66;82-88;106-111')

    Returns
    -------
    cdrs : list


    Examples
    --------
    >>> _extract_aligned_cdrs('GQGVEQ.P.DNLMSVEGTFARVNCTYSTSG......FNGLSWYQQREGHAPVFLSYVVL....DGLKDS.....GHFSTFLSRSN.GYSYLLLTELQIKDSASYLCAVR..',
 '28-39;57-66;82-88;106-111')
    ['TSG......FNG', 'VVL....DGL', 'SRSN.GY', 'CAVR..']
    """
    try:
        pos = cdr_columns.split(";")
    except AttributeError:
        return None
    pos2 = [list(map(int,x.split("-"))) for x in pos]
    cdrs = [aligned_protseqs[x[0]-1:x[1]] for x in pos2]
    return(cdrs)

def _extract_cdrs_without_gaps(aligned_protseqs, cdr_columns):
    """

    from db use the aligned_protseq and cdr_column to return the aligned cdrs with gaps removed


    Parameters
    ----------
    aligned_protseqs : string
        example: 'GQGVEQ.P.AKLMSVEGTFARVNCTYSTSG......FNGLSWYQQREGQAPVFLSYVVL....DGLKDS.....GHFSTFLSRSN.GYSYLLLTELQIKDSASYLCAVR..'
    cdr_columns : string
        example: '28-39;57-66;82-88;106-111')

    Returns
    -------
    cdrs : list


    Examples
    --------
    >>> _extract_cdrs_without_gaps('GQGVEQ.P.AKLMSVEGTFARVNCTYSTSG......FNGLSWYQQREGQAPVFLSYVVL....DGLKDS.....GHFSTFLSRSN.GYSYLLLTELQIKDSASYLCAVR..',
 '28-39;57-66;82-88;106-111')
    ['TSGFNG', 'VVLDGL', 'SRSNGY', 'CAVR']
    """
    try:
        pos = cdr_columns.split(";")
    except AttributeError:
        return None

    pos2 = [list(map(int,x.split("-"))) for x in pos]
    cdrs = [aligned_protseqs[x[0]-1:x[1]].replace(".","") for x in pos2]
    return(cdrs)


def generate_pbr_cdr(db_name= "alphabeta_db.tsv"):
    """
    From "alphabeta_db.tsv"  generates the equivalent of pb_cdr dictionary
    previously generated with:

    from tcrdist.cdr3s_human import pb_cdrs

    """
    pb_cdrs = d = {"mouse": OrderedDict(), "human": OrderedDict()}

    d = generate_dict_from_db(db_name = db_name)
    ids = [d['mouse'][k]["id"] for k in d['mouse'].keys()]
    aligned_protseqs = [d['mouse'][k]["aligned_protseq"] for k in d['mouse'].keys()]
    cdr_columns = [d['mouse'][k]["cdr_columns"] for k in d['mouse'].keys()]
    mouse_cdrs_inferred = list(map(lambda a,p: _extract_cdrs_without_gaps(a,p),
                                  aligned_protseqs[:], cdr_columns[:]))

    for i,id in enumerate(ids):
        if isinstance(mouse_cdrs_inferred[i], list):
            pb_cdrs['mouse'][id] = [(item, None, None) for item in mouse_cdrs_inferred[i]]

    ids = [d['human'][k]["id"] for k in d['human'].keys()]
    aligned_protseqs = [d['human'][k]["aligned_protseq"] for k in d['human'].keys()]
    cdr_columns = [d['human'][k]["cdr_columns"] for k in d['human'].keys()]
    human_cdrs_inferred = list(map(lambda a,p: _extract_cdrs_without_gaps(a,p),
                                  aligned_protseqs[:], cdr_columns[:]))

    for i,id in enumerate(ids):
        if isinstance(human_cdrs_inferred[i], list):
            pb_cdrs['human'][id] = [(item, None, None) for item in human_cdrs_inferred[i]]

    return(pb_cdrs)


def generate_dict_from_db(db_name):

    """
    generates a dictionary based on .tsv file name in the db folder

    Parameters
    ----------
    db_name : string

    Returns
    -------
    d : dictionary


    """
    db = open(db_name, 'r')
    header = db.readline()
    d = {"mouse": OrderedDict(), "human": OrderedDict()}
    for line in db:
        try:
            id,organism,chain,region,nucseq,frame,aligned_protseq,cdr_columns,cdrs = \
            line.strip().split('\t')
            d[organism][id] = {'id':id,
                     'organism':organism,
                     'chain':chain,
                     'region':region,
                     'nucseq':nucseq,
                     'frame':frame,
                     'aligned_protseq': aligned_protseq,
                     'cdr_columns': cdr_columns,
                     'cdrs':cdrs}
        except ValueError: # for entries of length 7
            id,organism,chain,region,nucseq,frame,aligned_protseq = \
            line.strip().split('\t')
            d[organism][id] = {'id':id,
                     'organism':organism,
                     'chain':chain,
                     'region':region,
                     'nucseq':nucseq,
                     'frame':frame,
                     'aligned_protseq': aligned_protseq,
                     'cdr_columns':None,
                     'cdrs':None}
    db.close()
    return(d)


def _extract_from_reference_db(d, organism, id, attr):
    return(d[organism][id][attr])

# LIEL functions start here.

def get_cdr_from_gene(gene_obj, cdr, raise_exception=True):
    """
    :param gene_obj: a TCRGene object from which the CDR should be extracted.
    :param cdr: string: CDR1, CDR2, CDR2.5 or CDR3
    :param raise_exception: raise an exception of TCRGene object contains no
                            cdrs_no_gaps, or whether it's a V gene with less than
                            4 cdrs, or a J gene without 1 cdr (!= 1).
    :return: CDR sequence.
    """
    cdr_names_dict_Vgene = {'cdr1': 0, 'cdr2': 1, 'cdr2.5': 2, 'pmhc':2, 'cdr3': 3}

    # gene_obj = all_genes[organism][gene]
    if gene_obj.region != 'D':
        if 'cdrs_no_gaps' not in gene_obj.__dict__: # no 'cdrs_no_gaps' in some gene objects
            message = 'get_cdr_from_gene: in gene {}, ' \
                       'no "cdrs_no_gaps"'.format(gene_obj.id)
            if raise_exception:
                raise Exception(message)
            else:
                print(message)

        else:
            cdrs = gene_obj.__dict__['cdrs_no_gaps']

            if gene_obj.region == 'V':
                if (len(cdrs)) < 4:
                    message = 'get_cdr_from_gene: in V gene {}, ' \
                              'there are less than 4 cdrs!'.format(gene_obj.id)
                    if raise_exception:
                        raise Exception(message)
                    else:
                        print(message)

                else:
                    return cdrs[cdr_names_dict_Vgene[cdr.lower()]]

            elif gene_obj.region == 'J':
                if (len(cdrs)) != 1:
                    message = 'get_cdr_from_gene: in J gene {}, there are ' \
                              '{} cdrs and not 1!'.format(len(cdrs), gene_obj.id)
                    if raise_exception:
                        raise Exception(message)
                    else:
                        print(message)

                else:
                    if cdr.lower() == 'cdr3': # this is the only CDR in J jenes
                        return cdrs[0]

    return None # if got here - could not return the required CDR.

def get_genes_with_cdr_motif(all_genes, organism, chain, cdr, motif):
    """

    :param all_genes: a RefGeneSet.all_genes object containing all genes
                      user wished to search.
    :param organism: the required organism to search all_genes
    :param chain: string: 'A' or 'B' or 'a' or 'b'. the required chain to search all_genes
    :param cdr: string: CDR1, CDR2, CDR2.5 or CDR3 - which CDR the user
                        wished to search the motif in
    :param motif: the motif to search
    :return: all genes in the organism, of the required TCR chain,
             which contain the motif in the required CDR.
             format: a dictionary with gene name as key, and TCRGene object as value.
    """
    genes_org = all_genes[organism]

    genes_with_motif = dict()
    for gene_name in genes_org:
        gene = genes_org[gene_name]
        if gene.chain == chain.upper():
            cdr_seq = get_cdr_from_gene(gene, cdr, raise_exception=False)
            if cdr_seq is not None:
                if motif.upper() in cdr_seq:
                    genes_with_motif[gene_name] = gene

    return genes_with_motif


def get_cdrs_df_from_geneset_dict(all_genes, organism='both', add_alleles_info=True):
    """
    Get dataframe with info for each gene in the all_genes dict:
    'name', 'organism', 'chain', 'region' (for all genes)
    'CDR3' (V and J genes)
    'CDR1', 'CDR2', 'CDR2.5' (V genes only)

    :param all_genes: dict TCRGeneTools.RefGeneSet(input_file).all_genes
    :param organism: 'human', 'mouse' or 'both'
    :param add_alleles_info: If TRUE, counts the number of alleles from each gene,
                            for each organism separately,
                            and writes it up in a column named 'num_alleles'.
                            Also, for V genes adds a column indicating if cdr1, 2 and 2.5
                            in all alleles are identical.
    :return: df
    """
    if organism not in ['human', 'mouse', 'both']:
        raise Exception('organism must be human, mouse or both')
    if organism == 'both':
        organisms = ['human', 'mouse']
    else:
        organisms = [organism]

    gdf = pd.DataFrame(columns=['name', 'organism', 'chain', 'region',
                                'CDR1', 'CDR2', 'CDR2.5', 'CDR3'])

    cdr_names = ['CDR1', 'CDR2', 'CDR2.5', 'CDR3']

    for organism in organisms:
        for gene in all_genes[organism]:
            gene_obj = all_genes[organism][gene]
            gene_info = {'name': gene, 'organism': organism,
                         'chain': gene_obj.chain, 'region': gene_obj.region}

            if gene_obj.region == 'J':
                gene_info['CDR3'] = get_cdr_from_gene(gene_obj,
                                            'CDR3', raise_exception=False)
            elif gene_obj.region == 'V':
                for cdr in cdr_names:
                    gene_info[cdr] = get_cdr_from_gene(gene_obj,
                                            cdr, raise_exception=False)

            gdf.loc[gdf.shape[0]] = pd.Series(gene_info)

    if add_alleles_info:
        gdf = cdrs_df_add_alleles_info(gdf)

    return gdf


def get_cdrs_df_from_db_path(input_file, organism='both', add_alleles_info=True):
    """
    Get dataframe with info for each gene in the all_genes dict
    created from db file in path:
    (for example 'alphabeta_gammadelta_db2_tcrdist3.txt')
    'name', 'organism', 'chain', 'region' (for all genes)
    'CDR3' (V and J genes)
    'CDR1', 'CDR2', 'CDR2.5' (V genes only)

    :param db path
    :param organism: 'human', 'mouse' or 'both'
    :param add_alleles_info: If TRUE, counts the number of alleles from each gene,
                        for each organism separately,
                        and writes it up in a column named 'num_alleles'.
                        Also, for V genes adds a column indicating if cdr1, 2 and 2.5
                        in all alleles are identical.
    :return: df
    """

    all_genes = RefGeneSet(input_file).all_genes
    gdf = get_cdrs_df_from_geneset_dict(all_genes, organism=organism,
                                        add_alleles_info=add_alleles_info)
    return gdf


def cdrs_df_add_alleles_info(cdrs_df):
    """
    Counts the number of alleles from each gene, for each organism separately,
    and writes it up in a new column named 'num_alleles'.
    Also, for V genes adds a column indicating if cdr1, 2 and 2.5
    in all alleles are identical.
    :param cdrs_df: df from function get_cdrs_df_from_db_path
                                or get_cdrs_df_from_geneset_dict
    :return: new df with new columns
    """
    gdf = cdrs_df.copy()

    gdf['name_no_allele'] = None
    gdf['num_alleles'] = None
    gdf['v_alleles_cdrs1-2.5_same'] = None

    for organism in gdf['organism'].unique():
        gdf_dict = gdf[gdf['organism'] == organism].to_dict(orient='index')

        for ind in gdf_dict:
            al_count = 0
            same = True  # assume all true. If one allele will not be identical, will be changed to false.

            gene_no_al = gdf_dict[ind]['name'].split('*')[0]
            is_v = gdf_dict[ind]['region'] == 'V'

            for ind2 in gdf_dict:
                gene2_no_al = gdf_dict[ind2]['name'].split('*')[0]

                if gene_no_al == gene2_no_al:  # same gene
                    al_count += 1

                    if is_v:  # compare cdrs other than cdr3
                        if gdf_dict[ind]['CDR1'] != gdf_dict[ind2]['CDR1'] or \
                                gdf_dict[ind]['CDR2'] != gdf_dict[ind2]['CDR2'] or \
                                gdf_dict[ind]['CDR2.5'] != gdf_dict[ind2]['CDR2.5']:
                            same = False  # if one of them isn't equal, set to false

            gdf.loc[ind, 'name_no_allele'] = gene_no_al
            gdf.loc[ind, 'num_alleles'] = al_count

            if is_v:
                gdf.loc[ind, 'v_alleles_cdrs1-2.5_same'] = same

    return gdf

# see 2021_03_25 Atwal data.py for example usage and no_allele structures