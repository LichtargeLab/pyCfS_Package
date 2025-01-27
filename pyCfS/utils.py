"""
Utility functions for the scripts
"""

from typing import Any
from collections.abc import Iterable
from scipy.stats import hypergeom
import pkg_resources
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csgraph, identity
from scipy.sparse.linalg import lgmres
import ast
from collections import Counter
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import io
from matplotlib_venn import venn2
from PIL import Image
from multiprocessing import Pool
from scipy.stats import ks_2samp
import seaborn as sns
import os

#region Statistical tests
def _hypergeo_overlap(background_size: int, query_genes:int, gs_genes:int, overlap:int) -> float:
    """
    Calculates the statistical significance (p-value) of the overlap between two gene sets
    using the hypergeometric distribution.

    This function is used to assess whether the observed overlap between a set of query genes
    and a gold standard gene set is statistically significant, given the total number of genes
    in the background set.

    Arguments:
    ---------
    background_size (int): The total number of genes in the background set.
    query_genes (int): The number of genes in the query set.
    gs_genes (int): The number of genes in the gold standard set.
    overlap (int): The number of genes that overlap between the query and gold standard sets.

    Returns:
    -------
    float: The p-value representing the statistical significance of the overlap. A lower p-value
           indicates a more significant overlap between the two gene sets.

    Note:
    -----
    The p-value is computed using the survival function ('sf') of the hypergeometric distribution
    from the SciPy stats library. The survival function provides the probability of observing
    an overlap as extreme as, or more extreme than, the observed overlap.
    """
    M = background_size
    N = query_genes
    n = gs_genes
    k = overlap
    pval = hypergeom.sf(k - 1, M, n, N)
    return pval
#endregion

#region Load shared background files
def _load_grch38_background(just_genes:bool = True) -> Any:
    """Return list of background genes from GRCh38

    Contains the following fields:
        chrom           object
        gene            str
        start           int64
        end             int64
    Returns
    -------
        list: List of genes annotated in GRCh38_v94
    """
    stream = pkg_resources.resource_stream(__name__, 'data/ENSEMBL-lite_GRCh38.v94.txt')
    df = pd.read_csv(stream, sep = '\t')
    if just_genes:
        return df['gene'].tolist()
    else:
        df.set_index('gene', inplace = True)
        return df

def _load_string(version:str) -> pd.DataFrame:
    """Return a dataframe of STRINGv11 protein-protein interactions

    Contains the following fields:
        node1               str
        node2               str
        neighborhood        int64
        fusion              int64
        cooccurrence        int64
        coexpression        int64
        experimental        int64
        database            int64
        textmining          int64
        combined_score      int64
    Returns
    -------
        pd.DataFrame of STRINGv11 protein interactions
    """
    if version not in ['v10.0', 'v11.0', 'v11.5', 'v12.0']:
        raise ValueError("Version must be 'v10.0', 'v11.0', 'v11.5', or 'v12.0'")
    stream = pkg_resources.resource_stream(__name__, f'data/9606.protein.links.detailed.{version}.feather')
    df_1 = pd.read_feather(stream)
    df_1 = df_1.dropna()
    return df_1

def _load_open_targets_mapping() -> pd.DataFrame:
    """
    Load the open targets mapping data from a file and return it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The mapping data with columns for ensgID and geneNames.
    """
    mapping_stream = pkg_resources.resource_stream(__name__, 'data/biomart_ensgID_geneNames_08162023.txt')
    mapping_df = pd.read_csv(mapping_stream, sep='\t')
    return mapping_df

def _load_reactome() -> list:
    """
    Load the Reactome pathways from the provided GMT file.

    Returns:
        list: A DataFrame containing the Reactome pathways.
    """
    reactomes_stream = pkg_resources.resource_stream(__name__, 'data/ReactomePathways_Mar2023.gmt')
    reactomes = reactomes_stream.readlines()
    reactomes = [x.decode('utf-8').strip('\n') for x in reactomes]
    reactomes = [x.split('\t') for x in reactomes]
    for x in reactomes:
        x.pop(1)
    return reactomes

def _get_open_targets_gene_mapping() -> dict:
    """
    Returns a tuple containing two dictionaries:
    1. A dictionary mapping Ensembl gene IDs to gene names

    Returns:
    -------
    tuple:
        A tuple containing two dictionaries:
        1. A dictionary mapping Ensembl gene IDs to gene names
    """
    mapping_df = _load_open_targets_mapping()
    mapping_dict = dict(zip(mapping_df['Gene stable ID'].tolist(), mapping_df['Gene name'].tolist()))
    return mapping_dict

def _load_pdb_et_mapping() -> pd.DataFrame:
    """
    Load the PDB to Ensembl gene ID mapping data from a file and return it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The mapping data with columns for PDB ID and ensgID.
    """
    mapping_stream = pkg_resources.resource_stream(__name__, 'data/PDB-AF_id_map.csv')
    mapping_df = pd.read_csv(mapping_stream, sep=',')
    return mapping_df
#endregion

#region General cleaning and formatting
def _validate_ea_thresh(ea_lower:int, ea_upper:int) -> None:
    """
    Validates the lower and upper thresholds for effect size.

    Args:
        ea_lower (int): The lower threshold for effect size.
        ea_upper (int): The upper threshold for effect size.

    Raises:
        ValueError: If the lower threshold is greater than the upper threshold,
                    or if the lower threshold is less than 0,
                    or if the upper threshold is greater than 100.
    """
    if ea_lower > ea_upper:
        raise ValueError("Lower threshold cannot be greater than the upper threshold.")
    if ea_lower < 0:
        raise ValueError("Lower threshold cannot be less than 0.")
    if ea_upper > 100:
        raise ValueError("Upper threshold cannot be greater than 100.")

def _validate_af_thresh(af_lower:float, af_upper:float) -> None:
    """
    Validates the allele frequency thresholds.

    Args:
        af_lower (float): The lower threshold for allele frequency.
        af_upper (float): The upper threshold for allele frequency.

    Raises:
        ValueError: If the lower threshold is greater than the upper threshold.
        ValueError: If the lower threshold is less than 0.
        ValueError: If the upper threshold is greater than 1.
    """
    if af_lower > af_upper:
        raise ValueError("Lower threshold cannot be greater than the upper threshold.")
    if af_lower < 0:
        raise ValueError("Lower threshold cannot be less than 0.")
    if af_upper > 1:
        raise ValueError("Upper threshold cannot be greater than 1.")

def _define_background_list(background_:Any, just_genes: bool = True, verbose:int = 0) -> (dict, str): # type: ignore
    """
    Defines the background list based on the input background parameter.

    Parameters:
    background_ (Any): The background parameter. It can be either 'reactome', 'ensembl', or a list of genes.
    just_genes (bool): A boolean flag indicating whether to include only genes in the background list. Default is True.

    Returns:
    tuple: A tuple containing the background dictionary and the background name.

    Raises:
    ValueError: If the background parameter is not 'reactome', 'ensembl', or a list of genes.

    """
    background_name = background_
    if isinstance(background_, str):
        if background_ == 'reactome':
            reactomes_bkgd = _load_reactome()
            reactomes_genes = [x[1:] for x in reactomes_bkgd]
            reactomes_genes = [item for sublist in reactomes_genes for item in sublist]
            reactomes_bkgd = list(set(reactomes_genes))
            reactomes_bkgd = [gene for gene in reactomes_bkgd if gene.isupper()]
            background_dict = {'reactome':reactomes_bkgd}
        elif background_ == 'ensembl':
            ensembl_bkgd = _load_grch38_background(just_genes)
            background_dict = {'ensembl':ensembl_bkgd}
        elif background_ == 'string_v12.0':
            _, string_v12_genes = _load_clean_string_network('v12.0', ['all'], 'low')
            background_dict = {'string_v12.0':string_v12_genes}
        elif background_ == 'string_v11.5':
            _, string_v11_5_genes = _load_clean_string_network('v11.5', ['all'], 'low')
            background_dict = {'string_v11.5':string_v11_5_genes}
        elif background_ == 'string_v11.0':
            _, string_v11_0_genes = _load_clean_string_network('v11.0', ['all'], 'low')
            background_dict = {'string_v11.0':string_v11_0_genes}
        elif background_ == 'string_v10.0':
            _, string_v10_0_genes = _load_clean_string_network('v10.0', ['all'], 'low')
            background_dict = {'string_v10.0':string_v10_0_genes}
        else:
            raise ValueError("Background must be either 'reactome', 'ensembl', 'string_v12.0', 'string_v11.5', 'string_v11.0', or 'string_v10.0'")
    # Custom background
    elif isinstance(background_, list):
        if verbose > 0:
            print(f"Custom background: {len(background_)} genes")
        background_dict = {'custom':background_}
        background_name = 'custom'
    else:
        raise ValueError("Background must be either 'reactome', 'ensembl' or list of genes")

    return background_dict, background_name

def _clean_genelists(lists: Iterable) -> list:
    """
    Normalize gene lists by ensuring all are lists and non-None.

    This function takes up to five gene sets and processes them to ensure that they are
    all list objects. `None` types are converted to empty lists, other iterables are
    converted to lists, and non-iterables are wrapped in a list.

    Parameters:
        lists (list): An iterable of lists that need to be converted to lists

    Returns:
        A list of 5 lists, corresponding to the input gene sets, sanitized toensure
        there are no None types and all elements are list objects.
    """
    clean_lists = []
    for x in lists:
        if x is None:
            # Convert None to an empty list
            clean_lists.append([])
        elif isinstance(x, Iterable) and not isinstance(x, str):
            # Convert iterables to a list, but exclude strings
            clean_lists.append(list(x))
        else:
            # Wrap non-iterables in a list
            clean_lists.append([x])
    clean_lists = [x for x in clean_lists if len(x) != 0]
    return clean_lists

def _format_scientific(value:float, threshold:float =9e-3) -> Any:
    """
    Formats a float in scientific notation if it's below a certain threshold.

    Args:
        value (float): The float value to be formatted.
        threshold (float): The threshold below which scientific notation is used. Defaults to 1e-4.

    Returns:
        str: The formatted float as a string.
    """
    if abs(value) < threshold:
        return f"{value:.1e}"
    else:
        return str(value)

def _fix_savepath(savepath:str) -> str:
    """
    Fixes the savepath by ensuring that it ends with a forward slash.

    Args:
        savepath (str): The savepath to be fixed.

    Returns:
        str: The fixed savepath.
    """
    if savepath[-1] != "/":
        savepath += "/"
    return savepath

def _get_avg_and_std_random_counts(random_counts_merged:dict) -> (dict, dict): # type: ignore
    """
    Calculates the average and standard deviation of the values in a dictionary of random counts.

    Args:
    random_counts_merged (dict): A dictionary containing the merged random counts.

    Returns:
    A tuple containing two dictionaries: the first dictionary contains the average values for each key in the input dictionary,
    and the second dictionary contains the standard deviation values for each key in the input dictionary.
    """
    avg_dict = {}
    std_dict = {}
    for k, v in random_counts_merged.items():
        avg_dict[k] = np.mean(v)
        std_dict[k] = np.std(v)
    return avg_dict, std_dict

def _merge_random_counts(random_counts_iterations:list) -> dict:
    """
    Merge the counts from multiple iterations of random sampling.

    Args:
    random_counts_iterations (list): A list of dictionaries, where each dictionary contains the counts for a single iteration of random sampling.

    Returns:
    dict: A dictionary containing the merged counts from all iterations of random sampling.
    """
    merged_counts = {}
    for i in random_counts_iterations:
        for k, v in i.items():
            if k in merged_counts:
                merged_counts[k].append(v)
            else:
                merged_counts[k] = [v]
    return merged_counts

def _filter_variants(variants: pd.DataFrame, gene: str, max_af:float, min_af:float, ea_lower:float, ea_upper:float, consequence: str) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Filters variants based on specified criteria.

    Args:
        variants (pd.DataFrame): DataFrame containing variant data.
        gene (str): Gene name to filter variants for.
        max_af (float): Maximum allele frequency threshold.
        ea_lower (float): Lower bound of effect allele frequency.
        ea_upper (float): Upper bound of effect allele frequency.

    Returns:
        tuple: A tuple containing two DataFrames - case_vars and cont_vars.
            case_vars: DataFrame containing filtered variants for case samples.
            cont_vars: DataFrame containing filtered variants for control samples.
    """
    if gene not in variants['gene'].unique():
        raise ValueError(f"Gene {gene} not found in the variant data.")
    case_vars = variants[
        (variants['gene'] == gene) &
        (variants['AF'] <= max_af) &
        (variants['AF'] >= min_af) &
        (variants['EA'] >= ea_lower) &
        (variants['EA'] <= ea_upper) &
        (variants['CaseControl'] == 1) &
        (variants['HGVSp'] != '.') &
        (variants['Consequence'].str.contains(consequence, na=False))
    ].reset_index(drop = True)
    cont_vars = variants[
        (variants['gene'] == gene) &
        (variants['AF'] <= max_af) &
        (variants['AF'] >= min_af) &
        (variants['EA'] >= ea_lower) &
        (variants['EA'] <= ea_upper) &
        (variants['CaseControl'] == 0) &
        (variants['HGVSp'] != '.') &
        (variants['Consequence'].str.contains(consequence, na=False))
    ].reset_index(drop = True)
    return case_vars, cont_vars

def _convert_amino_acids(df:pd.DataFrame, column_name:str="SUB") -> pd.DataFrame:
    """
    Convert three-letter amino acid codes to single-letter codes in a DataFrame column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be converted.
        column_name (str, optional): The name of the column to be converted. Defaults to "SUB".

    Returns:
        pandas.DataFrame: The DataFrame with the converted column.
    """
    # Create a dictionary to map three-letter codes to single-letter codes
    aa_codes = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V"
    }
    # Replace three-letter amino acid codes with single-letter codes
    df[column_name] = df[column_name].replace(aa_codes, regex=True)
    return df

def _clean_variant_formats(variants: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the variant formats in the given DataFrame.

    Args:
        variants (pd.DataFrame): The DataFrame containing the variants.

    Returns:
        pd.DataFrame: The cleaned DataFrame with updated variant formats.
    """
    new_variants = variants.copy()
    # Separate HGVSp into two columns
    new_variants[['ENSP', 'SUB']] = new_variants['HGVSp'].str.split(':', expand=True)
    # Remove leading 'p.' from SUB
    new_variants['SUB'] = new_variants['SUB'].str.replace('p.', '')
    new_variants = new_variants[~new_variants['SUB'].str.startswith("Ter")]
    # Convert three-letter amino acid codes to single-letter codes
    new_variants = _convert_amino_acids(new_variants)
    # Remove NA rows
    new_variants = new_variants.dropna()
    # Aggregate Zyg
    new_variants = new_variants.groupby(['SUB', 'EA']).agg(
        ENSP = ('ENSP', 'first'),
        SUB = ('SUB', 'first'),
        EA = ('EA', 'first'),
        AC = ('zyg', 'sum')
    )
    new_variants = new_variants.reset_index(drop=True)
    return new_variants

def _check_ensp_len(ensp:list, verbose:int = 0) -> bool:
    """
    Check if all ENSP IDs in a list have the same length.

    Args:
        ensp (list): A list of ENSP IDs.

    Returns:
        bool: True if all ENSP IDs have the same length, False otherwise.
    """
    if len(ensp) == 0:
        raise ValueError("List of ENSP IDs is empty.")
    elif len(ensp) == 1:
        return ensp
    elif len(ensp) > 1:
        if verbose > 0:
            print(f"Multiple ENSP IDs found: {ensp}. Using {ensp[0]}")
        return ensp
    else:
        return ensp
#endregion

#region STRING network functions
def _select_evidences(evidence_lst: list, network: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and returns specified columns from a network DataFrame.

    Given a list of evidence column names and a network DataFrame, this function extracts the specified columns
    along with the 'node1' and 'node2' columns from the network DataFrame and returns them as a new DataFrame.
    This is useful for filtering and focusing on specific aspects of a network represented in a DataFrame.

    Args:
        evidence_lst (list): A list of column names representing the evidence to be selected from the network DataFrame.
        network (pd.DataFrame): The network DataFrame containing at least 'node1' and 'node2' columns,
                                along with other columns that may represent different types of evidence or attributes.

    Returns:
        pd.DataFrame: A DataFrame consisting of the 'node1' and 'node2' columns from the original network DataFrame,
                      as well as the additional columns specified in evidence_lst.

    Note:
        The function assumes that the network DataFrame contains 'node1' and 'node2' columns and that all column
        names provided in evidence_lst exist in the network DataFrame. If any column in evidence_lst does not exist
        in the network DataFrame, a KeyError will be raised.
    """
    return network[['node1', 'node2'] + evidence_lst]

def _get_evidence_types(evidence_lst: list) -> list:
    """
    Processes and returns a list of evidence types for network analysis.

    This function takes a list of evidence types and, if 'all' is included in the list, replaces it with a predefined set
    of evidence types. It is primarily used to standardize and expand the evidence types used in network-based analyses.

    Args:
        evidence_lst (list): A list of strings indicating the types of evidence to be included. If the list contains 'all',
                             it is replaced with a complete set of predefined evidence types.

    Returns:
        list: A list of evidence types. If 'all' was in the original list, it is replaced by a comprehensive list of evidence types;
              otherwise, the original list is returned.

    Note:
        The predefined set of evidence types includes 'neighborhood', 'fusion', 'cooccurence', 'coexpression',
        'experimental', 'database', and 'textmining'. This function is particularly useful for initializing or
        configuring network analysis functions where a broad range of evidence types is desired.
    """
    if 'all' in evidence_lst:
        evidence_lst = ['neighborhood', 'fusion', 'cooccurence',
                        'coexpression', 'experimental', 'database', 'textmining']
    return evidence_lst

def _get_combined_score(net_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and appends a combined score for each row in a network DataFrame.

    This function takes a network DataFrame with various evidence scores and computes a combined score for each row.
    The combined score is a weighted measure based on the individual evidence scores present in the DataFrame, adjusted
    by a pre-defined probability value.

    Args:
        net_df (pd.DataFrame): A DataFrame containing network data. The first two columns are expected to be 'node1' and 'node2',
                               followed by columns representing different types of evidence scores.

    Returns:
        pd.DataFrame: The original DataFrame with an additional 'score' column appended, representing the combined score for each row.

    Note:
        The calculation of the combined score uses a fixed probability value (p = 0.041) to adjust the evidence scores.
        Scores are first normalized, then combined using a product method, and finally adjusted by the probability value.
        This function assumes that the evidence scores are represented as integers in the range 0-1000.
    """
    cols = net_df.columns.values.tolist()
    cols = cols[2:]
    p = 0.041
    for col in cols:
        net_df[col] = 1-((net_df[col]/1000) - p) / (1 -p)
        net_df[col] = np.where(net_df[col] > 1, 1, net_df[col])
    net_df['score'] = 1 - np.product([net_df[i] for i in cols], axis=0)
    net_df['score'] = net_df['score'] + p * (1 - net_df['score'])
    return net_df

def _get_edge_weight(edge_confidence:str) -> float:
    """
    Determines the weight of an edge based on its confidence level.

    This function assigns a numerical weight to an edge in a network based on a provided confidence level string.
    Different confidence levels correspond to different predefined weights.

    Args:
        edge_confidence (str): A string representing the confidence level of an edge. Accepted values are
                               'low', 'high', 'highest', and 'all'. Any other value is treated as a default case.

    Returns:
        float: The weight assigned to the edge. This is determined by the confidence level:
               - 'low' results in a weight of 0.2
               - 'high' results in a weight of 0.7
               - 'highest' results in a weight of 0.9
               - 'all' results in a weight of 0.0
               - Any other value defaults to a weight of 0.4.

    Note:
        This function is typically used in network analysis where edges have varying levels of confidence, and a numerical
        weight needs to be assigned for computational purposes.
    """
    if edge_confidence == 'low':
        weight = 0.15
    elif edge_confidence == 'high':
        weight = 0.7
    elif edge_confidence == 'highest':
        weight = 0.9
    elif edge_confidence == 'all':
        weight = 0.0
    else:
        weight = 0.4
    return weight

def _load_clean_string_network(version:str, evidences:list, edge_confidence:str) -> pd.DataFrame:
    """
    Load and clean the STRING network based on the provided evidences and edge confidence.

    Parameters:
    - evidences (list): List of evidence types to consider.
    - edge_confidence (str): Minimum confidence level for edges.

    Returns:
    - string_net (pd.DataFrame): Cleaned STRING network.
    - string_net_genes (list): List of genes present in the cleaned network.
    """
    # Load STRING
    string_net = _load_string(version)
    # Parse the evidences and edge confidence
    evidence_lst = _get_evidence_types(evidences)
    string_net = _select_evidences(evidence_lst, string_net)
    string_net = _get_combined_score(string_net)

    edge_weight = _get_edge_weight(edge_confidence)
    string_net = string_net[string_net['score'] >= edge_weight]

    # Get all genes
    string_net_genes = list(set(string_net['node1'].unique().tolist() + string_net['node2'].unique().tolist()))

    return string_net, string_net_genes
#endregion

#region GO network functions
def _get_go_terms(min_size: int, max_size: int) -> (dict, dict, dict): # type: ignore
    """
    Returns three dictionaries containing Gene Ontology (GO) terms for biological processes, cellular components, and molecular functions.

    Parameters:
    max_size (int): The maximum size of the GO term gene list.
    min_size (int): The minimum size of the GO term gene list.

    Returns:
    tuple: A tuple containing three dictionaries. The first dictionary contains GO terms for biological processes, the second dictionary contains GO terms for cellular components, and the third dictionary contains GO terms for molecular functions. Each dictionary maps a GO term name to a list of genes associated with that term.
    """
    #load go terms
    #goterms_stream = pkg_resources.resource_stream(__name__, 'data/GO_terms_parsed_12012022.csv')
    goterms_stream = pkg_resources.resource_stream(__name__, 'data/GO_terms_parsed_12.20.24.csv')
    goterms = pd.read_csv(goterms_stream)
    goterms['gene_lst'] = goterms['gene_lst'].apply(lambda x: list(ast.literal_eval(x)))
    goterms['Goterm/Name'] = goterms['GOterm'] + '/' + goterms['Name']
    goterms_bp = goterms[(goterms['Type']== 'namespace: biological_process') & (goterms['length']<= max_size) & (goterms['length']>= min_size)]
    goterms_bp_dict = dict(zip(goterms_bp['Goterm/Name'].tolist(), goterms_bp['gene_lst'].tolist()))
    goterms_cc = goterms[(goterms['Type']== 'namespace: cellular_component') & (goterms['length']<= max_size) & (goterms['length']>= min_size)]
    goterms_cc_dict = dict(zip(goterms_cc['Goterm/Name'].tolist(), goterms_cc['gene_lst'].tolist()))
    goterms_mf = goterms[(goterms['Type']== 'namespace: molecular_function') & (goterms['length']<= max_size) & (goterms['length']>= min_size)]
    goterms_mf_dict = dict(zip(goterms_mf['Goterm/Name'].tolist(), goterms_mf['gene_lst'].tolist()))
    return goterms_bp_dict, goterms_cc_dict, goterms_mf_dict

def _parse_true_go_terms(true_go_terms:list, min_size:int, max_size:int) -> dict:
    out_go = {'go_bp': [], 'go_cc': [], 'go_mf': []}
    len_go = {'go_bp': 0, 'go_cc': 0, 'go_mf': 0}
    all_go = {'go_bp': [], 'go_cc': [], 'go_mf': []}
    # Get the GO Term file
    goterms_bp_dict, goterms_cc_dict, goterms_mf_dict = _get_go_terms(min_size, max_size)
    # Get the number of GO terms
    len_go['go_bp'] = len(goterms_bp_dict.keys())
    len_go['go_cc'] = len(goterms_cc_dict.keys())
    len_go['go_mf'] = len(goterms_mf_dict.keys())
    # Get all the GO Ids for each
    all_go['go_bp'] = [x.split('/')[0] for x in goterms_bp_dict.keys()]
    all_go['go_cc'] = [x.split('/')[0] for x in goterms_cc_dict.keys()]
    all_go['go_mf'] = [x.split('/')[0] for x in goterms_mf_dict.keys()]
    for term in true_go_terms:
        if term in all_go['go_bp']:
            out_go['go_bp'].append(term)
        elif term in all_go['go_cc']:
            out_go['go_cc'].append(term)
        elif term in all_go['go_mf']:
            out_go['go_mf'].append(term)
    return out_go, len_go, all_go

def _plot_overlap_venn(query_len:int, goldstandard_len:int, overlap:list, pval:float, show_genes: bool, show_pval: bool, query_color:str, goldstandard_color:str, fontsize:int, fontface:str, goldstandard_name:str) -> None:
    """
        Plots a Venn diagram representing the overlap between two sets and returns the plot as an image.

        This function creates a Venn diagram to visualize the overlap between a query set and a gold standard set.
        It displays the overlap size, the p-value of the overlap, and the names of the overlapping items. If there
        are no overlapping genes or if the query set is empty, the function will print a relevant message and return False.

        Args:
            query_len (int): The number of elements in the query set.
            goldstandard_len (int): The number of elements in the gold standard set.
            overlap (list): A list of overlapping elements between the query and gold standard sets.
            pval (float): The p-value representing the statistical significance of the overlap.
            query_color (str): The color to be used for the query set in the Venn diagram.
            goldstandard_color (str): The color to be used for the gold standard set in the Venn diagram.
            fontsize (int): The font size to be used in the Venn diagram.
            fontface (str): The font face to be used in the Venn diagram.

        Returns:
            Image: An image object of the Venn diagram. If there is no overlap or the query is empty, returns False.
    """
    if overlap == 0:
        return False
    elif query_len == 0:
        return False
    # Create Venn Diagram
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _ = plt.figure(figsize=(10, 5))
    out = venn2(subsets=((query_len - overlap),
                        (goldstandard_len - overlap),
                        overlap),
                        set_labels=('Query', f'{goldstandard_name}'),
                        set_colors=('white', 'white'),
                        alpha=0.7)
    overlap1 = out.get_patch_by_id("A")
    overlap1.set_edgecolor(query_color)
    overlap1.set_linewidth(3)
    overlap2 = out.get_patch_by_id("B")
    overlap2.set_edgecolor(goldstandard_color)
    overlap2.set_linewidth(3)

    for text in out.set_labels:
        text.set_fontsize(fontsize + 2)
    for text in out.subset_labels:
        if text == None:
            continue
        text.set_fontsize(fontsize)
    if show_genes:
        plt.text(0, -0.78,
                ", ".join(overlap),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize-2)
    if show_pval:
        if pval < 0.01:
            plt.text(0, -0.7,
                    str("p = " + f"{pval:.2e}"),
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=fontsize-2)
        else:
            plt.text(0, -0.7,
                str("p = " + f"{pval:.2f}"),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize-2)
    #plt.title("Gold Standard Overlap", fontsize=fontsize+4)
    plt.tight_layout(pad = 2.0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _hypergeometric_overlap(query_phenotypes: list, true_phenotypes:list, total_phenotypes:int, omim_true_keyword:int, plot_fontsize:int, plot_fontface:str, plot_venn:bool = False,) -> (Image, float): # type: ignore
    """
    Returns a venn diagram and p-value for the hypergeometric overlap between the true phenotypes and all phenotypes.
    """
    # Calculate hypergeometric overlap
    overlapping_phenotypes = [phenotype for phenotype in query_phenotypes if phenotype in true_phenotypes]
    background_size = total_phenotypes
    len_query = len(query_phenotypes)
    len_gs = len(true_phenotypes)
    len_overlap = len(overlapping_phenotypes)
    p_val = hypergeom.sf(len_overlap - 1, background_size, len_gs, len_query)

    # Plot venn diagram
    if plot_venn:
        venn_img = _plot_overlap_venn(
            query_len = len_query,
            goldstandard_len = len_gs,
            overlap = len_overlap,
            pval = p_val,
            show_genes = False,
            show_pval = True,
            query_color = 'red',
            goldstandard_color = 'gray',
            fontsize = plot_fontsize,
            fontface = plot_fontface,
            goldstandard_name = f'OMIM:{omim_true_keyword}'
        )

    return venn_img, p_val

def _test_overlap_with_random(query_phenotypes: list, true_phenotypes:list, total_phenotypes:int, all_phenotypes:list, random_iter:int, omim_true_keyword:int, plot_fontsize:int, plot_fontface:str, plot_venn:bool = False) -> (Image, float): # type: ignore
    """
    Tests the overlap between the query phenotypes and the true phenotypes by pulling 100 random phenotypes of the same size and testing the overlap.
    """
    # Pull 100 sets of random phenotypes of same size
    len_query = len(query_phenotypes)
    true_overlap = len([phenotype for phenotype in query_phenotypes if phenotype in true_phenotypes])
    rando_phenotypes = []
    rando_overlaps = []
    rando_pvals = []
    for _ in range(random_iter):
        random_pheno = pd.Series(all_phenotypes).sample(len_query).tolist()
        rando_phenotypes.append(random_pheno)
        # Calculate the hypergeometric overlap
        overlapping_phenotypes = [phenotype for phenotype in random_pheno if phenotype in true_phenotypes]
        _, p_val = _hypergeometric_overlap(random_pheno, true_phenotypes, total_phenotypes, omim_true_keyword, plot_fontsize, plot_fontface, plot_venn = plot_venn)
        rando_pvals.append(p_val)
        rando_overlaps.append(len(overlapping_phenotypes))
    
    # Calculate the z-score
    z_score = (true_overlap - np.mean(rando_overlaps)) / np.std(rando_overlaps)

    # Plot the z-score distribution
    _, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.hist(rando_overlaps, color='gray', edgecolor='black', density=False, bins=20, alpha = 0.5, label = 'Random Overlap')
    ax.axvline(true_overlap, color='red', linestyle='dashed', label = 'True Overlap')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    ax.set_xlabel('# of phenotypes overlapping', size=14)
    ax.set_ylabel('Count', size=14)
    ax.set_title('Z-score: {:.2f}'.format(z_score), size=14)
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    
    return image, z_score, rando_phenotypes, rando_overlaps

def _load_go_network(go_type:str):
    # LOad the graphml file
    if go_type == 'go_bp':
        graph_stream = pkg_resources.resource_stream(__name__, 'data/GO_biological_process_graph_12.19.24.graphml')
    elif go_type == 'go_cc':
        graph_stream = pkg_resources.resource_stream(__name__, 'data/GO_cellular_component_graph_12.19.24.graphml')
    elif go_type == 'go_mf':
        graph_stream = pkg_resources.resource_stream(__name__, 'data/GO_molecular_function_graph_12.19.24.graphml')
    G = nx.read_graphml(graph_stream)
    # Get all nodes
    nodes = list(G.nodes())
    return G, nodes

def _get_go_graph(go_net:nx.Graph):
    graph_node = list(go_net.nodes())
    adj_matrix = nx.to_scipy_sparse_array(go_net)
    node_degree = dict(nx.degree(go_net))
    g_degree = node_degree.values()
    return graph_node, adj_matrix, node_degree, g_degree

def _go_term_ndiffusion(go_type:str, query_phenotypes: list, true_keyword:int, true_id_terms:list, set_1_name:str, n_iter: int = 100, cores:int =1, savepath:str = False, verbose: int = 0) -> (Image, float, Image, float): # type: ignore
    """
        Performs network diffusion analysis between two sets of genes.

        Args:
            - set_1 (list): List of genes in set 1.
            - set_2 (list): List of genes in set 2.
            - set_1_name (str, optional): Name of set 1. Defaults to 'Set_1'.
            - set_2_name (str, optional): Name of set 2. Defaults to 'Set_2'.
            - string_version (str, optional): STRING version to use. Defaults to 'v11.0'. Options include 'v10.0', 'v11.0', 'v11.5', 'v12.0'.
            - evidences (list, optional): List of evidence types to consider. Defaults to ['all']. Options include 'experiments', 'databases', 'textmining', 'coexpression', 'neighborhood', 'fusion', 'cooccurrence'.
            - edge_confidence (str, optional): Confidence level for edges. Defaults to 'all'. Options include 'all', 'low' (>0.15), 'medium' (0.4), 'high' (0.7), 'highest' (0.9).
            - custom_background (Any, optional): Custom background gene set. Defaults to 'string'. Options include 'string', 'ensembl', 'reactome'.
            - n_iter (int, optional): Number of diffusion iterations. Defaults to 100.
            - cores (int, optional): Number of cores to use for parallel processing. Defaults to 1.
            - savepath (str, optional): Path to save the results. Defaults to False.
            - verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            Image: AUROC plot for show_1 - "from Set1Exclusive to Set2"; if there is no overlap, then "from Set1 to Set2"
            float: AUROC value for show_1 - randomized set1, degree-matched
            Image: AUROC plot for show_2 - "from Set2Exclusive to Set1"; if there is no overlap, then "from Set2 to Set1"
            float: AUROC value for show_2 - randomized set2, degree-matched
    """
    # Set parameters
    group1_name = set_1_name
    group2_name = str(true_keyword)

    # Load MGI Network
    go_net, _ = _load_go_network(go_type)

    # Get network and diffusion parameters
    graph_node, adj_matrix, node_degree, g_degree = _get_go_graph(go_net)

    ps = _get_diffusion_param(adj_matrix)
    graph_node_index = _get_index_dict(graph_node)
    gp1_only_dict, gp2_only_dict, overlap_dict, other_dict =_parse_gene_input(
        query_phenotypes, true_id_terms, graph_node, graph_node_index, node_degree, verbose = verbose
    )
    degree_nodes = _get_degree_node(g_degree, node_degree, other_dict['node'])
    gp1_all_dict, gp2_all_dict, exclusives_dict = _check_overlap_dict(overlap_dict, gp1_only_dict, gp2_only_dict)

    # Run diffusion
    # If there is no overlap, no genes specific to set_1, and no genes specific to set_2
    if overlap_dict['node'] != [] and gp1_only_dict['node'] != [] and gp2_only_dict['node'] != []:
        # From group 1 exclusive to group 2 all:
        r_gp1o_gp2 = _get_results(
            gp1_only_dict, gp2_all_dict, group1_name+'Excl', group2_name, show = '__SHOW_1_',
            degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_1_plot = r_gp1o_gp2[1][0]
        show_1_z = r_gp1o_gp2[0][1][1]
        # From group 2 exclusive to group 1 all:
        r_gp2o_gp1 = _get_results(
            gp2_only_dict, gp1_all_dict, group2_name+'Excl', group1_name, show = '__SHOW_2_',
            degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_2_plot = r_gp2o_gp1[1][0]
        show_2_z = r_gp2o_gp1[0][1][1]
        # From group 1 exclusive to group 2 exclusive:
        r_gp1o_gp2o = _get_results(
            gp1_only_dict, gp2_only_dict, group1_name+'Excl', group2_name+'Excl',
            degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        # From group 2 exclusive to group 1 exclusive:
        r_gp2o_gp1o = _get_results(
            gp2_only_dict, gp1_only_dict, group2_name+'Excl', group1_name+'Excl',
            degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        # From group 1 exclusive to the overlap
        r_gp1o_overlap = _get_results(
            gp1_only_dict, overlap_dict, group1_name+'Excl', 'Overlap', degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        # From group 2 exclusive to the overlap
        r_gp2o_overlap = _get_results(
            gp2_only_dict, overlap_dict, group2_name+'Excl', 'Overlap', degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        # From overlap to (group 1 exclusive and group 2 exlusive)
        r_overlap_exclusives = _get_results(
            overlap_dict, exclusives_dict,'Overlap', 'Exclus', degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        # Record results to not write
        r_gp1_gp2 = False
        r_gp2_gp1 = False
        r_overlap_gp1o = False
        r_overlap_gp2o = False
    # For when group 2 is entirely part of group 1
    elif overlap_dict['node'] != [] and gp2_only_dict['node'] == []:
        # From group 1 exclusive to overlap/group 2
        r_gp1o_overlap = _get_results(
            gp1_only_dict, overlap_dict, group1_name+'Excl', 'Overlap or'+group2_name, degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_1_plot = r_gp1o_overlap[1][0]
        show_1_z = r_gp1o_overlap[0][1][1]
        # From overlap/group 2 to group 1 exclusive
        r_overlap_gp1o = _get_results(
            overlap_dict, gp1_only_dict,'Overlap or'+group2_name, group1_name+'Excl', degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_2_plot = r_overlap_gp1o[1][0]
        show_2_z = r_overlap_gp1o[0][1][1]
        # Record results to not write
        r_gp1o_gp2 = False
        r_gp2o_gp1 = False
        r_gp1o_gp2o = False
        r_gp2o_gp1o = False
        r_gp2o_overlap = False
        r_overlap_exclusives = False
        r_gp1_gp2 = False
        r_gp2_gp1 = False
        r_overlap_gp2o = False
    # For when group 1 is entirely part of group 2
    elif overlap_dict['node'] != [] and gp1_only_dict['node'] == []:
        # From group 2 exclusive to overlap/group 1
        r_gp2o_overlap = _get_results(
            gp2_only_dict, overlap_dict, group2_name+'Excl', 'Overlap or '+group1_name, degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_1_plot = r_gp2o_overlap[1][0]
        show_1_z = r_gp2o_overlap[0][1][1]
        # From overlap/group 1 to group 2 exclusive
        r_overlap_gp2o = _get_results(
            overlap_dict, gp2_only_dict, 'Overlap or'+group1_name, group2_name+'Excl', degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_2_plot = r_overlap_gp2o[1][0]
        show_2_z = r_overlap_gp2o[0][1][1]
        # Record what to save
        r_gp1o_gp2 = False
        r_gp2o_gp1 = False
        r_gp1o_gp2o = False
        r_gp2o_gp1o = False
        r_gp1o_overlap = False
        r_overlap_exclusives = False
        r_gp1_gp2 = False
        r_gp2_gp1 = False
        r_overlap_gp1o = False
    # For when there is no overlap b/w two groups
    else:
        # From group 1 to group 2:
        r_gp1o_gp2o = _get_results(
            gp1_only_dict, gp2_only_dict, group1_name, group2_name, show = '__SHOW_1_',
            degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_1_plot = r_gp1o_gp2o[1][0]
        show_1_z = r_gp1o_gp2o[0][1][1]
        # From group 2 to group 1:
        r_gp2o_gp1o = _get_results(
            gp2_only_dict, gp1_only_dict, group2_name, group1_name, show = '__SHOW_2_',
            degree_nodes = degree_nodes, other_dict = other_dict, graph_node_index = graph_node_index, graph_node = graph_node, ps = ps, cores = cores, repeat = n_iter
        )
        show_2_plot = r_gp2o_gp1o[1][0]
        show_2_z = r_gp2o_gp1o[0][1][1]
        # Record what to save
        r_gp1o_gp2 = False
        r_gp2o_gp1 = False
        r_gp1o_overlap = False
        r_gp2o_overlap = False
        r_overlap_exclusives = False
        r_gp1_gp2 = False
        r_gp2_gp1 = False
        r_overlap_gp1o = False
        r_overlap_gp2o = False

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'nDiffusion_GOterms_{true_keyword}/')
        os.makedirs(new_savepath, exist_ok=True)
        _write_sum_txt(
            new_savepath, group1_name, group2_name, gp1_only_dict, gp2_only_dict, overlap_dict,
            r_gp1o_gp2 = r_gp1o_gp2,
            r_gp2o_gp1 = r_gp2o_gp1,
            r_gp1o_gp2o = r_gp1o_gp2o,
            r_gp2o_gp1o = r_gp2o_gp1o,
            r_gp1o_overlap = r_gp1o_overlap,
            r_gp2o_overlap = r_gp2o_overlap,
            r_overlap_exclusives = r_overlap_exclusives,
            r_gp1_gp2 = r_gp1_gp2,
            r_gp2_gp1 = r_gp2_gp1,
            r_overlap_gp1o = r_overlap_gp1o,
            r_overlap_gp2o = r_overlap_gp2o
        )
    return show_1_plot, show_1_z, show_2_plot, show_2_z
#endregion

#region nDiffusion functions
def _get_graph(network: pd.DataFrame) -> (nx.Graph, list, np.array, dict, list): # type: ignore
    """
    Constructs a graph from a given network dataframe.

    Parameters:
        network (pd.DataFrame): The network dataframe containing the edges and weights.

    Returns:
        G (nx.Graph): The constructed graph.
        graph_node (list): The list of nodes in the graph.
        adj_matrix (np.array): The adjacency matrix of the graph.
        node_degree (dict): The dictionary containing the degree of each node in the graph.
        g_degree (list): The list of degrees of all nodes in the graph.
    """
    in_network = network.copy()
    in_network.rename(columns = {'score':'weight'}, inplace = True)
    G = nx.from_pandas_edgelist(in_network, 'node1', 'node2', ['weight'])
    graph_node = list(G.nodes())
    adj_matrix = nx.to_scipy_sparse_array(G)
    node_degree = dict(nx.degree(G))
    g_degree = node_degree.values()
    return G, graph_node, adj_matrix, node_degree, g_degree

def _get_diffusion_param(adj_matrix: np.array) -> csr_matrix:
    """
    Calculates the diffusion parameter for a given adjacency matrix.

    Parameters:
    adj_matrix (np.array): The adjacency matrix.

    Returns:
    np.array: The diffusion parameter matrix.
    """
    adj_matrix = csr_matrix(adj_matrix)
    L = csgraph.laplacian(adj_matrix, normed=True)
    n = adj_matrix.shape[0]
    I = identity(n, dtype='int8', format='csr')
    axis_sum = coo_matrix.sum(np.abs(L), axis=0)
    sum_max = np.max(axis_sum)
    diffusion_parameter = (1 / float(sum_max))
    ps = (I + (diffusion_parameter * L))
    return ps

def _get_index_dict(graph_node: list) -> dict:
    """
    Create a dictionary that maps each element in the graph_node list to its index.

    Args:
        graph_node (list): A list of graph nodes.

    Returns:
        dict: A dictionary mapping each element in graph_node to its index.
    """
    graph_node_index = {}
    for i in range(len(graph_node)):
        graph_node_index[graph_node[i]] = i
    return graph_node_index

def _get_index(lst:list, graph_node_index:dict) -> list:
    """
    Get the index of each element in a list based on a given index dictionary.

    Args:
        lst (list): A list of elements.
        graph_node_index (dict): A dictionary mapping each element to its index.

    Returns:
        list: A list of indices corresponding to the elements in the input list.
    """
    index = []
    for i in lst:
        ind = graph_node_index[i]
        index.append(ind)
    return index

def _get_degree(pred_node:list, node_degree:dict) -> dict:
    """
    Get the degree of each node in a given list.

    Args:
        pred_nodes (list): A list of nodes.
        node_degree (dict): A dictionary containing the degree of each node in the graph.

    Returns:
        dict: A dictionary containing the degree of each node in the input list.
    """
    pred_degree = []
    for i in pred_node:
        pred_degree.append(node_degree[i])
    pred_degree_count = dict(Counter(pred_degree))
    return pred_degree_count

def _parse_gene_input(fl1:list, fl2:list, graph_node:list, graph_node_index:dict, node_degree:dict, verbose: int = 0) -> (dict, dict, dict, dict): # type: ignore
    """
    Parses the input files and maps genes into the network.

    Args:
        fl1 (list): List of genes in file 1.
        fl2 (list): List of genes in file 2.
        graph_node (list): List of genes in the network.
        graph_node_index (dict): Dictionary mapping genes to their indexes in the network.
        node_degree (dict): Dictionary mapping genes to their connectivity degrees.
        graph_gene (list): List of genes to be included in the network.

    Returns:
        tuple: A tuple containing four dictionaries:
            - gp1_only_dict: Dictionary containing information about genes only in file 1.
            - gp2_only_dict: Dictionary containing information about genes only in file 2.
            - overlap_dict: Dictionary containing information about genes that overlap between file 1 and file 2.
            - other_dict: Dictionary containing information about genes not in file 1 or file 2.

    """
    ### Parsing input files
    group1 = set(fl1)
    group2 = set(fl2)
    fl1_name = "Set_1"
    fl2_name = "Set_2"
    overlap = list(set(group1).intersection(group2))
    group1_only = list(set(group1)-set(overlap))
    group2_only = list(set(group2)-set(overlap))
    ### Mapping genes into the network
    group1_node = list(set(group1).intersection(graph_node))
    group2_node = list(set(group2).intersection(graph_node))
    overlap_node = list(set(overlap).intersection(graph_node))
    other = list(set(graph_node) - set(group1_node) - set(group2_node))
    group1_only_node = list(set(group1_node)-set(overlap_node))
    group2_only_node = list(set(group2_node)-set(overlap_node))
    if verbose > 0:
        print("{} genes are mapped (out of {}) in {}\n {} genes are mapped (out of {}) in {}\n {} are overlapped and mapped (out of {})\n".format(len(group1_node), len(group1), fl1_name, len(group2_node), len(group2), fl2_name, len(overlap_node), len(overlap)))
    ### Getting indexes of the genes in the network node list
    group1_only_index = _get_index(group1_only_node, graph_node_index)
    group2_only_index = _get_index(group2_only_node, graph_node_index)
    overlap_index = _get_index(overlap_node, graph_node_index)
    other_index = list(set(range(len(graph_node))) - set(group1_only_index) - set(group2_only_index)-set(overlap_index))
    ### Getting counter dictionaries for the connectivity degrees of the genes
    group1_only_degree_count = _get_degree(group1_only_node, node_degree)
    group2_only_degree_count = _get_degree(group2_only_node, node_degree)
    overlap_degree_count = _get_degree(overlap_node, node_degree)
    ### Combining these features into dictionaries
    gp1_only_dict={'orig': group1_only, 'node':group1_only_node, 'index':group1_only_index, 'degree': group1_only_degree_count}
    gp2_only_dict={'orig': group2_only,'node':group2_only_node, 'index':group2_only_index, 'degree': group2_only_degree_count}
    overlap_dict={'orig': overlap, 'node':overlap_node, 'index':overlap_index, 'degree': overlap_degree_count}
    other_dict={'node':other, 'index':other_index}
    return gp1_only_dict, gp2_only_dict, overlap_dict, other_dict

def _get_degree_node(g_degree: list, node_degree: dict, other: list) -> dict:
    """
    Returns a dictionary mapping each degree value to a list of nodes with that degree.

    Parameters:
    g_degree (list): A list of degree values.
    node_degree (dict): A dictionary mapping nodes to their degree values.
    other (list): A list of nodes to consider.

    Returns:
    dict: A dictionary mapping each degree value to a list of nodes with that degree.
    """
    degree_nodes = {}
    for i in set(g_degree):
        degree_nodes[i] = []
        for y in node_degree:
            if node_degree[y] == i and y in other:
                degree_nodes[i].append(y)
        degree_nodes[i] = list(set(degree_nodes[i]))
        random.shuffle(degree_nodes[i])
    return degree_nodes

def _merge_degree_dict(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries by summing the values of common keys.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary with merged values.

    """
    merge_dict = {}
    for k in dict1:
        try:
            merge_dict[k] = dict1[k] + dict2[k]
        except:
            merge_dict[k] = dict1[k]
    for k in dict2:
        try:
            _ = dict1[k]
        except:
            merge_dict[k] = dict2[k]
    return merge_dict

def _combine_group(gp1_dict:dict, gp2_dict:dict) -> dict:
    """
    Combines two group dictionaries into a single dictionary.

    Parameters:
    gp1_dict (dict): The first group dictionary.
    gp2_dict (dict): The second group dictionary.

    Returns:
    dict: The combined group dictionary.
    """
    combine_dict = {}
    combine_dict['orig'] = gp1_dict['orig']+gp2_dict['orig']
    combine_dict['node'] = gp1_dict['node']+gp2_dict['node']
    combine_dict['index'] = gp1_dict['index']+gp2_dict['index']
    combine_dict['degree'] = _merge_degree_dict(gp1_dict['degree'], gp2_dict['degree'])
    return combine_dict

def _check_overlap_dict(overlap_dict: dict, gp1_only_dict:dict, gp2_only_dict:dict) -> (dict, dict, dict): # type: ignore
    """
    Checks the overlap between three dictionaries and combines them accordingly.

    Args:
        overlap_dict (dict): A dictionary representing the overlap between two groups.
        gp1_only_dict (dict): A dictionary representing the elements unique to group 1.
        gp2_only_dict (dict): A dictionary representing the elements unique to group 2.

    Returns:
        tuple: A tuple containing three dictionaries:
            - gp1_all_dict: A dictionary combining gp1_only_dict and overlap_dict.
            - gp2_all_dict: A dictionary combining gp2_only_dict and overlap_dict.
            - exclusives_dict: A dictionary combining gp1_only_dict and gp2_only_dict.
    """
    if overlap_dict['node'] != []:
        gp1_all_dict = _combine_group(gp1_only_dict, overlap_dict)
        gp2_all_dict = _combine_group(gp2_only_dict, overlap_dict)
        exclusives_dict = _combine_group(gp1_only_dict, gp2_only_dict)
        return gp1_all_dict, gp2_all_dict, exclusives_dict
    else:
        return {}, {}, {}

def _diffuse(label_vector:list, ps:csr_matrix) -> lil_matrix:
    """
    Diffuses the label vector using the given sparse matrix.

    Parameters:
    label_vector (list): The label vector to be diffused.
    ps (csr_matrix): The sparse matrix used for diffusion.

    Returns:
    lil_matrix: The diffused label vector.
    """
    sv_sum = label_vector.sum()
    if sv_sum == 0:
        lil_matrix_d = np.zeros(len(label_vector))
        return lil_matrix_d
    y = label_vector
    f = lgmres(ps, y, tol=1e-10)[0]
    return f

def _performance_run(from_index:list, to_index:list, graph_node:list, ps:csr_matrix, exclude:list = [], diffuse_matrix:csr_matrix = False) -> dict:
    """
    Calculates performance metrics for a given set of indices.

    Args:
        from_index (list): List of indices representing the source nodes.
        to_index (list): List of indices representing the target nodes.
        graph_node (list): List of graph nodes.
        ps (csr_matrix): Sparse matrix representing the diffusion process.
        exclude (list, optional): List of indices to exclude. Defaults to [].
        diffuse_matrix (csr_matrix, optional): Sparse matrix representing the diffusion matrix. Defaults to False.

    Returns:
        dict: Dictionary containing the performance metrics.
            - 'classify': List of binary classifications.
            - 'score': List of diffusion scores.
            - 'scoreTP': List of diffusion scores for true positive nodes.
            - 'genes': List of genes associated with the graph nodes.
            - 'diffuseMatrix': Sparse matrix representing the diffusion matrix.
            - 'fpr': List of false positive rates for ROC curve.
            - 'tpr': List of true positive rates for ROC curve.
            - 'auROC': Area under the ROC curve.
            - 'precision': List of precision values for precision-recall curve.
            - 'recall': List of recall values for precision-recall curve.
            - 'auPRC': Area under the precision-recall curve.
    """
    results = {}
    if exclude == []:
        exclude = from_index
    if isinstance(diffuse_matrix, bool) == True:
        label = np.zeros(len(graph_node))
        for i in from_index:
            label[i] = 1
        diffuse_matrix = _diffuse(label, ps)

    score, classify, score_tp, gene_write = [], [], [], []
    for i in range(len(graph_node)):
        if i not in exclude:
            gene_write.append(graph_node[i])
            score.append(diffuse_matrix[i])
            if i in to_index:
                classify.append(1)
                score_tp.append(diffuse_matrix[i])
            else:
                classify.append(0)
    results['classify'], results['score'], results['scoreTP'], results['genes'] = classify, score, score_tp, gene_write
    results['diffuseMatrix'] = diffuse_matrix
    results['fpr'], results['tpr'], _ = roc_curve(classify, score, pos_label=1)
    results['auROC']= auc(results['fpr'], results['tpr'])
    results['precision'], results['recall'], _ = precision_recall_curve(classify, score, pos_label=1)
    results['auPRC'] = auc(results['recall'], results['precision'])
    return results

def _plot_performance(x_axis:list, y_axis:list, auc_:float, title:str = '', type:str='ROC', plotting:bool= True) -> (list, Image): # type: ignore
    """
        Plots the performance curve for a given classification model.

        Args:
            x_axis (list): The values for the x-axis.
            y_axis (list): The values for the y-axis.
            auc_ (float): The area under the curve (AUC) value.
            type (str, optional): The type of performance curve to plot. Defaults to 'ROC'.
            plotting (bool, optional): Whether to plot the curve or not. Defaults to True.

        Returns:
            tuple: A tuple containing the raw data used for plotting and the image of the performance curve.
    """
    raw_data = pd.DataFrame(np.column_stack((y_axis,x_axis)))
    if type == 'ROC':
          x_axis_name, y_axis_name = 'FPR', 'TPR'
    elif type == 'PRC':
          x_axis_name, y_axis_name = 'Recall', 'Precision'
    if plotting == True:
        # header = '%20s\t%30s'%(y_axis_name,x_axis_name)
        plt.figure()
        lw = 2
        plt.plot(x_axis, y_axis, color='darkorange', lw=lw, label='AU'+type+' = %0.2f' % auc_)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(x_axis_name, fontsize='x-large')
        plt.ylabel(y_axis_name, fontsize='x-large')
        plt.legend(loc='lower right',fontsize='xx-large')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format = 'png', dpi = 300)
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close()
    else:
        image = None
    return raw_data, image

def _write_ranking(genes:list, score:list, classify:list, group2_name:str) -> pd.DataFrame:
    """
    Writes the ranking of genes based on diffusion scores and classification into a DataFrame.

    Parameters:
    genes (list): List of gene names.
    score (list): List of diffusion scores.
    classify (list): List of gene classifications.
    group2_name (str): Name of the group.

    Returns:
    pd.DataFrame: DataFrame containing the gene ranking, diffusion scores, and gene classification.
    """
    # Get the percentile for each scor

    result_data = {
        'Gene': genes,
        'Diffusion score (Ranking)': score,
        'Is the gene in {}? (1=yes)'.format(group2_name): classify
    }
    df = pd.DataFrame(result_data)
    df_sorted = df.sort_values(by='Diffusion score (Ranking)', ascending=False)
    df_sorted['Percentile_Rank'] = df_sorted['Diffusion score (Ranking)'].rank(pct=True) * 100
    return df_sorted

def _get_rand_uniform(pred_degree_count: list, other: dict) -> list:
    """
    Returns a list of randomly selected nodes from the 'other' dictionary,
    based on the counts specified in the 'pred_degree_count' list.

    Args:
        pred_degree_count (list): A list of counts specifying the number of nodes to select.
        other (dict): A dictionary containing the nodes to select from.

    Returns:
        list: A list of randomly selected nodes from the 'other' dictionary.
    """
    number_rand = sum(pred_degree_count.values())
    rand_node = random.sample(other, number_rand)
    return rand_node

def _get_rand_degree(pred_degree_count: list, degree_nodes: list, iteration: int = 1) -> list:
    """
    Randomly selects nodes based on their degree from the given degree_nodes dictionary.

    Args:
        pred_degree_count (list): A list of predicted degree counts.
        degree_nodes (list): A dictionary containing nodes grouped by their degree.
        iteration (int, optional): The number of iterations to perform. Defaults to 1.

    Returns:
        list: A list of randomly selected nodes.

    """
    rand_node, rand_degree = [], {}
    for i in pred_degree_count:
        # Initialize the list for each degree
        rand_degree[i] = []
        count = pred_degree_count[i] * iteration
        lst = []
        modifier = 0
        cnt = 0
        if float(i) <= 100:
            increment = 1
        elif float(i) <= 500:
            increment = 5
        else:
            increment = 10
        while len(lst) < count and modifier <= float(i) / 10 and cnt <= 500:
            degree_select = [n for n in degree_nodes.keys() if n <= i + modifier and n >= i - modifier]
            node_select = []
            for m in degree_select:
                node_select += degree_nodes[m]
            node_select = list(set(node_select))
            random.shuffle(node_select)
            try:
                lst += node_select[0:(count - len(lst))]
            except:
                pass
            # Increase the degree bin size we accept as "matching" by increment
            modifier += increment
            # Increase the counter to prevent infinite loops
            cnt += 1
            # Remove the nodes that are already in rand_node
            overlap = set(rand_node).intersection(lst)
            for item in overlap:
                lst.remove(item)
        rand_node += lst
        rand_degree[i] += lst
    return rand_node

def _run_rand_parallelized(node_degree_count:list, node_index:list, degree_nodes:list, other:dict, graph_node_index:list, graph_node:list, ps:csr_matrix, rand_type:str, node_type:str, repeat:int, diffuse_matrix:bool=False, cores:int = 1) -> (list, list, list): # type: ignore
    """
        Runs the _run_rand function in parallel using multiple processes.

        Args:
            node_degree_count (list): List of node degree counts.
            node_index (list): List of node indices.
            degree_nodes (list): List of degree nodes.
            other (dict): Dictionary containing other parameters.
            graph_node_index (list): List of graph node indices.
            graph_node (list): List of graph nodes.
            ps (csr_matrix): CSR matrix.
            rand_type (str): Randomization type.
            node_type (str): Node type.
            repeat (int): Number of times to repeat the randomization.
            diffuse_matrix (bool, optional): Whether to diffuse the matrix. Defaults to False.
            cores (int, optional): Number of CPU cores to use for parallelization. Defaults to 1.

        Returns:
            tuple: A tuple containing the lists of aurocs, auprcs, and score_tps.
    """
    aurocs, auprcs, score_tps = [], [], []

    args = [(node_degree_count, node_index, degree_nodes, other, graph_node_index, graph_node, ps, rand_type, node_type, repeat, diffuse_matrix) for _ in range(repeat)]
    with Pool(cores) as p:
        results = p.starmap(_run_rand, args)

    for result in results:
        aurocs.append(result['auROC'])
        auprcs.append(result['auPRC'])
        score_tps += result['scoreTP']

    return aurocs, auprcs, score_tps

def _run_rand(node_degree_count:list, node_index:list, degree_nodes:list, other:dict, graph_node_index:list, graph_node:list, ps:csr_matrix, rand_type:str, node_type:str, repeat:int, diffuse_matrix:bool=False) -> dict:
    """
    Runs the randomization process for a given node.

    Args:
        node_degree_count (list): List of node degree counts.
        node_index (list): List of node indices.
        degree_nodes (list): List of degree nodes.
        other (dict): Other parameters.
        graph_node_index (list): List of graph node indices.
        graph_node (list): List of graph nodes.
        ps (csr_matrix): CSR matrix.
        rand_type (str): Type of randomization.
        node_type (str): Type of node.
        repeat (int): Number of repetitions.
        diffuse_matrix (bool, optional): Whether to diffuse the matrix. Defaults to False.

    Returns:
        dict: Results of the randomization process.
    """
    if rand_type == 'uniform':
        rand_node = _get_rand_uniform(node_degree_count, other)
    elif rand_type == 'degree':
        rand_node = _get_rand_degree(node_degree_count, degree_nodes)
    
    rand_index = _get_index(rand_node, graph_node_index)
    if node_type == 'TO':
        results = _performance_run(node_index, rand_index, graph_node, ps, diffuse_matrix=diffuse_matrix)
    elif node_type == 'FROM':
        results = _performance_run(rand_index, node_index, graph_node, ps)
    return results

def _z_scores(exp:float, randf_degree:list, randt_degree:list, randf_uniform:list, randt_uniform:list) -> (float, float, float, float): # type: ignore
    """
    Computing z-scores of experimental AUC against random AUCs

    Parameters:
    exp (float): The experimental value.
    randf_degree (list): List of random samples for the degree distribution in the forward direction.
    randt_degree (list): List of random samples for the degree distribution in the reverse direction.
    randf_uniform (list): List of random samples for the uniform distribution in the forward direction.
    randt_uniform (list): List of random samples for the uniform distribution in the reverse direction.

    Returns:
    tuple: A tuple containing the z-scores for the degree distribution in the forward direction,
        the degree distribution in the reverse direction, the uniform distribution in the forward direction,
        and the uniform distribution in the reverse direction.
    """
    try: zf_degree = '%0.2f' %((exp-np.mean(randf_degree))/np.std(randf_degree))
    except: zf_degree = np.nan
    try: zt_degree = '%0.2f' %((exp-np.mean(randt_degree))/np.std(randt_degree))
    except: zt_degree = np.nan
    try: zf_uniform = '%0.2f' %((exp-np.mean(randf_uniform))/np.std(randf_uniform))
    except: zf_uniform = np.nan
    try: zt_uniform = '%0.2f' %((exp-np.mean(randt_uniform))/np.std(randt_uniform))
    except: zt_uniform = np.nan
    return zf_degree, zt_degree, zf_uniform, zt_uniform

def _dist_stats(exp:list, randf_degree:list, randt_degree:list, randf_uniform:list, randt_uniform:list) -> (str, str, str, str): # type: ignore
    """
    Performing KS test to compare distributions of diffusion values

    Parameters:
    exp (list): The experimental data.
    randf_degree (list): Random samples generated using the degree distribution.
    randt_degree (list): Random samples generated using the degree distribution.
    randf_uniform (list): Random samples generated using the uniform distribution.
    randt_uniform (list): Random samples generated using the uniform distribution.

    Returns:
    tuple: A tuple containing the p-values for the statistical distances between the experimental data and each set of random samples.
    """
    try: pf_degree ='{:.2e}'.format(ks_2samp(exp, randf_degree)[1])
    except ValueError: pf_degree = np.nan
    try: pt_degree ='{:.2e}'.format(ks_2samp(exp, randt_degree)[1])
    except ValueError: pt_degree = np.nan
    try: pf_uniform ='{:.2e}'.format(ks_2samp(exp, randf_uniform)[1])
    except ValueError: pf_uniform = np.nan
    try: pt_uniform ='{:.2e}'.format(ks_2samp(exp, randt_uniform)[1])
    except ValueError: pt_uniform = np.nan
    return pf_degree, pt_degree, pf_uniform, pt_uniform

def _plot_auc_rand (roc_exp:list, roc_rands:list, z_text:str, type:str = 'density', title:str = '', raw_input:bool = True) -> (Image, pd.DataFrame): # type: ignore
    """
    Plots the density or histogram of random AUCs and annotates the experimental AUC and z-score.

    Parameters:
    roc_exp (list): List of experimental AUC values.
    roc_rands (list): List of random AUC values.
    z_text (str): The z-score value.
    name (str): Name of the plot.
    type (str, optional): Type of plot. Can be 'density' or 'hist'. Defaults to 'density'.
    raw_input (bool, optional): Whether to include raw input data in the returned DataFrame. Defaults to True.

    Returns:
    Image: The plot as an Image object.
    pd.DataFrame: The random AUC values as a DataFrame.
    """
    if type == 'density':
          sns.kdeplot(np.array(roc_rands) , color="gray", fill = True)
          _, top = plt.ylim()
          plt.annotate('AUC = %0.2f\nz = {}'.format(z_text) %roc_exp, xy = (roc_exp, 0), xytext = (roc_exp,0.85*top),color = 'orangered',fontsize = 'x-large', arrowprops = dict(color = 'orangered',width = 2, shrink=0.05),va='center',ha='right')
          plt.xlim([0,1])
          plt.xlabel("Random AUCs", fontsize='x-large')
          plt.ylabel("Density", fontsize='x-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
    elif type == 'hist':
          plt.hist(roc_rands, color = 'gray', bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
          plt.annotate('AUC = %0.2f\nz = {}'.format(z_text) %roc_exp, xy = (roc_exp, 0), xytext = (roc_exp,10),color = 'orangered',fontsize = 'x-large', arrowprops = dict(color = 'orangered',width = 2, shrink=0.05),va='center',ha='right')
          plt.xlim([0.0, 1.0])
          plt.xlabel('Random AUCs', fontsize='x-large')
          plt.ylabel('Count', fontsize='x-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
    if raw_input == True:
        roc_rands_array = np.array(roc_rands)
        df = pd.DataFrame(roc_rands_array, columns=['AUROC'])
    else:
        df = pd.DataFrame()
    plt.title(title)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image, df

def _plot_dist (exp_dist:list, rand_frd:list, rand_tod:list, rand_fru:list, rand_tou:list, from_gp_name:str, to_gp_name:str, title:str = "") -> (Image, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Plots the distribution of diffusion values for different groups and saves the plot as an image.

    Parameters:
    exp_dist (list): List of diffusion values for the experiment group.
    rand_frd (list): List of diffusion values for the randomly generated group (degree-matched to the experiment group).
    rand_tod (list): List of diffusion values for the randomly generated group (degree-matched to the target group).
    rand_fru (list): List of diffusion values for the randomly generated group (uniform distribution).
    rand_tou (list): List of diffusion values for the randomly generated group (uniform distribution).
    from_gp_name (str): Name of the experiment group.
    to_gp_name (str): Name of the target group.
    raw_input (bool, optional): Flag indicating whether to return the raw data as pandas DataFrames. Defaults to True.

    Returns:
    image (PIL.Image.Image): The plot as an image.
    exp_dist_log10_df (pd.DataFrame): DataFrame containing the log10-transformed diffusion values for the experiment group.
    rand_frd_log10_df (pd.DataFrame): DataFrame containing the log10-transformed diffusion values for the randomly generated group (degree-matched to the experiment group).
    rand_tod_log10_df (pd.DataFrame): DataFrame containing the log10-transformed diffusion values for the randomly generated group (degree-matched to the target group).
    rand_fru_log10_df (pd.DataFrame): DataFrame containing the log10-transformed diffusion values for the randomly generated group (uniform distribution).
    rand_tou_log10_df (pd.DataFrame): DataFrame containing the log10-transformed diffusion values for the randomly generated group (uniform distribution).
    """
    exp_dist = np.array(exp_dist, dtype = np.float32)
    rand_frd = np.array(rand_frd, dtype = np.float32)
    rand_tod = np.array(rand_tod, dtype = np.float32)
    rand_fru = np.array(rand_fru, dtype = np.float32)
    rand_tou = np.array(rand_tou, dtype = np.float32)
    arrays = {
        'exp_dist': [exp_dist, 'red', 'Experiment'],
        'rand_frd': [rand_frd, 'darkgreen', "Randomize "+from_gp_name+" (degree-matched)"],
        'rand_tod': [rand_tod, 'darkblue', "Randomize "+to_gp_name+" (degree-matched)"],
        'rand_fru': [rand_fru, 'lightgreen', "Randomize "+from_gp_name+" (uniform)"],
        'rand_tou': [rand_tou, 'lightskyblue', "Randomize "+to_gp_name+" (uniform)"]
    }
    dfs = {}
    for key, value in arrays.items():
        array = value[0]
        color = value[1]
        label = value[2]
        # Create dataframe
        df = pd.DataFrame(array, columns=['log10 (diffusion value)'])
        dfs[key] = df
        # Plot if length > 0
        if len(array) == 0:
            continue
        array = np.log10(array, where=(array!=0))
        array[(array==0) | (np.isnan(array))] = np.nanmin(array)
        array[np.isinf(array)] = np.nanmax(array)
        sns.kdeplot(array, color=color, label=label, fill = True)

    plt.title(title)
    plt.legend(loc = "upper left")
    plt.xlabel("log10 (diffusion value)")
    plt.ylabel("Density")
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image, dfs['exp_dist'], dfs['rand_frd'], dfs['rand_tod'], dfs['rand_fru'], dfs['rand_tou']

def _run_run(from_dict:dict, to_dict:dict, group1_name:str, group2_name:str, show:str, degree_nodes:dict, other:dict, graph_node_index:dict, graph_node:list, ps:csr_matrix, cores:int, exclude:list=[], repeat:int=100) -> (tuple, tuple, tuple): # type: ignore
    """
    Run the CFS analysis and perform degree-matched randomization and uniform randomization.

    Args:
        from_dict (dict): A dictionary containing information about the 'from' group.
        to_dict (dict): A dictionary containing information about the 'to' group.
        group1_name (str): Name of the 'from' group.
        group2_name (str): Name of the 'to' group.
        show (str): Show name for saving plots.
        degree_nodes (dict): Dictionary containing degree information for nodes.
        other (dict): Other parameters for the analysis.
        graph_node_index (dict): Dictionary containing index information for graph nodes.
        graph_node (list): List of graph nodes.
        ps (csr_matrix): Diffusion matrix.
        exclude (list, optional): List of nodes to exclude. Defaults to [].
        repeat (int, optional): Number of repetitions for randomization. Defaults to 100.

    Returns:
        tuple: A tuple containing the results of the analysis, plots, and additional data.
    """
    name = 'from {} to {}'.format(group1_name, group2_name)
    #region Experimental results
    results = _performance_run(from_dict['index'], to_dict['index'], graph_node, ps, exclude = exclude)
    auroc_df, auroc_plot = _plot_performance(results['fpr'], results['tpr'], results['auROC'], title = name, type = 'ROC')
    auprc_df, auprc_plot = _plot_performance(results['recall'], results['precision'],results['auPRC'], title = name, type = 'PRC')
    ranking = _write_ranking(results['genes'], results['score'], results['classify'], group2_name)
    #endregion

    ### Degree-matched randomization
    #### Randomizing nodes where diffusion starts
    aurocs_from_degree, auprcs_from_degree, score_tps_from_degree = _run_rand_parallelized(
        from_dict['degree'], to_dict['index'], degree_nodes, other, graph_node_index, graph_node, ps, rand_type='degree', node_type='FROM', repeat = repeat, cores = cores
    )
    #### Randomizing nodes which are true positive
    aurocs_to_degree, auprcs_to_degree, score_tps_to_degree = _run_rand_parallelized(
        to_dict['degree'], from_dict['index'], degree_nodes, other, graph_node_index, graph_node, ps, rand_type='degree', node_type='TO', diffuse_matrix=results['diffuseMatrix'], repeat=repeat, cores = cores
    )

    ### Uniform randomization
    #### Randomizing nodes where diffusion starts
    aurocs_from_uniform, auprcs_from_uniform, score_tps_from_uniform = _run_rand_parallelized(
        from_dict['degree'], to_dict['index'], degree_nodes, other, graph_node_index, graph_node, ps, rand_type='uniform', node_type='FROM', repeat=repeat, cores = cores
    )
    #### Randomizing nodes which are true positive
    aurocs_to_uniform, auprcs_to_uniform, score_tps_to_uniform = _run_rand_parallelized(
        to_dict['degree'], from_dict['index'], degree_nodes, other, graph_node_index, graph_node, ps, rand_type='uniform', node_type='TO', diffuse_matrix=results['diffuseMatrix'], repeat=repeat, cores = cores
    )

    ### Computing z-scores when comparing AUROC and AUPRC against random
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform
    z_auc = _z_scores(results['auROC'], aurocs_from_degree, aurocs_to_degree, aurocs_from_uniform, aurocs_to_uniform)
    z_prc = _z_scores(results['auPRC'], auprcs_from_degree, auprcs_to_degree, auprcs_from_uniform, auprcs_to_uniform)

    ### Computing KS test p-values when comparing distribution of diffusion values against random
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform
    pval = _dist_stats(results['scoreTP'], score_tps_from_degree, score_tps_to_degree, score_tps_from_uniform, score_tps_to_uniform)

    to_degree_auroc_plot, to_degree_auroc_df = _plot_auc_rand(
        results['auROC'], aurocs_to_degree, z_auc[1], title = show+'_1 randomize ' + group2_name + ': diffusion ' + name
    )
    from_degree_auroc_plot, from_degree_auroc_df = _plot_auc_rand(
        results['auROC'], aurocs_from_degree, z_auc[0], title = show+'_2 randomize' + group1_name + ': diffusion ' + name
    )

    #### CHECK THE SIZE OF THE INPUTS HERE AND OMIT IF THEY ARE EMPTY
    # name = show+'_3 randomize ' + group2_name + ': diffusion ' + name
    rand_dist_plot, exp_dist_log10_df, rand_frd_log10_df, rand_tod_log10_df, rand_fru_log10_df, rand_tou_log10_df = _plot_dist(results['scoreTP'], score_tps_from_degree, score_tps_to_degree, score_tps_from_uniform, score_tps_to_uniform, group1_name, group2_name, title = show+'_3 randomize ' + group2_name + ': diffusion ' + name)

    return ('%0.2f' %results['auROC'], z_auc, '%0.2f' %results['auPRC'], z_prc, pval), (auroc_plot, auprc_plot, to_degree_auroc_plot, from_degree_auroc_plot, rand_dist_plot), (auroc_df, auprc_df, ranking, to_degree_auroc_df, from_degree_auroc_df, exp_dist_log10_df, rand_frd_log10_df, rand_tod_log10_df, rand_fru_log10_df, rand_tou_log10_df)

def _get_results(gp1:dict, gp2:dict, gp1_name:str, gp2_name:str, degree_nodes:dict, other_dict:dict, graph_node_index:dict, graph_node:list, ps:csr_matrix, cores:int, repeat:int, show:str = '', exclude:list=[]) -> (tuple, tuple, tuple): # type: ignore
    """
    Calculate various scores and statistics for two groups of data.

    Args:
        gp1 (dict): Group 1 data.
        gp2 (dict): Group 2 data.
        gp1_name (str): Name of Group 1.
        gp2_name (str): Name of Group 2.
        degree_nodes (dict): Degree nodes.
        show (str, optional): Show option. Defaults to ''.
        exclude (list, optional): List of nodes to exclude. Defaults to [].
        other_dict (dict): Other dictionary.
        graph_node_index (dict): Graph node index.
        graph_node (list): Graph node.
        ps (csr_matrix): CSR matrix.
        repeat (int): Number of repetitions.

    Returns:
        tuple: A tuple containing the scores, plots, and dataframes.
    """
    #### auroc, z-scores for auc, auprc, z-scores for auprc, KS pvals
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform
    # original scores: auroc, z_auc, auprc, z_prc, pval
    scores, plots, dfs = _run_run(
        gp1, gp2, gp1_name, gp2_name, show,
        degree_nodes, other_dict['node'], graph_node_index, graph_node, ps, cores, exclude=exclude, repeat=repeat
    )
    return scores, plots, dfs

def _write_sum_txt(result_fl: str, group1_name: str, group2_name: str, gp1_only_dict: dict, gp2_only_dict: dict, overlap_dict: dict, r_gp1o_gp2: list = [], r_gp2o_gp1: list = [], r_gp1o_gp2o: list = [], r_gp2o_gp1o: list = [], r_gp1o_overlap: list = [], r_gp2o_overlap: list = [], r_overlap_exclusives: list = [], r_gp1_gp2: Any = [], r_gp2_gp1: Any = [], r_overlap_gp1o: list = [], r_overlap_gp2o: list = []) -> None:
    """
    Writes the summary file containing the results of the analysis.

    Args:
        result_fl (str): The file path to write the summary file.
        group1_name (str): The name of group 1.
        group2_name (str): The name of group 2.
        gp1_only_dict (dict): A dictionary containing the genes exclusive to group 1.
        gp2_only_dict (dict): A dictionary containing the genes exclusive to group 2.
        overlap_dict (dict): A dictionary containing the overlapping genes between group 1 and group 2.
        r_gp1o_gp2 (list, optional): The results for group 1 exclusive genes compared to group 2. Defaults to [].
        r_gp2o_gp1 (list, optional): The results for group 2 exclusive genes compared to group 1. Defaults to [].
        r_gp1o_gp2o (list, optional): The results for group 1 exclusive genes compared to group 2 exclusive genes. Defaults to [].
        r_gp2o_gp1o (list, optional): The results for group 2 exclusive genes compared to group 1 exclusive genes. Defaults to [].
        r_gp1o_overlap (list, optional): The results for group 1 exclusive genes compared to overlap genes. Defaults to [].
        r_gp2o_overlap (list, optional): The results for group 2 exclusive genes compared to overlap genes. Defaults to [].
        r_overlap_exclusives (list, optional): The results for overlap genes compared to exclusive genes. Defaults to [].
        r_gp1_gp2 (list, optional): The results for group 1 genes compared to group 2 genes. Defaults to [].
        r_gp2_gp1 (list, optional): The results for group 2 genes compared to group 1 genes. Defaults to [].
        r_overlap_gp1o (list, optional): The results for overlap genes or group 2 exclusive genes compared to group 1 exclusive genes. Defaults to [].
        r_overlap_gp2o (list, optional): The results for overlap genes or group 1 exclusive genes compared to group 2 exclusive genes. Defaults to [].
    """
    # Set up summary file needs
    ks_result = []
    ks_result.append(['Seeds','Recipients','Randomize Seeds (degree-matched)','Randomize Recipients (degree-matched)','Randomize Seeds (uniform)','Randomize Recipients (uniform)'])
    roc_result=[]
    roc_result.append(['Seeds','Recipients','AUROC','Z-score for Random Seeds (degree-matched)','Z-score for Random Recipients (degree-matched)','Z-score for Random Seeds (uniform)','Z-score for Random Recipients (uniform)'])
    prc_result=[]
    prc_result.append(['Seeds','Recipients','AUPRC','Z-score for Random Seeds (degree-matched)','Z-score for Random Recipients (degree-matched)','Z-score for Random Seeds (uniform)','Z-score for Random Recipients (uniform)'])
    # Create dictionary of saving parameters
    save_dict = {
        "r_gp1o_gp2": [r_gp1o_gp2, group1_name + "Exclusive", group2_name],
        "r_gp2o_gp1": [r_gp2o_gp1, group2_name + "Exclusive", group1_name],
        "r_gp1o_gp2o": [r_gp1o_gp2o, group1_name + "Exclusive", group2_name + "Exclusive"],
        "r_gp2o_gp1o": [r_gp2o_gp1o, group2_name + "Exclusive", group1_name + "Exclusive"],
        "r_gp1o_overlap": [r_gp1o_overlap, group1_name + "Exclusive", "Overlap"],
        "r_gp2o_overlap": [r_gp2o_overlap, group2_name + "Exclusive", "Overlap"],
        "r_overlap_exclusives": [r_overlap_exclusives, "Overlap", "Exclusive"],
        "r_gp1_gp2": [r_gp1_gp2, group1_name, group2_name],
        "r_gp2_gp1": [r_gp2_gp1, group2_name, group1_name],
        "r_overlap_gp1o": [r_overlap_gp1o, "Overlap", group1_name + "Exclusive"],
        "r_overlap_gp2o": [r_overlap_gp2o, "Overlap", group2_name + "Exclusive"]
    }
    # Save the results
    for name, values in save_dict.items():
        result = values[0]
        new_group1_name = values[1]
        new_group2_name = values[2]
        if not result:
            continue
        # Parse the results
        scores = result[0]
        plots = result[1]
        dfs = result[2]
        # Create saving folder
        new_save_folder = os.path.join(os.path.dirname(result_fl), new_group1_name + '_vs_' + new_group2_name+'/')
        os.makedirs(new_save_folder, exist_ok=True)
        # Save the plots
        plot_save_folder = os.path.join(new_save_folder, 'plots/')
        os.makedirs(plot_save_folder, exist_ok=True)
        plots[0].save(os.path.join(plot_save_folder, 'AUROC.png'))
        plots[1].save(os.path.join(plot_save_folder, 'AUPRC.png'))
        plots[2].save(os.path.join(plot_save_folder, 'AUROC_randomize_to_degree_matched.png'))
        plots[3].save(os.path.join(plot_save_folder, 'AUROC_randomize_from_degree_matched.png'))
        plots[4].save(os.path.join(plot_save_folder, 'Diffusion_distribution.png'))
        # Save the dataframes
        df_save_folder = os.path.join(new_save_folder, 'dataframes/')
        os.makedirs(df_save_folder, exist_ok=True)
        dfs[0].to_csv(os.path.join(df_save_folder, 'AUROC.csv'), index=False)
        dfs[1].to_csv(os.path.join(df_save_folder, 'AUPRC.csv'), index=False)
        dfs[2].to_csv(os.path.join(df_save_folder, 'ranking.csv'))
        dfs[3].to_csv(os.path.join(df_save_folder, 'AUROC_randomize_to_degree_matched.csv'), index=False)
        dfs[4].to_csv(os.path.join(df_save_folder, 'AUROC_randomize_from_degree_matched.csv'), index=False)
        dfs[5].to_csv(os.path.join(df_save_folder, 'Exp_distribution.csv'), index=False)
        dfs[6].to_csv(os.path.join(df_save_folder, 'Rand_from_degree_distribution.csv'), index=False)
        dfs[7].to_csv(os.path.join(df_save_folder, 'Rand_to_degree_distribution.csv'), index=False)
        dfs[8].to_csv(os.path.join(df_save_folder, 'Rand_from_uniform_distribution.csv'), index=False)
        dfs[9].to_csv(os.path.join(df_save_folder, 'Rand_to_uniform_distribution.csv'), index=False)
        # Append results to output files
        ks_result.append([new_group1_name, new_group2_name, scores[4][0], scores[4][1], scores[4][2], scores[4][3]])
        roc_result.append([new_group1_name, new_group2_name, scores[0], scores[1][0], scores[1][1], scores[1][2], scores[1][3]])
        prc_result.append([new_group1_name, new_group2_name, scores[2], scores[3][0], scores[3][1], scores[3][2], scores[3][3]])

    ### Mapping results
    gene_result=[]
    gene_result.append(['**','#Total', '# Mapped in the network','Not mapped genes',' '])
    if overlap_dict['node'] != []:
        gene_result.append([group1_name+' Exclusive', len(gp1_only_dict['orig']), len(gp1_only_dict['node']),
                            ';'.join(str(x) for x in set(gp1_only_dict['orig']).difference(gp1_only_dict['node']))])
        gene_result.append([group2_name+' Exclusive', len(gp2_only_dict['orig']), len(gp2_only_dict['node']),
                            ';'.join(str(x) for x in set(gp2_only_dict['orig']).difference(gp2_only_dict['node']))])
        gene_result.append(['Overlap', len(overlap_dict['orig']), len(overlap_dict['node']),
                            ';'.join(str(x) for x in set(overlap_dict['orig']).difference(overlap_dict['node']))])
    else:
        gene_result.append([group1_name, len(gp1_only_dict['orig']), len(gp1_only_dict['node']),
                            ';'.join(str(x) for x in set(gp1_only_dict['orig']).difference(gp1_only_dict['node']))])
        gene_result.append([group2_name, len(gp2_only_dict['orig']), len(gp2_only_dict['node']),
                            ';'.join(str(x) for x in set(gp2_only_dict['orig']).difference(gp2_only_dict['node']))])

    ### Write files
    file_hand = open('{}/diffusion_result.txt'.format(result_fl), 'w')
    file_hand.write('Comparing distributions of experimental and random diffusion values (p-values for KS tests)\n')
    for line in ks_result:
            val = "\t".join(str(v) for v in line)
            file_hand.writelines("%s\n" % val)
    file_hand.write('\n')
    file_hand.write('\n')
    file_hand.write('\n')
    file_hand.write('Evaluating how well {} genes are linked to {} genes, comparing against random\n'.format(group1_name, group2_name))
    file_hand.write('\n')
    file_hand.write('ROC results\n')
    for line in roc_result:
            val = "\t".join(str(v) for v in line)
            file_hand.writelines("%s\n" % val)
    file_hand.write('\n')
    file_hand.write('PRC results\n')
    for line in prc_result:
            val = "\t".join(str(v) for v in line)
            file_hand.writelines("%s\n" % val)
    file_hand.write('\n')
    file_hand.write('Z-scores are computed for the experimental area under ROC or PRC based on distributions of the random areas under these curves\n')
    file_hand.write('Seeds: Genes where diffusion signal starts FROM)\nRecipients: Genes that receive the diffusion signal and that are in the other validated group\n')
    file_hand.write('Random genes are selected either uniformly or degree matched\n')
    file_hand.write('\n')
    file_hand.write('\n')
    file_hand.write('\n')
    file_hand.write('Number of genes\n')
    for gene in gene_result:
            val = "\t".join(str(v) for v in gene)
            file_hand.writelines("%s\n" % val)
    file_hand.close()

#endregion