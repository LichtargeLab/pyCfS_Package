"""
Utility functions for the scripts
"""

from typing import Any
from collections.abc import Iterable
from scipy.stats import hypergeom
import pkg_resources
import pandas as pd
import numpy as np

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
    return pd.read_feather(stream)

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
