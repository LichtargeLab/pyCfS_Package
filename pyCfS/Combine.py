"""Collection of functions looking at previous genetic overlap recovery

Functions:

"""

import pkg_resources
import pandas as pd
import numpy as np
from typing import Any
from collections import defaultdict
import matplotlib.pyplot as plt
import upsetplot as up
from PIL import Image
import io
from matplotlib_venn import venn2
import networkx as nx
import uuid
import ast
import time
import markov_clustering as mc
import warnings
from statsmodels.stats.multitest import multipletests
import itertools
import multiprocessing as mp
from scipy.stats import hypergeom, percentileofscore
import os
from .utils import _fix_savepath, _get_edge_weight, _get_combined_score, _select_evidences, _get_evidence_types, _load_string, _load_reactome, _define_background_list, _go_term_ndiffusion, _hypergeometric_overlap, _test_overlap_with_random, _parse_true_go_terms, _get_go_terms

#region Consensus
def _format_input_dict(list_names:list, gene_lists:list) -> dict:
    """
    Formats input gene lists into a dictionary with deduplication and NaN filtering.

    This function takes two lists as input: one with names for the gene lists and another
    with the gene lists themselves. It creates a dictionary where each entry corresponds to
    a gene list, keyed by its name from the `list_names`. Each gene list is deduplicated and
    filtered to exclude NaN values or empty strings. If `list_names` is not provided, the keys
    are generated as 'list_0', 'list_1', etc.

    Parameters:
    -----------
    list_names : list
        A list containing names for each gene list. If empty, default names are generated.
    gene_lists : list
        A list of gene lists. Each gene list is a list of gene identifiers.

    Returns:
    --------
    dict
        A dictionary with keys as list names and values as cleaned and deduplicated gene lists.

    Notes:
    ------
    - Deduplication preserves the order of first occurrence of each gene in the lists.
    - NaN values are identified as empty strings or the literal string 'nan'.

    Example:
    --------
    >>> list_names = ['list1', 'list2']
    >>> gene_lists = [['gene1', 'gene2', 'gene1'], ['gene3', 'nan', 'gene4']]
    >>> formatted_dict = _format_input_dict(list_names, gene_lists)
    """
    gene_dict = {}
    if list_names:
        for j in range(len(list_names)):
            seen = set()
            gene_list = [x for x in gene_lists[j] if not (x in seen or seen.add(x))]
            gene_dict[list_names[j]] = [gene for gene in gene_list if gene and str(gene) != 'nan']
    else:
        for j in range(len(gene_lists)):
            if gene_lists[j]:
                seen = set()
                gene_list = [x for x in gene_lists[j] if not (x in seen or seen.add(x))]
                gene_dict[f"list_{j}"] = [gene for gene in gene_list if gene and str(gene) != 'nan']
    return gene_dict

def _upset_plot(result_dict:dict, fontsize:int, fontface:str) -> Image:
    """
    Generates an upset plot from the provided data and returns it as an image.

    This function creates an upset plot using the 'upsetplot' library. It takes a dictionary
    of results, where keys are categories and values are lists of elements in each category.
    The function sets custom font size and font face for the plot, generates the plot,
    and then saves it to a BytesIO buffer as a PNG image, which is then returned.

    Parameters:
    -----------
    result_dict : dict
        A dictionary with keys representing categories and values being lists of elements
        in each category. This is used to generate the upset plot.
    fontsize : int
        The font size to be used in the plot.
    fontface : str
        The font face (font family) to be used in the plot.

    Returns:
    --------
    Image
        An image object representing the generated upset plot.

    Notes:
    ------
    - The function requires the 'matplotlib' and 'upsetplot' libraries for plot generation
      and the 'PIL' library for image handling.
    - The plot is generated in memory and returned as an Image object, without saving to disk.

    Example:
    --------
    >>> result_dict = {'Category1': ['Item1', 'Item2'], 'Category2': ['Item2', 'Item3']}
    >>> image = _upset_plot(result_dict, 12, 'Arial')
    """
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    # Set up plot
    from_contents = up.from_contents(result_dict)
    up.plot(from_contents, sort_by = 'degree', show_counts = True)
    # Saveplot
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image

def consensus(genes_1:list = False, genes_2:list = False, genes_3: list = False, genes_4:list = False, genes_5:list = False, genes_6:list = False, gene_dict:dict = False, list_names:Any = False, plot_fontface:str='Avenir', plot_fontsize:int = 14, savepath: Any = False) -> (pd.DataFrame, Image): # type: ignore
    """
    Combines multiple lists of genes, excluding NaNs, counts occurrences of each gene, and tracks the lists they came from.

    Parameters:
    -----------
    - genes_1, genes_2, genes_3, genes_4, genes_5, genes_6 (list) : Lists of genes to be combined and analyzed. NaNs, None values, or empty strings are excluded.
    - gene_dict (dict) : dictionary of gene lists as alternative to defining individual gene lists. {list_name: gene_list}
    - list_names (list) : list of names for each gene list. If not provided, default names are generated.
    - plot_fontface (str) : Font face to be used in the upset plot (default is 'Avenir').
    - plot_fontsize (int) : Font size to be used in the upset plot (default is 14).
    - savepath (Any, optional) : If provided, the path where the resulting DataFrame will be saved as a CSV file (default is False).

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns 'gene', 'occurrences', and 'lists', detailing each unique gene,
        its number of occurrences, and the lists it appeared in.
    Image
        Upset plot showing overlap between gene lists.

    Example:
    --------
    >>> genes_1 = ['SLC30A8', 'GENE1', 'GENE2']
    >>> genes_2 = ['SLC30A8', 'GENE3', 'GENE4']
    >>> consensus_df = consensus(genes_1, genes_2)
    """
    result_dict = defaultdict(lambda: {'count':0, 'lists':set()})
    if isinstance(gene_dict, bool):
        if isinstance(genes_1, bool):
            raise ValueError("At least one gene list must be provided. Define genes_1 or gene_dict.")
        gene_lists = [genes_1, genes_2, genes_3, genes_4, genes_5, genes_6]
        gene_dict = _format_input_dict(list_names, gene_lists)
    # Plot upset plot
    upset_plot = _upset_plot(gene_dict, plot_fontsize, plot_fontface)
    # Count gene occurrences
    for list_name, gene_list in gene_dict.items():
        if gene_list:
            for gene in gene_list:
                result_dict[gene]['count'] += 1
                result_dict[gene]['lists'].add(list_name)
    # Create dataframe
    data = [{'gene': gene,
             'occurrences': info['count'],
             'lists': ','.join(sorted(info['lists']))} for gene, info in result_dict.items()]
    df = pd.DataFrame(data, columns=['gene', 'occurrences', 'lists'])
    df = df.sort_values(by='occurrences', ascending=False)
    # Save and output
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'Consensus/')
        os.makedirs(new_savepath, exist_ok=True)
        df.to_csv(new_savepath + "ConsensusGenes.csv", index=False)
        upset_plot.save(new_savepath + "UpsetPlot.png", bbox_inches='tight', pad_inches=0.5)
    return df, upset_plot
#endregion



#region Functional Clustering
def _load_clean_network(version:str, evidences:list, edge_confidence:str) -> (pd.DataFrame, list, pd.DataFrame): # type: ignore
    """
    Load and customize STRINGv11 network for analysis.

    Args:
    - evidences (list): List of evidence types to include in the network.
    - edge_confidence (str): Edge confidence threshold for filtering the network.

    Returns:
    - string_net (pd.Dataframe): Filtered STRINGv11 network.
    - string_net_all_genes (list): List of all genes in the network.
    - string_net_degree_df (pd.Dataframe): Dataframe containing degree connectivity for each gene in the network.
    """
    # load and customize STRINGv11 network for analysis (evidence types, edge weight)
    string_net = _load_string(version)
    string_net_all_genes = list(set(string_net['node1'].unique().tolist() + string_net['node2'].unique().tolist()))
    evidence_lst = _get_evidence_types(evidences)
    string_net = _select_evidences(evidence_lst, string_net)
    string_net = _get_combined_score(string_net)

    # Filtering network for edge weight
    edge_weight = _get_edge_weight(edge_confidence)
    string_net = string_net[string_net['score'] >= edge_weight]

    # get degree connectivity after edgeweight filtering
    g_string_net = nx.from_pandas_edgelist(string_net[['node1', 'node2']], 'node1', 'node2')
    g_string_net_degree = dict(g_string_net.degree)
    string_net_degree_df = pd.DataFrame(index=string_net_all_genes)
    string_net_degree_df['degree'] = string_net_degree_df.index.map(g_string_net_degree)
    # fillna with zeros for genes w/o connections meeting edge weight filter (edgeWeight variable)
    string_net_degree_df.fillna(0, inplace=True)
    # round each degree to nearest 10
    string_net_degree_df['degree_rounded'] = string_net_degree_df['degree'].apply(lambda x: round(x / 10) * 10)

    return string_net, string_net_all_genes, string_net_degree_df

def _get_reactomes(min_size: int, max_size: int) -> dict:
    """
    Returns a dictionary of Reactome pathways with genes as values, filtered by size.

    Args:
        min_size (int): Minimum number of genes in a Reactome pathway.
        max_size (int): Maximum number of genes in a Reactome pathway.

    Returns:
        dict: A dictionary with Reactome pathway names as keys and gene lists as values.
    """
    reactomes = _load_reactome()
    #subtract 1 b/c name of Reactome is included with genes
    reactomes = [x for x in reactomes if len(x) - 1 >= min_size]
    reactomes = [x for x in reactomes if len(x) - 1 <= max_size]
    reactomes_names = [x[0] for x in reactomes]
    reactomes_genes = [x[1:] for x in reactomes]
    reactome_dict = dict(zip(reactomes_names, reactomes_genes))
    return reactome_dict

def _get_kegg_pathways(min_size:int, max_size:int) -> dict:
    """
    Returns a dictionary of KEGG pathways with gene sets between min_size and max_size.

    Args:
    min_size (int): Minimum size of gene set.
    max_size (int): Maximum size of gene set.

    Returns:
    dict: A dictionary of KEGG pathways with gene sets between min_size and max_size.
    """
    kegg_stream = pkg_resources.resource_stream(__name__, 'data/KEGGpathways_20230620_Homo_sapiens.gmt')
    kegg = kegg_stream.readlines()
    kegg = [x.decode('utf-8').strip('\n') for x in kegg]
    kegg = [x.split('\t') for x in kegg]
    kegg_names = [x[0] for x in kegg]
    kegg_genes = [x[1].split(' ') for x in kegg]
    kegg_dict = dict(zip(kegg_names, kegg_genes))

    kegg_dict_filtered = {}
    for k, v in kegg_dict.items():
        if len(v) >= min_size and len(v) <= max_size:
            kegg_dict_filtered[k] = v
    return kegg_dict_filtered

def _get_wikipathways(min_size:int, max_size:int) -> dict:
    """
    Returns a dictionary of WikiPathways gene sets with sizes between min_size and max_size.

    Args:
    - min_size (int): minimum size of gene set
    - max_size (int): maximum size of gene set

    Returns:
    - wiki_dict (dict): dictionary of WikiPathways gene sets with sizes between min_size and max_size
    """
    wiki_stream = pkg_resources.resource_stream(__name__, 'data/wikipathways_20230610_Homo_sapiens_genes.gmt')
    wiki = wiki_stream.readlines()
    wiki = [x.decode('utf-8').strip('\n') for x in wiki]
    wiki = [x.split('\t') for x in wiki]
    # Subtract 1 b/c name of Wiki is included with genes
    wiki = [x for x in wiki if len(x) - 1 >= min_size]
    wiki = [x for x in wiki if len(x) - 1 <= max_size]
    wiki_names = [x[0] for x in wiki]
    wiki_genes = [x[1:] for x in wiki]
    wiki_dict = dict(zip(wiki_names, wiki_genes))
    return wiki_dict

def _get_functional_sets(min_size: int, max_size: int) -> (list, list): # type: ignore
    """
    Returns a list of functional groups and their names based on the minimum and maximum size of the groups.

    Parameters:
    min_size (int): The minimum size of the functional groups.
    max_size (int): The maximum size of the functional groups.

    Returns:
    tuple: A tuple containing a list of functional groups and their names.
    """
    # load reactomes
    reactomes_dict = _get_reactomes(min_size, max_size)
    # Load GO Terms
    goterms_bp_dict, goterms_cc_dict, goterms_mf_dict = _get_go_terms(min_size, max_size)
    # Load KEGG pathways
    kegg_dict = _get_kegg_pathways(min_size, max_size)
    # Load Wikipathways
    wiki_dict = _get_wikipathways(min_size, max_size)
    # Format output
    functional_groups = [reactomes_dict, goterms_bp_dict, goterms_cc_dict, goterms_mf_dict, kegg_dict, wiki_dict]
    functional_groups_names = ['reactomes', 'go_bp', 'go_cc', 'go_mf', 'kegg','wiki']
    return functional_groups, functional_groups_names

def _clean_query(gene_list:list, source_names:Any) -> (dict, list): # type: ignore
    """
    Given a list of genes, creates a dictionary where each key is a set name and each value is a gene from the input list.

    Args:
    gene_list (list): A list of genes.

    Returns:
    dict: A dictionary where each key is a set name and each value is a gene from the input list.
    """
    gene_list = [x for x in gene_list if x != False]
    if source_names:
        sets = source_names
    else:
        sets = [f'set_{i+1}' for i in range(len(gene_list))]
    gene_dict = {}
    for i in range(len(gene_list)):
        gene_dict[sets[i]] = gene_list[i]
    return gene_dict, sets

def _get_gene_sources(set_dict: dict) -> (dict, list): # type: ignore
    """
    Given a dictionary of sets, where each set contains proteins, returns a dictionary where the keys are proteins and the values are lists of sets that contain the protein.
    Also returns a list of all proteins in the dictionary.

    Args:
        set_dict (dict): A dictionary where each key is a set of proteins.

    Returns:
        A tuple containing two elements:
        - A dictionary where the keys are proteins and the values are lists of sets that contain the protein.
        - A list of all proteins in the dictionary.
    """
    gene_source = {}
    for k, v in set_dict.items():
        for protein in set_dict[k]:
            if protein in gene_source:
                gene_source[protein].append(k)
            else:
                gene_source[protein] = [k]
    return gene_source, list(gene_source.keys())

def _check_connection(x:str, y:str, gene_lst:list) -> str:
    """
    Check if two genes are present in the given gene list.

    Args:
    x (str): Name of first gene.
    y (str): Name of second gene.
    gene_lst (list): List of genes to check against.

    Returns:
    str: 'yes' if both genes are present in the list, 'no' otherwise.
    """
    if x in gene_lst and y in gene_lst:
        out = 'yes'
    else:
        out = 'no'
    return out

def _get_node_pair(x:str, y:str) -> str:
    """
    Given two node names, returns a string representation of the pair in sorted order.

    Args:
    x (str): The first node name.
    y (str): The second node name.

    Returns:
    str: A string representation of the pair in sorted order.
    """
    pair = [x, y]
    pair.sort()
    return str(pair)

def _get_input_gene_network(gene_lst: list, network: pd.DataFrame) -> pd.DataFrame:
    """
    Given a list of genes and a gene network, returns a subnetwork containing only the genes in the input list and their
    direct neighbors in the network.

    Args:
        gene_lst (list): A list of gene names.
        network (pd.DataFrame): A pandas DataFrame representing a gene network, with columns 'node1', 'node2', and
        'weight'.

    Returns:
        pd.DataFrame: A pandas DataFrame representing the subnetwork containing only the input genes and their direct
        neighbors in the input network, with columns 'node1', 'node2', 'weight', 'inputGenePair', and 'pair'.
    """
    n_df = network.copy()
    n_df1 = n_df[n_df['node1'].isin(gene_lst)]
    n_df2 = n_df[n_df['node2'].isin(gene_lst)]
    n_df_final = pd.concat([n_df1, n_df2])
    n_df_final['inputGenePair'] = n_df_final.apply(lambda x: _check_connection(x['node1'], x['node2'], gene_lst), axis=1)
    n_df_final = n_df_final[n_df_final['inputGenePair'] == 'yes']
    n_df_final.drop_duplicates(inplace=True)
    n_df_final['pair'] = n_df_final.apply(lambda x: _get_node_pair(x['node1'], x['node2']), axis=1)
    n_df_final.sort_values(by='node1', inplace=True)
    n_df_final.drop_duplicates(subset=['pair'], inplace=True, keep='first')

    return n_df_final

def _mcl_analysis(network_df:pd.DataFrame, inflation:Any, verbose:int = 0) -> (list, float, nx.Graph, dict): # type: ignore
    """
    Runs the MCL algorithm on a given network and returns the clusters, the inflation parameter used, the graph, and a dictionary of clusters.

    Args:
    - network_df: a pandas DataFrame containing the edges of the network
    - inflation: the inflation parameter to use for the MCL algorithm. If None, the function will identify the best inflation parameter using modularity (Q).

    Returns:
    - clusters: a list of clusters, where each cluster is a list of nodes
    - max_q_inflation: the inflation parameter used for the MCL algorithm
    - G: a NetworkX graph object representing the network
    - clusters_dict: a dictionary where the keys are the names of the clusters and the values are lists of nodes in each cluster
    """
    G = nx.from_pandas_edgelist(network_df[['node1', 'node2']], 'node1', 'node2')
    # Build adjacency matrix
    #A = nx.to_numpy_matrix(G)
    A = nx.to_numpy_array(G)

    # check for user defined inflation parameter
    if inflation != None:
        max_q_inflation = float(inflation)
        print('Using manually set inflation parameter:', str(max_q_inflation))
        # Run MCL algorithm
        try:
            result = mc.run_mcl(A, inflation = max_q_inflation)
            clusters = mc.get_clusters(result)
        except ValueError:
            raise ValueError('Query network has no edges. Please adjust input genes or edge confidence threshold.')

        #export cluster proteins
        nodes_list = np.array(list(G.nodes()))
        clusters_dict = {}
        count = 0
        for i in clusters:
            cluster_genes = nodes_list[list(i)]
            clusters_dict['cluster_'+ str(count+1)] = list(cluster_genes)
            count += 1
    else:
        mod_values = {}
        #identify best inflation paramater using modularity (Q)
        for inflation in [i / 10 for i in range(15, 30)]:
            try: result = mc.run_mcl(A, inflation=float(inflation))
            except ValueError: raise ValueError('Query network has no edges. Please adjust input genes or edge confidence threshold.')
            matrix = np.matrix(result)
            clusters = mc.get_clusters(matrix)
            Q = mc.modularity(matrix=matrix, clusters=clusters)
            mod_values[inflation] = Q
        # identify inflation parameter with highest modularity
        max_q_inflation = 1.0
        max_q = 0
        for k, v in mod_values.items():
            if v > max_q:
                max_q = v
                max_q_inflation = k
                response = "Using algorithmically determined inflation parameter: " + str(max_q_inflation)
        if max_q_inflation == 1:
            response = "Optimal inflation could not be determined due to too few query edges: Using default inflation parameter: 3.0"
            max_q_inflation = 3.0
            max_q = 3.0
        # Run MCL algorithm with optimized inflation parameter
        if verbose > 0:
            print(response)
        result = mc.run_mcl(A, inflation = max_q_inflation)
        matrix = np.matrix(result)
        clusters = mc.get_clusters(matrix)
        # export cluster proteins
        nodes_list = np.array(list(G.nodes()))
        clusters_dict = {}
        count = 0
        for i in clusters:
            cluster_genes = nodes_list[list(i)]
            clusters_dict['cluster_' + str(count + 1)] = list(cluster_genes)
            count += 1

    return clusters, max_q_inflation, G, clusters_dict

def _matrix_fdr(matrix:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the false discovery rate (FDR) adjusted p-values (q-values) for a given matrix of p-values.

    Args:
    matrix (pd.DataFrame): A pandas DataFrame containing the p-values to be adjusted.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the adjusted p-values (q-values).
    """
    df = matrix.copy()
    df.sort_values(by = 'pval', inplace = True)
    pvals = list(df['pval'])
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=True)[1]
    df['qval'] = qvals
    return df

def _cluster_functional_enrichment(biological_groups:list, biological_groups_names:list, cluster_dict:dict, all_string_genes:list) -> dict:
    """
    Calculates functional enrichment for each cluster in the cluster_dict based on the biological groups provided.

    Args:
    - biological_groups (list): A list of dictionaries where each dictionary contains a set of genes for a specific biological group.
    - biological_groups_names (list): A list of names for each biological group.
    - cluster_dict (dict): A dictionary where each key is a cluster name and each value is a list of genes in that cluster.
    - all_string_genes (list): A list of all genes in the background.

    Returns:
    - enrichment_dict_final (dict): A dictionary where each key is a biological group name and each value is a dictionary of cluster names and their corresponding functional enrichment dataframes.
    """
    enrichment_dict_final = {}
    for i in range(len(biological_groups)):
        groups = biological_groups[i]
        all_paths = list(groups.keys())

        enrichment_df_dict = {}
        # testing enrichment across all pathways for each cluster
        for k, v in cluster_dict.items():
            summary_df = pd.DataFrame(index=all_paths)
            summary_df['#Genes_Bkgd'] = len(all_string_genes)
            summary_df['#clusterGenes'] = len(v)
            summary_df['PathwayGenes'] = summary_df.index.map(groups)
            summary_df['#PathwayGenes'] = summary_df['PathwayGenes'].apply(lambda x: len(x))
            summary_df['clusterGeneOverlap'] = summary_df['PathwayGenes'].apply(
                lambda x: len(list(set(x).intersection(set(v))))
            )
            summary_df['pval'] = hypergeom.sf(
                summary_df['clusterGeneOverlap'] - 1,
                summary_df['#Genes_Bkgd'],
                summary_df['#PathwayGenes'],
                summary_df['#clusterGenes']
            )
            summary_df = _matrix_fdr(summary_df)
            # k = cluster name; v = df with pathway enrichments
            enrichment_df_dict[k] = summary_df
        enrichment_dict_final[biological_groups_names[i]] = enrichment_df_dict
    return enrichment_dict_final

def _get_node_degree_dict(unique_genes:list, degree_df:pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the degree_df DataFrame with only the rows corresponding to the uniqueGenes list.

    Parameters:
    uniqueGenes (list): A list of unique gene names.
    degree_df (pd.DataFrame): A DataFrame containing the degree of each gene in a network.

    Returns:
    pd.DataFrame: A copy of the degree_df DataFrame with only the rows corresponding to the uniqueGenes list.
    """
    df = degree_df.copy()
    df = df.loc[unique_genes]
    return df

def _load_reactome_genes(min_group_size:int, max_group_size:int) -> list:
    """
    Load genes from the Reactome database that belong to functional sets with group sizes between minGroupSize and maxGroupSize.

    Args:
    - minGroupSize (int): minimum group size of functional sets to include genes from
    - maxGroupSize (int): maximum group size of functional sets to include genes from

    Returns:
    - genes (list): list of unique genes belonging to functional sets with group sizes between minGroupSize and maxGroupSize
    """
    groups_lst = _get_functional_sets(min_group_size, max_group_size)
    reactomes = groups_lst[0]
    allgenes = []
    for k, v in reactomes.items():
        allgenes.extend(v)
    return list(set(allgenes))

def _create_random_degree_matched_set(gene_sets:dict, background_genes:list, string_net_all_genes:list, string_net_degree_df:pd.DataFrame, named_sources:list, min_group_size:int, max_group_size:int) -> dict:
    """
    Creates a random set of genes that are degree-matched to the input gene set.

    Args:
    - gene_sets (dict): A dictionary where the keys are gene set names and the values are lists of genes.
    - string_net_all_genes (list): A list of all genes in the STRING network.
    - string_net_degree_df (pd.DataFrame): A pandas DataFrame containing the degree of each gene in the STRING network.
    - named_sources (list): A list of sources for each gene set.
    - min_group_size (int): The minimum size of each random gene set.
    - max_group_size (int): The maximum size of each random gene set.

    Returns:
    - random_sets (dict): A dictionary where the keys are gene set names and the values are lists of randomly selected genes that are degree-matched to the input gene set.
    """
    random_sets = {}
    # Filter for your background set of genes
    degree_df = string_net_degree_df.copy()
    degree_df = degree_df[degree_df.index.isin(background_genes)]
    for k, v in gene_sets.items():
        unique_mapped_genes = v
        # filter for genes mapped to STRING
        unique_mapped_genes = [x for x in unique_mapped_genes if x in string_net_all_genes]
        unique_mapped_genes_degree_df = _get_node_degree_dict(
            unique_mapped_genes,
            string_net_degree_df
        )
        unique_mapped_genes_degree_df = pd.DataFrame(unique_mapped_genes_degree_df.groupby('degree_rounded')['degree'].count())
        #in this dictionary: key is degree, value is count of genes with that degree
        unique_mapped_genes_degree_dict = dict(zip(unique_mapped_genes_degree_df.index.tolist(), unique_mapped_genes_degree_df['degree'].tolist()))
        # add filter step to check source of true gene set
        set_num = int(named_sources.index(k))
        random_seed = int(time.time()) + (uuid.uuid4().int & (1<<32)-1)
        random_genes = []
        if str(named_sources[set_num]) == 'Reactomes':
            reactome_genes = _load_reactome_genes(min_group_size, max_group_size)
            rng = np.random.default_rng(seed = random_seed)
            for k1, v1 in unique_mapped_genes_degree_dict.items():
                degree_df_matched = degree_df[degree_df['degree_rounded']==k1]
                degree_df_matched = degree_df_matched[degree_df_matched.index.isin(reactome_genes)]
                degree_matched_genes = degree_df_matched.index.tolist()
                x = rng.choice(degree_matched_genes, v1, replace=False).tolist()
                random_genes.extend(x)
        else:
            rng = np.random.default_rng(seed = random_seed)
            for k1, v1 in unique_mapped_genes_degree_dict.items():
                degree_df_matched = degree_df[degree_df['degree_rounded']==k1]
                degree_matched_genes = degree_df_matched.index.tolist()
                x = rng.choice(degree_matched_genes, v1, replace=False).tolist()
                random_genes.extend(x)
        random_sets[k] = random_genes
    return random_sets

def _mcl_analysis_random(network_df:pd.DataFrame, true_inflat_parameter:float) -> (list, dict): # type: ignore
    """
    Runs Markov Clustering Algorithm (MCL) on a given network and returns the clusters and their constituent genes.

    Args:
    - network_df (pd.DataFrame): A pandas DataFrame containing the edges of the network.
    - true_inflat_parameter (float): The inflation parameter used in the MCL algorithm.

    Returns:
    - clusters (list): A list of sets, where each set contains the indices of the genes in a cluster.
    - clusters_dict (dict): A dictionary where the keys are the names of the clusters and the values are lists of the genes in each cluster.
    """
    G = nx.from_pandas_edgelist(network_df[['node1', 'node2']], 'node1', 'node2')
     # Build adjacency matrix
    A = nx.to_numpy_array(G)
    try: result = mc.run_mcl(A, inflation = true_inflat_parameter)
    except ValueError:
        return [()], {}
    clusters = mc.get_clusters(result)

    # export cluster proteins
    nodes_list = np.array(list(G.nodes()))
    clusters_dict = {}
    count = 0
    for i in clusters:
        cluster_genes = nodes_list[list(i)]
        clusters_dict['cluster_' + str(count + 1)] = list(cluster_genes)
        count += 1

    return clusters, clusters_dict

def _sort_cluster_dict(cluster_dict:dict, sources:dict) -> (dict, pd.DataFrame): # type: ignore
    """
    Sorts a dictionary of gene clusters by the number of genes in each cluster, and returns a new dictionary
    with the same clusters, but with new names based on their size (e.g. "cluster_0", "cluster_1", etc.).
    Also returns a DataFrame with two columns: "gene" and "cluster", where each row represents a gene and its
    corresponding cluster.

    Args:
        cluster_dict (dict): A dictionary where the keys are cluster names and the values are lists of genes.

    Returns:
        A tuple containing two objects:
        - A dictionary with the same clusters as `cluster_dict`, but with new names based on their size.
        - A DataFrame with two columns: "gene" and "cluster", where each row represents a gene and its
          corresponding cluster.
    """
    hold_dict = {}
    for k, v in cluster_dict.items():
        if len(v) > 1:
            hold_dict[k] = len(v)
    hold_df = pd.DataFrame.from_dict(hold_dict, orient='index')
    hold_df = hold_df.sort_values(by=0, ascending=False)
    # Rename clusters ordered by their size
    sorted_cluster_dict = {}
    for i in range(len(hold_df)):
        old_cluster = hold_df.index[i]
        sorted_cluster_dict[f"cluster_{i+1}"] = cluster_dict[old_cluster]
    # Expand dict length-wise
    data = [(gene, cluster) for cluster, genes in sorted_cluster_dict.items() for gene in genes]
    sorted_cluster_df = pd.DataFrame(data, columns=['gene', 'cluster'])
    # Add a new column 'source' to the DataFrame
    sorted_cluster_df['source'] = sorted_cluster_df['gene'].map(sources)
    return sorted_cluster_dict, sorted_cluster_df

def _multiprocessing_mcl(gene_sets:dict, background_genes:list, string_net_all_genes:list, string_net_degree_df:pd.DataFrame, source_names:list, min_group_size:int, max_group_size:int, string_net:pd.DataFrame, true_inflation_parameter:float, functional_groups:list, functional_groups_names:list) -> list:
    """
    Runs MCL analysis on a given set of gene sets and returns the functional enrichment of the resulting clusters.

    Args:
    - gene_sets (dict): A dictionary of gene sets.
    - string_net_all_genes (list): A list of all genes in the STRING network.
    - string_net_degree_df (pd.DataFrame): A DataFrame containing the degree of each gene in the STRING network.
    - source_names (list): A list of source names.
    - min_group_size (int): The minimum size of a cluster.
    - max_group_size (int): The maximum size of a cluster.
    - string_net (pd.DataFrame): The STRING network.
    - true_inflation_parameter (float): The true inflation parameter.
    - functional_groups (list): A list of functional groups.
    - functional_groups_names (list): A list of names for the functional groups.

    Returns:
    - A list containing the functional enrichment of the resulting clusters and the random sets clusters.
    """
    random_sets_clusters = []
    random_sets = _create_random_degree_matched_set(
        gene_sets,
        background_genes,
        string_net_all_genes,
        string_net_degree_df,
        source_names,
        min_group_size,
        max_group_size
    )
    _, random_input_genes = _get_gene_sources(random_sets)
    random_gene_network = _get_input_gene_network(
        random_input_genes,
        string_net
    )
    random_clusters, random_clusters_dict = _mcl_analysis_random(
        random_gene_network,
        true_inflation_parameter
    )
    random_sets_clusters.append(random_clusters)
    random_clusters_enrichment = _cluster_functional_enrichment(functional_groups, functional_groups_names, random_clusters_dict, string_net_all_genes)
    return [random_clusters_enrichment, random_sets_clusters]

def _multiprocessing_randomization(arg:tuple) -> list:
    """
    This function performs multiprocessing randomization.

    Args:
    - arg (tuple): A tuple containing the following arguments:
        - gene_sets (list): A list of gene sets.
        - string_net_all_genes (list): A list of all genes in the STRING network.
        - string_net_degree_df (pandas.DataFrame): A pandas DataFrame containing the degree of each gene in the STRING network.
        - args (Namespace): A Namespace object containing the command-line arguments.
        - min_group_size (int): The minimum size of a gene set.
        - max_group_size (int): The maximum size of a gene set.
        - string_net (networkx.Graph): A networkx Graph object representing the STRING network.
        - true_inflation_parameter (float): The true inflation parameter.
        - functional_groups (dict): A dictionary containing functional groups.
        - functional_groups_names (list): A list of functional group names.

    Returns:
    - out_lst (list): A list of output values.
    """
    gene_sets = arg[0]
    background_genes = arg[1]
    string_net_all_genes = arg[2]
    string_net_degree_df = arg[3]
    source_names = arg[4]
    min_group_size = arg[5]
    max_group_size = arg[6]
    string_net = arg[7]
    true_inflation_parameter = arg[8]
    functional_groups = arg[9]
    functional_groups_names = arg[10]
    out_lst = _multiprocessing_mcl(
        gene_sets,
        background_genes,
        string_net_all_genes,
        string_net_degree_df,
        source_names,
        min_group_size,
        max_group_size,
        string_net,
        true_inflation_parameter,
        functional_groups,
        functional_groups_names
    )
    return out_lst

def _pool_multiprocessing_randomization(random_iter:int, gene_sets:dict, background_genes:list, string_net_all_genes:list, string_net_degree_df:pd.DataFrame, source_names:list, min_group_size:int, max_group_size:int, string_net:pd.DataFrame, true_inflation_parameter:float, functional_groups:list, functional_groups_names:list, cores:int) -> list:
    """
    This function performs multiprocessing randomization of gene sets.

    Args:
    - random_iter (int): Number of times to randomly shuffle gene sets.
    - gene_sets (dict): Dictionary of gene sets.
    - string_net_all_genes (list): List of all genes in the STRING network.
    - string_net_degree_df (pd.DataFrame): DataFrame of STRING network degree for each gene.
    - source_names (list): List of source names.
    - min_group_size (int): Minimum group size.
    - max_group_size (int): Maximum group size.
    - string_net (pd.DataFrame): DataFrame of STRING network.
    - true_inflation_parameter (float): True inflation parameter.
    - functional_groups (list): List of functional groups.
    - functional_groups_names (list): List of functional group names.
    - cores (int): Number of cores to use for multiprocessing.

    Returns:
    - output (list): List of outputs from multiprocessing randomization.
    """
    args_ = tuple(zip(
        [gene_sets] * random_iter,
        [background_genes] * random_iter,
        [string_net_all_genes] * random_iter,
        [string_net_degree_df] * random_iter,
        [source_names] * random_iter,
        [min_group_size] * random_iter,
        [max_group_size] * random_iter,
        [string_net] * random_iter,
        [true_inflation_parameter] * random_iter,
        [functional_groups] * random_iter,
        [functional_groups_names] * random_iter))
    pool = mp.Pool(processes=cores)
    output = pool.map(_multiprocessing_randomization, args_)
    pool.close()
    pool.join()
    return output

def _get_top_pval_random_clusters(random_sets_clusters_enrichment:list) -> (list, list, list, list, list, list): # type: ignore
    """
    Given a list of nested dictionaries containing random cluster enrichment data, returns the top p-value for each
    pathway across all iterations for each biological group type.

    Args:
    - random_sets_clusters_enrichment (list): A list of nested dictionaries containing random cluster enrichment data.

    Returns:
    - A tuple of six lists, each containing a dictionary of the top p-value for each pathway across all iterations for
    each biological group type. The six biological group types are reactomes, go_bp, go_mf, go_cc, kegg, and wiki.
    """
    top_reactome_pval = []
    top_go_bp_pval = []
    top_go_mf_pval = []
    top_go_cc_pval = []
    top_kegg_pval = []
    top_wiki_pval = []
    # Set of nested dicts; length matches number randomizations; 4 dicts per randomization
    for random_iter in random_sets_clusters_enrichment:
        for biological_group_type, df_dict in random_iter.items():
            top_pathway_pval = {}
            # Each sub-dict is random cluster (key) with biological group enrichment df (v)
            for k, v in df_dict.items():
                df = v
                pathways = df.index.tolist()
                for path in pathways:
                    pval = df.loc[path, 'pval']
                    # collect the top pvalue for each biological group across each iteration
                    if path in top_pathway_pval:
                        if top_pathway_pval[path] < pval:
                            continue
                        top_pathway_pval[path] = pval
                    else:
                        top_pathway_pval[path] = pval
            if biological_group_type == 'reactomes':
                top_reactome_pval.append(top_pathway_pval)
            elif biological_group_type == 'go_bp':
                top_go_bp_pval.append(top_pathway_pval)
            elif biological_group_type == 'go_cc':
                top_go_cc_pval.append(top_pathway_pval)
            elif biological_group_type == 'go_mf':
                top_go_mf_pval.append(top_pathway_pval)
            elif biological_group_type == 'kegg':
                top_kegg_pval.append(top_pathway_pval)
            elif biological_group_type == 'wiki':
                top_wiki_pval.append(top_pathway_pval)
    return top_reactome_pval, top_go_bp_pval, top_go_mf_pval, top_go_cc_pval, top_kegg_pval, top_wiki_pval

def _merge_pvals(group_lst:list) -> dict:
    """
    Merge p-values from multiple groups into a single dictionary.

    Args:
    - group_lst (list): A list of dictionaries, where each dictionary contains p-values for a group.

    Returns:
    - output_dict (dict): A dictionary where each key is a p-value and each value is a list of p-values from all groups.
    """
    output_dict = {}
    for i in group_lst:
        for k, v in i.items():
            if k in output_dict:
                output_dict[k].append(v)
            else:
                output_dict[k] = [v]
    return output_dict

def _merge_top_pvals_random_clusters(top_reactome_pval:list, top_go_bp_pval:list, top_go_mf_pval:list, top_go_cc_pval:list, top_kegg_pval:list, top_wiki_pval:list, functional_groups_names:list) -> dict:
    """
    Merge the top p-values for each functional group into a single dictionary.

    Args:
    top_reactome_pval (list): List of top p-values for Reactome.
    top_go_bp_pval (list): List of top p-values for Gene Ontology Biological Process.
    top_go_mf_pval (list): List of top p-values for Gene Ontology Molecular Function.
    top_go_cc_pval (list): List of top p-values for Gene Ontology Cellular Component.
    top_kegg_pval (list): List of top p-values for KEGG.
    top_wiki_pval (list): List of top p-values for WikiPathways.
    functional_groups_names (list): List of functional group names.

    Returns:
    dict: A dictionary containing the merged top p-values for each functional group.
    """
    top_reactome_pval_merged = _merge_pvals(top_reactome_pval)
    top_go_bp_pval_merged = _merge_pvals(top_go_bp_pval)
    top_go_cc_pval_merged = _merge_pvals(top_go_cc_pval)
    top_go_mf_pval_merged = _merge_pvals(top_go_mf_pval)
    top_kegg_pval_merged = _merge_pvals(top_kegg_pval)
    top_wiki_pval_merged = _merge_pvals(top_wiki_pval)
    output_dict = {}
    output_dict[functional_groups_names[0]] = top_reactome_pval_merged
    output_dict[functional_groups_names[1]] = top_go_bp_pval_merged
    output_dict[functional_groups_names[2]] = top_go_cc_pval_merged
    output_dict[functional_groups_names[3]] = top_go_mf_pval_merged
    output_dict[functional_groups_names[4]] = top_kegg_pval_merged
    output_dict[functional_groups_names[5]] = top_wiki_pval_merged

    return output_dict

def _annotate_percentile_of_score(a:float , b:int) -> float:
    """
    Given a score and a list of scores, returns the percentile of the score in the list.

    Args:
        a (float): The score to calculate the percentile of.
        b (int): The list of scores to use for the percentile calculation.

    Returns:
        float: The percentile of the score in the list.
    """
    if np.isnan(b).all():
        warnings.warn("Warning: All background tests contain no edges. Please decrease edge confidence")
        b = [1.0]
    return percentileofscore(b, a)

def _annotated_true_clusters_enrich_sig(true_clusters_enrich_df_dict:dict, pval_merged_tuple_set:dict) -> dict:
    """
    Annotate true clusters with their significance level based on a set of random iterations.

    Parameters:
    true_clusters_enrich_df_dict (dict): A dictionary of dataframes containing gene set enrichment results for each true cluster.
    pval_merged_tuple_set (dict): A dictionary of tuples containing the top p-values for each biological group.

    Returns:
    dict: A dictionary of dataframes containing annotated gene set enrichment results for each true cluster.
    """
    # Biological group = pathways (i.e. "reactomes")
    # cluster_dict = (i.e. {'cluster_1': dataframe of reactomes gene set enrichment})
    new_dict = {}
    for biological_group, cluster_dict in true_clusters_enrich_df_dict.items():
        for k, v in cluster_dict.items():
            summary_df = v
            summary_df['RandomIterationTopPvals'] = summary_df.index.map(pval_merged_tuple_set[biological_group])
            summary_df['TrueClusterRanking'] = summary_df.apply(lambda x: _annotate_percentile_of_score(x['pval'], x['RandomIterationTopPvals']), axis = 1)
            if k not in new_dict:
                new_dict[k] = {}
            new_dict[k][biological_group] = summary_df
    # Get the combo of all biological groups
    for key, value in new_dict.items():
        new_dict[key]['combo'] = pd.concat([value['reactomes'], value['go_bp'], value['go_cc'], value['go_mf'], value['kegg'], value['wiki']], axis = 0)
    return new_dict

def functional_clustering(genes_1: list = False, genes_2: list = False, genes_3: Any = False, genes_4: Any = False, genes_5: Any = False, source_names: Any = False, gene_dict: Any = False, true_go_terms: Any = False, string_version:str = 'v11.0', evidences:list = ['all'], edge_confidence:str = 'highest', custom_background:Any = 'string', random_iter:int = 100, inflation:Any = None, pathways_min_group_size:int = 5, pathways_max_group_size: int = 100, cores:int = 1, savepath: Any = False, verbose:int = 0) -> (pd.DataFrame, pd.DataFrame, dict): # type: ignore
    """
        Perform functional clustering analysis on a set of genes.

        Args:
        - genes_1 (list): List of genes to be analyzed.
        - genes_2 (list): List of genes to be analyzed.
        - genes_3 (Any, optional): List of genes to be analyzed. Defaults to False.
        - genes_4 (Any, optional): List of genes to be analyzed. Defaults to False.
        - genes_5 (Any, optional): List of genes to be analyzed. Defaults to False.
        - source_names (Any, optional): List of source names for the genes. Defaults to False.
        - gene_dict (Any, optional): Dictionary of gene sets (as alternative to above; e.g. {'gene_set_1': ['gene1', 'gene2', ...], 'gene_set_2': ['gene3', 'gene4', ...]}). Defaults to False.
        - true_go_terms (list, optional): List of GO terms to perform hypergeometric overlap and nDiffusion on. Should define as a list of GO terms IDs (e.g. ['GO:0006955', 'GO:0006954', 'GO:0006956']). Defaults to False.
        - string_version (str, optional): STRING version to use. Defaults to 'v11.0'. Options include 'v10.0', 'v11.0', 'v11.5', 'v12.0'.
        - evidences (list, optional): List of evidences to be used for network cleaning. Defaults to ['all'].
        - edge_confidence (str, optional): Edge confidence level for network cleaning. Defaults to 'highest'.
        - custom_background (Any, optional): Custom background gene set. Defaults to 'string'. Options include 'string', 'ensembl', 'reactome', or user defined list.
        - random_iter (int, optional): Number of random iterations for cluster enrichment analysis. Defaults to 100.
        - inflation (Any, optional): Inflation parameter for MCL clustering. Defaults to None.
        - pathways_min_group_size (int, optional): Minimum group size for functional enrichment. Defaults to 5.
        - pathways_max_group_size (int, optional): Maximum group size for functional enrichment. Defaults to 100.
        - cores (int, optional): Number of cores to use for multiprocessing. Defaults to 1.
        - savepath (Any, optional): Path to save output files. Defaults to False.
        - verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
        - true_gene_network (pandas.DataFrame): DataFrame of the true gene network.
        - true_cluster_df (pandas.DataFrame): DataFrame of the true gene clusters.
        - true_clusters_enrichment_df_dict (dict): Dictionary of DataFrames containing functional enrichment results for each true gene cluster.
    """
    # Clean the network
    string_net, string_net_all_genes, string_net_degree_df = _load_clean_network(
        string_version,
        evidences,
        edge_confidence
    )
    # load manually curated gene sets for functional enrichment of clusters
    functional_groups, functional_groups_names = _get_functional_sets(
        pathways_min_group_size,
        pathways_max_group_size
    )
    # Determine background gene set
    if custom_background == 'string':
        background_genes = string_net_all_genes
    else:
        background_dict, background_name = _define_background_list(custom_background)
        background_genes = background_dict[background_name]
    # true gene sets and set of all input genes
    if genes_1:
        gene_sets, source_names = _clean_query([genes_1, genes_2, genes_3, genes_4, genes_5], source_names)
    else:
        gene_sets = gene_dict
        source_names = list(gene_dict.keys())
    gene_sources, input_genes = _get_gene_sources(gene_sets)
    # returns network with connections b/w all input_genes
    true_gene_network = _get_input_gene_network(input_genes, string_net)
    _, true_inflation_parameter, _, true_cluster_dict = _mcl_analysis(true_gene_network, inflation, verbose = verbose)
    true_cluster_dict, true_cluster_df = _sort_cluster_dict(true_cluster_dict, gene_sources)
    true_cluster_df = true_cluster_df.set_index('gene')
    # Perform True Cluster Functional Enrichment
    true_clusters_enrichment_df_dict = _cluster_functional_enrichment(
        functional_groups, functional_groups_names, true_cluster_dict, background_genes
    )
    # Perform Random Cluster Functional Enrichment
    randomization_output = _pool_multiprocessing_randomization(
        random_iter, gene_sets, background_genes, string_net_all_genes, string_net_degree_df, source_names, pathways_min_group_size, pathways_max_group_size, string_net, true_inflation_parameter, functional_groups, functional_groups_names, cores
    )
    random_sets_clusters = []
    random_sets_clusters_enrichment = []
    for i in randomization_output:
        random_sets_clusters_enrichment.append(i[0])
        random_sets_clusters.append(i[1][0])
    # Collect top p-value for each pathway term for each randomization iteration (
    # if there are 100 interations, then there will be 100 top pvalues for each pathway term)
    top_reactome_pval, top_go_bp_pval, top_go_mf_pval, top_go_cc_pval, top_kegg_pval, \
    top_wiki_pval = _get_top_pval_random_clusters(random_sets_clusters_enrichment)
    # merge top pvalues for each biological group type across each randomization iteration
    pval_merged_tuple_set = _merge_top_pvals_random_clusters(top_reactome_pval, top_go_bp_pval, top_go_mf_pval, top_go_cc_pval, top_kegg_pval, top_wiki_pval,functional_groups_names)
    # REMOVED - find zscores across different cluster size thresholds -- clusters_zscores, clusters_zscores_at_each_cluster_size
    # annotated true cluster enrichments with randomSets cluster enrichment pvalues
    true_clusters_enrichment_df_dict = _annotated_true_clusters_enrich_sig(
        true_clusters_enrichment_df_dict,
        pval_merged_tuple_set
    )
    # Assess true go term overlap with true_go_terms
    if true_go_terms:
        if savepath:
            val_savepath = os.path.join(savepath, 'Functional_Clustering/Validation/')
            os.makedirs(val_savepath, exist_ok=True)
        else:
            pass
        
        # parse out molecular function, biological process, and cellular
        true_go_map, len_go, all_go = _parse_true_go_terms(true_go_terms, pathways_min_group_size, pathways_max_group_size)

        # Loop through each set to test true positives
        for true_go_type, true_go_ids in true_go_map.items():
            if len(true_go_ids) == 0:
                continue

            for cluster_num, sub_dict in true_clusters_enrichment_df_dict.items():
                if savepath:
                    sub_savepath = os.path.join(val_savepath, f'{cluster_num}/{true_go_type}/')
                    os.makedirs(sub_savepath, exist_ok=True)
                # Filter for significant enrichment
                query_enrich = sub_dict[true_go_type]
                # Split the index into pathway_id and pathway_name
                query_enrich['pathway_id'] = query_enrich.index.str.split('/').str[0]
                # Annotate if the GO term is a true positive term
                query_enrich['true_positive'] = query_enrich['pathway_id'].apply(lambda x: 1 if x in true_go_ids else 0)
                sub_dict[true_go_type] = query_enrich
                query_enrich = query_enrich[query_enrich['qval'] < 0.01]
                query_terms = query_enrich['pathway_id'].unique().tolist()
                if len(query_terms) == 0:
                    continue

                # Assess overlap with the true go terms
                venn_img, p_val = _hypergeometric_overlap(query_terms, true_go_ids, len_go[true_go_type], f'{true_go_type}-({len(true_go_ids)})', plot_venn = True, plot_fontsize = 12, plot_fontface = 'sans-serif')
                if not isinstance(venn_img, bool):
                    venn_img.save(sub_savepath + f'{true_go_type}_overlap_venn.png')
                
                # Test how significant the overlap is by pulling 100 random phenotypes of same size
                if p_val != 1:
                    z_dist_plot, z_score, rando_phenotypes, rando_overlaps = _test_overlap_with_random(query_terms, true_go_ids, len_go[true_go_type], all_go[true_go_type], 100, f'{true_go_type}-({len(true_go_ids)})', 12, 'sans-serif', plot_venn = True)
                    z_dist_plot.save(sub_savepath + f'{true_go_type}_z_dist_plot.png', bbox_inches = 'tight', pad_inches = 0.5)
                    pd.Series(rando_phenotypes).to_csv(sub_savepath + f'{true_go_type}_rando_phenotypes.csv', index = False, header = False)
                    pd.Series(rando_overlaps).to_csv(sub_savepath + f'{true_go_type}_rando_overlaps.csv', index = False, header = False)

                # Run nDiffusion for GO terms
                show_1_plot, show_1_z, show_2_plot, show_2_z = _go_term_ndiffusion(
                    go_type = true_go_type,
                    query_phenotypes = query_terms,
                    true_id_terms = true_go_ids,
                    true_keyword = true_go_type,
                    set_1_name = f"Query GO Terms ({len(query_terms)})",
                    n_iter = 100,
                    cores = cores,
                    savepath = sub_savepath
                )

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'Functional_Clustering/')
        os.makedirs(new_savepath, exist_ok=True)
        # Save true gene network
        true_gene_network.to_csv(new_savepath + 'GeneSetNetwork.csv', index = False)
        # save true_cluster_dict
        true_cluster_df.to_csv(new_savepath + 'TrueClusters.csv', index = True)
        # save random_sets
        with open(new_savepath + 'RandomSets_Clusters.txt', 'w') as f:
            for item in random_sets_clusters:
                f.write("%s\n" % item)
        # Save cluster enrichment dataframes - true_clusters_enrichment_df_dict
        enrich_save_path = new_savepath + 'Enrichment/'
        if not os.path.exists(enrich_save_path):
            os.makedirs(enrich_save_path)
        for cluster_num, sub_dict in true_clusters_enrichment_df_dict.items():
            combo_df = False
            for biological_group, df in sub_dict.items():
                df['biological_group'] = biological_group
                df['pathway'] = df.index
                column_order = ['pathway'] + [col for col in df if col != 'pathway']
                df = df[column_order]
                if combo_df is False:
                    combo_df = df
                else:
                    combo_df = pd.concat([combo_df, df], axis=0, ignore_index=True)
                df.to_csv(enrich_save_path + cluster_num + '_' + biological_group + '_enrichment.csv', index = False)
            combo_df = combo_df.sort_values(by=['qval', 'biological_group'], ascending = True)
            combo_df.to_csv(enrich_save_path + cluster_num + '_combo_enrichment.csv', index = False)

    return true_gene_network, true_cluster_df, true_clusters_enrichment_df_dict
#endregion



#region Statistical combo
def _check_df_format(df: pd.DataFrame) -> bool:
    """
    Validates if the DataFrame has exactly two columns,
    with the first column being of type str and the second column of type float.
    If the DataFrame does not meet these conditions, raises a ValueError.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.

    Raises:
    ValueError: If the DataFrame does not meet the specified conditions.
    """
    if isinstance(df, bool):
        return False
    # Check if the DataFrame has exactly two columns
    elif df.shape[1] != 2:
        raise ValueError("DataFrame must have exactly two columns")
    # Check if the first column is of type str
    elif not pd.api.types.is_string_dtype(df.iloc[:, 0]):
        raise ValueError("The first column of the DataFrame must be of type str")
    # Check if the second column is of type float
    elif not pd.api.types.is_float_dtype(df.iloc[:, 1]):
        raise ValueError("The second column of the DataFrame must be of type float")
    else:
        return True

def _merge_p_inputs(dfs:dict) -> pd.DataFrame:
    """
    Merges multiple DataFrame objects into a single DataFrame based on a common column.

    This function iterates over a dictionary of DataFrame objects. For each DataFrame, it first checks
    if it meets specific format criteria (using the _check_df_format function). It then filters the DataFrame
    to retain rows where the second column's value is less than or equal to 1. The function also renames the
    first column to 'gene' and the second column to a unique name based on its key in the dictionary.
    Finally, it merges these DataFrames on the 'gene' column, performing an inner join.

    Parameters:
    -----------
    dfs : dict
        A dictionary where keys are strings and values are DataFrame objects. Each DataFrame is expected
        to have a specific format (validated by _check_df_format).

    Returns:
    --------
    pd.DataFrame
        A single merged DataFrame containing the combined data from the input DataFrames, with rows filtered
        and columns renamed as described above.

    Raises:
    -------
    ValueError
        If any DataFrame does not meet the format criteria set by _check_df_format.

    Notes:
    ------
    - The _check_df_format function is used to validate each DataFrame. It should ensure that each DataFrame
      has exactly two columns with specific data types and possibly other criteria.
    - If the dictionary is empty or none of the DataFrames meet the format criteria, the function will return an empty DataFrame.

    Example:
    --------
    >>> dfs = {'df1': df1, 'df2': df2}
    >>> merged_df = _merge_p_inputs(dfs)
    """
    new_df = False
    for key, df in dfs.items():
        if (isinstance(df, pd.DataFrame)) and (_check_df_format(df)):
            df = df.copy()
            df = df[df.iloc[:, 1] <= 1]
            df.rename(columns = {df.columns[0]: "gene", df.columns[1]: f"p_{key}"}, inplace = True)
            df = df.set_index('gene')
            if isinstance(new_df, pd.DataFrame):
                new_df = df.merge(new_df, left_index = True, right_index = True, how = 'outer')
            else:
                new_df = df
    return new_df

def _cauchy_combination_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a Cauchy combination test on a given DataFrame.

    This function applies the Cauchy combination test to combine p-values from different tests
    (represented as columns in the DataFrame) for each row. The Cauchy combination test is a method
    to combine p-values when the number of combined tests varies across the tests, which is
    particularly useful in meta-analyses. The function calculates a combined p-value for each row
    by applying a transformation to the p-values and then averaging these transformed values.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame where the first column is an identifier (such as a gene name) and the remaining columns
        are p-values from different tests that need to be combined.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional column 'p.cauchy' that contains the combined p-value
        for each row calculated using the Cauchy combination test.

    Notes:
    ------
    - The input DataFrame is expected to have more than one column, with the first column as an identifier
      and the rest as numeric p-values.
    - The Cauchy combination test is sensitive to uniform p-values. Thus, it is suitable in scenarios where
      the null distribution of the p-values is close to uniform.

    Example:
    --------
    >>> df = pd.DataFrame({'gene': ['gene1', 'gene2'], 'study1': [0.01, 0.05], 'study2': [0.03, 0.02]})
    >>> combined_df = _cauchy_combination_test(df)
    """
    trial = np.matrix(df.iloc[:, 0:].astype(np.float64))
    t_not = np.mean(np.tan((0.5 - trial) * np.pi), axis=1)
    p_cau = 0.5 - np.arctan(t_not) / np.pi
    df['p.cauchy'] = [np.float64(x) for x in p_cau]
    return df

def _min_p(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the minimum p-value for each row in a DataFrame.

    This function iterates through each row of the input DataFrame and computes the minimum p-value
    across columns that contain p-values. The columns containing p-values are identified by the
    prefix 'p_'. It then adds these minimum p-values as a new column to the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame where specific columns contain p-values. These columns are identified by
        having a prefix 'p_' in their column names.

    Returns:
    --------
    pd.DataFrame
        The same input DataFrame with an additional column 'p.min', which contains the minimum
        p-value for each row across the specified p-value columns.

    Notes:
    ------
    - The function assumes that the DataFrame has multiple columns with p-values prefixed by 'p_'.
      These columns should contain numeric values.
    - If no columns with the 'p_' prefix are found, the 'p.min' column will contain NaN values.

    Example:
    --------
    >>> df = pd.DataFrame({'gene': ['gene1', 'gene2'], 'p_study1': [0.01, 0.05], 'p_study2': [0.03, 0.02]})
    >>> min_p_df = _min_p(df)
    """
    p_val = np.matrix(df[[x for x in df.columns if "p_" in x]])
    p_min = [np.min(p_val[i]) for i in range(len(p_val))]
    df['p.min'] = p_min
    return df

def _mcm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Minimum of Combined p-Values (MCM) method to a DataFrame.

    This function calculates the MCM for each row in the input DataFrame based on two specific
    p-value columns: 'p.cauchy' and 'p.min'. The MCM is calculated as the minimum of three values:
    1, twice the value in 'p.cauchy', and twice the value in 'p.min'. The resulting MCM values
    are added to the DataFrame as a new column 'p.mcm'.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing at least two columns, 'p.cauchy' and 'p.min', which hold the p-values
        that need to be combined using the MCM method.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional column 'p.mcm', containing the MCM values computed
        for each row.

    Notes:
    ------
    - The 'p.cauchy' and 'p.min' columns are assumed to contain valid p-values. If these columns
      do not exist, the function will raise an error.
    - This method is used in statistical meta-analysis to combine p-values from different sources
      in a conservative manner.

    Example:
    --------
    >>> df = pd.DataFrame({'gene': ['gene1', 'gene2'], 'p.cauchy': [0.02, 0.05], 'p.min': [0.03, 0.04]})
    >>> mcm_df = _mcm(df)
    """
    df["p.mcm"] = df[["p.cauchy", "p.min"]].apply(lambda x: min([1, 2*x[0], 2*x[1]]), axis=1)
    return df

def _cmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Combined Minimum and Complementary (CMC) p-values method to a DataFrame.

    This function calculates the CMC p-values for each row in the input DataFrame using two specific
    p-value columns: 'p.cauchy' and 'p.min'. The CMC method combines these p-values by transforming
    them via the tangent function, averaging these transformed values, and then applying the arctangent
    transformation. This process is designed to balance the conservative nature of the minimum p-value
    method with its complementary p-value, providing a more nuanced approach to combining p-values.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing at least two columns, 'p.cauchy' and 'p.min', which hold the p-values
        that need to be combined using the CMC method.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional column 'p.cmc', containing the CMC p-values computed
        for each row.

    Notes:
    ------
    - The 'p.cauchy' and 'p.min' columns are assumed to contain valid p-values. The function assumes
      that these columns are present and will raise an error if they are not found.
    - The CMC method is particularly useful in statistical meta-analysis and genomic studies where
      the combination of p-values from various tests is required.

    Example:
    --------
    >>> df = pd.DataFrame({'gene': ['gene1', 'gene2'], 'p.cauchy': [0.02, 0.05], 'p.min': [0.03, 0.04]})
    >>> cmc_df = _cmc(df)
    """
    t_not = np.mean(np.tan((0.5 - df[["p.cauchy", "p.min"]].values) * np.pi), axis=1)
    df["p.cmc"] = 0.5 - np.arctan(t_not) / np.pi
    return df

def _p_multiply(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies p-values across specified columns in a DataFrame.

    This function calculates the product of p-values for each row in the DataFrame. It identifies
    columns to include in the calculation by looking for columns with names that start with 'p_'.
    The product of these p-values is then stored in a new column, 'p.multiply'.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing one or more columns with p-values. These columns should be named
        such that they start with 'p_'.

    Returns:
    --------
    pd.DataFrame
        The same input DataFrame with an additional column 'p.multiply', which contains the product
        of p-values for each row across the specified p-value columns.

    Notes:
    ------
    - The function operates on columns that have a prefix 'p_'. It's assumed that these columns
      contain numeric p-values.
    - This method of multiplying p-values can be useful in certain statistical analyses where
      the joint significance of multiple tests is assessed.

    Example:
    --------
    >>> df = pd.DataFrame({'gene': ['gene1', 'gene2'], 'p_study1': [0.01, 0.05], 'p_study2': [0.03, 0.02]})
    >>> prod_p_df = _p_multiply(df)
    """
    sub_df = df[[x for x in df.columns if "p_" in x]]
    df['p.multiply'] = sub_df.prod(axis=1)
    return df

def statistical_combination(df_1:pd.DataFrame = pd.DataFrame(), df_2:pd.DataFrame = pd.DataFrame(), df_3:pd.DataFrame = pd.DataFrame(), df_4:pd.DataFrame = pd.DataFrame(), df_5:pd.DataFrame = pd.DataFrame(), df_6:pd.DataFrame = pd.DataFrame(), gene_df:pd.DataFrame = pd.DataFrame(), list_names:Any = False, savepath:Any = False) -> pd.DataFrame:
    """
    Combines statistical data from multiple DataFrames using various statistical methods.

    This function takes up to six DataFrames as input and combines them using a series of statistical
    methods. These methods include the Cauchy Combination Test (CCT), minimum p-value (minP) calculation,
    Minimum of Combined p-Values (MCM), Combined Minimum and Complementary p-values (CMC), and p-value
    multiplication. The results are optionally saved to a CSV file.

    Parameters:
    -----------
    df_1 : pd.DataFrame
        The first DataFrame to be included in the statistical combination.
    df_2 : pd.DataFrame
        The second DataFrame to be included in the statistical combination.
    df_3 : Any, optional
        The third DataFrame to be included in the statistical combination (default is False).
    df_4 : Any, optional
        The fourth DataFrame to be included in the statistical combination (default is False).
    df_5 : Any, optional
        The fifth DataFrame to be included in the statistical combination (default is False).
    df_6 : Any, optional
        The sixth DataFrame to be included in the statistical combination (default is False).
    list_names : Any, optional
        The names of the DataFrames to be included in the statistical combination (default is False).
    gene_df : pd.DataFrame
        Dataframe of p-values if more than 6 (or alternative to above). Index = gene names. Columns 1-x = 'p_{list_name}' with p-values. Do not fille NAs with 1s.
    savepath : Any, optional
        If provided, the path where the resulting DataFrame will be saved as a CSV file (default is False).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the results of the statistical combination of the input DataFrames.

    Notes:
    ------
    - The function assumes that the input DataFrames are formatted correctly for the statistical methods
      used. This includes having appropriate columns for p-values.
    - Only DataFrames that are not set to False will be included in the combination process.

    Example:
    --------
    >>> df_combined = statistical_combination(df_1, df_2, df_3=df_3, savepath="path/to/save/")
    """
    # Prepare input
    if not isinstance(gene_df, pd.DataFrame):
        raise ValueError("Please provide a DataFrame for gene_df with the format of gene name as index, and columns as p-values formatted as 'p_{method 1}'")
    if gene_df.empty:
        df_dict = {}
        df_list = [x for x in [df_1, df_2, df_3, df_4, df_5, df_6] if not x.empty]
        if list_names:
            for j in range(len(list_names)):
                df_dict[list_names[j]] = df_list[j]
        else:
            for i in range(len(df_list)):

                df_dict[i+1] = df_list[i]
            #df_dict = {1:df_1, 2:df_2, 3:df_3, 4:df_4, 5:df_5, 6:df_6}
        clean_df = _merge_p_inputs(df_dict)
    else:
        clean_df = gene_df.copy()
    # Calculate CCT, minP
    df_cols = [x for x in clean_df.columns if "p_" in x]
    results = []

    # Loop through column combinations, starting with the largest
    for r in range(len(df_cols), 1, -1):
        for comb in itertools.combinations(df_cols, r):
            # Filter the DataFrame for rows where the selected combination is not NaN
            subset_df = clean_df[list(comb)].dropna()
            # Apply the sequence of functions
            df = _cauchy_combination_test(subset_df)
            df = _min_p(df)
            df = _mcm(df)
            df = _cmc(df)
            df = _p_multiply(df)
            # Add a column to keep track of which combination was used
            df['combination'] = str(comb)
            df['num_columns'] = len(comb)
            # Append the processed DataFrame to the results list
            results.append(df)

    # Combine all the results
    combined_df = pd.concat(results, axis=0, ignore_index=False)

    # Sort to prioritize combinations with more columns
    combined_df = combined_df.sort_values(by=['num_columns'], ascending=[False])

    # Remove duplicates, keeping the one with the most columns
    final_df = combined_df[~combined_df.index.duplicated(keep='first')]
    final_df = final_df.sort_values(by = 'p.cauchy', ascending = True)
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'Statistical_Consensus/')
        os.makedirs(new_savepath, exist_ok=True)
        final_df.to_csv(new_savepath + "StatisticalCombination.csv", index = True)

    return final_df
#endregion

