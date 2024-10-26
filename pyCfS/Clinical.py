"""
Experiments focused on clinical data
"""

import pkg_resources
import pandas as pd
import numpy as np
from typing import Any
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm, mannwhitneyu
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import requests
from adjustText import adjust_text
import seaborn as sns
from PIL import Image
from typing import Any
import io
import os
import math
import warnings
import multiprocessing
from multiprocessing import Manager
from .utils import _fix_savepath, _load_open_targets_mapping, _get_avg_and_std_random_counts, _merge_random_counts, _define_background_list, _get_open_targets_gene_mapping

#region Mouse Phenotype
def _load_mouse_phenotypes() -> (pd.DataFrame, list): # type: ignore
    """
    Loads mouse phenotype data from a parquet file and returns a dataframe of the data, a list of unique gene IDs, and a
    dictionary of background information.

    Returns:
        mgi_df (pd.DataFrame): A dataframe of the mouse phenotype data.
        mgi_genes (list): A list of unique gene IDs.
    """
    mgi_stream = pkg_resources.resource_stream(__name__, 'data/mousePhenotypes/part-00000-acbeac24-79db-4a95-8d79-0ae045cb6538-c000.snappy.parquet')
    mgi_df = pd.read_parquet(mgi_stream, engine='pyarrow')
    # Get the mappings
    mapping_dict = _get_open_targets_gene_mapping()
    mgi_df['gene_ID'] = mgi_df['targetFromSourceId'].map(mapping_dict)
    mgi_df['upperPheno'] = mgi_df['modelPhenotypeClasses'].apply(lambda x: x[0]['label'])
    mgi_genes = mgi_df['gene_ID'].unique().tolist()

    return mgi_df, mgi_genes

def _get_pheno_counts(matrix:pd.DataFrame, gene_list:list) -> dict:
    """
    Given a matrix of gene expression data and a list of gene IDs, returns a dictionary
    containing the frequency of each phenotype label in the matrix for the given gene IDs.

    Args:
    - matrix: pandas DataFrame containing gene expression data
    - gene_list: list of gene IDs to filter the matrix by

    Returns:
    - df_dict: dictionary containing the frequency of each phenotype label in the matrix
        for the given gene IDs
    """
    matrix = matrix[matrix['gene_ID'].isin(gene_list)]
    df = pd.DataFrame(matrix.groupby('modelPhenotypeLabel')['gene_ID'].count())
    df.rename(columns={'gene_ID': 'freq'}, inplace=True)
    df_dict = df.to_dict()['freq']
    return df_dict

def _process_gene(mgi_df:pd.DataFrame, g:str, model_pheno_label_gene_mappings: dict, lock: multiprocessing.Lock) -> None: # type: ignore
    """
    Process a gene and update the model_pheno_label_gene_mappings dictionary.

    Args:
        g (str): The gene ID.

    Returns:
        None
    """
    #mgi_df, g, model_pheno_label_gene_mappings, lock = args
    df = mgi_df[mgi_df['gene_ID'] == g]
    df_phenos = df['modelPhenotypeLabel'].tolist()
    for p in df_phenos:
        with lock:
            model_pheno_label_gene_mappings[p] = model_pheno_label_gene_mappings.get(p, []) + [g]

def _get_pheno_gene_mappings(mgi_df: pd.DataFrame, mgi_genes: list, num_cores:int) -> dict:
    """
    Given a pandas DataFrame containing MGI gene information and a list of MGI gene IDs,
    returns a dictionary mapping each model phenotype label to a list of genes associated with that label.

    Args:
        mgi_df (pd.DataFrame): A pandas DataFrame containing MGI gene information.
        mgi_genes (list): A list of MGI gene IDs.

    Returns:
        dict: A dictionary mapping each model phenotype label to a list of genes associated with that label.
    """
    manager = Manager()
    model_pheno_label_gene_mappings = manager.dict()
    lock = manager.Lock()
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.starmap(_process_gene, [(mgi_df, g, model_pheno_label_gene_mappings, lock) for g in mgi_genes])

    model_pheno_label_gene_mappings = dict(model_pheno_label_gene_mappings)

    return model_pheno_label_gene_mappings

def _get_gene_representation(matrix: pd.DataFrame, gene_list: list) -> pd.DataFrame:
    """
    Returns a DataFrame containing the representation of genes in a given list.

    Args:
        matrix (pd.DataFrame): A DataFrame containing gene IDs and model phenotype labels.
        gene_list (list): A list of gene IDs to retrieve representation for.

    Returns:
        pd.DataFrame: A DataFrame containing the representation of genes in the gene_list.
    """
    df = matrix[matrix['gene_ID'].isin(gene_list)]
    df = pd.DataFrame(df.groupby('gene_ID')['modelPhenotypeLabel'].count())
    df.rename(columns={'modelPhenotypeLabel': 'representation'}, inplace=True)
    df['rep_rounded'] = df['representation'].apply(lambda x: round(x / 10) * 10)
    return df

def _create_target_gene_representation_counts(all_mgi_genes: list, mgi_rep_df: pd.DataFrame, target_genes: list) -> dict:
    """
    Given a list of all MGI genes, a dataframe of MGI gene representations, and a list of target genes,
    returns a dictionary with the frequency of each rounded MGI representation value as the key and the
    count of target genes with that representation as the value.

    Args:
        all_mgi_genes (list): List of all MGI genes.
        mgi_rep_df (pd.DataFrame): Dataframe of MGI gene representations.
        target_genes (list): List of target genes.

    Returns:
        dict: Dictionary with the frequency of each rounded MGI representation value as the key and the
        count of target genes with that representation as the value.
    """
    # filter for genes tested by MGI
    target_gene_set_filtered = [x for x in target_genes if x in all_mgi_genes]
    target_gene_set_filtered_represent = mgi_rep_df[
        mgi_rep_df.index.isin(target_gene_set_filtered)]
    # get number of genes per each rounded MGI representation value as df
    target_gene_set_filtered_represent = pd.DataFrame(
        target_gene_set_filtered_represent.groupby('rep_rounded')['representation'].count())
    # this dictionary has representation frequency as index and count of target genes with that
    # representation as the value
    target_gene_set_filtered_represent_dict = dict(zip(
        target_gene_set_filtered_represent.index.tolist(),
        target_gene_set_filtered_represent['representation'].tolist()
    ))
    return target_gene_set_filtered_represent_dict

def _get_representation_matched_random_gene_set(background_genes:list, mgi_rep_df:pd.DataFrame, target_gene_set_filtered_represent_dict:dict, seed:int) -> list:
    """
    Returns a list of randomly selected genes that are matched to the representation of the target gene set in the background gene set.

    Parameters:
    background_genes (list): A list of background genes to select from.
    mgi_rep_df (pd.DataFrame): A pandas dataframe containing the representation of each gene in the background gene set.
    target_gene_set_filtered_represent_dict (dict): A dictionary containing the representation of the target gene set in the background gene set.

    Returns:
    list: A list of randomly selected genes that are matched to the representation of the target gene set in the background gene set.
    """
    random_genes = []
    df_filtered = mgi_rep_df.copy()
    df_filtered = df_filtered[df_filtered.index.isin(background_genes)]
    for k, v in target_gene_set_filtered_represent_dict.items():
        df = df_filtered.copy()
        df = df[df['rep_rounded'] == k]
        rep_matched_genes = df.index.tolist()
        generator = np.random.default_rng(seed = seed **3)
        x = generator.choice(rep_matched_genes, v, replace=False).tolist()
        random_genes.extend(x)
    return random_genes

def _process_iteration(i:int, random_bkgd:list, mgi_gene_representation:pd.DataFrame, target_representation_counts:dict, mgi_df:pd.DataFrame) -> dict:
    """
    Process an iteration of the algorithm.

    Args:
        i (int): The iteration number.
        random_bkgd (list): The list of random background genes.
        mgi_gene_representation (pd.DataFrame): The gene representation dataframe.
        target_representation_counts (dict): The target representation counts.
        mgi_df (pd.DataFrame): The MGI dataframe.

    Returns:
        dict: The counts of phenotypes for the random genes.
    """
    random_genes = _get_representation_matched_random_gene_set(
        random_bkgd,
        mgi_gene_representation,
        target_representation_counts,
        i
    )
    random_counts = _get_pheno_counts(mgi_df, random_genes)
    return random_counts

def _parallelize_iterations(random_iter:int, random_bkgd:list, mgi_gene_representation:pd.DataFrame, target_representation_counts:dict, mgi_df:pd.DataFrame, num_cores:int) -> list:
    """
    Parallelizes the iterations of the _process_iteration function using multiprocessing.Pool.

    Args:
        random_iter (int): The number of iterations to perform.
        random_bkgd (list): The list of random background samples.
        mgi_gene_representation (pd.DataFrame): The gene representation dataframe.
        target_representation_counts (dict): The target representation counts.
        mgi_df (pd.DataFrame): The MGI dataframe.
        num_cores (int): The number of CPU cores to use for parallel processing.

    Returns:
        list: The merged random counts from all iterations.
    """
    with multiprocessing.Pool(processes = num_cores) as pool:
        random_counts_iterations = pool.starmap(_process_iteration, [(i, random_bkgd, mgi_gene_representation, target_representation_counts, mgi_df) for i in range(random_iter)])

    random_counts_merged = _merge_random_counts(random_counts_iterations)
    return random_counts_merged

def _get_random_iteration_pheno_counts(background_type: str, background_dict: dict, mgi_gene_representation: pd.DataFrame, target_representation_counts: dict, mgi_df: pd.DataFrame, random_iter:int, num_cores:int) -> dict:
    """
    Returns the average and standard deviation of phenotype counts for a given number of random iterations.

    Args:
    - background_type (str): The type of background to use for random gene selection.
    - background_dict (dict): A dictionary containing the background data for each background type.
    - mgi_gene_representation (pd.DataFrame): A dataframe containing the MGI gene representation data.
    - target_representation_counts (dict): A dictionary containing the target representation counts for each phenotype.
    - mgi_df (pd.DataFrame): A dataframe containing the MGI data.
    - random_iter (int): The number of random iterations to perform.

    Returns:
    - A dictionary containing the average and standard deviation of phenotype counts for the given number of random iterations.
    """
    random_bkgd = background_dict[background_type]
    random_counts_merged = _parallelize_iterations(random_iter, random_bkgd, mgi_gene_representation, target_representation_counts, mgi_df, num_cores)

    random_counts_avg, random_counts_std = _get_avg_and_std_random_counts(random_counts_merged)
    return {'random_avg':random_counts_avg, 'random_std':random_counts_std}

def _z_score(avg: float, std: float, x: float) -> Any:
    """
    Calculates the z-score of a given value x, given the mean (avg) and standard deviation (std).

    Args:
    - avg (float): the mean value
    - std (float): the standard deviation
    - x (float): the value to calculate the z-score for

    Returns:
    - z (float or str): the z-score of x, or 'no_z-score' if a ValueError occurs during calculation
    """
    if x != 0 and avg != 0 and std != 0:
        z = (x - avg) / std
        if math.isnan(z):
            z = 'no_z-score'
    else:
        z = 'no_z-score'
    return z

def _or_fdr(matrix:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the false discovery rate (FDR) for a given matrix of p-values.

    Args:
    - matrix (pd.DataFrame): A pandas DataFrame containing p-values.

    Returns:
    - pd.DataFrame: A pandas DataFrame with an additional column for FDR values.
    """
    matrix = matrix.sort_values(by='pvalue')
    pvals = list(matrix['pvalue'])
    if len(pvals) == 0:
        matrix['fdr'] = np.nan
        return matrix
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=True)[1]
    # changed method for qval calculation
    matrix['fdr'] = qvals
    return matrix

def _get_output_matrix(target_counts:dict, random_dict:dict, model_pheno_label_gene_mappings:dict, target_genes:list, mgi_df:pd.DataFrame) -> pd.DataFrame:
    """
    This function takes in the following parameters:
    target_counts: a dictionary containing the frequency of each target gene
    random_dict: a dictionary containing the random average and standard deviation of each target gene
    model_pheno_label_gene_mappings: a dictionary containing the gene mappings for each model phenotype label
    target_genes: a list of target genes
    mgi_df: a pandas DataFrame containing model phenotype labels and classes

    It returns a pandas DataFrame containing the following columns:
    - Freq: frequency of each target gene
    - RandomAvgFreq: random average frequency of each target gene
    - RandomStdFreq: random standard deviation frequency of each target gene
    - z-score: z-score of each target gene
    - pvalue: p-value of each target gene
    - geneMappings: gene mappings for each target gene
    - UpperLevelLabel: upper level label for each model phenotype label
    """
    summary_df = pd.DataFrame.from_dict(target_counts, orient='index', columns=['Freq'])
    summary_df['RandomAvgFreq'] = summary_df.index.map(random_dict['random_avg'])
    summary_df['RandomStdFreq'] = summary_df.index.map(random_dict['random_std'])
    summary_df['z-score'] = summary_df.apply(
        lambda x: _z_score(
            x['RandomAvgFreq'],
            x['RandomStdFreq'],
            x['Freq']
        ), axis=1
    )

    summary_df = summary_df[summary_df['z-score'] != 'no_z-score']
    summary_df['z-score'] = summary_df['z-score'].astype(float)
    is_nan_mask = np.isnan(summary_df['z-score'])
    summary_df['pvalue'] = np.where(is_nan_mask, np.nan, norm.sf(np.abs(summary_df['z-score'])) * 2)
    summary_df = summary_df.dropna(subset=['pvalue'], axis=0)
    summary_df = summary_df[summary_df['RandomStdFreq'] != 0]
    summary_df = _or_fdr(summary_df)
    summary_df['geneMappings'] = summary_df.index.map(model_pheno_label_gene_mappings)
    summary_df['geneMappings'] = summary_df['geneMappings'].apply(lambda x: [y for y in x if y in target_genes])
    summary_df['geneMappings'] = [",".join(x) for x in summary_df['geneMappings']]

    # Add the HighLevelPhenotypes
    mgi_df['UpperLevelLabel'] = mgi_df['modelPhenotypeClasses'].apply(lambda x: [item['label'] for item in x])
    mgi_df['UpperLevelLabel'] = mgi_df['UpperLevelLabel'].apply(lambda x: x[0])
    summary_df['original_index'] = summary_df.index
    summary_df = pd.merge(
        summary_df,
        mgi_df[['modelPhenotypeLabel', 'UpperLevelLabel']],
        left_index = True,
        right_on = 'modelPhenotypeLabel',
        how = 'inner'
    )
    summary_df.set_index('original_index', inplace=True)
    summary_df = summary_df.drop_duplicates()
    summary_df = summary_df.sort_values(by = 'fdr', ascending = True)
    return summary_df

def _process_df(df:pd.DataFrame, q_cut:float) -> pd.DataFrame:
    """
    Process a pandas DataFrame by transforming the fdr column to -log10(FDR), counting how many phenotypes are significant,
    counting how many phenotypes are in a higher class, sorting the values by number of significant phenotypes, the total number
    of phenotypes, and the name (to induce randomness in upper-level).

    Args:
    - df: pandas DataFrame
    - q_cut: float

    Returns:
    - pandas DataFrame
    """
    # Transform to FDR
    df['-log10(FDR)'] = -np.log10(df['fdr'])

    # Count how many phenotypes are significant
    df_sig_count = df.groupby('UpperLevelLabel').apply(lambda x: (x['fdr'] < q_cut).sum()).reset_index(name = 'sig_count')
    df = df.merge(df_sig_count, on = 'UpperLevelLabel', how = 'left')

    # Count how many phenotypes are in a higher class
    df_all_count = df.groupby('UpperLevelLabel').size().reset_index(name = 'total_count')
    df = df.merge(df_all_count, on = 'UpperLevelLabel', how = 'left')

    # Sort the values by number of significant phenotypes, the total number of phenotypes, and the name (to induce randomness in upper-level )
    df.sort_values(by = ['sig_count', 'total_count'], ascending = False, inplace = True)

    return df

def _mgi_strip_plot(df:pd.DataFrame, pheno_of_interest:list, sig_dot_color:str, show_labels:bool, q_cut:float, fontface:str, fontsize:int) -> Image:
    """
    Creates a strip plot of the given dataframe with significant dots colored in a specified color.
    Args:
    - df: pandas DataFrame containing the data to be plotted
    - pheno_of_interest: list of phenotypes to be labeled in the plot
    - sig_dot_color: color to be used for significant dots
    - show_labels: boolean indicating whether to show labels for selected dots
    - q_cut: float indicating the cutoff for significance
    Returns:
    - Image object of the plot
    """
    # Pre_process the df
    df = _process_df(df, q_cut)

    # Subset for only higher level phenotypes that have significant lower-level phenotype
    df = df[df.sig_count > 0].reset_index(drop = True)
    unique_labels = df['UpperLevelLabel'].unique()
    # Create color column
    df['Significance'] = np.where(df['fdr'] < q_cut, f'FDR<{q_cut}', f'FDR>{q_cut}')
    custom_palette = {f'FDR<{q_cut}':sig_dot_color, f'FDR>{q_cut}':'grey'}

    # Strip plot
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _, ax = plt.subplots(figsize = (20, 5))
    ax = sns.stripplot(data = df,
                        y = '-log10(FDR)',
                        x = 'UpperLevelLabel',
                        hue = 'Significance',
                        palette = custom_palette,
                        edgecolor = 'black',
                        s = 10,
                        linewidth = 0.5
                        )
    # Set the x tick labels
    clean_unique_labels = [x.replace(" phenotype", "").capitalize() for x in unique_labels]
    plt.xticks(ticks = ax.get_xticks(), labels = clean_unique_labels, rotation = 45, ha = 'right', va= 'top', size = 16)
    plt.xlabel("High Level Phenotypes", size = 16)
    # Set y labels
    plt.yticks(size = 16)
    plt.ylabel("-log10(FDR)", size = 16)
    # Plot a line for significance
    cutoff_line = -np.log10(q_cut)
    plt.axhline(y = cutoff_line, linestyle = '--', linewidth = 2, color = 'black', label = f"FDR = {q_cut}")
    # Set the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index("FDR<0.05")] + [i for i in range(len(labels)) if i != labels.index("FDR<0.05")]
    legend = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize = 16)
    for handle in legend.legendHandles:
        try: handle.set_sizes([100])
        except AttributeError:
            continue
    plt.tight_layout()

    # Plot names of selected dots if you want
    if show_labels == False:
        buffer = io.BytesIO()
        plt.savefig(buffer, format = 'png', dpi = 300)
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close()
    else:
        points_of_interest = df[df['Unnamed: 0'].isin(pheno_of_interest)]
        texts = []
        for _,row in points_of_interest.iterrows():
            clean_genes = row['geneMappings'].split(",")
            clean_genes = ", ".join(clean_genes)
            if row["-log10(FDR)"] == max(df['-log10(FDR)']):
                y_offset = -20
            else:
                y_offset = 100
            texts.append(plt.annotate(
                f"{row['Unnamed: 0'].capitalize()}\n{clean_genes}",
                (row['UpperLevelLabel'],
                 row['-log10(FDR)']),
                textcoords='offset points',
                xytext = (0.01, y_offset),
                ha = 'center', va = 'center',
                arrowprops=dict(arrowstyle='-', color = 'black'),
                bbox=dict(boxstyle='round,pad=0.1', edgecolor = 'black', facecolor = 'white'),
                size = 12
            ))
        adjust_text(texts,
                    force_points = (0.02, 100),
                    force_text = (10, 10),
                    ensure_inside_axes = True
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format = 'png', dpi = 300)
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close()
    return image

def _get_zdist_plot(summary_matrix:pd.DataFrame, bin_size:float, rounder:int)->Image:
    """
    Generates a histogram plot of the z-scores in the summary_matrix dataframe.

    Args:
    summary_matrix (pd.DataFrame): A pandas dataframe containing the summary statistics for a set of z-scores.
    bin_size (float): The size of each bin in the histogram.
    rounder (int): The rounding factor for the minimum and maximum bin values.

    Returns:
    Image: A PIL Image object containing the histogram plot.
    """
    scores = summary_matrix['z-score'].tolist()

    max_bin = round(np.max(scores) / rounder, 0) * rounder
    min_bin = round(np.min(scores) / rounder, 0) * rounder
    bins = np.arange(min_bin, max_bin, bin_size)

    _, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.hist(scores, color='#bababa', edgecolor='black', density=False, bins=bins)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axvline(np.mean(scores), color='black', linestyle='dashed')

    plt.yticks(size=14)
    plt.xticks(np.arange(0, max_bin, 5))
    plt.xlabel('Z-score', size=14)
    plt.ylabel('Count', size=14)

    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _get_pval_plot(summary_matrix:pd.DataFrame, bin_size:float, x_min:float, x_max:float)->Image:
    """
    Returns a histogram plot of p-values from a summary matrix.

    Args:
    summary_matrix (pd.DataFrame): A pandas dataframe containing summary statistics.
    bin_size (float): The size of each bin in the histogram.
    x_min (float): The minimum value of the x-axis.
    x_max (float): The maximum value of the x-axis.

    Returns:
    Image: A histogram plot of p-values.
    """
    scores = summary_matrix['pvalue'].tolist()
    bins = np.arange(0, 1, bin_size)

    _, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.hist(scores, color='#bababa', edgecolor='black', density=False, bins=bins)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.yticks(size=14)
    plt.xlim(x_min, x_max)
    plt.xlabel('p-value', size=14)
    plt.ylabel('Count', size=14)
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _get_fdr_plot(summary_matrix:pd.DataFrame, bin_size:float, x_min:float, x_max:float) -> Image:
    """
    Generates a histogram plot of the false discovery rate (FDR) values in the summary matrix.

    Args:
    summary_matrix (pd.DataFrame): A pandas DataFrame containing the summary statistics of the FDR values.
    bin_size (float): The size of each bin in the histogram.
    x_min (float): The minimum value of the x-axis.
    x_max (float): The maximum value of the x-axis.

    Returns:
    Image: A PIL Image object of the histogram plot.
    """
    scores = summary_matrix['fdr'].tolist()

    bins = np.arange(0, 1, bin_size)

    _, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    n, bins, _ = ax.hist(scores, color='#bababa', edgecolor='black', density=False, bins=bins)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.yticks(size=14)
    plt.xlim(x_min, x_max)
    plt.ylim(0, round(n[0]/10)*10 + 5)
    plt.xlabel('FDR', size=14)
    plt.ylabel('Count', size=14)
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def mouse_phenotype_enrichment(query:list, custom_background:Any = 'ensembl', random_iter:int = 5000, plot_sig_color:str = 'red', plot_q_threshold:float = 0.05, plot_show_labels:bool = False, plot_labels_to_show: list = [False], plot_fontface:str = 'Avenir', plot_fontsize:int = 14, cores:int = 1, savepath:Any = False, verbose:int = 0) -> (pd.DataFrame, Image, Image): # type: ignore
    """
    Performs a phenotype enrichment analysis on a list of genes using the Mouse Genome Informatics (MGI) database.

    Args:
    - query (list): A list of gene symbols to be tested for enrichment.
    - background (str): The gene set to be used as the background for the enrichment analysis. Default is 'ensembl'.
    - random_iter (int): The number of random iterations to perform for the enrichment analysis. Default is 5000.
    - plot_sig_color (str): The color to use for significant points in the enrichment plot. Default is 'red'.
    - plot_q_threshold (float): The q-value threshold for significance in the enrichment plot. Default is 0.05.
    - plot_show_labels (bool): Whether to show labels on the enrichment plot. Default is False.
    - plot_labels_to_show (list): A list of phenotype labels to show on the enrichment plot. Default is [False].
    - savepath (Any): The path to save the output files. Default is False.

    Returns:
    - summary_df (pandas.DataFrame): A summary dataframe of the enrichment analysis results.
    - strip_plot (matplotlib.pyplot): A strip plot of the enrichment analysis results.
    - fdr_plot (matplotlib.pyplot): A plot of the false discovery rate (FDR) distribution for the enrichment analysis results.
    """
    background_dict, background_name = _define_background_list(custom_background, just_genes = True, verbose = verbose)
    mgi_df, mgi_genes = _load_mouse_phenotypes()
    target_genes_pheno_counts = _get_pheno_counts(mgi_df, query)
    model_pheno_label_gene_mappings = _get_pheno_gene_mappings(mgi_df, mgi_genes, cores)
    # Summary - Write to file
    if verbose > 0:
        print('# of target genes tested in MGI: {}/{}'.format(
            len([x for x in query if x in mgi_genes]),
            len(query)
        ))

    #find number of times gene has a documented phenotype in MGI (similar to degree count in network-based analyses)
    mgi_genes_representation = _get_gene_representation(mgi_df, mgi_genes)
    #find how often each target gene is covered in MGI; tallied/grouped by nearest 10 value
    target_genes_representation_counts = _create_target_gene_representation_counts(
        mgi_genes,
        mgi_genes_representation,
        query
    )
    #random iterations set at 1000x
    random_iter_dict = _get_random_iteration_pheno_counts(
        background_name,
        background_dict,
        mgi_genes_representation,
        target_genes_representation_counts,
        mgi_df,
        random_iter,
        cores
    )
    summary_df = _get_output_matrix(
        target_genes_pheno_counts,
        random_iter_dict,
        model_pheno_label_gene_mappings,
        query,
        mgi_df
    )
    # _process_df? How does it change the output?
    # Generate plots
    strip_plot = _mgi_strip_plot(
        summary_df,
        plot_labels_to_show,
        plot_sig_color,
        plot_show_labels,
        plot_q_threshold,
        plot_fontface,
        plot_fontsize
    )
    z_dist_plot = _get_zdist_plot(summary_df, 0.5, 10)
    p_val_plot = _get_pval_plot(summary_df, 0.0005, 0, 0.01)
    fdr_plot = _get_fdr_plot(summary_df, 0.0005, 0, 0.01)

    # Output files
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'MGI_Mouse_Phenotypes/')
        os.makedirs(new_savepath, exist_ok=True)
        summary_df.to_csv(new_savepath + 'MGI_Lower-Level_PhenoEnrichment.csv')
        strip_plot.save(new_savepath + "MGI_Lower-Level_PhenoEnrichment.png")
        z_dist_plot.save(new_savepath + "MGI_Lower-Level_PhenoEnrichment_ZscoreDist.png")
        p_val_plot.save(new_savepath + "MGI_Lower-Level_PhenoEnrichment_PvalDist.png")
        fdr_plot.save(new_savepath + "MGI_Lower-Level_PhenoEnrichment_FdrDist.png")

    return summary_df, strip_plot, fdr_plot
#endregion


#region Protein Family
def _get_level_class(x_array:list, level:str) -> str:
    """
    Given an array of protein family classes, returns the class at the specified level.

    Args:
        x_array (list): A list of protein family classes.
        level (str): The level to retrieve.

    Returns:
        str: The protein family class at the specified level.
    """
    try:
        for i in x_array:
            if i['level'] == level:
                out = i['label']
                break
            else:
                out = 'no_level_assignment'
    except:
        out = 'no_level_assignment'
    return out

def _get_protein_class_data(targets_df: pd.DataFrame, mapping_dict:dict) -> (pd.DataFrame, list): # type: ignore
    """
    Retrieves protein class data from the given targets dataframe and mapping dictionary.

    Args:
        targets_df (pd.DataFrame): The dataframe containing the target data.
        mapping_dict (dict): The dictionary mapping gene IDs to target classes.

    Returns:
        pd.DataFrame: The modified targets dataframe with additional columns for level 1 to level 5 classes.
        list: The list of unique gene IDs corresponding to protein class targets.
    """
    targets_df = targets_df[
        ['id', 'approvedSymbol', 'biotype',
         'targetClass', 'approvedName']
    ].copy()
    targets_df['level1_class'] = targets_df['targetClass'].apply(
        lambda x: _get_level_class(x, 'l1')
    )
    targets_df['level2_class'] = targets_df['targetClass'].apply(
        lambda x: _get_level_class(x, 'l2')
    )
    targets_df['level3_class'] = targets_df['targetClass'].apply(
        lambda x: _get_level_class(x, 'l3')
    )
    targets_df['level4_class'] = targets_df['targetClass'].apply(
        lambda x: _get_level_class(x, 'l4')
    )
    targets_df['level5_class'] = targets_df['targetClass'].apply(
        lambda x: _get_level_class(x, 'l5')
    )
    targets_df['gene_ID'] = targets_df['id'].map(mapping_dict)
    # the following ENSG IDs have multiple targetClass assignments for the same gene name
    # these will be omitted so correction is not needed; these represent about 10 genes total
    ensg_to_drop = [
        'ENSG00000290203', 'ENSG00000285723', 'ENSG00000285508', 'ENSG00000125863', 'ENSG00000285437','ENSG00000168255', 'ENSG00000287542', 'ENSG00000138641', 'ENSG00000280987', 'ENSG00000015479', 'ENSG00000275596', 'ENSG00000288357', 'ENSG00000258724', 'ENSG00000254093', 'ENSG00000233024',
        'ENSG00000183889'
    ]
    targets_df = targets_df[~targets_df['id'].isin(ensg_to_drop)]
    targets_df = targets_df[targets_df['biotype'] == 'protein_coding']
    targets_class_proteins_genes = targets_df['gene_ID'].unique().tolist()

    return targets_df, targets_class_proteins_genes

def _get_family_files() -> (pd.DataFrame, list): # type: ignore
    """
    Load protein family data and clean the protein family table.

    Parameters:
    background (str): The background data.

    Returns:
    targets_df (DataFrame): The cleaned protein family table.
    targets_class_proteins_genes (list): List of unique gene IDs in the protein family table.
    background_dict (dict): The background data dictionary.
    """
    # Load background data
    mapping_dict = _get_open_targets_gene_mapping()
    # Load protein family data
    family_ref_stream = pkg_resources.resource_stream(__name__, 'data/targetsFileList.txt')
    family_ref_df = family_ref_stream.readlines()
    family_ref_df = [x.decode('utf-8').strip() for x in family_ref_df]
    targets_df = pd.DataFrame()
    for f in family_ref_df:
        target_df_stream = pkg_resources.resource_stream(__name__, 'data/targets/' + f)
        df = pd.read_parquet(target_df_stream, engine='pyarrow')
        targets_df = pd.concat([targets_df, df], axis=0)
    # Get protein class data
    targets_class_df, targets_class_df_gene_list = _get_protein_class_data(targets_df, mapping_dict)

    return targets_class_df, targets_class_df_gene_list

def _get_level_gene_mappings(targets_df:pd.DataFrame) -> dict:
    """
    Returns a dictionary containing mappings of gene IDs based on different levels of classes.

    Parameters:
    targets_df (pd.DataFrame): DataFrame containing the target data.

    Returns:
    dict: A dictionary containing mappings of gene IDs based on different levels of classes.
    """
    level_class_mappings = {}
    level_lst = ['level1_class', 'level2_class', 'level3_class', 'level4_class', 'level5_class']
    for level in level_lst:
        class_mappings = {}
        classes = targets_df[level].unique().tolist()
        for x in classes:
            df = targets_df[targets_df[level] == x]
            class_mappings[x] = df['gene_ID'].tolist()
        level_class_mappings[level] = class_mappings
    return level_class_mappings

def _map_level(level: list) -> list:
    """
    Maps the input level list to the corresponding levels.

    Args:
        level (list): The input level list.

    Returns:
        list: The mapped levels.
    """
    if 'all' in level:
        levels = ['level1', 'level2', 'level3', 'level4', 'level5']
    else:
        levels = level
    return levels

def _get_target_class_counts(matrix: pd.DataFrame, gene_list: list, class_level: str) -> dict:
    """
    Get the counts of target classes for a given gene list at a specific class level.

    Args:
        matrix (pd.DataFrame): The input matrix containing gene information.
        gene_list (list): The list of genes to consider.
        class_level (str): The class level to calculate the counts for.

    Returns:
        dict: A dictionary containing the counts of target classes.

    """
    level_dict = {'level1': 'level1_class', 'level2': 'level2_class', 'level3': 'level3_class', 'level4': 'level4_class',
                  'level5': 'level5_class'}
    matrix = matrix[matrix['gene_ID'].isin(gene_list)]
    df = pd.DataFrame(matrix.groupby(level_dict[class_level])['gene_ID'].count())
    df.rename(columns={'gene_ID': 'freq'}, inplace=True)
    df_dict = df.to_dict()['freq']
    return df_dict

def _get_random_gene_set(background_genes:list, target_genes:list, iter:int) -> list:
    """
    Returns a random gene set from the background genes list with the same length as the target genes list.

    Parameters:
    background_genes (list): List of background genes to choose from.
    target_genes (list): List of target genes to match the length of the random gene set.
    iter (int): Seed value for random number generation.

    Returns:
    list: Random gene set with the same length as the target genes list.
    """
    rng = np.random.default_rng(seed=iter**2)
    return rng.choice(background_genes, len(target_genes), replace=False).tolist()

def _get_random_iteration_class_counts(background_type:str, background_dict:dict, target_genes:list, targets_df:pd.DataFrame, level:str, random_iter:int, cores:int) -> dict:
    """
    Get the average and standard deviation of target class counts for multiple random iterations.

    Parameters:
    background_type (str): The type of background for random gene selection.
    background_dict (dict): A dictionary containing different types of background gene sets.
    target_genes (list): A list of target genes.
    targets_df (pd.DataFrame): A DataFrame containing target gene information.
    level (str): The level of target gene classification.
    random_iter (int): The number of random iterations.
    cores (int): The number of worker processes to use for parallelization.

    Returns:
    dict: A dictionary containing the average and standard deviation of target class counts for random iterations.
    """
    random_bkgd = background_dict[background_type]
    random_counts_iterations = []

    with multiprocessing.Pool(processes=cores) as pool:
        results = []
        for i in range(random_iter):
            result = pool.apply_async(_get_random_gene_set, (random_bkgd, target_genes, i))
            results.append(result)

        for result in results:
            random_genes = result.get()
            random_counts = _get_target_class_counts(targets_df, random_genes, level)
            random_counts_iterations.append(random_counts)

    random_counts_merged = _merge_random_counts(random_counts_iterations)

    random_counts_avg, random_counts_std = _get_avg_and_std_random_counts(random_counts_merged)

    return {'random_avg':random_counts_avg, 'random_std':random_counts_std}

def _get_family_output_matrix(target_counts:dict, random_dict:dict, target_class_level_gene_mappings:dict, target_genes:list, level:str) -> pd.DataFrame:
    """
    Generate a summary DataFrame for family output matrix.

    Args:
        target_counts (dict): A dictionary containing target counts.
        random_dict (dict): A dictionary containing random averages and standard deviations.
        target_class_level_gene_mappings (dict): A dictionary containing target class level gene mappings.
        target_genes (list): A list of target genes.
        level (str): The level of the class.

    Returns:
        pd.DataFrame: A summary DataFrame for family output matrix.
    """
    summary_df = pd.DataFrame.from_dict(target_counts, orient='index', columns=['Freq'])
    summary_df['RandomAvgFreq'] = summary_df.index.map(random_dict['random_avg'])
    summary_df['RandomStdFreq'] = summary_df.index.map(random_dict['random_std'])
    summary_df['z-score'] = summary_df.apply(
        lambda x: _z_score(
            x['RandomAvgFreq'],
            x['RandomStdFreq'],
            x['Freq']
        ), axis=1
    )

    summary_df = summary_df[summary_df['z-score'] != 'no_z-score']
    summary_df['z-score'] = summary_df['z-score'].apply(pd.to_numeric, errors='coerce')
    summary_df['pvalue'] = norm.sf(abs(summary_df['z-score'])) * 2
    summary_df = summary_df.dropna(subset=['pvalue'], axis=0)
    summary_df = summary_df[summary_df['RandomStdFreq'] != 0]
    summary_df = _or_fdr(summary_df)

    level_dict = {'level1':'level1_class', 'level2':'level2_class', 'level3':'level3_class', 'level4':'level4_class',
                  'level5':'level5_class'}
    summary_df['geneMappings'] = summary_df.index.map(
        target_class_level_gene_mappings[level_dict[level]]
    )
    summary_df['geneMappings'] = summary_df['geneMappings'].apply(
        lambda x: [y for y in x if y in target_genes]
    )
    return summary_df

def _protein_class_strip_plot(df:pd.DataFrame, q_cut:float, sig_dot_color:str, fontsize:int, fontface:str) -> Image:
    """
    Generate a strip plot to visualize protein class levels and their significance.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing protein class data.
    q_cut (float): The significance threshold for determining significance.
    sig_dot_color (str): The color of significant dots in the plot.
    fontsize (int): The font size for the plot.
    fontface (str): The font face for the plot.

    Returns:
    Image: The generated strip plot as an Image object.
    """
    # clean
    df = df.copy()
    levels_name_map = {'level1':'Level 1', 'level2':'Level 2', 'level3':'Level 3', 'level4':'Level 4', 'level5':'Level 5'}
    df['Level'] = df['Level'].map(levels_name_map)
    df = df[df.index != 'no_level_assignment']
    df['-log10(FDR)'] = -np.log10(df.fdr)
    df['Significance'] = np.where(df['fdr'] < q_cut, f'FDR<{q_cut}', f'FDR>{q_cut}')
    custom_palette = {f'FDR<{q_cut}':sig_dot_color, f'FDR>{q_cut}':'grey'}

    # Plot results
    plt.rcParams.update({'font.size': fontsize,
                        'font.family': fontface})
    _, ax = plt.subplots(figsize = (5,7))
    ax = sns.stripplot(data = df,
                        y = 'Level',
                        x = '-log10(FDR)',
                        hue = 'Significance',
                        palette = custom_palette,
                        edgecolor = 'black',
                        s = 10,
                        linewidth = 0.5
                        )
    # Set axis labels
    ax.set_ylabel('Protein Class Levels')
    ax.set_xlabel(ax.get_xlabel())

    # Plot significance threshold
    q_plot = -np.log10(q_cut)
    plt.axvline(x = q_plot, linestyle = '--', color = 'black', linewidth = 1, label = f"FDR = {q_cut}")

    # Fix legend
    #legend = plt.legend(loc='center left', bbox_to_anchor=(1.5, 0.6))
    legend = ax.legend(loc = 'upper right')
    for handle in legend.legendHandles:
        try: handle.set_sizes([100])
        except AttributeError:
            continue
    plt.tight_layout()
    # Save plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def protein_family_enrichment(query:list, custom_background:Any = 'ensembl', level:list = ['all'], random_iter:int = 5000, plot_q_cut:float = 0.05, plot_sig_dot_color: str = 'red', plot_fontface:str = 'Avenir', plot_fontsize:int = 14, cores:int = 1, savepath:Any = False, verbose: int = 0) -> (pd.DataFrame, Image): # type: ignore
    """
    Perform protein family enrichment analysis.

    Args:
        query (list): List of genes to analyze.
        background (str, optional): Background gene set. Defaults to 'ensembl'.
        level (list, optional): List of levels to analyze. Defaults to ['all'].
        random_iter (int, optional): Number of random iterations. Defaults to 5000.
        plot_q_cut (float, optional): Q-value cutoff for plotting. Defaults to 0.05.
        plot_sig_dot_color (str, optional): Color of significant dots in the plot. Defaults to 'red'.
        plot_fontface (str, optional): Font face for the plot. Defaults to 'Avenir'.
        plot_fontsize (int, optional): Font size for the plot. Defaults to 14.
        cores (int, optional): Number of CPU cores to use. Defaults to 1.
        savepath (Any, optional): Path to save the results. Defaults to False.

    Returns:
        pd.DataFrame: Summary dataframe of the enrichment analysis.
        Image: Plot of the enrichment analysis.
    """
    # Get input data
    background_dict, background_name = _define_background_list(custom_background, just_genes = True, verbose = verbose)
    targets_class_df, targets_class_df_genelist = _get_family_files()
    # Get Level mappings
    target_class_level_gene_mappings = _get_level_gene_mappings(targets_class_df)
    if verbose > 0:
        print('{}/{} target genes in Open Targets'.format(
            len([x for x in query if x in targets_class_df_genelist]),
            len(query))
        )
    # Loop through each level
    levels = _map_level(level)
    out_level_dict = {}
    for j in levels:
        # Get target class counts
        target_genes_class_counts = _get_target_class_counts(targets_class_df, query, j)
        ##### How do I do all the levels? I thought this ran everything
        #random iterations set at 5000x
        random_iter_dict = _get_random_iteration_class_counts(
            background_name,
            background_dict,
            query,
            targets_class_df,
            j,
            random_iter,
            cores
        )
        # Get summary matrix
        summary_df = _get_family_output_matrix(
            target_genes_class_counts,
            random_iter_dict,
            target_class_level_gene_mappings,
            query,
            j
        )

        summary_df['Level'] = j
        out_level_dict[j] = summary_df
    # Merge all levels
    summary_df = pd.concat(out_level_dict.values(), axis=0)
    # Generate plots
    plot = _protein_class_strip_plot(summary_df, plot_q_cut, plot_sig_dot_color, plot_fontsize, plot_fontface)
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'Protein_Family_Enrichment/')
        os.makedirs(new_savepath, exist_ok=True)
        summary_df.to_csv(new_savepath + 'Open-Targets_ProteinClass-' + ",".join(level) + '_Enrichment.csv')
        plot.save(new_savepath + "Open-Targets_ProteinClass-" + ",".join(level) + "_Enrichment.png")
    return summary_df, plot

#endregion


#region Tissue Expression
#endregion


#region DepMap Scores
def _load_depmap() -> (pd.DataFrame, list): # type: ignore
    """
    Load the DepMap data from a feather file and return a DataFrame and a list of all DepMap IDs.

    Returns:
        df (pd.DataFrame): The loaded DepMap data as a DataFrame.
        all_depmap (list): A list of all DepMap IDs.
    """
    stream = pkg_resources.resource_stream(__name__, 'data/CRISPR_(DepMap_22Q2_Public+Score,_Chronos)_.feather')
    df = pd.read_feather(stream)
    all_depmap = df['depmap_id'].tolist()
    df.set_index('depmap_id', inplace = True)
    return df, all_depmap

def _check_cancer_type(df:pd.DataFrame, cancer_type:list, verbose:int = 0) -> list:
    """
    Check if the given cancer types exist in any of the lineage columns of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the lineage columns.
    cancer_type (list): The list of cancer types to check.

    Returns:
    list: The list of valid cancer types that exist in the lineage columns.
    """
    valid = []
    for name in cancer_type:
        # Check if the name exists in any of the lineage columns
        exists_in_lineage = (
            df['lineage_1'].str.contains(name).any() |
            df['lineage_2'].str.contains(name).any() |
            df['lineage_3'].str.contains(name).any() |
            df['lineage_4'].str.contains(name).any()
        )
        # If the name does not exist, print it
        if not exists_in_lineage:
            warnings.warn(f"{name} does not exist in any 'lineage' columns. Skipping.")
        else:
            valid.append(name)
    if not valid:
        raise ValueError("No valid cancer types found. Please check the provided cancer types.")
    if verbose > 0:
        print(f"Valid cancer types: {valid}")
    return valid

def _cancer_specific_depmap(df: pd.DataFrame, cancer_type: list, verbose:int = 0) -> pd.DataFrame:
    """
    Filter the given DataFrame for specific cancer types.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cancer_type (list): A list of cancer types to filter for.

    Returns:
        pd.DataFrame: The filtered DataFrame without the unneeded columns.
    """
    # Filter for cancer types
    cancer_type = _check_cancer_type(df, cancer_type, verbose = verbose)
    names = '|'.join(cancer_type)
    depmap_filt = df[
        df['lineage_1'].str.contains(names) |
        df['lineage_2'].str.contains(names) |
        df['lineage_3'].str.contains(names) |
        df['lineage_4'].str.contains(names)
    ]
    if depmap_filt.empty:
        raise ValueError("No cell lines found for the given cancer type. Please check the provided cancer type.")
    # Drop unneeded columns
    depmap_filt = depmap_filt.drop(columns=['cell_line_display_name', 'lineage_1', 'lineage_2', 'lineage_3', 'lineage_4'])
    total_genes = len(depmap_filt.columns)
    return depmap_filt, total_genes

def _gene_specific_depmap(df:pd.DataFrame, genes:list) -> pd.DataFrame:
    """
    Filter the input DataFrame for specific genes and calculate the average scores for each gene.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    genes (list): The list of genes to filter for.

    Returns:
    df_sub (pd.DataFrame): The filtered DataFrame containing only the specified genes.
    avg_scores (list): The average scores for each gene.
    """
    if not genes:
        raise ValueError("No genes provided. Please provide a list of genes to filter for.")
    # Filter for genes
    filt_genes_no_missing = [x for x in genes if x in df.columns]
    df_sub = df[filt_genes_no_missing]
    # Get the average scores
    avg_scores = df_sub.mean(axis=0).tolist()
    return df_sub, avg_scores, filt_genes_no_missing

def _plot_depmap_distributions(query_avg_score:list, control_avg_score:list, p_value: float, cancer_type:list, fontface:str, fontsize:int, plot_query_color:str, plot_background_color:str) -> Image:
    """
    Plots the distributions of query and control average scores and returns the image.

    Args:
        query_avg_score (list): List of query average scores.
        control_avg_score (list): List of control average scores.
        p_value (float): The p-value.
        cancer_type (list): List of cancer types.
        fontface (str): Font face for the plot.
        fontsize (int): Font size for the plot.
        plot_query_color (str): Color for the query genes plot.
        plot_background_color (str): Color for the background plot.

    Returns:
        Image: The plotted image.

    """
    plt.rcParams.update({'font.size': fontsize,
                    'font.family': fontface})
    _, ax = plt.subplots(figsize = (6, 4), tight_layout = True)
    ax.hist(
        [query_avg_score, control_avg_score],
        color = [plot_query_color, plot_background_color],
        density = True,
        bins = np.arange(-2.6, 0.51, 0.1)
    )
    # Set the legend
    custom_lines = [Line2D([0], [0], color=plot_query_color, lw=4),
                    Line2D([0], [0], color=plot_background_color, lw=4),
                    Line2D([0], [0], color='none')]
    # Update legend with the p-value
    ax.legend(custom_lines, [
        f'Query Genes, Mean = {np.mean(query_avg_score):.2f}',
        f'Control Genes, Mean = {np.mean(control_avg_score):.2f}',
        f'P-value = {p_value:.2e}'
    ], fontsize=fontsize)
    # Adjust plot axes
    plt.xlabel(f'Average {",".join(cancer_type)} Chronos Score', fontsize = 16)
    plt.ylabel('Normalized Gene Count', fontsize = 16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.xticks(size = fontsize)
    plt.yticks(size = fontsize)
    # Save plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def depmap_enrichment(query:list, cancer_type:list, custom_background:Any = 'depmap', plot_fontface:str = 'Avenir', plot_fontsize:int = 14, plot_query_color:str = 'red', plot_background_color:str = 'gray', savepath:str = False, verbose:int = 0) -> (float, Image): # type: ignore
    """
    Performs enrichment analysis using DepMap scores for a given query gene list and cancer type.

    Args:
        query (list): List of query genes.
        cancer_type (list): List of cancer types.
        control_genes (list, optional): List of control genes. Defaults to an empty list.
        plot_fontface (str, optional): Font face for the plot. Defaults to 'Avenir'.
        plot_fontsize (int, optional): Font size for the plot. Defaults to 14.
        plot_query_color (str, optional): Color for the query gene distribution in the plot. Defaults to 'red'.
        plot_background_color (str, optional): Color for the background gene distribution in the plot. Defaults to 'gray'.
        savepath (str, optional): Path to save the output files. Defaults to False.

    Returns:
        tuple: A tuple containing the p-value and the plot image.
    """
    # Load depmap scores
    depmap_df, _ = _load_depmap()
    # Clean and filter depmap_df for cancer type
    depmap_filt, depmap_total_genes = _cancer_specific_depmap(depmap_df, cancer_type, verbose = verbose)

    # Subset for genes of interest
    _, query_avg_score, query_genes_clean = _gene_specific_depmap(depmap_filt, query)
    # Save scores
    query_df = pd.DataFrame({
        'Genes': query_genes_clean,
        'DepMap Score' : query_avg_score
    })
    query_df = query_df.set_index('Genes')

    # Get control or background genes
    if custom_background == 'depmap':
        background_name = 'DepMap'
        background_genes = [x for x in depmap_filt.columns.tolist() if x not in query]
        _, control_avg_score, control_genes_clean = _gene_specific_depmap(depmap_filt, background_genes)
        total_control = depmap_total_genes
    else:
        background_dict, background_name = _define_background_list(custom_background, just_genes = True, verbose = verbose)
        control_genes = background_dict[background_name]
        _, control_avg_score, control_genes_clean = _gene_specific_depmap(depmap_filt, control_genes)
        total_control = len(control_genes)

    # Calculate the p-value
    _, p_val = mannwhitneyu(query_avg_score, control_avg_score)

    # Plot distributions
    plot = _plot_depmap_distributions(query_avg_score, control_avg_score, p_val, cancer_type, plot_fontface, plot_fontsize, plot_query_color, plot_background_color)

    # Save the files
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'DepMap_Enrichment')
        os.makedirs(new_savepath, exist_ok=True)
        # Save p-value
        with open(os.path.join(new_savepath, 'p_value.txt'), 'w') as f:
            f.write(f"Number of cell lines for {','.join(cancer_type)}: {len(depmap_filt)}\n")
            f.write(f'Number of query genes with scores: {len(query_avg_score)}/{len(query)}\n')
            f.write(f'Number of control genes (from {background_name}) with scores: {len(control_avg_score)}/{total_control}\n')
            f.write(f'Mann-Whitney U p-value: {p_val:.6e}')

        # Save scores
        query_df.sort_values(by='DepMap Score', ascending = True).to_csv(os.path.join(new_savepath, 'Query_Avg_Score.txt'), sep = '\t', header = True, index = True)
        # Control dataframe
        control_df = pd.DataFrame({
            'Genes': control_genes_clean,
            'DepMap Score' : control_avg_score
        })
        control_df.sort_values(by = 'DepMap Score', ascending = True).to_csv(os.path.join(new_savepath, 'Control_Avg_Score.txt'), sep = '\t', header = True, index = False)
        # Save plot
        plot.save(os.path.join(new_savepath, 'DepMap_Enrichment.png'))

    return query_df, p_val, plot
#endregion


#region DrugBank/OpenTargets Drugs
def _parse_dgi_db(data:dict) -> pd.DataFrame:
    """
    Parses the DGI database data and returns a DataFrame with the extracted information.

    Parameters:
    data (dict): The DGI database data.

    Returns:
    pandas.DataFrame: A DataFrame containing the parsed information from the DGI database.
    """
    flattened_data = []
    # Loop through each gene's interactionss
    for gene_data in data['data']['genes']['nodes']:
        gene_name = gene_data['name']
        interactions = gene_data['interactions']
        # Loop through each interaction
        for interaction_idx in range(len(interactions)):
            # Get important holder values
            interaction_holder = interactions[interaction_idx]
            drug_name = interaction_holder['drug']['name']

            ### Parse the interaction attributes
            # Mechanism of Action
            mechanism_of_action = np.nan
            for attribute in interaction_holder['interactionAttributes']:
                if attribute['name'] == 'Mechanism of Action':
                    mechanism_of_action = attribute['value']
            # Source databases
            source_dbs = []
            for attribute in interaction_holder['sources']:
                source_dbs.append(attribute['sourceDbName'])
            # Publications
            pmids = []
            for attribute in interaction_holder['publications']:
                pmids.append(str(attribute['pmid']))
            # Drug aliases
            aliases = []
            for attribute in interaction_holder['drug']['drugAliases']:
                if attribute['alias'] != drug_name:
                    aliases.append(attribute['alias'])
            # Approval Ratings
            approval_ratings = [x['rating'] for x in interaction_holder['drug']['drugApprovalRatings']]
            approval_ratings = [x for x in approval_ratings if x not in ['Prescribable', 'Approved', 'Prescription', 'Max Phase 4', 'Not Approved']]
            approval_ratings = ";".join(approval_ratings)
            # Drug Class, Indication, and Year of Approval, Developer, Clinical Trial ID
            drug_class, indications, year_of_app, developer, clinical_trial_id = [], [], [], [], []
            for attribute in interaction_holder['drug']['drugAttributes']:
                if attribute['name'] == 'Drug Class':
                    drug_class.append(attribute['value'])
                elif attribute['name'] == 'Indication':
                    indications.append(attribute['value'])
                elif attribute['name'] == 'Year of Approval':
                    year_of_app.append(attribute['value'])
                elif attribute['name'] == 'Developer':
                    developer.append(attribute['value'])
                elif attribute['name'] == 'Clinical Trial ID':
                    clinical_trial_id.append(attribute['value'])

            # Concat results from parsing
            try:
                interaction_type = interaction_holder['interactionTypes'][0]['type']
                directionality_type = interaction_holder['interactionTypes'][0]['directionality']
                interaction_type = f"{interaction_type} ({directionality_type})"
            except:
                interaction_type = np.nan

            # Create entry dictionary
            interaction_entry = {
                'gene': gene_name,
                'drug': drug_name,
                'drug_concept_id': interaction_holder['drug']['conceptId'],
                'aliases': ";".join(aliases),
                'approvalRatings' : approval_ratings,
                'isApproved' : bool(interaction_holder['drug']['approved']),
                'drug_class': ";".join(drug_class),
                'indications': ";".join(indications),
                'year_of_approval': ";".join(year_of_app),
                'developer': ";".join(developer),
                'clinical_trial_id': ";".join(clinical_trial_id),
                'interaction_score': interaction_holder['interactionScore'],
                'interaction_direction': interaction_type,
                'mechanism_of_action': mechanism_of_action,
                'source_dbs': ";".join(source_dbs),
                'num_pmids': len(pmids),
                'pmids': ";".join(pmids)
            }

            flattened_data.append(interaction_entry)

    df = pd.DataFrame(flattened_data)
    return df

def _get_dgidb_data(query:list, min_interaction_citations:int, approved: bool) -> pd.DataFrame:
    """
    Retrieves interaction data from DGIdb API for a given list of genes.
    https://www.dgidb.org/api

    Args:
        query (list): List of gene names to query.
        min_interaction_citations (int): Minimum number of interaction citations required.
        approved (bool): Whether to only include approved drugs.

    Returns:
        pd.DataFrame: DataFrame containing the retrieved interaction data.
    """
    url = "https://dgidb.org/api/graphql"
    gene_array = "[" + ", ".join(f'"{gene}"' for gene in query) + "]"
    api_query = f"""
        {{
        genes(names: {gene_array}) {{
            nodes {{
            name
            interactions {{
                drug {{
                name
                conceptId
                approved
                drugAliases {{
                    alias
                }}
                drugApprovalRatings {{
                    rating
                }}
                drugAttributes {{
                    name
                    value
                }}
                }}
                interactionScore
                interactionTypes {{
                type
                directionality
                }}
                interactionAttributes {{
                name
                value
                }}
                publications {{
                pmid
                }}
                sources {{
                sourceDbName
                }}
            }}
            }}
        }}
        }}
        """

    # Make the POST request
    response = requests.post(url, json={'query': api_query})

    # Check for errors
    if response.status_code == 200:
        data = response.json()
        df = _parse_dgi_db(data)
        df = df[df['num_pmids'] >= min_interaction_citations]
        df = df[df['isApproved'] == approved]
        df = df.sort_values(by = ['num_pmids'], ascending = False)
        df = df.set_index('gene')
    else:
        raise ValueError("Query failed to run by returning code of {}.".format(response.status_code))
    return df

def _load_open_targets_drugs() -> pd.DataFrame:
    """
    Load the Open Targets drugs data from a parquet file and return it as a DataFrame.

    Returns:
        pd.DataFrame: The Open Targets drugs data as a DataFrame.
    """
    ot_drug_stream = pkg_resources.resource_stream(__name__, 'data/opentarget_drugtargets_08242023.parquet')
    ot_drug_df = pd.read_parquet(ot_drug_stream, engine='pyarrow')

    columns_to_decode = ['linkedTargets.rows', 'linkedDiseases.rows', 'childChemblIds', 'crossReferences', 'synonyms', 'tradeNames']
    for col in columns_to_decode:
        if any(isinstance(x, bytes) for x in ot_drug_df[col]):
            ot_drug_df[col] = ot_drug_df[col].apply(lambda x: eval(x.decode("utf-8")) if pd.notnull(x) and isinstance(x, bytes) else x)
    return ot_drug_df

def _get_open_targets_drugs(query:list, approved: bool, verbose:int = 0) -> pd.DataFrame:
    """
    Retrieves drugs from Open Targets database that target the specified genes.

    Args:
        query (list): List of gene names to search for.
        approved (bool): Value to filter drugs by (FDA-approved drugs = True).

    Returns:
        pd.DataFrame: Filtered dataframe containing drugs that target at least one gene from the query list.
    """
    # Load Open Targets files
    drug_targets = _load_open_targets_drugs()
    genes_to_ensembl = _load_open_targets_mapping()

    # Convert genes to Ensembl ID
    pd.options.mode.chained_assignment = None
    ensemble_ids = genes_to_ensembl[genes_to_ensembl['Gene name'].isin(query)]
    if verbose > 0:
        print(f"OpenTargets - {len(ensemble_ids['Gene name'].unique())}/{len(query)} genes found in Ensembl mapping.")

    # Find drugs which target at least one gene
    filtered_targets = drug_targets[drug_targets['linkedTargets.rows'].apply(lambda x: any(gene in ensemble_ids['Gene stable ID'].tolist() for gene in x))]

    # Add columns for all target genes and overlapping target genes
    filtered_targets['linkedTargets.genes'] = filtered_targets['linkedTargets.rows'].apply(lambda x: [genes_to_ensembl.loc[genes_to_ensembl['Gene stable ID'] == ensg, 'Gene name'].values[0] if ensg in genes_to_ensembl['Gene stable ID'].tolist() else ensg for ensg in x])
    overlapping_genes_list = []
    for genes_list in filtered_targets['linkedTargets.genes']:
        genes_set = set(genes_list)
        overlap = genes_set.intersection(set(query))
        overlapping_genes_list.append(overlap)
    filtered_targets['overlapping_genes'] = overlapping_genes_list

    # Explode filtered_targets
    filtered_targets['overlapping_genes'] = filtered_targets['overlapping_genes'].apply(list)
    filtered_targets= filtered_targets.explode('overlapping_genes')

    # Modify column values
    filtered_targets['isApproved'] = filtered_targets['isApproved'].apply(lambda x: True if x == 1 else False)
    filtered_targets['ApprovalRatings'] = [f"Max Phase {x}" for x in filtered_targets['maximumClinicalTrialPhase']]
    filtered_targets = filtered_targets.drop('maximumClinicalTrialPhase', axis=1)

    # Filter dataframe
    filtered_targets = filtered_targets[filtered_targets['isApproved'] == approved]

    # Clean column names
    filtered_targets = filtered_targets.rename(columns = {'name':'drug'})
    filtered_targets = filtered_targets.set_index('overlapping_genes')

    pd.options.mode.chained_assignment = 'warn'

    return filtered_targets

def drug_gene_interactions(query: list, drug_source:list = ['OpenTargets'], dgidb_min_citations:int = 1, approved:bool = True, savepath:Any = False, verbose: int = 0) -> dict:
    """
    Retrieves drug-gene interactions based on the given query.

    Parameters:
    - query (list): List of genes for which drug-gene interactions are to be retrieved.
    - drug_source (str): Source of drug-gene interaction data. Valid options are 'OpenTargets' and 'DGIdb'. Default is 'OpenTargets'.
    - dgidb_min_citations (int): Minimum number of citations required for a drug-gene interaction to be considered. Default is 1.
    - approved (bool): Whether to filter drugs by approval status. Default is True.
    - savepath (Any): Path to save the drug-gene interaction data. If False, the data will not be saved. Default is False.

    Returns:
    - df (pandas.DataFrame): DataFrame containing the drug-gene interaction data.

    Raises:
    - ValueError: If the drug_source parameter is not 'OpenTargets' or 'DGIdb'.
    """
    # Set saving parameters
    method_data = {}

    # Check if drug source is valid
    check = [x for x in drug_source if x not in ['OpenTargets', 'DGIdb']]
    if len(check) > 0:
        raise ValueError("Invalid drug source. Please define list with options: 'OpenTargets' or 'DGIdb'.")

    # Get drug-gene interactions
    if 'OpenTargets' in drug_source:
        ot_drugs_df = _get_open_targets_drugs(query, approved, verbose = verbose)
        if verbose > 0:
            print("OpenTargets - {}/{} genes with interacting drugs. {} unique drugs found with {} approval".format(len(ot_drugs_df.index.unique()), len(query), len(ot_drugs_df.drug.unique()), approved))
        method_data['OpenTargets'] = ot_drugs_df

    if 'DGIdb' in drug_source:
        dgi_db = _get_dgidb_data(query, dgidb_min_citations, approved)
        if verbose > 0:
            print("DGIdb - {}/{} genes with interacting drugs. {} unique drugs found with {} approval".format(len(dgi_db.index.unique()), len(query), len(dgi_db.drug.unique()), approved))
        method_data['DGIdb'] = dgi_db

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'DrugGeneInteractions/')
        os.makedirs(new_savepath, exist_ok=True)
        # Save files
        for key, value in method_data.items():
            value.to_csv(new_savepath + f'{key}_DrugGeneInteractions.csv', index = True)

    return method_data

#endregion
