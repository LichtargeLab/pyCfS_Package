"""
Collection of functions looking at previous genetic overlap recovery

Functions:

"""

import pkg_resources
import io
import os
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import requests
import time
from multiprocessing import Pool
from collections import Counter
from typing import Any
import random
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from Bio import Entrez
import seaborn as sns
from PIL import Image
from venn import venn
from scipy.stats import norm, fisher_exact, ks_2samp
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csgraph, identity
from scipy.sparse.linalg import lgmres
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import networkx as nx
import concurrent.futures
from itertools import repeat
from .utils import _hypergeo_overlap, _fix_savepath, _define_background_list, _clean_genelists, _load_grch38_background,_load_clean_string_network, _get_edge_weight

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


#region OpenTargets Gold Standard generator
def _validate_efo_term(efo_id:str) -> None:
    """
        Validate the given EFO or MONDO term ID.

        This function checks if the provided ID starts with "EFO_" or "MONDO_".
        If the ID does not start with either of these prefixes, a ValueError is raised.

        Args:
            efo_id (str): The EFO or MONDO term ID to validate.

        Raises:
            ValueError: If the provided ID does not start with "EFO_" or "MONDO_".
    """
    # Check if EFO or MONDO in ID
    if efo_id.startswith("EFO_") or efo_id.startswith("MONDO_"):
        pass
    else:
        raise ValueError("Invalid EFO ID - Please provide a valid EFO ('EFO_######') or MONDO ID ('MONDO_######')")

def _def_opentargets_base_url() -> str:
    """
    Returns the base URL for the Open Targets API.

    This function provides the base URL for accessing the Open Targets API,
    which is used for querying biomedical data. API documentation - https://platform-docs.opentargets.org/data-access/graphql-api

    Returns:
        str: The base URL for the Open Targets API.
    """
    return "https://api.platform.opentargets.org/api/v4/graphql"

def _query_target_disease_associations(efo_id: str) -> pd.DataFrame:
    """
        Queries the Open Targets platform for target-disease associations based on the provided EFO ID.
        Args:
            efo_id (str): Experimental Factor Ontology (EFO) ID of the disease to query.
        Returns:
            pd.DataFrame: A DataFrame containing the target-disease association data, including target IDs, approved symbols, overall scores, datatype scores, and datasource scores.
            str: The name of the disease associated with the provided EFO ID.
        Raises:
            ValueError: If the HTTP request to the Open Targets platform fails.
    """
    base_url = _def_opentargets_base_url()
    
    query = f'''
        query associatedTargets {{
            disease(efoId: "{efo_id}") {{
                id
                name
                associatedTargets(page: {{ index: 0, size: 10000 }}) {{
                    count
                    rows {{
                        target {{
                            id
                            approvedSymbol
                        }}
                        score
                        datatypeScores {{
                            id
                            score
                        }}
                        datasourceScores {{
                            id
                            score
                        }}
                    }}
                }}
            }}
        }}
    '''
    response = requests.post(base_url, json={'query': query})
    if response.status_code != 200:
        # Wait for 1 minute and retry
        time.sleep(60)
        response = requests.post(base_url, json={'query': query})
        if response.status_code != 200:
            raise ValueError("Error: " + response.text)
    
    data = response.json()
    disease = data['data']['disease']['name']
    data_parsed = data['data']['disease']['associatedTargets']
    target_data = data_parsed['rows']

    # Parse each row in target_data
    rows_list = []
    for row in target_data:
        # Extract basic target info and score
        target_id = row['target']['id']
        approved_symbol = row['target']['approvedSymbol']
        score = row['score']
        
        # Extract scores for each datatype and datasource as a flattened list
        datatype_scores = {ds['id']: ds['score'] for ds in row.get('datatypeScores', [])}
        datasource_scores = {ds['id']: ds['score'] for ds in row.get('datasourceScores', [])}
        
        # Combine all extracted data into one dictionary
        row_data = {
            'target_id': target_id,
            'approved_symbol': approved_symbol,
            'overall_score': score,
            **datatype_scores,
            **datasource_scores
        }
        
        rows_list.append(row_data)

    # Create DataFrame from rows_list
    target_df = pd.DataFrame(rows_list)
    return target_df, disease

def _plot_association_scores(df: pd.DataFrame, bins:list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]) -> Image:
    """
        Plots the association scores between overall association scores and genetic association scores.
        Args:
            df (pd.DataFrame): DataFrame containing the data with 'overall_score' and 'genetic_association' columns.
            bins (list, optional): List of bin edges for categorizing the scores. Defaults to [0, 0.2, 0.4, 0.6, 0.8, 1.0].
        Returns:
            Image: An image object containing the plotted bar chart.
    """
    # Define custom colors for each bin
    colors = ['blue', 'lightblue', 'lightgray', 'lightcoral', 'red']
    
    # Create bin labels
    labels = []
    for bin in bins:
        try:
            i = bins.index(bin)
            if bin != 0.8:
                labels.append(f'[{bin}, {bins[i+1]})')
            else:
                labels.append(f'[{bin}, {bins[i+1]}]')
        except IndexError:
            continue
    
    # Group labels
    df['globalScore_group'] = pd.cut(
        df['overall_score'], bins=bins, labels=labels, include_lowest=True, right=False
    )
    df['geneticAssociations_group'] = pd.cut(
        df['genetic_association'], bins=bins, labels=labels, include_lowest=True, right=False
    )
    
    # Calculate counts and percentages
    counts = df.groupby(['globalScore_group', 'geneticAssociations_group']).size().unstack()
    percentages = counts.divide(counts.sum(axis=1), axis=0)
    labels = percentages.index
    
    # Plot setup
    fig, ax = plt.subplots()
    bottom = [0] * len(labels)

    # Keep track of used positions for each x-tick to detect overlaps
    used_positions = {i: [] for i in range(len(labels))}

    # Plot each bar segment with colors and labels
    for column_idx, (column, color) in enumerate(zip(percentages.columns, colors)):
        ax.bar(labels, percentages[column], bottom=bottom, label=column, color=color, edgecolor='black', linewidth=0.5)
        
        for i, (value, total) in enumerate(zip(counts[column], bottom)):
            if value <= 0:
                continue

            # Calculate y-position and check for overlap
            y_position = total + percentages[column][i] / 2
            # Check for overlap with previously used positions at the same x-tick
            adjustment = 0
            for pos in used_positions[i]:
                if abs(y_position - pos) < 0.06:  # Adjust this threshold as needed
                    adjustment += 0.06  # Increment adjustment to avoid overlap
            y_position += adjustment
            if y_position < 0.03:
                y_position = 0.03

            # Add the position to used positions list for this x-tick
            used_positions[i].append(y_position)

            # Add the text label
            text_color = 'black' if color != 'blue' else 'white'
            ax.text(
                i, y_position,
                f"{column} \n (n={value})",
                ha='center', va='center', size=8, color=text_color
            )
        
        # Update the bottom values for stacking
        bottom += percentages[column]

    # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles),
        reversed(labels),
        title="Genetic Association Score",
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0
    )
    
    # Add labels and title
    plt.ylabel('% of Genes by Genetic Association Score')
    plt.xlabel('Overall Association Score')
    plt.title('Genetic Association Score by Overall Association Score')
    plt.tight_layout()
    
    # Save plot to image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _check_coding_variants(df: pd.DataFrame, efo_id: str) -> list:
    """
        Check for coding variant associations for a given disease using the Open Targets API.
        Parameters:
        df (pd.DataFrame): DataFrame containing gene IDs in a column named 'target_id'.
        efo_id (str): EFO ID of the disease to query.
        Returns:
        pd.DataFrame: DataFrame containing the evidence of coding variant associations with the following columns:
            - target_id: ID of the target gene.
            - approved_symbol: Approved symbol of the target gene.
            - disease_id: ID of the disease.
            - disease_name: Name of the disease.
            - datasource_id: ID of the data source.
            - odds_ratio: Odds ratio of the variant.
            - odds_ratio_ci_lower: Lower confidence interval of the odds ratio.
            - odds_ratio_ci_upper: Upper confidence interval of the odds ratio.
            - variant_effect: Effect of the variant.
            - beta: Beta value of the variant.
            - beta_ci_lower: Lower confidence interval of the beta value.
            - beta_ci_upper: Upper confidence interval of the beta value.
            - variant_id: ID of the variant.
            - variant_rs_id: RS ID of the variant.
            - direction_on_trait: Direction of the variant's effect on the trait.
            - variant_func_consequence_label: Functional consequence label of the variant.
            - literature: Literature references associated with the variant.
        Raises:
        ValueError: If the API request fails.
    """
    # Get gene IDs
    gene_ids = df['target_id'].tolist()  # List of gene IDs
    gene_ids = [f'"{gene_id}"' for gene_id in gene_ids]  # Wrap gene IDs in quotes

    # Query Open Targets API to check for coding variant associations
    base_url = _def_opentargets_base_url()
    query = f'''
        query targetDiseaseCodingVariantEvidence {{
            disease(efoId: "{efo_id}") {{
                id
                name
                evidences(ensemblIds: [{",".join(gene_ids)}], datasourceIds: ["ot_genetics_portal", "gene_burden"], size: 10000) {{
                    rows {{
                        target {{
                            id
                            approvedSymbol
                        }}
                        disease {{
                            id
                            name
                        }}
                        datasourceId
                        oddsRatioConfidenceIntervalLower
                        oddsRatioConfidenceIntervalUpper
                        oddsRatio
                        beta
                        betaConfidenceIntervalLower
                        betaConfidenceIntervalUpper
                        variantId
                        variantRsId
                        literature
                        variantAminoacidDescriptions
                        variantFunctionalConsequence {{
                            label
                        }}
                        variantEffect
                        directionOnTrait
                    }}
                }}
            }}
        }}
    '''

    response = requests.post(base_url, json={'query': query})
    if response.status_code != 200:
        # Wait for 1 minute and retry
        time.sleep(60)
        response = requests.post(base_url, json={'query': query})
        if response.status_code != 200:
            raise ValueError("Error: " + response.text)

    data = response.json()
    disease_name = data['data']['disease']['name']
    evidences = data['data']['disease']['evidences']['rows']
    # Parse each row in evidences
    rows_list = []
    for evidence in evidences:
        # Basic information
        target_id = evidence['target']['id']
        approved_symbol = evidence['target']['approvedSymbol']
        disease_id = evidence['disease']['id']
        disease_name = evidence['disease']['name']
        datasource_id = evidence['datasourceId']
        odds_ratio = evidence.get('oddsRatio')
        odds_ratio_ci_lower = evidence.get('oddsRatioConfidenceIntervalLower')
        odds_ratio_ci_upper = evidence.get('oddsRatioConfidenceIntervalUpper')
        beta = evidence.get('beta')
        beta_ci_lower = evidence.get('betaConfidenceIntervalLower')
        beta_ci_upper = evidence.get('betaConfidenceIntervalUpper')
        variant_id = evidence.get('variantId')
        variant_rs_id = evidence.get('variantRsId')
        literature = evidence.get('literature')
        if isinstance(literature, list):
            literature = ":".join(literature)
        variant_effect = evidence.get('variantEffect')
        direction_on_trait = evidence.get('directionOnTrait')
        
        # Handle nested fields like variantFunctionalConsequence and variantFunctionalConsequenceFromQtlId
        variant_func_consequence = evidence.get('variantFunctionalConsequence')
        variant_func_consequence_label = variant_func_consequence.get('label') if variant_func_consequence else None
        
        # Collect data into a dictionary
        row_data = {
            'target_id': target_id,
            'approved_symbol': approved_symbol,
            'disease_id': disease_id,
            'disease_name': disease_name,
            'datasource_id': datasource_id,
            'odds_ratio': odds_ratio,
            'odds_ratio_ci_lower': odds_ratio_ci_lower,
            'odds_ratio_ci_upper': odds_ratio_ci_upper,
            'variant_effect': variant_effect,
            'beta': beta,
            'beta_ci_lower': beta_ci_lower,
            'beta_ci_upper': beta_ci_upper,
            'variant_id': variant_id,
            'variant_rs_id': variant_rs_id,
            'direction_on_trait': direction_on_trait,
            'variant_func_consequence_label': variant_func_consequence_label,
            'literature': literature,
        }
        rows_list.append(row_data)
    
    # Create DataFrame from rows_list
    evidence_df = pd.DataFrame(rows_list)
    return evidence_df

def _prioritize_goldstandards(df: pd.DataFrame, efo_id: str, min_goldstandards: int) -> (pd.DataFrame, list): # type: ignore
    """
        Prioritize gold standard genes based on genetic association and global score.

        This function filters and prioritizes genes from a given DataFrame based on their genetic association 
        and global score. It looks for coding variant evidence in the genes and returns a DataFrame of prioritized 
        genes along with a list of these genes and the threshold used for prioritization.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing gene data.
        efo_id (str): The EFO (Experimental Factor Ontology) ID to check for coding variants.
        min_goldstandards (int): The minimum number of gold standard genes required.

        Returns:
        pd.DataFrame: A DataFrame containing the prioritized genes with their associated data.
        list: A list of prioritized gene symbols.
        str: The threshold used for genetic association and global score.
    """
    # Generate matrix of global and genetic association scores
    coding_var_genes = []
    coding_var_rows = []
    min_genetic_association = None
    min_global_score = None
    # Loop through data to find coding variant evidence
    for genetic_association in ['[0.8, 1.0]', '[0.6, 0.8)', '[0.4, 0.6)', '[0.2, 0.4)', '[0.0, 0.2)']:
        for global_score in ['[0.8, 1.0]', '[0.6, 0.8)', '[0.4, 0.6)', '[0.2, 0.4)', '[0.0, 0.2)']:
            min_genetic_association = genetic_association
            min_global_score = global_score
            # Grab subset of data
            try:
                sub_df = df[(df['geneticAssociations_group'].astype(str) == genetic_association) & (df['globalScore_group'].astype(str) == global_score)]
                if sub_df.empty:
                    continue
            except KeyError as e:
                continue
            # Loop through each gene to see if there is coding variant evidence
            for gene in sub_df['approved_symbol']:
                gene_data = df[df['approved_symbol'] == gene]
                data_evidences = _check_coding_variants(gene_data, efo_id)
                if data_evidences.empty:
                    continue
                # Check for coding variant evidence
                if "gene_burden" in data_evidences['datasource_id'].tolist() or data_evidences['variant_func_consequence_label'].str.contains('missense_variant|inframe_deletion|inframe_insertion|frameshift_variant', na = False).any():
                    # Grab coding variant gene burden literature evidence
                    gene_burden_literature = data_evidences[data_evidences['datasource_id'] == 'gene_burden']['literature'].unique().tolist()
                    gene_burden_literature = [x for x in gene_burden_literature if x is not None]
                    gene_burden_literature = ":".join(gene_burden_literature) if len(gene_burden_literature) > 0 else None
                    # Grab GWAS missense variant literature evidence
                    missense_variant_literature = data_evidences[data_evidences['variant_func_consequence_label'].str.contains('missense_variant|inframe_deletion|inframe_insertion|frameshift_variant', na = False)]['literature'].unique().tolist()
                    missense_variant_literature = [x for x in missense_variant_literature if x is not None]
                    missense_variant_literature = ":".join(missense_variant_literature) if len(missense_variant_literature) > 0 else None
                    # Add the gene to the coding_var_genes list
                    coding_var_genes.append(gene)
                    # Create row for output df
                    coding_var_rows.append({
                        'approved_symbol': gene,
                        'genetic_association': gene_data['genetic_association'].values[0],
                        'overall_score': gene_data['overall_score'].values[0],
                        'gene_burden_literature': gene_burden_literature,
                        'coding_variant_gwas_literature': missense_variant_literature,
                        'genetic_association_group': genetic_association,
                        'overall_score_group': global_score
                    })
            if len(coding_var_genes) >= min_goldstandards:
                break
        if len(coding_var_genes) >= min_goldstandards:
            break
    coding_var_evidence_df = pd.DataFrame(coding_var_rows)
    threshold = f"Genetic Association Score: {min_genetic_association}, Overall Association Score: {min_global_score}"
    return coding_var_evidence_df, coding_var_genes, threshold

def generate_ot_goldstandards(efo_id:str, min_goldstandards:int = 10, min_genetic_association:float = 0.0, min_overall_association:float = 0.0, filter_for_coding:bool = True, savepath:Any = False) -> (list, pd.DataFrame): # type: ignore
    """
        Generate a gold standard list of genes associated with a given EFO (Experimental Factor Ontology) term using the OpenTargets API.

        Args:
            efo_id (str): The EFO ID for the disease or phenotype of interest.
            min_goldstandards (int, optional): Minimum number of gold standard genes to prioritize. Defaults to 10.
            filter_for_coding (bool, optional): Whether to filter for coding variant evidence. Defaults to True.
            savepath (Any, optional): Path to save the results. If False, results are not saved. Defaults to False.

        Returns:
            list: A list of prioritized gold standard gene symbols.
            pd.DataFrame: A DataFrame containing the prioritized gold standard genes and their association scores.

        Notes:
            API documentation - https://platform-docs.opentargets.org/data-access/graphql-api
    """
    # Verify EFO id
    _validate_efo_term(efo_id)

    # Query associations from OpenTargets API
    gene_associations, disease = _query_target_disease_associations(efo_id)
    gene_associations = gene_associations.sort_values(by = ['genetic_association', 'overall_score'], ascending = False).reset_index(drop = True)

    # Plot overall association scores against genetic scores
    image = _plot_association_scores(gene_associations)

    # Filter associations for genetic association and overall score
    gene_associations_prior = gene_associations[(gene_associations['genetic_association'] >= min_genetic_association) & (gene_associations['overall_score'] >= min_overall_association)]
    if gene_associations_prior.empty:
        raise ValueError("No genes found with the specified genetic association and overall score thresholds")
    if min_genetic_association > 0.0 or min_overall_association > 0.0:
        min_goldstandards = 10_000 # Set to essentially no limit on the number of values

    # Prioritize top genes by association and coding variant evidence
    if filter_for_coding:
        prioritized_goldstandards_df, prioritized_goldstandards, threshold = _prioritize_goldstandards(gene_associations_prior, efo_id, min_goldstandards)
    else:
        prioritized_goldstandards_df = gene_associations.head(min_goldstandards)
        prioritized_goldstandards = prioritized_goldstandards_df['approved_symbol'].tolist()
        threshold = "No filtering for coding variant evidence"

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'OpenTargets_GoldStandard/')
        os.makedirs(new_savepath, exist_ok=True)
        # Save the overall gene associations
        gene_associations.to_csv(new_savepath + f"OpenTargets-Export-{disease}.csv", index = False)
        # Save image
        image.save(new_savepath + f"OpenTargets-Association-Scores-{disease}.png", bbox_inches = 'tight', pad_inches = 0.5)
        # Save coding variant evidence
        prioritized_goldstandards_df.to_csv(new_savepath + f"OpenTargets-Coding-Variant-Evidence-{disease}.csv", index = False)
        # Save prioritized gold standards
        with open(new_savepath + f"OpenTargets-GoldStandards-{disease}.txt", 'w') as f:
            f.write("\n".join(prioritized_goldstandards))
        # Write threshold
        with open(new_savepath + f"OpenTargets-Threshold-{disease}.txt", 'w') as f:
            f.write(threshold)

    return prioritized_goldstandards, prioritized_goldstandards_df
#endregion



#region Gold Standard Overlap
def _get_overlap(list1:list, list2:list, background_list:list) -> (list, float): # type: ignore
    """
    Calculates the overlap between two lists and computes the p-value for the overlap.

    This function identifies the common elements between two lists and calculates the
    statistical significance (p-value) of this overlap using a hypergeometric test.
    The background gene set for the hypergeometric test is loaded from a predefined
    function `_load_grch38_background`.

    Args:
        list1 (list): The first list, typically representing a set of query genes.
        list2 (list): The second list, usually representing a gene set (e.g., from a gene set database).

    Returns:
        tuple:
            - list: A list containing the overlapping elements between `list1` and `list2`.
            - float: The p-value indicating the statistical significance of the overlap,
                     calculated using the hypergeometric test.
    """
    # Get overlap and calculate p-val
    overlap = [x for x in list1 if x in list2]
    if background_list:
        background = background_list
    else:
        background = _load_grch38_background()
    pval = _hypergeo_overlap(background_size = len(background), query_genes = len(list1), gs_genes = len(list2), overlap = len(overlap))

    return overlap, pval

def _plot_overlap_venn(query_len:int, goldstandard_len:int, overlap:list, pval:float, show_genes_pval: bool, query_color:str, goldstandard_color:str, fontsize:int, fontface:str) -> Image:
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
    if len(overlap) == 0:
        warnings.warn("No overlapping genes to plot")
        return False
    elif query_len == 0:
        warnings.warn("No genes found in query")
        return False
    # Create Venn Diagram
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _ = plt.figure(figsize=(10, 5))
    out = venn2(subsets=((query_len - len(overlap)),
                        (goldstandard_len - len(overlap)),
                        len(overlap)),
                        set_labels=('Query', 'Goldstandard'),
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
    if show_genes_pval:
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
        plt.text(0, -0.78,
                ", ".join(overlap),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize-2)
    #plt.title("Gold Standard Overlap", fontsize=fontsize+4)
    plt.tight_layout(pad = 2.0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    #buffer.close()
    plt.close()

    return image

def goldstandard_overlap(query: list, goldstandard:list, custom_background:Any = 'ensembl', plot_query_color:str = 'red', plot_goldstandard_color:str = 'gray', plot_show_gene_pval:bool = True, plot_fontsize:int = 14, plot_fontface:str = 'Avenir', savepath:Any = False) -> (list, float, Image): # type: ignore
    """
    Analyzes the overlap between a query gene list and a gold standard gene list, plots a Venn diagram of the overlap,
    and optionally saves the plot and summary.

    This function first cleans the input gene lists, then calculates the overlap between the query and gold standard
    gene lists, including the p-value for this overlap. It then creates a Venn diagram to visually represent the overlap.
    If a save path is provided, the function will save the Venn diagram image and a summary text file with the overlapping
    genes and the p-value.

    Args:
        - query (list): A list of genes representing the query set.
        - goldstandard (list): A list of genes representing the gold standard set.
        - custom_background (Any, optional): Background gene set. Defaults to 'ensembl'. Options include 'reactome', 'ensembl', 'string_v12.0', 'string_v11.5', 'string_v11.0', 'string_v10.0'. STRING background is all interactions > 0.15.
        - plot_query_color (str, optional): Color for the query set in the Venn diagram. Defaults to 'red'.
        - plot_goldstandard_color (str, optional): Color for the gold standard set in the Venn diagram. Defaults to 'gray'.
        - plot_show_gene_pval (bool, optional): Whether to show the overlapping genes and p-value in the Venn diagram. Defaults to True.
        - plot_fontsize (int, optional): Font size for the text in the Venn diagram. Defaults to 14.
        - plot_fontface (str, optional): Font face for the text in the Venn diagram. Defaults to 'Avenir'.
        - savepath (Any, optional): Path to save the output files. If False, files are not saved. Defaults to False.

    Returns:
        tuple:
            - list: A list of genes that overlap between the query and gold standard sets.
            - float: The p-value indicating the statistical significance of the overlap.
            - Image: An image object of the Venn diagram. None if there is no overlap or the query set is empty.

    Note:
        The function depends on the '_clean_genelists', '_get_overlap', and '_plot_overlap_venn' functions being
        available in the same code base and correctly implemented.
    """
    cleaned = _clean_genelists([query, goldstandard])
    query, goldstandard = cleaned[0], cleaned[1]
    # Get overlap
    background_dict, background_name = _define_background_list(custom_background)
    background_list = background_dict[background_name]
    overlapping_genes, pval = _get_overlap(query, goldstandard, background_list)
    # Plot Venn diagram
    image = _plot_overlap_venn(len(query), len(goldstandard), overlapping_genes, pval, plot_show_gene_pval, plot_query_color, plot_goldstandard_color, plot_fontsize, plot_fontface)
    # Output files
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'GoldStandard_Overlap/')
        os.makedirs(new_savepath, exist_ok=True)
        if image:
            image.save(new_savepath + "GoldStandard_Overlap.png", bbox_inches = 'tight', pad_inches = 0.5)
        with open(new_savepath + "GoldStandard_Overlap_Summary.txt", 'w') as f:
            f.write(f"Overlapping genes: {overlapping_genes}\n")
            f.write(f"P-value: {pval}")
        # Write gold standard file for prioritization table
        with open(new_savepath + "GoldStandards.txt", 'w') as f:
            f.write("\n".join(goldstandard))

    return overlapping_genes, pval, image
#endregion



#region PPI Significance
def _base_string_api(version:str) -> str:
    """
    Returns the base URL for the STRING-DB API.

    Returns:
        str: The base URL for the STRING-DB API.
    """
    if version == 'v10.0':
        return "http://version10a.string-db.org/api/"
    elif version == 'v11.0':
        return "https://version-11-0.string-db.org/api/"
    elif version == 'v11.5':
        return "https://version-11-5.string-db.org/api/"
    elif version == 'v12.0':
        return "https://version-12-0.string-db.org/api/"
        #return "https://string-db.org/api/"
    else:
        raise ValueError("Invalid version - Please choose 'v11.0', 'v11.5', or 'v12.0'")

def _format_species(species:int) -> str:
    """
    Formats the species parameter for the API request.

    Args:
        species (int): The species ID.

    Returns:
        str: The formatted species parameter for the API request.
    """
    return f"&species={species}"

def _format_genes(genes:list) -> str:
    """
    Formats a list of genes into a string.

    Args:
        genes (list): A list of genes.

    Returns:
        str: A formatted string containing the gene identifiers.
    """
    genes = "%0d".join(genes)
    return f"identifiers={genes}"

def _format_method(method: str) -> str:
    """
    Formats the given method string into the corresponding API endpoint.

    Args:
        method (str): The method to be formatted.

    Returns:
        str: The formatted API endpoint.

    Raises:
        ValueError: If the method is not one of the valid options.
    """
    if method == 'network_image':
        return "highres_image/network?"
    elif method == 'network_interactions':
        return "tsv/network?"
    elif method == 'ppi_enrichment':
        return "tsv/ppi_enrichment?"
    elif method == 'functional_enrichment':
        return "tsv/enrichment?"
    elif method == 'version':
        return "tsv/version?"
    else:
        raise ValueError("Invalid method - Please choose 'network_image', 'network_interactions', 'network_enrichment', 'functional_enrichment', or 'version'")

def _format_score_threshold(score: float) -> str:
    """
    Formats the score threshold for the required score parameter.

    Args:
        score (float): The score threshold to be formatted.

    Returns:
        str: The formatted score threshold as a query parameter for the required score.

    """
    score *= 1000
    return f"&required_score={score}"

def _plot_enrichment(enrichment_df: pd.DataFrame, plot_fontsize:int, plot_fontface:str) -> dict:
    """
    Plot enrichment analysis results.

    Parameters:
    enrichment_df (pd.DataFrame): The DataFrame containing the enrichment analysis results.
    plot_fontsize (int): The font size of the plot.
    plot_fontface (str): The font face of the plot.

    Returns:
    dict: A dictionary containing the generated enrichment plots.
    """
    # Clean the enrichment dataframe
    enrichment_df = enrichment_df[enrichment_df['fdr'] <= 0.05]
    enrichment_df['-log10(FDR)'] = -1 * enrichment_df['fdr'].apply(np.log10)
    enrichment_df = enrichment_df.sort_values(by='-log10(FDR)', ascending=False)
    enrichment_df['Enrichment Term'] = enrichment_df['term'] + "~" + enrichment_df['description']
    enrichment_df = enrichment_df.rename(columns = {'number_of_genes':'Gene Count'})
    # Set up saving parameters
    categories = enrichment_df.category.unique().tolist()
    enrichment_plots = {}
    # Set plotting parameters
    plt.rcParams.update({'font.size': plot_fontsize,
                        'font.family': plot_fontface})
    def _calculate_plot_height(n_rows:pd.DataFrame, plot_fontsize:int) -> float:
        """
        Calculate the height of a plot based on the number of rows in a DataFrame and the font size of the plot.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data for the plot.
        plot_fontsize (int): The font size of the plot.

        Returns:
        float: The calculated height of the plot.
        """
        plot_height = (n_rows * plot_fontsize) / 50
        if plot_height < 6:
            return 6.0
        else:
            return plot_height
    def _calculate_plot_width(enrichment_terms: pd.Series, plot_fontsize: int) -> float:
        """
        Calculate the plot width based on the maximum term length in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the enrichment terms.
            plot_fontsize (int): The font size used in the plot.

        Returns:
            float: The calculated plot width.
        """
        max_term_length = enrichment_terms.str.len().max()
        plot_width = (max_term_length * plot_fontsize) / 61
        return plot_width
    def _check_viable_plot_size(width: float, height: float, plot_fontsize:int, dpi: int = 300) -> (float, float): # type: ignore
        """
        Check if the calculated plot size is viable and adjust if necessary.

        Args:
            width (float): The calculated plot width.
            height (float): The calculated plot height.

        Returns:
            tuple: A tuple containing the adjusted plot width and height.
        """
        num_pixels = (width * dpi) * (height * dpi)
        if num_pixels > 178956970:
            width /= 2
            height /= 2
            plot_fontsize = np.nan
        return width, height, plot_fontsize
    # Create plots
    for category in categories:
        sub_df = enrichment_df[enrichment_df.category == category]
        # Create 'Gene Count' color map
        norm = Normalize(vmin=sub_df["Gene Count"].min(), vmax=sub_df["Gene Count"].max())
        sm = ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        # Create bar plot
        plot_height = _calculate_plot_height(sub_df.shape[0], plot_fontsize)
        plot_width = _calculate_plot_width(sub_df['Enrichment Term'], plot_fontsize)
        plot_width, plot_height, new_plot_fontsize = _check_viable_plot_size(plot_width, plot_height, plot_fontsize)
        _, ax = plt.subplots(figsize=(plot_width, plot_height))
        sns.barplot(
            x="-log10(FDR)",
            y="Enrichment Term",
            data=sub_df,
            palette="RdBu_r",
            dodge=False,
            hue="Gene Count",
            edgecolor='.2',
            linewidth=1.5,
            legend = False
        )
        # Add significance line
        plt.axvline(x=-np.log10(0.05), color="black", linestyle="--", linewidth=2)
        # Add color bar
        cbar = plt.colorbar(sm, ax = ax, shrink = 0.5)
        cbar.set_label('Gene Count', fontsize=new_plot_fontsize+4)

        #Set labels
        plt.xlabel("-log10(FDR)", fontsize=new_plot_fontsize+4)
        plt.ylabel("Enrichment Term", fontsize=new_plot_fontsize+4)
        plt.title(f"{category} Enrichment", fontsize=new_plot_fontsize+6)
        plt.tight_layout(pad = 5.0)
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format = 'png', dpi = 300)
            buffer.seek(0)
            image = Image.open(buffer)
            enrichment_plots[category] = image
        except Image.DecompressionBombError:
            warnings.warn("Decompression Bomb Error: Image too large to save")
        plt.close()

    return enrichment_plots

def _string_api_call(version:str, genes:list, method:str, score_threshold:float, species:int) -> bytes:
    """
    Makes an API call to retrieve data based on the given parameters.

    Args:
        genes (list): A list of genes.
        method (str): The method to be used for the API call.
        score_threshold (float): The score threshold for the API call.
        species (int): The species for the API call.

    Returns:
        bytes: The response content from the API call.

    Raises:
        ValueError: If the API call returns a non-200 status code.
    """
    query = _base_string_api(version) + \
        _format_method(method) + \
        _format_genes(genes) + \
        _format_species(species=species) + \
        _format_score_threshold(score = score_threshold)
    response = requests.get(query)
    if response.status_code != 200:
        raise ValueError("Error: " + response.text)
    return response.content

def string_enrichment(query:list, string_version:str = 'v11.0', edge_confidence:str = 'medium', species:int = 9606, plot_fontsize:int = 14, plot_fontface:str = 'Sans Serif', savepath:Any = False, verbose:int = 0) -> tuple:
    """
    Performs STRING enrichment analysis for a given list of genes.

    Args:
        - query (list): List of genes for enrichment analysis.
        - string_version(str, optional): STRING database version. Defaults to 'v11.0'. Options include 'v10.0', 'v11.0', 'v11.5', and 'v12.0'.
        - edge_confidence (str, optional): Confidence score for STRING interactions. Defaults to 'medium'. Options include 'all', 'low'(>0.15), 'medium'(0.4), 'high'(0.7), and 'highest'(0.9).
        - species (int, optional): Species ID for STRING database. Defaults to 9606.
        - plot_fontsize (int, optional): Font size for enrichment plots. Defaults to 14.
        - plot_fontface (str, optional): Font face for enrichment plots. Defaults to 'Avenir'.
        - savepath (Any, optional): Path to save the results. Defaults to False.
        - verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        tuple: A tuple containing the following:
            - network_df (pd.DataFrame): DataFrame containing STRING network interactions.
            - p_value (float): P-value for the enrichment analysis.
            - network_image (PIL.Image.Image): Image of the STRING network.
            - functional_enrichment_df (pd.DataFrame): DataFrame containing functional enrichment results.
            - enrichment_plots (dict): Dictionary of enrichment plots.
    """
    # Get the right edge weight value
    edge_weight = _get_edge_weight(edge_confidence)

    # Get the STRING API version
    if string_version != 'v10.0':
        version_b = _string_api_call(version = string_version, genes = query, method = 'version', score_threshold = edge_weight, species = species)
        version = pd.read_csv(io.StringIO(version_b.decode('utf-8')), sep = '\t')
        version = version.loc[0, 'string_version']
        if verbose > 0:
            print(f"STRING API version: {version}")
    else:
        version = string_version
        raise ValueError("STRING v10.0 is archived and API endpoints have been disconnected. Please select another version")
        if verbose > 0:
            print(f"STRING API version: {version}")

    # Get the STRING interactions network file
    network = _string_api_call(version = string_version, genes = query, method = 'network_interactions', score_threshold = edge_weight, species = species)
    network_df = pd.read_csv(io.StringIO(network.decode('utf-8')), sep = '\t')

    # Get the STRING PPI enrichment values
    enrichment = _string_api_call(version = string_version, genes = query, method = 'ppi_enrichment', score_threshold = edge_weight, species = species)
    enrichment_df = pd.read_csv(io.StringIO(enrichment.decode('utf-8')), sep = '\t')
    p_value = enrichment_df['p_value'][0]

    # Get the STRING network
    network_image = _string_api_call(version = string_version, genes = query, method = 'network_image', score_threshold = edge_weight, species = species)
    network_image = Image.open(io.BytesIO(network_image))

    # Get the functional enrichment of the gene set
    functional_enrichment = _string_api_call(version = string_version, genes = query, method = 'functional_enrichment', score_threshold = edge_weight, species = species)
    functional_enrichment_df = pd.read_csv(io.StringIO(functional_enrichment.decode('utf-8')), sep = '\t')
    enrichment_plots = _plot_enrichment(functional_enrichment_df, plot_fontsize, plot_fontface)

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'STRING_Enrichment/')
        os.makedirs(new_savepath, exist_ok=True)
        # Write p-value to stats
        with open(new_savepath + "STRING_Stats.txt", "w") as f:
            f.write(f"STRING Metrics: confidence: {edge_confidence}, species: {species}, version: {version}\n")
            f.write(f"STRING PPI p-value: {p_value}\n")
        network_df.to_csv(new_savepath + "STRING_Network.csv", index = False)
        network_image.save(new_savepath + "STRING_Network.png", bbox_inches = 'tight', pad_inches = 0.5)
        enrichment_df.to_csv(new_savepath + "STRING_Enrichment.csv", index = False)
        functional_enrichment_df.to_csv(new_savepath + "STRING_Functional_Enrichment.csv", index = False)
        # Save plot images
        plot_save_path = os.path.join(new_savepath, 'Enrichment_Plots/')
        os.makedirs(plot_save_path, exist_ok=True)
        for category in enrichment_plots:
            enrichment_plots[category].save(plot_save_path +  f"STRING_{category}_Enrichment.png", bbox_inches = 'tight', pad_inches = 0.5)

    return network_df, p_value, network_image,functional_enrichment_df, enrichment_plots

#endregion



#region nDiffusion
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
    result_data = {
        'Gene': genes,
        'Diffusion score (Ranking)': score,
        'Is the gene in {}? (1=yes)'.format(group2_name): classify
    }
    df = pd.DataFrame(result_data)
    df_sorted = df.sort_values(by='Diffusion score (Ranking)', ascending=False)
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
        rand_degree[i] = []
        count = pred_degree_count[i] * iteration
        lst = []
        modifier = 1
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
            modifier += increment
            cnt += 1
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

def ndiffusion(set_1: list, set_2: list, set_1_name:str = 'Set_1', set_2_name:str = 'Set_2', string_version:str = 'v11.0', evidences:list = ['all'], edge_confidence:str = 'all', custom_background:Any = 'string', n_iter: int = 100, cores:int =1, savepath:str = False, verbose: int = 0) -> (Image, float, Image, float): # type: ignore
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
    group2_name = set_2_name

    # Load STRING network
    string_net, string_net_all_genes = _load_clean_string_network(string_version, evidences, edge_confidence)
    # Define and clean by background gene set
    if custom_background == 'string':
        background_genes = string_net_all_genes
    else:
        background_dict, background_name = _define_background_list(custom_background)
        background_genes = background_dict[background_name]
    string_net = string_net[(string_net['node1'].isin(background_genes)) & (string_net['node2'].isin(background_genes))]

    # Get network and diffusion parameters
    _, graph_node, adj_matrix, node_degree, g_degree = _get_graph(string_net)
    ps = _get_diffusion_param(adj_matrix)
    graph_node_index = _get_index_dict(graph_node)
    gp1_only_dict, gp2_only_dict, overlap_dict, other_dict =_parse_gene_input(
        set_1, set_2, graph_node, graph_node_index, node_degree, verbose = verbose
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
        new_savepath = os.path.join(savepath, 'nDiffusion/')
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



#region First neighbor connectivity
def _load_genelists(set_1:list, set_2:list, set_3:list, set_4:list, set_5: list) -> dict:
    """
    Loads gene lists and organizes them into a dictionary.

    Each gene list is associated with a key following the naming convention 'set1', 'set2', etc.
    Only non-None lists are included in the returned dictionary.

    Parameters
    ----------
    set_1 : list
        The first gene list.
    set_2 : list
        The second gene list.
    set_3 : list
        The third gene list.
    set_4 : list
        The fourth gene list.
    set_5 : list
        The fifth gene list.

    Returns
    -------
    dict
        A dictionary where each key corresponds to a 'set' label (e.g., 'set1') and each value
        is one of the input gene lists. Only non-None gene lists are included.

    Examples
    --------
    >>> gene_lists = _load_genelists(['gene1', 'gene2'], ['gene3', 'gene4'], None, None, None)
    >>> print(gene_lists)
    {'set1': ['gene1', 'gene2'], 'set2': ['gene3', 'gene4']}
    """
    clean_lists = _clean_genelists([set_1, set_2, set_3, set_4, set_5])
    sets = ['set'+str(i+1) for i in range(len(clean_lists))]
    gene_dict = {}
    for i in range(len(clean_lists)):
        gene_dict[sets[i]] = clean_lists[i]
    return gene_dict

def _get_gene_sources(set_dict: dict) -> dict:
    """
    Maps each gene to the sets it belongs to in a given dictionary.

    This function processes a dictionary where each key is a set (e.g., a specific condition or category) and
    the corresponding value is a list of genes (proteins) associated with that set. It inverts this relationship
    to create a new dictionary where each gene is mapped to a list of sets it belongs to.

    Args:
        set_dict (dict): A dictionary with set names as keys and lists of genes (proteins) as values.

    Returns:
        dict: A dictionary where each key is a gene (protein), and the corresponding value is a list of set names
              that the gene is associated with.

    Note:
        This function is useful in contexts where the association between genes and various sets (like experimental
        conditions, categories, or any grouping criteria) needs to be analyzed. If a gene is associated with multiple sets,
        all such associations are captured in the list corresponding to that gene.
    """
    gene_source = {}
    for k, v in set_dict.items():
        for protein in set_dict[k]:
            if protein in gene_source:
                gene_source[protein].append(k)
            else:
                gene_source[protein] = [k]
    return gene_source

def _get_unique_genes(source_dict:dict) -> dict:
    """
    Extracts unique genes from the source dictionary.

    This function iterates through the source dictionary and collects genes that
    are associated with only one source.

    Arguments:
    ---------
        source_dict (dict): A dictionary with keys as sources and values as lists of gene names.

    Returns:
    -------
        dict: A dictionary containing unique genes and their associated source.
    """
    unique_genes = {}
    for k, v in source_dict.items():
        if len(v) == 1:
            unique_genes[k] = v
    return unique_genes

def _check_connection(x:str, y:str, gene_lst:list) -> str:
    """
    Checks if both specified genes are present in a given gene list.

    This function examines whether both of the two specified genes (x and y) are present in a provided list of genes.
    It is primarily used to determine if there is a connection or association between two genes within the context of a given gene list.

    Args:
        x (str): The first gene to check for presence in the gene list.
        y (str): The second gene to check for presence in the gene list.
        gene_lst (list): A list of gene symbols (strings) to search within.

    Returns:
        str: Returns 'yes' if both genes are found in the gene list; otherwise, returns 'no'.

    Note:
        This function is useful in network analysis or genetic studies where determining the co-occurrence or connection of genes
        within a specific list or dataset is needed. It performs a simple presence check and does not infer any biological or functional relationship.
    """
    if x in gene_lst and y in gene_lst:
        out = 'yes'
    else:
        out = 'no'
    return out

def _get_node_pair(x:str, y:str) -> str:
    """
    Creates a sorted string representation of a pair of nodes (genes or proteins).

    This function takes two node identifiers (such as gene or protein symbols) and returns a string representation
    of the node pair. The two nodes are sorted alphabetically and combined into a single string, which facilitates
    consistent identification of node pairs regardless of the order in which they are provided.

    Args:
        x (str): The identifier for the first node.
        y (str): The identifier for the second node.

    Returns:
        str: A string representation of the node pair, where the node identifiers are sorted alphabetically
             and combined in a list format.

    Note:
        This function is useful in network analysis and graph operations where consistent identification of an
        edge or relationship between two nodes is required. It ensures that the pair 'x, y' is treated identically
        to the pair 'y, x', thereby avoiding duplicate representations of the same edge.
    """
    pair = [x, y]
    pair.sort()
    return str(pair)

def _get_unique_gene_network(gene_lst:list, network:pd.DataFrame) -> pd.DataFrame:
    """
    Generates a sub-network DataFrame containing only the connections between genes in a specified list.

    This function takes a list of genes and a network DataFrame, then filters and returns a new DataFrame representing
    the sub-network of connections exclusively among the genes in the list. It ensures that each connection (edge) is
    unique and appears only once in the sub-network.

    Args:
        gene_lst (list): A list of gene symbols (strings) to be included in the sub-network.
        network (pd.DataFrame): A DataFrame representing a network with columns 'node1' and 'node2',
                                indicating connections between nodes (genes or proteins).

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the connections between the genes specified in gene_lst.
                      Each row represents a unique connection, and duplicate connections are removed.

    Note:
        The function first filters the network to include connections where either 'node1' or 'node2' is in gene_lst,
        then further filters to include only connections where both nodes are in the list. It also ensures that each
        connection is represented only once, regardless of the order of nodes, by creating a sorted pair string for each
        connection and removing duplicates based on these pairs.
    """
    n_df = network.copy()
    n_df1 = n_df[n_df['node1'].isin(gene_lst)]
    n_df2 = n_df[n_df['node2'].isin(gene_lst)]
    n_df_final = pd.concat([n_df1, n_df2])
    n_df_final['bwSetConnection'] = n_df_final.apply(lambda x: _check_connection(x['node1'], x['node2'], gene_lst),
                                                     axis=1)
    n_df_final = n_df_final[n_df_final['bwSetConnection'] == 'yes']
    n_df_final.drop_duplicates(inplace=True)
    n_df_final['pair'] = n_df_final.apply(
        lambda x: _get_node_pair(x['node1'], x['node2']), axis=1)
    n_df_final.sort_values(by='node1', inplace=True)
    n_df_final.drop_duplicates(subset=['pair'], inplace=True, keep='first')
    return n_df_final

def _get_unique_gene_network_bw_method_connections(network:pd.DataFrame, unique_dict:dict) -> list:
    """
    Identifies between-method connections in a gene network.

    This function processes a network DataFrame and a dictionary of unique gene sets
    to find connections between genes that belong to different sets (methods).

    Arguments:
    ---------
        network (pd.DataFrame): A DataFrame containing the network edges with columns 'node1' and 'node2'.
        unique_dict (dict): A dictionary with keys as gene names and values as lists indicating the method/set they belong to.

    Returns:
    -------
        list: A list of unique between-method edge pairs.

    """
    net_genes = list(
        set(network['node1'].tolist() + network['node2'].tolist()))
    bw_method_edges = []

    for g in net_genes:
        # Get all connections where node1 = g
        df1 = network[network['node1'] == g]
        # Get all genes connected to g in node2
        df1_genes = df1['node2'].tolist()
        # Loop through all genes connected to g in node2
        for p in df1_genes:
            # If the genes is not from the same set, add to bw_method_edges
            if unique_dict[p][0] != unique_dict[g][0]:
                bw_method_edges.append([g, p])

        df2 = network[network['node2'] == g]
        df2_genes = df2['node1'].tolist()

        for p in df2_genes:
            if unique_dict[p][0] != unique_dict[g][0]:
                bw_method_edges.append([g, p])

    bw_method_edges = [sorted(x) for x in bw_method_edges]
    bw_method_edges = [str(x) for x in bw_method_edges]
    bw_method_edges = list(set(bw_method_edges))

    return bw_method_edges

def _get_unique_gene_counts(unique_gene_dict: dict) -> dict:
    """
    Counts the occurrences of unique genes across different sets in a dictionary.

    This function processes a dictionary where each key represents a gene or a set of genes, and the value is a list
    with the first element typically being a gene name or identifier. It then creates a new dictionary where each key
    is a unique gene, and the value is a list of keys from the input dictionary where the gene is present.

    Args:
        unique_gene_dict (dict): A dictionary with keys representing gene sets or similar constructs, and values are lists,
                                 with the first element of each list being a gene name or identifier.

    Returns:
        dict: A dictionary where each key is a unique gene, and the corresponding value is a list of keys from the
              input dictionary where the gene is mentioned.

    Note:
        This function is particularly useful in situations where one needs to map each gene to the sets or categories
        it belongs to, based on a given dictionary structure. It assumes that the first element of each list in the
        values of `unique_gene_dict` is the gene identifier.
    """
    unique_genes_per_set = {}
    for k, v in unique_gene_dict.items():
        if v[0] in unique_genes_per_set:
            unique_genes_per_set[v[0]].append(k)
        else:
            unique_genes_per_set[v[0]] = [k]
    return unique_genes_per_set

def _get_node_degree_dict(unique_genes:list, degree_df:pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to include only the degrees of specified nodes (genes).

    This function takes a list of unique genes and a DataFrame containing node degrees, then filters the DataFrame
    to include only the rows corresponding to the unique genes in the list. The resulting DataFrame contains degree
    information exclusively for the specified genes.

    Args:
        unique_genes (list): A list of gene symbols (strings) whose degree information is to be extracted.
        degree_df (pd.DataFrame): A DataFrame containing degree information for nodes (genes). The DataFrame should
                                  have an index or a column that matches the gene symbols in the `unique_genes` list.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only the rows corresponding to the genes in `unique_genes`.
                      The structure of the DataFrame remains the same as `degree_df`, but only with the relevant rows.

    Note:
        This function assumes that the index or a key column of `degree_df` directly corresponds to gene symbols.
        It is essential that `unique_genes` contains valid entries that are present in `degree_df`. If a gene in
        `unique_genes` is not found in `degree_df`, it will not be included in the returned DataFrame.
    """
    df = degree_df.copy()
    df = df.loc[unique_genes]
    return df

def _create_random_degree_matched_set(unique_gene_sets:dict, background_genes:list, string_net_all_genes:list, string_net_degree_df:pd.DataFrame, seed:int) -> dict:
    """
    Generates random sets of genes matched by degree distribution to given gene sets.

    This function creates random gene sets that are degree-matched to the provided gene sets. For each gene set in
    `unique_gene_sets`, it generates a random set of genes from `string_net_all_genes` such that the degree distribution
    matches that of the original set. The degree information is sourced from `string_net_degree_df`.

    Args:
        unique_gene_sets (dict): A dictionary where keys are set identifiers and values are lists of genes in each set.
        string_net_all_genes (list): A list of all genes present in the STRING network.
        string_net_degree_df (pd.DataFrame): A DataFrame containing degree information for genes in the STRING network.
        seed (int): A seed for the random number generator to ensure reproducibility.

    Returns:
        dict: A dictionary where each key corresponds to a key in `unique_gene_sets`, and each value is a list of randomly
              selected genes. The random genes are selected such that their degree distribution matches that of the genes in
              the corresponding original set.

    Note:
        The function uses a numpy random number generator with the provided seed for reproducibility. It ensures that the
        randomly generated sets have a similar degree distribution as the original sets, which is crucial for certain types
        of network analysis where degree distribution is a significant factor.
    """
    random_sets = {}
    seed = seed *2
    rng = np.random.default_rng(seed)

    # Get full degree dataframe filtered
    degree_df = string_net_degree_df.copy()
    degree_df = degree_df[degree_df.index.isin(background_genes)]

    # Loop through unique gene sets
    for k, v in unique_gene_sets.items():
        unique_mapped_genes = v
        # need to filter for genes that are mapped to STRING appropriately
        unique_mapped_genes = [
            x for x in unique_mapped_genes if x in string_net_all_genes
        ]
        unique_mapped_genes_degree_df = _get_node_degree_dict(
            unique_mapped_genes, string_net_degree_df
        )
        unique_mapped_genes_degree_df = pd.DataFrame(
            unique_mapped_genes_degree_df.groupby('degree_rounded')['degree'].count()
        )
        #in this dictionary: key is degree, value is count of genes with that degree
        unique_mapped_genes_degree_dict = dict(
            zip(
                unique_mapped_genes_degree_df.index.tolist(),
                unique_mapped_genes_degree_df['degree'].tolist()
            )
        )

        random_genes = []
        # Loop through degree, # of genes and get random genes with the same degree
        for k1, v1 in unique_mapped_genes_degree_dict.items():
            degree_df_matched = degree_df[degree_df['degree_rounded'] == k1]
            degree_matched_genes = degree_df_matched.index.tolist()
            # Select v1 number of random genes from degree_matched_genes
            random_degree_matched_genes = rng.choice(degree_matched_genes, v1, replace = False).tolist()
            random_genes.extend(random_degree_matched_genes)
        random_sets[k] = random_genes
    return random_sets

def _plot_results_norm(random_dist:list, true_connections:list, fontface: str = 'Avenir', fontsize: int = 14, random_color: str = 'gray', true_color: str = 'red') -> Image:
    """
        Plots the normalized distribution of random distances and marks the average and true connection count.
    The plot is returned as a PIL Image object.

    Parameters
    ----------
    random_dist : List[float]
        A list of numerical values representing the random distribution.
    true_connections : List[str]
        A list of strings representing the true connections.
    fontface : str, optional
        The font face to use for all text in the plot, by default 'Avenir'.
    random_color : str, optional
        The color for the random distribution elements in the plot, by default 'gray'.
    true_color : str, optional
        The color for the true connections line in the plot, by default 'red'.

    Returns
    -------
    Image
        A PIL Image object of the plot.

    Examples
    --------
    >>> image = _plot_results_norm([0.5, 0.6, 0.7], ['conn1', 'conn2'])
    >>> image.show()  # This will display the image.
    >>> image.save('my_plot.png')  # This will save the image to a file.
    """
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _, ax = plt.subplots(tight_layout=True)
    ax.hist(random_dist, color=random_color, density=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.axvline(np.mean(random_dist), linestyle='dashed', color=random_color)
    plt.axvline(len(true_connections), linestyle='dashed', color=true_color)
    plt.xlabel('Number Intermethod Edges', size=14)
    plt.ylabel('Random Analysis Count', size=14)

    mu, sigma = norm.fit(random_dist)
    z_fit = (len(true_connections) - mu)/sigma
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax)
    plt.text(xmin + 5, 0.002, 'Z-score: ' + str(round(z_fit, 2)) + '\nNumber True Connections: ' + str(len(true_connections)),
             bbox={'facecolor': 'blue', 'alpha': 1.0, 'pad': 2}, color='white')

    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, linewidth=2)
    # Use PIL to save the plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    #buffer.close()
    plt.close()

    return image

def _output_unique_gene_network(unique_gene_network:pd.DataFrame, unique_genes:list) -> None:
    """
    Processes a gene network DataFrame to identify and save unique gene interactions.

    This function maps a list of unique genes to their sources in the network,
    filters out self-interactions, and saves the resulting network to a CSV file.

    Parameters
    ----------
    unique_gene_network : pd.DataFrame
        A DataFrame containing the gene network data with at least 'node1' and 'node2' columns.
    unique_genes : list
        A list mapping indices to gene names to identify the source of nodes.
    savepath : str
        The file path where the resulting CSV file will be saved.

    Returns
    -------
    None
        This function does not return a value; it writes output to a file.

    Examples
    --------
    >>> network_df = pd.DataFrame({
    ...     'node1': ['geneA', 'geneB', 'geneC'],
    ...     'node2': ['geneD', 'geneE', 'geneA'],
    ...     'score': [0.9, 0.8, 0.85]})
    >>> unique_genes = {'geneA': 'source1', 'geneB': 'source2', 'geneC': 'source3', 'geneD': 'source4', 'geneE': 'source5'}
    >>> _output_unique_gene_network(network_df, unique_genes, './')
    # This will save 'UniqueGeneNetwork.csv' to the current directory.
    """
    unique_gene_network['node1_source'] = unique_gene_network['node1'].map(
        unique_genes)
    unique_gene_network['node2_source'] = unique_gene_network['node2'].map(
        unique_genes)
    unique_gene_network = unique_gene_network[unique_gene_network['node1_source']
                                          != unique_gene_network['node2_source']]
    unique_gene_network = unique_gene_network[[
        'node1', 'node2', 'score', 'node1_source', 'node2_source']]
    return unique_gene_network

def _gene_set_overlap(gene_sets:dict) -> Image:
    """
    Generates and exports a Venn diagram visualizing the overlap between gene sets.

    This function takes a dictionary of gene sets, converts lists to sets for unique representation,
    generates a Venn diagram to visualize the overlaps, and saves the figure to a specified path.

    Parameters
    ----------
    gene_sets : dict
        A dictionary where each key is the name of the gene set and each value is a list of genes.
    savepath : str
        The file path where the Venn diagram figure will be saved, including the file name and extension.

    Returns
    -------
    None
        This function does not return a value; it exports a figure to the given save path.

    Examples
    --------
    >>> gene_sets = {
    ...     'Set1': ['gene1', 'gene2', 'gene3'],
    ...     'Set2': ['gene2', 'gene3', 'gene4'],
    ...     'Set3': ['gene1', 'gene4', 'gene5']
    ... }
    >>> _gene_set_overlap(gene_sets, 'gene_set_overlap.png')
    # This will save the Venn diagram as 'gene_set_overlap.png'.
    """
    gene_sets_ = {}
    for k, v in gene_sets.items():
        gene_sets_[k] = set(v)
    venn(gene_sets_, fontsize=8, legend_loc="upper left")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image

def _process_random_set(iteration:int, unique_gene_sets:dict, background_genes:list, string_net_all_genes:pd.DataFrame, string_net_degree_df:pd.DataFrame, string_net:pd.DataFrame) -> Any:
    """
    Processes a single iteration for creating and analyzing a random degree-matched gene set.

    This function performs a series of operations to create a random degree-matched gene set based on a given iteration
    index, and then analyzes this set to determine the number of connections in the network. It encapsulates the workflow
    of generating random sets, mapping genes, and evaluating network connections.

    Args:
        iteration (int): The iteration index used as a seed for randomization.
        unique_gene_sets (dict): A dictionary of unique gene sets.
        string_net_all_genes (list): A list of all genes in the STRING network.
        string_net_degree_df (pd.DataFrame): A DataFrame containing degree information for genes in the STRING network.
        string_net (pd.DataFrame): A DataFrame representing the STRING network.

    Returns:
        int or None: The number of connections in the random unique gene network. Returns None if an error occurs.

    Note:
        This function is designed to be used in parallelized execution for multiple iterations of random set generation
        and analysis. It handles exceptions internally and prints error messages without halting execution.
    """
    try:
        random_sets = _create_random_degree_matched_set(unique_gene_sets, background_genes, string_net_all_genes, string_net_degree_df, iteration)
        random_gene_sources = _get_gene_sources(random_sets)
        random_unique_genes = _get_unique_genes(random_gene_sources)
        random_unique_gene_network = _get_unique_gene_network(list(random_unique_genes.keys()), string_net)
        random_connections = _get_unique_gene_network_bw_method_connections(random_unique_gene_network, random_unique_genes)
        return len(random_connections)
    except Exception as e:
        warnings.warn(f"An error occurred during processing: {e}")
        return None

def _parallel_random_enrichment(unique_gene_sets:dict, background_genes:list, string_net_all_genes:pd.DataFrame, string_net_degree_df:pd.DataFrame, string_net:pd.DataFrame, num_iterations:int, num_processes:int) -> list:
    """
    Executes random enrichment analysis in parallel over multiple iterations.

    This function orchestrates the parallel execution of random set generation and network analysis across multiple iterations.
    It uses multiprocessing to distribute the task across multiple processes, thereby speeding up the computation significantly.

    Args:
        unique_gene_sets (dict): A dictionary of unique gene sets.
        string_net_all_genes (list): A list of all genes in the STRING network.
        string_net_degree_df (pd.DataFrame): A DataFrame containing degree information for genes in the STRING network.
        string_net (pd.DataFrame): A DataFrame representing the STRING network.
        num_iterations (int): The number of iterations to perform the random enrichment analysis.
        num_processes (int): The number of processes to use in parallel execution.

    Returns:
        list: A list of integers, each representing the number of connections in the random unique gene network for each iteration.

    Note:
        It is crucial to use this function within a 'if __name__ == "__main__":' block for safe parallel execution, especially
        on Windows. The number of processes (`num_processes`) should be chosen based on the system's capabilities.
    """
    args = [(i, unique_gene_sets, background_genes, string_net_all_genes, string_net_degree_df, string_net) for i in range(num_iterations)]

    with Pool(num_processes) as pool:
        random_sets_connections = pool.starmap(_process_random_set, args)

    return random_sets_connections

def interconnectivity(set_1:list, set_2:list, set_3:list = None, set_4:list = None, set_5:list = None, string_version:str = 'v11.0', custom_background:Any = 'string', savepath:Any = False, evidences:list = ['all'], edge_confidence:str = 'highest', num_iterations: int = 250, cores: int = 1, plot_fontface:str = 'Avenir', plot_fontsize:int = 14, plot_background_color:str = 'gray', plot_query_color: str = 'red', verbose:int = 0) -> (Image, Image, list, pd.DataFrame, dict): # type: ignore
    """
    Analyzes gene set interconnectivity and visualizes the results, returning multiple outputs
    including images, lists, and data structures.

    Args:
        - set_1 (list): Strings of gene names to network embed.
        - set_2 (list): Strings of gene names to network embed.
        - set_3 (list, optional): Strings of gene names to network embed. Defaults - to None.
        - set_4 (list, optional): Strings of gene names to network embed. Defaults to None.
        - set_5 (list, optional): Strings of gene names to network embed. Defaults to None.
        - string_version (str, optional): Version of STRING network to use. Defaults to 'v11.0'. Options include 'v10.0', 'v11.0', 'v11.5', 'v12.0'.
        - custom_background (Any, optional): Custom background gene list. Defaults to 'string'. Options include 'string', 'ensembl', 'reactome', or a user defined list.
        - savepath (str, optional): Path to the save directory. Defaults to './'.
        - evidences (list, optional): Evidences list to calculate interaction score. Defaults to ['all']. Options include 'all', 'textmining', 'experiments', 'databases', 'coexpression', 'neighborhood', 'gene_fusion', 'cooccurence'.
        - edge_confidence (str, optional): Level of interactions to include. Defaults to 'highest'. Options include 'all', 'low' (>0.15), 'medium' (0.4), 'high' (0.7), 'highest' (0.9).
        - num_iterations (int, optional): Number of random samplings from STRING network. Defaults to 250
        - cores (int, optional): Number of cores for parallel jobs. Defaults to 1.
        - plot_background_color (str, optional): Color of the background distribution. Defaults to 'gray'.
        - plot_query_color (str, optional): Color of the query line. Defaults to 'red'.
        - plot_fontface (str, optional): Font face for plot text. Defaults to 'Avenir'.
        - plot_fontsize (int, optional): Font size for plot text. Defaults to 12.
        - verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Image: A PIL Image object of the interconnectivity enrichment plot.
        Image: A PIL Image object of the Venn diagram illustrating gene set overlaps.
        list: A list of the number of connections in randomly generated gene sets.
        pd.DataFrame: A DataFrame detailing the unique gene network.
        dict: A dictionary of gene sources.

    Raises:
        ValueError: If the gene sets provided do not fit the expected input structure.

    Examples:
        >>> enrich_plot, venn_plot, random_set_connections, true_connection_df, query_gene_sources = _interconnectivity_enrichment(['geneA', 'geneB'], ['geneC', 'geneD'])
        This will perform the interconnectivity analysis and return the results.
        >>> resized_image = enrich_plot.resize((3, 4))
        Display the new image

    Note:
        If `savepath` is provided and not 'None', the function will save the results to disk
        at the specified path.
    """
    #load and customize STRINGv11 network for analysis (evidence types, edge weights)
    string_net, string_net_all_genes = _load_clean_string_network(string_version, evidences, edge_confidence)

    #get degree connectivity after edgeweight filtering
    # network is already edge weight filtered
    g_string_net = nx.from_pandas_edgelist(
        string_net[['node1', 'node2']], 'node1', 'node2')
    g_string_net_degree = dict(g_string_net.degree)
    string_net_degree_df = pd.DataFrame(index=string_net_all_genes)
    string_net_degree_df['degree'] = string_net_degree_df.index.map(
        g_string_net_degree)
    # fillna with zeros for genes w/o appropriate edge weights
    string_net_degree_df.fillna(0, inplace=True)
    # Round each degree to nearest 10
    string_net_degree_df['degree_rounded'] = string_net_degree_df['degree'].apply(
        lambda x: round(x/10)*10)
    #true gene sets b/w set unique gene connectivity
    gene_sets = _load_genelists(set_1, set_2, set_3, set_4, set_5)
    query_gene_sources = _get_gene_sources(gene_sets)
    query_unique_genes = _get_unique_genes(query_gene_sources)
    query_unique_gene_network = _get_unique_gene_network(
        list(query_unique_genes.keys()), string_net
    )
    true_connections = _get_unique_gene_network_bw_method_connections(query_unique_gene_network, query_unique_genes)
    # random gene sets b/w set unique gene connectivity w/ degree matching
    if custom_background == 'string':
        background_genes = string_net_all_genes
    else:
        background_dict, background_name = _define_background_list(custom_background)
        background_genes = background_dict[background_name]
    # dictionary with unique genes (values) per each set (keys) in true gene lists
    unique_gene_sets = _get_unique_gene_counts(query_unique_genes)
    # Perform random enrichment-parallelized
    random_sets_connections = _parallel_random_enrichment(unique_gene_sets, background_genes, string_net_all_genes, string_net_degree_df, string_net, num_iterations, cores)
    # Generate z score
    mu, sigma = norm.fit(random_sets_connections)
    z_fit = (len(true_connections) - mu)/sigma
    # Plot results
    enrich_image = _plot_results_norm(random_sets_connections, true_connections, fontface = plot_fontface, fontsize = plot_fontsize, random_color = plot_background_color, true_color = plot_query_color)
    # Get output files
    venn_image = _gene_set_overlap(gene_sets)
    unique_gene_network = _output_unique_gene_network(query_unique_gene_network, query_unique_genes)

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'GoldStandard_Interconnectivity/')
        os.makedirs(new_savepath, exist_ok=True)
        enrich_image.save(new_savepath + "Interconnectivity_Norm_Plot.png", format = "PNG")
        venn_image.save(new_savepath + "Interconnectivity_Venn.png", format = "PNG")
        pd.DataFrame({'col': random_sets_connections}).to_csv(new_savepath + "RandomSetConnections.csv", header = False, index = False)
        unique_gene_network.to_csv(new_savepath + "UniqueGeneNetwork.csv", header = True, index = False)
        pd.DataFrame.from_dict(query_gene_sources, orient='index').to_csv(new_savepath + "GeneSet_Sources.csv", header = False)
        # Write stats
        with open(new_savepath + "EnrichmentStats.txt", "w") as f:
            f.write('Number of b/w set connects in real gene sets:'+ str(len(true_connections)) + '\n')
            f.write('Z-score based on curve fit:' + str(z_fit))
        f.close()
    if verbose > 0:
        print('Number of b/w set connects in real gene sets:', len(true_connections))
        print('Z-score based on curve fit:', z_fit)
    return venn_image, enrich_image, random_sets_connections, unique_gene_network, query_gene_sources
#endregion



#region GWAS Colocalization
def _map_mondo_efo_id(mondo_id: str) -> Any:
    """
    Maps a MONDO ID to its corresponding EFO (Experimental Factor Ontology) ID using the OLS (Ontology Lookup Service) API.

    This function queries the OLS API to find the exact mapping of a given MONDO(Mendelian Inheritance in Man Ontology) ID to an EFO ID.
    It returns the first matching EFO ID found.

    Args:
        mondo_id (str): The MONDO ID to be mapped to an EFO ID. The MONDO ID should be in the format 'MONDO:XXXXXXX'.

    Returns:
        str: The corresponding EFO ID in the format 'EFO_XXXXXXX'. Returns False if no matching EFO ID is found.

    Raises:
        HTTPError: If the HTTP request to the OLS API fails.
        JSONDecodeError: If the response from the OLS API is not in JSON format or is otherwise malformed.

    Note:
        The function assumes that the MONDO ID is correctly formatted and that the OLS API endpoint is accessible and functioning.
        It retrieves the first exact match found and does not account for multiple possible mappings.
    """
    # Define the OLS API endpoint and parameters
    ols_api_url = "https://www.ebi.ac.uk/ols/api/terms"
    params = {
        "obo_id": mondo_id,
        "ontology": "efo",
        "mapping_types": "exact",
    }

    # Query the OLS API
    response = requests.get(ols_api_url, params=params)
    response.raise_for_status()
    data = response.json()

    # Check if data is available
    if not data.get("_embedded"):
        return False
    else:
        efo_term = data["_embedded"]["terms"][0]["obo_id"]
        efo_term = efo_term.replace(":", "_")
        return efo_term

def _pull_gwas_catalog(mondo_id:str, p_upper:float, verbose: int = 0) -> pd.DataFrame:
    """
    Pulls GWAS (Genome-Wide Association Studies) catalog data for a given MONDO ID and filters the results based on a p-value threshold.

    This function first checks if the input MONDO ID is in the correct format, and if not, attempts to map it to the correct EFO (Experimental Factor Ontology) ID using the _map_mondo_efo_id function. It then queries the GWAS Catalog API to download association data for the given MONDO/EFO ID. The data is filtered to include only records with a p-value less than or equal to the specified upper limit (p_upper).

    Args:
        mondo_id (str): The MONDO ID for which GWAS data is to be pulled. It can be in MONDO format or a format that can be mapped to EFO.
        p_upper (float): The upper limit for p-values to include in the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing GWAS catalog data filtered by the specified p-value threshold.

    Raises:
        ValueError: If no corresponding EFO term is found for the input MONDO ID, or if the GWAS Catalog data cannot be downloaded.

    Notes:
        The function requires an internet connection to access the GWAS Catalog API. The API URL is hardcoded for 'associations/download' endpoint with 'includeBgTraits' and 'includeChildTraits' set to True.
    """
    if "MONDO_" not in mondo_id:
        mondo_id = _map_mondo_efo_id(mondo_id)
    if not mondo_id:
        raise ValueError(f"No EFO term found for {mondo_id}, please download and add the path to the table.")
    if verbose > 0:
        print(f"Querying GWAS Catalog API for {mondo_id}")
    api_url = f"https://www.ebi.ac.uk/gwas/api/v2/efotraits/{mondo_id}/associations/download?includeBgTraits=True&includeChildTraits=True"
    response = requests.get(api_url)

    if response.status_code == 200:
        content_file = io.BytesIO(response.content)
        df = pd.read_csv(content_file, sep = '\t')
        if df.empty:
            raise ValueError(f"Data pull for {mondo_id} returned no values, please download and add the path to the table using 'gwas_summary_path'.")
        df = df[df.pValue <= p_upper]
        df_temp = df['locations'].str.split(":", expand = True)
        df['CHR_ID'] = df_temp[0]
        df['CHR_POS'] = df_temp[1]
        df.drop(columns = ['locations'], inplace = True)
        download_columns = ['STRONGEST SNP-RISK ALLELE', 'P-VALUE', 'P-VALUE (TEXT)', 'RISK ALLELE FREQUENCY', 'OR VALUE', 'BETA', '95% CI (TEXT)', 'MAPPED_GENE', 'DISEASE/TRAIT','MAPPED_TRAIT', 'BG_TRAITS', 'STUDY ACCESSION', 'PUBMEDID', 'FIRST AUTHOR', 'CHR_ID', 'CHR_POS',]
        df.columns = download_columns
        df.set_index('STRONGEST SNP-RISK ALLELE', inplace = True)
        return df, mondo_id
    else:
        warnings.warn(f"Status code exited with error: {response.status_code}")
        raise ValueError(f"Cannot download summary statistics for {mondo_id}, please download and add the path to the table.")

def _split_row(row: pd.Series) -> pd.DataFrame:
    """
    Splits the values in the given row of a pandas DataFrame into multiple rows.

    Parameters:
        row (pd.Series): The row to be split.

    Returns:
        pd.DataFrame: A DataFrame with the split values.

    """
    snp_alleles = str(row.name).split(',')
    chr_pos = str(row['CHR_POS']).split(',')
    genes = str(row['MAPPED_GENE'])

    max_len = len(snp_alleles)

    return pd.DataFrame(index = snp_alleles, data = {
        'CHR_ID': [row['CHR_ID']] * max_len,
        'CHR_POS': chr_pos[:max_len] + [np.nan] * (max_len - len(chr_pos)),
        'MAPPED_GENE': genes
    })

def _check_query_against_index(query_genes:list, ref_gene_index:list) -> list:
    """
    Checks if query genes are present in a reference gene index and returns a list of matching genes.

    This function compares a list of query genes against a reference gene index and returns a list of genes that are present in both lists. It is useful for verifying the presence of query genes in a reference dataset.

    Args:
        query_genes (list): A list of gene symbols to query.
        ref_gene_index (list): A list of gene symbols to reference against.

    Returns:
        list: A list of gene symbols that are present in both the query genes and the reference gene index.

    Note:
        The function performs a case-insensitive comparison of gene symbols to ensure that genes are matched accurately regardless of the case.
    """
    matching_genes = [gene for gene in query_genes if gene.upper() in ref_gene_index]
    unmatched_genes = [gene for gene in query_genes if gene.upper() not in ref_gene_index]
    if unmatched_genes:
        warnings.warn(f"The following genes were not found in the reference gene index: {', '.join(unmatched_genes)}")
    return matching_genes

def _find_snps_within_range(query_genes:list, ref_gene_index:list, geneloci:pd.DataFrame, gwasloci:pd.DataFrame, query_distance:int) -> dict:
    """
    Identifies Single Nucleotide Polymorphisms (SNPs) within a specified distance from given query genes.

    This function iterates through a list of query genes and identifies SNPs from a GWAS (Genome-Wide Association Studies) loci
    DataFrame that are within a specified distance from each gene. It returns adictionary where each query gene is mapped to a list
    of SNPs (and their distances) that are within the defined range of that gene.
    Args:
        query_genes (list): A list of gene symbols to query.
        ref_gene_index (list): A list of SNP identifiers to reference against.
        geneloci (pd.DataFrame): A pandas DataFrame containing gene loci information with columns 'chrom', 'start', and 'end'.
        gwasloci (pd.DataFrame): A pandas DataFrame containing GWAS loci information with columns 'CHR_ID' and 'CHR_POS'.
        query_distance (int): The distance within which to search for SNPs near each query gene, in base pairs.
    Returns:
        dict: A dictionary mapping each query gene to a list of SNPs (and distances) that are within the specified distance of the gene.
    Note:
        The function checks for chromosome number compatibility (autosomes or sex chromosomes) and calculates distances in kilobase pairs (Kbp).
        It handles exceptions for genes or SNPs not found in the provided DataFrames and skips to the next gene/SNP if an issue is encountered.
    """
    # Get gene location information
    gene_loci = geneloci.loc[query_genes, ['chrom', 'start', 'end']]
    # Get SNP location information
    snp_loci = gwasloci.loc[ref_gene_index, ['CHR_ID', 'CHR_POS', 'MAPPED_GENE']]
    # Initialize the output dictionary
    gene_dict = {}
    # Iterate through genes and SNPs, and calculate the distance between them
    for gene, gene_data in gene_loci.iterrows():
        snps_within_range = []
        for snp, snp_data in snp_loci.iterrows():
            if gene_data['chrom'] == snp_data['CHR_ID']:
                try:
                    dist1 = abs(int(gene_data['start']) - int(snp_data['CHR_POS']))
                    dist2 = abs(int(gene_data['end']) - int(snp_data['CHR_POS']))
                except ValueError:
                    continue
                if dist1 < query_distance or dist2 < query_distance:
                    min_dist = min(dist1, dist2)
                    snps_within_range.append(
                        f"{snp} ({snp_data['MAPPED_GENE']}; {round(min_dist / 1000, 1)} Kbp)")
        gene_dict[gene] = snps_within_range

    return gene_dict

def _get_genes_with_snps(gene_snp_dict:dict) -> list:
    """
    Retrieves a list of genes that have associated SNPs within a specified distance.

    This function processes a dictionary where genes are mapped to SNPs and returns a list of those genes that have
    one or more SNPs associated with them. It is particularly useful for identifying genes with nearby genetic variants.

    Args:
        gene_snp_dict (dict): A dictionary where keys are gene symbols and values are lists of SNPs.
                              Each SNP is considered to be associated with the gene as a nearby genetic variant.

    Returns:
        list: A list of genes that have at least one associated SNP.

    Note:
        The function only includes genes that have associated SNPs. Genes without any associated SNPs are excluded from the returned list.
    """
    genes = [gene for gene, snps in gene_snp_dict.items() if snps]
    return genes

def _run_parallel_query(function:callable, chunks:list, gwas_snp_ids:pd.Series, geneloci:pd.DataFrame, gwasloci:pd.DataFrame, distance:int, num_processes:int) -> list:
    """
    Executes a given function in parallel across multiple processes.

    This function takes a user-defined function and applies it to each chunk of data in parallel using Python's
    multiprocessing capabilities. It is designed to perform computationally intensive tasks more efficiently by
    distributing the workload across multiple CPU cores.

    Args:
        function (callable): The function to be executed in parallel. This function should accept the parameters
                             as defined in the tuple structure used in the starmap call.
        chunks (list): A list of data chunks on which the function will be executed. Each chunk is processed
                       independently in a separate process.
        gwas_snp_ids (list): A list of SNP identifiers used in the function.
        geneloci (pd.DataFrame): A pandas DataFrame containing gene loci information.
        gwasloci (pd.DataFrame): A pandas DataFrame containing GWAS loci information.
        distance (int or float): The distance parameter to be passed to the function.
        num_processes (int): The number of parallel processes to use.

    Returns:
        list: A list of results obtained from executing the function on each chunk of data.

    Note:
        The user-defined function passed to this function must be capable of handling the data structure
        contained in each chunk and must accept the same parameters as those passed to this function.
    """
    with Pool(num_processes) as pool:
        results = pool.starmap(function, [(chunk, gwas_snp_ids, geneloci, gwasloci, distance) for chunk in chunks])
        pool.close()
        pool.join()
    return results

def _chunk_data(data:list, num_chunks:int) -> list:
    """
    Splits a list of data into smaller, approximately equal-sized chunks.

    This function divides a given list into a specified number of chunks. It is primarily used to partition data for
    parallel processing, ensuring that each chunk is as evenly distributed as possible.

    Args:
        data (list): The list of data to be chunked. This can be a list of any type of elements.
        num_chunks (int): The number of chunks to divide the data into. It should be a positive integer.

    Returns:
        list: A list of chunks, where each chunk is a sublist of the original data. The size of each chunk is approximately
              equal to the total size of the data divided by `num_chunks`.

    Note:
        If the total size of the data is not evenly divisible by `num_chunks`, the last chunk may be smaller than the others.
        This function does not shuffle or alter the order of the original data elements.
    """
    chunk_size = len(data) // num_chunks + 1
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def _combine_dicts(dicts_list:list) -> dict:
    """
    Merges a list of dictionaries into a single dictionary.

    This function takes a list of dictionaries and combines them into one dictionary. In the resulting dictionary,
    each key-value pair from each dictionary in the input list is preserved. If the same key appears in multiple
    dictionaries, the value from the last dictionary with that key will be used in the merged dictionary.

    Args:
        dicts_list (list): A list of dictionaries to be merged. Each element in the list should be a dictionary.

    Returns:
        dict: A single dictionary that contains the combined key-value pairs from all dictionaries in the input list.

    Note:
        If there are duplicate keys across the dictionaries, the value associated with the key in the last dictionary
        containing that key will overwrite any previous value. The order of dictionaries in the list impacts the final content.
    """
    return {k: v for d in dicts_list for k, v in d.items()}

def _calculate_fishers_exact(gene_snp_dict:dict, all_gene_dict:dict, final_genes:list) -> (int, int, int, int, float): # type: ignore
    """
    Calculates Fisher's Exact Test for a set of genes and their associated SNPs.

    This function computes the Fisher's Exact Test on a 2x2 contingency table formed from gene and SNP associations.
    It calculates the counts of true positives, false positives, false negatives, and true negatives based on the provided dictionaries.

    Args:
        gene_snp_dict (dict): A dictionary mapping genes to SNPs, used to determine the counts of true and false positives.
        all_gene_dict (dict): A dictionary mapping all genes to their associated SNPs, used for the total count of gene-SNP associations.
        final_genes (list): A list of genes considered as true positives.

    Returns:
        tuple: A tuple containing the counts of true positives (tp), false positives (fp), false negatives (fn), true negatives (tn),
               and the p-value (pval) from Fisher's Exact Test. The structure is (tp, fp, fn, tn, pval).

    Note:
        The function assumes that 'final_genes' are the true positives. It calculates the true positives (tp) as the length of 'final_genes',
        false positives (fp) as the number of genes in 'gene_snp_dict' not in 'final_genes', false negatives (fn) as genes in
        'all_gene_dict' with SNPs but not in 'final_genes', and true negatives (tn) as genes not associated with any SNPs in both dictionaries.
        The p-value is calculated using the 'greater' alternative hypothesis in Fisher's Exact Test, indicating a one-tailed test.
    """
    tp = len(final_genes)
    fp = len(gene_snp_dict.keys()) - tp
    fn = len([k for k, v in all_gene_dict.items() if len(v) > 0]) - tp
    tn = len([x for x in all_gene_dict.keys() if x not in final_genes and x not in gene_snp_dict.keys()])
    _, pval = fisher_exact(np.array([[tp, fp], [fn, tn]]), alternative='greater')
    return tp, fp, fn, tn, pval

def gwas_catalog_colocalization(query:list, mondo_id:str = False, gwas_summary_path:str = False, gwas_p_thresh: float = 5e-8, distance_mbp:float = 0.5, custom_background:Any = 'ensembl', cores:int = 1, savepath:Any = False, save_summary_statistics:bool = False, verbose: int = 0) -> (pd.DataFrame, float): # type: ignore
    """
    Performs colocalization analysis between a list of query genes and GWAS catalog SNPs.

    This function identifies SNPs from a GWAS catalog that are within a specified physical distance from the query genes. It can
    optionally perform this analysis for a background gene set to assess the statistical significance of the colocalization using
    Fisher's Exact Test.

    Args:
        - query (list): A list of query gene symbols.
        - mondo_id (str, optional): The MONDO ID to pull GWAS catalog data. If provided, GWAS catalog data will be pulled based on this ID.
        - gwas_summary_path (str, optional): The file path to a pre-downloaded GWAS summary statistics file. Used if 'mondo_id' is not provided.
        - gwas_p_thresh (float, optional): The p-value threshold for filtering GWAS catalog SNPs. Defaults to 5e-8.
        - distance_mbp (float, optional): The distance in megabase pairs (Mbp) within which to search for SNPs around each gene. Defaults to 0.5 Mbp.
        - custom_background (Any, optional): A custom background gene list for colocalization analysis. Defaults to 'ensembl'. Options include 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', 'ensembl', 'reactome', or a user-defined list. STRING versions will use edges with weight > 0.15
        - cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        - savepath (Any, optional): The path to save output files. If False, no files are saved.
        - save_summary_statistics (bool, optional): Flag to save GWAS summary statistics. Effective only if 'savepath' is provided.
        - verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame with genes and their associated SNPs within the specified distance.
        float: The p-value from Fisher's Exact Test assessing the significance of the colocalization.

    Note:
        The function uses multiprocessing for parallel computation if 'cores' is greater than 1. Ensure that the function is called
        within a 'if __name__ == "__main__":' block for safe parallel execution, especially on Windows platforms.
    """
    # Load GWAS Summary Stats
    if mondo_id:
        gwas_catalog, mondo_id = _pull_gwas_catalog(mondo_id, gwas_p_thresh, verbose = verbose)
    elif gwas_summary_path:
        gwas_catalog = pd.read_csv(gwas_summary_path, sep = '\t')
        gwas_catalog = gwas_catalog[gwas_catalog['P-VALUE'] <= gwas_p_thresh]
        gwas_catalog.set_index('STRONGEST SNP-RISK ALLELE', inplace = True)
        mondo_id = gwas_summary_path.split("/")[-1].split("-")[5]
    # Set parameters for colocalization
    distance_bp = distance_mbp * 1000000
    gene_locations = _load_grch38_background(just_genes=False)
    query = _check_query_against_index(query, gene_locations.index)
    # Clean the GWAS catalog result
    gwas_catalog = gwas_catalog[['CHR_ID', 'CHR_POS', 'MAPPED_GENE']].drop_duplicates().dropna()
    gwas_catalog['needs_split'] = gwas_catalog.index.str.contains(',', na = False)
    split_rows = gwas_catalog[gwas_catalog['needs_split']].apply(_split_row, axis = 1)
    if len(split_rows.tolist()) > 0:
        split_df = pd.concat(split_rows.tolist())
        split_df = split_df[split_df['CHR_POS'].str.len() >= 3]
        gwas_catalog = pd.concat([gwas_catalog[~gwas_catalog['needs_split']], split_df], ignore_index = False)
    gwas_catalog = gwas_catalog.drop('needs_split', axis = 1).dropna(subset = ['CHR_POS', 'CHR_ID']).drop_duplicates()
    gwas_catalog['CHR_POS'] = gwas_catalog['CHR_POS'].astype(int)
    gwas_catalog['CHR_ID'] = gwas_catalog['CHR_ID'].astype(str)
    # Run colocalization for query
    query_chunks = _chunk_data(query, cores)
    query_gene_dicts = _run_parallel_query(_find_snps_within_range, query_chunks, gwas_catalog.index, gene_locations, gwas_catalog, distance_bp, cores)
    query_snp_dict = _combine_dicts(query_gene_dicts)
    query_snp_df = pd.DataFrame(query_snp_dict.items(), columns=['Gene', 'SNPs'])
    query_snp_df = query_snp_df.set_index('Gene')
    final_genes = _get_genes_with_snps(query_snp_dict)
    # Run colocalization for background
    background_dict, background_name = _define_background_list(custom_background)
    background_genes = background_dict[background_name]
    clean = [x for x in background_genes if x in gene_locations.index]
    if verbose > 0:
        print(f"Background genes mapped: {len(clean)}/{len(background_genes)}")
    bg_new = gene_locations[gene_locations.index.isin(clean)]
    # Chunk the background genes for faster parsing
    bg_chunks = _chunk_data(bg_new.index.tolist(), cores)
    bg_gene_dicts = _run_parallel_query(_find_snps_within_range, bg_chunks, gwas_catalog.index, gene_locations, gwas_catalog[['CHR_ID', 'CHR_POS', 'MAPPED_GENE']].drop_duplicates(), distance_bp, cores)
    bg_gene_dict = _combine_dicts(bg_gene_dicts)
    # Test significance
    tp, fp, fn, tn, pval = _calculate_fishers_exact(query_snp_dict, bg_gene_dict, final_genes)
    # Output values
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'GWAS_Colocalization/')
        os.makedirs(new_savepath, exist_ok=True)
        if save_summary_statistics:
            gwas_catalog.to_csv(new_savepath + f"GWAS_Colocalization_{mondo_id}_p-{gwas_p_thresh}.csv", index = True)
        query_snp_df.to_csv(new_savepath +f"GWAS_Colocalization_{mondo_id}_p-{gwas_p_thresh}_TP.csv", index = True)
        with open(new_savepath + f"GWAS_Colocalization_{mondo_id}_p-{gwas_p_thresh}_Summary.txt", "w") as f:
            f.write(f"TP = {tp}\n")
            f.write(f"FP = {fp}\n")
            f.write(f"FN = {fn}\n")
            f.write(f"TN = {tn}\n")
            f.write(f"P-value = {pval}")
            f.close()

    return query_snp_df, pval
#endregion



#region PubMed Co-mentions
def _parse_field(field: str) -> Any:
    """
    Parses the input field and returns the appropriate query field.

    Args:
    ----
    field (str): The field specified by the user.

    Returns:
    -------
    str: The processed field value for the query.

    Raises:
    ------
    ValueError: If the provided field is not a valid selection.
    """
    valid_fields = ['all', 'title/abstract', 'title']
    if field not in valid_fields:
        raise ValueError(f"Invalid field selection '{field}'. Select a permitted field type: {', '.join(valid_fields)}")
    if field == 'all':
        return None
    return field

def _entrez_search(gene:str, disease:str, custom_terms:str, email:str, api_key:str, field:str) -> 'Entrez.Parser.DictionaryElement':
    """
    Conducts a search on the Entrez (PubMed) database for articles related to a specific gene and disease.

    Utilizes the Biopython Entrez API to query the PubMed database for articles
    containing the specified gene and disease terms within the specified field.

    Args:
        gene (str): The gene of interest.
        disease (str): The disease of interest.
        field (str, optional): The field within the PubMed article to search. Defaults to "all".
        email (str): The email address associated with the Entrez account.
        api_key (str): The API key associated with the Entrez account.

    Returns:
        Bio.Entrez.Parser.DictionaryElement: A dictionary-like object containing the Entrez search results.

    Raises:
        ValueError: If the provided field is not a valid selection.
    """
    Entrez.email = email
    Entrez.api_key = api_key
    field = _parse_field(field)
    if not any([gene, disease, custom_terms]):
        raise ValueError("At least one of gene, disease, or custom_term must be provided.")
    # Construct query
    if gene and disease:
        new_query = f'"{gene}" AND ("{disease}") AND (("gene") OR ("protein"))'
    if gene and custom_terms:
        new_query = f'"{gene}" AND {custom_terms}'
    retries = 3
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(
                db='pubmed',
                sort='relevance',
                retmax='100000',
                retmode='xml',
                field=field,
                term=new_query
            )
            results = Entrez.read(handle)
            handle.close()
            return results
        except HTTPError as e:
            if e.code == 502:
                # Specific action for 502 Bad Gateway error
                raise RuntimeError('Received a 502 Bad Gateway error from the PubMed server. Please try again later.')
            elif attempt < retries - 1:
                time.sleep(10)
            else:
                raise RuntimeError(f'Failed to retrieve data after {retries} attempts due to error: {e}')
        except (IndexError, URLError, IncompleteRead) as e:
            if attempt < retries - 1:
                time.sleep(10)
            else:
                warnings.warn(f'Error: {e}')
                return None

def _parse_entrez_result(result:dict) -> (str, int): # type: ignore
    """
    Parses the result from an Entrez (PubMed) API query to extract gene information and the count of papers.

    This function attempts to extract the gene name and the number of papers associated with it from the result
    of a PubMed query. It handles different scenarios where the expected fields might not be directly available
    or are located in different parts of the result structure due to warnings or errors in the query.

    Args:
        result (dict): The result dictionary returned by an Entrez (PubMed) API query.

    Returns:
        tuple:
            - str: The extracted gene name from the query result. If the gene name cannot be found directly,
                   it attempts to extract it from the 'WarningList' or 'QueryTranslation'.
            - int: The number of papers associated with the gene. This is NaN if the count cannot be
                   determined due to a warning or an error in the query.

    Raises:
        KeyError: If the necessary keys are not found in the result dictionary.
        IndexError: If the extraction from 'QueryTranslation' fails due to index errors.

    Note:
        The function assumes that the 'result' dictionary follows the structure of a response from the
        Entrez (PubMed) API. It is designed to handle cases where the query might have warnings or errors.
    """
    try:
        gene = result['TranslationStack'][0]['Term'].split('"')[1]
        n_paper_dis = int(result['Count'])
    except KeyError:
        try:
            gene = result['WarningList']['QuotedPhraseNotFound'][0].split('"')[1]
            n_paper_dis = 0
        except KeyError:
            gene = result['QueryTranslation'].split('"')[1]
            n_paper_dis = int(result['Count'])
        except IndexError:
            gene = result['QueryTranslation'].split('"')[1]
            n_paper_dis = int(result['Count'])
    return gene, n_paper_dis

def _fetch_query_pubmed(query: list, keyword: str, custom_terms:str, email: str, api_key: str, field:str, cores: int) -> pd.DataFrame:
    """
    Fetches publication data from PubMed for a list of genes related to a specific keyword.
    This function concurrently queries PubMed for each gene and compiles the results into
    a DataFrame which includes the count of papers and the PMIDs of papers related to each gene.

    Args:
        query (List[str]): A list of genes to be queried.
        keyword (str): A keyword to be used in conjunction with genes for querying PubMed.
        email (str): The email address associated with the NCBI account for Entrez queries.
        api_key (str): The API key for making requests to the NCBI Entrez system.
        cores (int): The number of worker threads to use for concurrent requests.

    Returns:
        pd.DataFrame: A DataFrame indexed by gene names with the count of related papers and PMIDs.
    """
    # Initialize data frames to store the query results and output data
    col_name = keyword if keyword else custom_terms
    out_df = pd.DataFrame(columns=['Count', 'PMID for Gene + ' + col_name], index=query)

    # Check field validity
    if field not in ['all', 'title/abstract', 'title']:
        raise ValueError(f"Invalid field selection '{field}'. Select a permitted field type: all, title/abstract, title")

    # Execute concurrent API calls to PubMed
    with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
        # Add tqdm for progress bar
        results = executor.map(_entrez_search, query, repeat(keyword), repeat(custom_terms), repeat(email), repeat(api_key), repeat(field))
        for result in tqdm(results, total=len(query), desc="Fetching PubMed data"):
            gene, n_paper_dis = _parse_entrez_result(result)
            # Populate the data frames with the results
            out_df.loc[gene, 'PMID for Gene + ' + col_name] = "; ".join(result.get('IdList', []))
            out_df.loc[gene, 'Count'] = n_paper_dis

    # Sort the output data frame by the count of related papers in descending order
    sorted_out_df = out_df.sort_values(by='Count', ascending=False)

    # Clean up the file
    sorted_out_df = sorted_out_df.rename(columns = {'Count': 'PubMed_CoMentions-' + col_name})

    return sorted_out_df

def _fetch_random_pubmed(query: list, disease_query: str, custom_terms: str, email: str, api_key: str, cores: int,
                         field: str, trials: int, background_genes: list, verbose: int = 0) -> list:
    """
    Performs PubMed queries on random sets of genes and records the number of papers
    associated with a disease for each gene in the set.
    This function is used to establish a baseline or control for comparison with
    a set of query genes of interest.

    Args:
        query (List[str]): A list of genes to be used as a query size reference.
        background_genes (List[str]): A list of background genes from which random samples are drawn.
        disease_query (str): A disease term to be used in conjunction with genes for querying PubMed.
        email (str): The email address associated with the NCBI account for Entrez queries.
        api_key (str): The API key for making requests to the NCBI Entrez system.
        cores (int): The number of worker threads to use for concurrent requests.
        trials (int, optional): The number of random gene sets to query. Defaults to 100.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each containing the count of papers for a random gene set.
    """
    randfs = []
    if len(background_genes) == 0:
        background_genes = _load_grch38_background()
    if custom_terms:
        out_query = custom_terms
    else:
        out_query = disease_query
    # Add a progress bar using tqdm
    for i in tqdm(range(trials), desc="Fetching random PubMed data", ncols=100):
        if i % 10 == 0 and verbose > 0:
            print(f" Random Trial : {i}")
        rng = np.random.default_rng(i * 3)
        randgenes = rng.choice(background_genes, size=len(query), replace=False).tolist()
        tempdf = pd.DataFrame(columns=['Count'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
            # Map the search function over the random genes
            results = executor.map(_entrez_search, randgenes, repeat(disease_query), repeat(custom_terms),
                                   repeat(email), repeat(api_key), repeat(field))
            # Process the results and update the temporary DataFrame
            for result in results:
                gene, n_paper_dis = _parse_entrez_result(result)
                n_paper_dis = result.get('Count', 0)
                tempdf.loc[gene, 'Count'] = int(n_paper_dis)
            tempdf = tempdf.rename(columns={'Count': 'PubMed_CoMentions-' + out_query})

        # Append the temporary DataFrame to the list
        randfs.append(tempdf)

    return randfs

def _calculate_z_score(observation: int, background: list) -> float:
    """
    Calculates the Z-score for the observed count against a background distribution.

    Args:
        observation (int): The observed count of genes.
        background (list): A list of counts from random gene sets for comparison.

    Returns:
        float: The calculated Z-score.
    """
    back_mean = np.mean(background)
    back_std = np.std(background)
    return (observation - back_mean) / back_std if back_std > 0 else np.nan

def _plot_results(disease_query: str, background: list, observation: int, query:list, paper_thrshld: list, z: float, random_color:str, query_color: str, fontsize: int, fontface:str) -> None:
    """
    Plots the results of the enrichment analysis.

    Args:
        disease_query (str): The disease query used in the enrichment analysis.
        background (list): A list of counts from random gene sets for comparison.
        observation (int): The observed count of genes.
        paper_thrshld (list): The threshold range for paper counts.
        z (float): The calculated Z-score.
        query (list): The list of query genes.
    """
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _, _ = plt.subplots(figsize=(6, 3.5), facecolor='white')
    y,_, _ = plt.hist(background, color=random_color)
    plt.axvline(x=observation, ymax=0.5, linestyle='dotted', color=query_color)
    # Orient the annotation
    values = background
    values.append(observation)
    if observation >= np.mean(values):
        ha_hold = 'right'
    else:
        ha_hold = 'left'
    plt.text(x=observation*0.99, y=(y.max()/1.8), s='{}/{} (Z = {})'.format(observation, len(query), round(z, 2)), color='red', ha=ha_hold)
    plt.xlabel('# of Genes with {}-{} Co-Mentions with "{}"'.format(paper_thrshld[0]+1, paper_thrshld[1], disease_query), fontsize=15)
    plt.ylabel('# of Random Occurrences', fontsize=15)
    plt.tight_layout()
    # Return image
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image

def pubmed_comentions(query:list, keyword: str = False, custom_terms: str = False, custom_background: Any = 'ensembl', field:str = 'all', email:str = 'kwilhelm95@gmail.com', api_key: str = '3a82b96dc21a79d573de046812f2e1187508',enrichment_trials: int = 100, workers: int = 15, run_enrichment:bool = True, enrichment_cutoffs:list = [[-1,0], [0,5], [5,15], [15,50], [50,100000]], plot_background_color:str = 'gray', plot_query_color: str = 'red', plot_fontface:str = 'Avenir', plot_fontsize:int = 14, savepath:Any = False, verbose:int = 0) -> (pd.DataFrame, dict, dict): # type: ignore
    """
    Searches PubMed for comention of genes within articles related to a given field and
    performs a randomization test to compute Z-scores for observed mention counts.

    Args:
        - query (list): A list of genes/other words to query in PubMed.
        - keyword (str): A keyword so search with (i.e. "Type 2 Diabetes")
        - custom_terms (str): A custom search query for PubMed. Function will add and "AND" to the end of the gene and then add your custom term after. For example, the custom_term = '(("adipose") OR ("diabetes")) AND (("gene") OR ("protein"))' would produce a search of '("{gene}") AND (("adipose") OR ("diabetes")) AND (("gene") OR ("protein"))'
        - custom_background (Any, optional): A custom background gene list for colocalization analysis. Defaults to 'ensembl'. Options include 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', 'ensembl', 'reactome', or a user-defined list. STRING versions will use edges with weight > 0.15
        - field (str, optional): The field within the PubMed article to search. Defaults to "all".
        - email (str, optional): The email address associated with the Entrez account.
        - api_key (str, optional): The API key associated with the Entrez account.
        - enrichment_trials (int, optional): The number of random trials to perform for calculating enrichment. Default is 100.
        - workers (int, optional): Number of workers for querying PubMed. Default is 15.
        - run_enrichment (bool, optional): False user only wants query co-mentions without enrichment. Default is True.
        - enrichment_cutoffs (list, optional): Cutoffs for enrichment analyses. Logic = >1st number, <=2nd number. Default is [[-1,0], [0,5], [5,15], [15,50], [50,100000]].
        - plot_background_color (str, optional): Color for random samplings in histogram. Default = gray.
        - plot_query_color (str, optional): Color for query findings in histogram. Default = red.
        - plot_fontface (str, optional): Fontface for plot. Default = Avenir.
        - plot_fontsize (int, optional): Font size for plot. Default = 14.
        - savepath (Any, optional): Path to save files. If undeclared, files are not saved and only returned.
        - verbose (int, optional): Verbosity level. Default is 0.
        
    Returns:
        pd.DataFrame : Dataframe of query words, number of co-mentions, and PMIDs.
        dict : Dictionary, where keys = enrichment_cutoff values and values = (number of query genes in subset, z_score)
        dict : Dictionary, where keys = enrichment_cutoff values and values = enrichment plot.
    """
    print('Running PubMed CoMentions')
    output_name = keyword if keyword else custom_terms
    # Pull the query co_mentions with keyword
    query_comention_df = _fetch_query_pubmed(query, keyword, custom_terms, email, api_key, field, workers)
    # Pull co_mentions for a random set of genes
    if run_enrichment:
        background_dict, background_name = _define_background_list(custom_background)
        background_genes = background_dict[background_name]
        rand_dfs = _fetch_random_pubmed(query, keyword, custom_terms, email, api_key, workers, field, enrichment_trials, background_genes, verbose = verbose)
        enrich_results, enrich_images = {}, {}
        rand_result_df = pd.DataFrame({'Iteration': range(0, len(rand_dfs))})
        for min_thresh, max_thresh in enrichment_cutoffs:
            observation = query_comention_df[(query_comention_df['PubMed_CoMentions-' + output_name] > min_thresh) & (query_comention_df['PubMed_CoMentions-' + output_name] <= max_thresh)].shape[0]
            background = [tmp[(tmp['PubMed_CoMentions-'+output_name] > min_thresh) & (tmp['PubMed_CoMentions-' + output_name] <= max_thresh)].shape[0] for tmp in rand_dfs]
            rand_result_df[f"{min_thresh + 1},{max_thresh}"] = background
            # Calculate Z scores
            z_score = _calculate_z_score(observation, background)
            # Plot results
            image = _plot_results(keyword, background, observation, query, [min_thresh, max_thresh], z_score, plot_background_color, plot_query_color, plot_fontsize, plot_fontface)
            # save results in dicts for output
            enrich_results[(min_thresh, max_thresh)] = (observation, z_score)
            enrich_images[(min_thresh, max_thresh)] = image
    else:
        enrich_results, enrich_images = {}, {}

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'PubMed_Comentions/{output_name}/')
        os.makedirs(new_savepath, exist_ok=True)
        query_comention_df.to_csv(new_savepath + f"PubMedQuery_keyword-{output_name}_field-{field}.csv", index = True)
        if run_enrichment:
            rand_result_df.to_csv(new_savepath + f"PubMedQueryRandomResults_keyword-{output_name}_field-{field}.csv", index = False)
            for key, value in enrich_images.items():
                value.save(new_savepath + f"PubMedQueryPlot_keyword-{output_name}_field-{field}_thresh-[>{key[0]},<={key[1]}].png")
            # Write results to file
            with open(new_savepath + f"PubMedQueryResults_keyword-{output_name}_field-{field}.txt", 'w') as f:
                for key,value in enrich_results.items():
                    f.write(f">{key[0] + 1} & <={key[1]} Comentions = {value[0]} (Z = {value[1]})\n")
                f.close()
    return query_comention_df, enrich_results, enrich_images
#endregion
