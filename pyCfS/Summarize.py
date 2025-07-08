"""
Collection of functions looking at previous genetic overlap recovery

Functions:

"""

import pkg_resources
import io
import os
import pandas as pd
import ast
import numpy as np
import warnings
from .utils import _fix_savepath


# Ignore SettingWithCopyWarning
pd.options.mode.chained_assignment = None


#region normalized_rank
def _validate_dict_keys(result_dict:dict, valid_keys_and_types: dict) -> None:
    """
    Validates that a dictionary contains specific keys with the correct types.

    Parameters:
    result_dict (dict): The dictionary to validate. It must contain the following keys with the specified types:
        - 'p_value': pd.DataFrame
        - 'p_value_col': str
        - 'gold_standard': list

    Raises:
    ValueError: If any of the required keys are missing or if the values are not of the expected types.
    """
    for key, value in result_dict.items():
        if key not in valid_keys_and_types.keys():
            raise ValueError(f"Missing key: {key}")
        if not isinstance(value, valid_keys_and_types[key]):
            raise ValueError(f"Invalid type for key: {key}. Type must be {value}")

def _load_results(result_path:str, result_experiments:list, valid_keys_and_types:dict, odds_ratios_directories:list, drug_source:str = 'all', valid_cluster_threshold:int = 2) -> dict:
    """
    Loads the results of specified experiments from the given result path.

    Parameters:
    result_path (str): The path to the directory containing the result files.
    result_experiments (list): A list of experiment names to load results for.
    valid_keys_and_types (dict): A dictionary of valid experiment names and their corresponding data types.

    Returns:
    dict: A dictionary containing the loaded results for the specified experiments.

    Notes:
    - The function validates the experiment names in `result_experiments` against `valid_keys_and_types`.
      Invalid experiment names are skipped with a warning.
    - The function loads different types of results based on the specified experiments:
      - 'p_value': Placeholder for future implementation.
      - 'gold_standard': Loads a list of genes from "Gold Standard Overlap/GoldStandards.txt".
      - 'colocalization': Loads colocalization data from files in "GWAS_Colocalization/".
      - 'interconnections': Loads interconnections data from "GoldStandard_Interconnectivity/UniqueGeneNetwork.csv".
      - 'clusters': Loads true clusters from "Functional_Clustering/TrueClusters.csv" and cluster enrichment data from "Functional_Clustering/Enrichment/".
    """
    # Validate the result experiments
    for value in result_experiments:
        if value not in valid_keys_and_types.keys():
            warnings.warn(f"WARNING: {value} is not a valid experiment. Skipping...")
            result_experiments.remove(value)
    # Load the results
    result_path = _fix_savepath(result_path)
    return_dict = {}

    if 'p_value' in result_experiments:
        p_value = pd.read_csv(result_path + "BP_gene_results.csv", index_col = 0)
        if isinstance(p_value.index[0], float):
            p_value = p_value.set_index('gene')
        return_dict['p_value'] = p_value
    
    if 'consensus' in result_experiments:
        consensus = pd.read_csv(result_path + "ConsensusGenes.csv",
                                index_col = 0)
        return_dict['consensus'] = consensus

    if 'goldstandard_overlap' in result_experiments:
        gs = pd.read_csv(result_path + "GoldStandard_Overlap/GoldStandards.txt", sep = '\t', header = None, names = ['Gene'])
        return_dict['goldstandard_overlap'] = gs['Gene'].tolist()

    if 'gwas_catalog_colocalization' in result_experiments:
        files = os.listdir(result_path + 'GWAS_Colocalization/')
        for file in files:
            if "_TP.csv" in file:
                colocalization = pd.read_csv(result_path + 'GWAS_Colocalization/' + file, index_col = 0)
                return_dict['gwas_catalog_colocalization'] = colocalization

    if 'interconnectivity' in result_experiments:
        # Load the gold standard list
        gs = pd.read_csv(result_path + "GoldStandard_Overlap/GoldStandards.txt", sep = '\t', header = None, names = ['Gene'])
        return_dict['goldstandard_overlap'] = gs['Gene'].tolist()
        # Load the interconnections dataframe
        interconnections = pd.read_csv(result_path + 'GoldStandard_Interconnectivity/UniqueGeneNetwork.csv')
        return_dict['interconnectivity'] = interconnections

    if 'functional_clustering' in result_experiments:
        # Load the gold standards
        gs = pd.read_csv(result_path + "GoldStandard_Overlap/GoldStandards.txt", sep = '\t', header = None, names = ['Gene'])
        return_dict['goldstandard_overlap'] = gs['Gene'].tolist()
        # Load the true clusters
        true_clusters = pd.read_csv(result_path + "Functional_Clustering/TrueClusters.csv", index_col = 0)
        return_dict['functional_clustering'] = true_clusters

        # Get valid clusters if they contain 2 or more genes
        valid_clusters = true_clusters.groupby('cluster').filter(lambda x: len(x) >= valid_cluster_threshold)
        valid_clusters = valid_clusters['cluster'].tolist()
        # Load cluster enrichment
        cluster_enrichment_path = result_path + 'Functional_Clustering/Enrichment/'
        enrichment_files = os.listdir(cluster_enrichment_path)
        return_dict['functional_clustering_enrichment'] = {}
        for file in enrichment_files:
            if '_combo_' in file:
                cluster_number = file.split('_')[0:2]
                cluster_number = '_'.join(cluster_number)
                if cluster_number in valid_clusters:
                    cluster_enrichment = pd.read_csv(cluster_enrichment_path + file)
                    return_dict['functional_clustering_enrichment'][cluster_number] = {}
                    return_dict['functional_clustering_enrichment'][cluster_number]['combo'] = cluster_enrichment

    if 'odds_ratios' in result_experiments:
        if odds_ratios_directories == []:
            or_files = os.listdir(result_path + "OddsRatios/")
            or_files = [x for x in or_files if x[0] != '.']
            if len(or_files) > 2:
                warnings.warn("WARNING: More than 2 odds ratio files found. Aggregating all files may result in over-counted results. Please define the specific files to load with or_directories.")
        else:
            or_files = odds_ratios_directories

        or_df_all = pd.DataFrame()
        for directory in or_files:
            # Load file
            or_df = pd.read_csv(result_path + "OddsRatios/" + directory + "/odds_ratio_results.csv")
            if or_df_all.empty:
                or_df_all = or_df
            else:
                or_df_all = pd.concat([or_df_all, or_df], axis = 0)
        return_dict['odds_ratios'] = or_df_all

    if 'risk_prediction' in result_experiments:
        risk_files = os.listdir(result_path + "RiskPrediction/")
        for file_1 in risk_files:
            if ".DS_Store" in file_1 or 'IntermediateFiles' in file_1 or "best_optimization" in file_1:
                continue
            # Check the individual models
            model_files = os.listdir(result_path + "RiskPrediction/" + file_1)
            for file_2 in model_files:
                if "TestingSamples" in file_2:
                    model_weights = pd.read_csv(result_path + "RiskPrediction/" + file_1 + "/" + file_2 + "/feature_weights.csv", index_col = 0)
        return_dict['risk_prediction'] = model_weights

    if 'ndiffusion' in result_experiments:
        try:
            receiver_rankings = pd.read_csv(result_path + "nDiffusion/Set_2Exclusive_vs_Set_1/dataframes/ranking.csv", index_col = 'Gene')
            #receiver_rankings = pd.read_csv(result_path + "nDiffusion/Set_2Exclusive_vs_Set_1Exclusive/dataframes/ranking.csv", index_col = 'Gene')
            # Filter for the query genes
            col_interest = [x for x in receiver_rankings.columns if "Is the gene in" in x][0]
            receiver_rankings = receiver_rankings[receiver_rankings[col_interest] == 1]
            return_dict['ndiffusion'] = receiver_rankings
        except:
            try:
                receiver_rankings = pd.read_csv(result_path + "nDiffusion/Set_2_vs_Set_1/dataframes/ranking.csv", index_col = 'Gene')
                return_dict['ndiffusion'] = receiver_rankings
            except:
                warnings.warn("WARNING: No nDiffusion file found. Skipping...")

    if 'mouse_phenotype_enrichment' in result_experiments:
        return_dict['mouse_phenotype_enrichment'] = {}
        # Load the significant query phenotypes
        query_mgi_pheno = pd.read_csv(result_path + "MGI_Mouse_Phenotypes/MGI_Lower-Level_PhenoEnrichment.csv", index_col = 0)
        # Look for the query phenotype nDiffusion to get rankings
        try:
            mgi_dir = os.listdir(result_path + "MGI_Mouse_Phenotypes")
            for file in mgi_dir:
                if "Validation" in file:
                    mgi_ndiff_dir = os.listdir(result_path + "MGI_Mouse_Phenotypes/" + file + "/nDiffusion_MGI")
                    for file2 in mgi_ndiff_dir:
                        # Use this for gold standard inclusive method
                        if "Exclusive_vs_Query Phenotypes" in file2 and file2.count("Exclusive") == 1:
                            receiver_rankings = pd.read_csv(result_path + "MGI_Mouse_Phenotypes/" + file + "/nDiffusion_MGI/" + file2 + "/dataframes/ranking.csv", index_col = 0)
                            col_interest = [x for x in receiver_rankings.columns if "Query Phenotypes" in x][0]
                            receiver_rankings = receiver_rankings[receiver_rankings[col_interest] == 1]
                            
        except:
            warnings.warn("WARNING: No nDiffusion file found. Skipping...")
        return_dict['mouse_phenotype_enrichment']['query_enrichment'] = query_mgi_pheno
        return_dict['mouse_phenotype_enrichment']['receiver_rankings'] = receiver_rankings

    if 'go_term_enrichment' in result_experiments:
        return_dict['go_term_enrichment'] = None
        query_go_terms = pd.read_csv(result_path + "STRING_Enrichment/STRING_Functional_Enrichment.csv", index_col = 1)
        # Look for the query phenotype nDiffusion to get rankings
        string_dir = os.listdir(result_path + 'STRING_Enrichment/')
        for file in string_dir:
            if "Validation" in file:
                # Load the true GO terms
                true_go_terms = pd.read_csv(result_path + 'STRING_Enrichment/Validation/go_bp_true_go_ids.csv', header = None, names = ['GO_Term'])
                true_go_terms = true_go_terms['GO_Term'].tolist()

                string_ndiff_dir = os.listdir(result_path + 'STRING_Enrichment/' + file + '/nDiffusion_GOterms_go_bp/')
                # Get the receiver rankings
                for file2 in string_ndiff_dir:
                    # Use this for gold standard inclusive method
                    if "go_bpExclusive_vs_Query GO Terms" in file2 and file2.count("Exclusive") == 1: ####
                        receiver_rankings = pd.read_csv(result_path + "STRING_Enrichment/" + file + "/nDiffusion_GOterms_go_bp/" + file2 + "/dataframes/ranking.csv", index_col = 1)
                        col_interest = [x for x in receiver_rankings.columns if "Query GO" in x][0]
                        receiver_rankings = receiver_rankings[receiver_rankings[col_interest] == 1]
        # Annotate true GO terms
        query_go_terms['is true GO term'] = np.where(query_go_terms.index.isin(true_go_terms), 1, 0)
        query_go_terms = query_go_terms[query_go_terms['category'] == 'Process']
        true_go_terms_to_mask = query_go_terms[query_go_terms['is true GO term'] == 1]['description'].tolist()

        # Transform the inputGenes column into a list of genes and annotate them
        query_go_terms_t = query_go_terms.copy()
        query_go_terms_t["GeneList"] = query_go_terms_t["inputGenes"].str.split(",")
        query_go_terms_t = query_go_terms_t.explode("GeneList")
        query_go_terms_t["GeneList"] = query_go_terms_t["GeneList"].str.strip()
        query_go_terms_t = query_go_terms_t.set_index("GeneList")[["is true GO term", "description"]]
        query_go_terms_t.columns = ["Flag_1_0", "PathwayDescription"]
        query_go_terms_t = (
            query_go_terms_t
            .groupby(level=0)  # group by the 'GeneList' index
            .agg({
            'Flag_1_0': 'max',  # or 'first', 'any', etc. depending on how you want to combine 1/0
            'PathwayDescription': lambda x: x.unique().tolist()
            })
        )
        # Pull the true GO terms into their own column
        query_go_terms_t['true_go_terms'] = query_go_terms_t['PathwayDescription'].apply(lambda x: [y for y in x if y in true_go_terms_to_mask])
        # Remove the masked GO terms from the PathwayDescription column
        query_go_terms_t['PathwayDescription'] = query_go_terms_t['PathwayDescription'].apply(lambda x: [y for y in x if y not in true_go_terms_to_mask])

        # Transform the receiver rankings to gene-based
        receiver_rankings = receiver_rankings.merge(query_go_terms[['inputGenes']], left_index = True, right_index = True)
        receiver_rankings_t = receiver_rankings.copy()
        receiver_rankings_t['inputGenes'] = receiver_rankings_t['inputGenes'].str.split(",")
        receiver_rankings_t = receiver_rankings_t.explode("inputGenes")
        receiver_rankings_t = receiver_rankings_t.set_index("inputGenes")[['Percentile_Rank', 'Diffusion score (Ranking)']]
        receiver_rankings_t = receiver_rankings_t.groupby(level = 0).agg({'Percentile_Rank': 'max', 'Diffusion score (Ranking)': 'max'})

        # Merge them together
        merged = receiver_rankings_t.merge(query_go_terms_t, left_index = True, right_index = True, how = 'left')
        # Renome the columns
        merged = merged.rename(columns = {'Flag_1_0': 'GO Terms - True GO Term Overlap', 'PathwayDescription': 'GO Terms - Non-Overlapping GO Terms', 'Percentile_Rank': 'GO Terms - Diffusion Score (Percentile Ranking)', 'Diffusion score (Ranking)': 'GO Terms - Diffusion Score', 'true_go_terms': 'GO Terms - Overlapping True GO Terms'})
        merged = merged[['GO Terms - True GO Term Overlap', 'GO Terms - Overlapping True GO Terms', 'GO Terms - Diffusion Score (Percentile Ranking)', 'GO Terms - Diffusion Score', 'GO Terms - Non-Overlapping GO Terms']]
        # Sort by Diffusion score (Ranking)
        return_dict['go_term_enrichment'] = merged

    if 'drug_gene_interactions' in result_experiments:
        if 'dgidb' in drug_source:
            dgidb = pd.read_csv(result_path + "DrugGeneInteractions/DGIdb_DrugGeneInteractions.csv", index_col = 0)
            dgidb = dgidb[['drug']]
            return_dict['drugs'] = dgidb
        if 'opentargets' in drug_source:
            opentargets = pd.read_csv(result_path + "DrugGeneInteractions/OpenTargets_DrugGeneInteractions.csv", index_col = 0)
            opentargets = opentargets[['drug']]
            return_dict['drugs'] = opentargets
        if drug_source == 'all':
            dgidb = pd.read_csv(result_path + "DrugGeneInteractions/DGIdb_DrugGeneInteractions.csv", index_col = 0)
            dgidb = dgidb[['drug']]

            opentargets = pd.read_csv(result_path + "DrugGeneInteractions/OpenTargets_DrugGeneInteractions.csv")
            opentargets = opentargets[['drug']]

            drug_df = pd.concat([dgidb, opentargets], axis = 0)
            drug_df = drug_df.drop_duplicates().dropna()
            return_dict['drug_gene_interactions'] = drug_df

    if 'depmap_enrichment' in result_experiments:
        query_values = pd.read_csv(result_path + "DepMap_Enrichment/Query_Avg_Score.txt", sep = '\t', index_col = 0, dtype = {'Genes': str, 'DepMap Score': float})
        return_dict['depmap_enrichment'] = query_values

    if 'pubmed_comentions' in result_experiments:
        keywords = os.listdir(result_path + "PubMed_Comentions/")
        comention_df = pd.DataFrame()
        for keyword in keywords:
            if ".DS_Store" in keyword:
                continue
            files = os.listdir(result_path + "PubMed_Comentions/" + keyword)
            for file in files:
                if ".csv" in file and "Random" not in file:
                    comentions = pd.read_csv(result_path + "PubMed_Comentions/" + keyword + "/" + file, index_col = 0)
                    if comention_df.empty:
                        comention_df = comentions
                    else:
                        comention_df = comention_df.merge(comentions, left_index = True, right_index = True, how = 'left')
                else:
                    continue
        return_dict['pubmed_comentions'] = comention_df

    return return_dict

def _annotate_consensus(df:pd.DataFrame, consensus:pd.DataFrame, show_indiv_scores:bool = True, score_method:str = 'rank') -> pd.DataFrame:
    """
    Annotates a DataFrame with consensus information and calculates prioritization scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    consensus (pd.DataFrame): A DataFrame containing consensus information.
    show_indiv_scores (bool): Whether to show individual scores for consensus. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with added prioritization scores.
    """
    out_df = df.copy()
    consensus = consensus.rename(columns = {'occurrences': 'Consensus Count'})
    out_df = out_df.merge(consensus, left_index = True, right_index = True, how = 'left')
    # Add score
    out_df['InverseRanking'] = out_df['Consensus Count'].rank(ascending = True, method = 'min').astype(int)
    out_df['Score-Consensus'] = out_df['InverseRanking'] / len(out_df)
    out_df['Score'] += out_df['Score-Consensus']
    if show_indiv_scores:
        out_df = out_df.drop(columns = ['InverseRanking', 'lists'])
    else:
        out_df = out_df.drop(columns = ['InverseRanking', 'Score-Consensus', 'lists'])
    return out_df

def _annotate_p_value(df:pd.DataFrame, p_value_df: pd.DataFrame, p_value_column:str, show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with p-values and calculates prioritization scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    p_value_df (pd.DataFrame): A DataFrame containing p-values.
    p_value_column (str): The column name in p_value_df that contains the p-values.
    show_indiv_scores (bool): Whether to show individual scores for discovery significance. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with added prioritization scores.
    """
    out_df = df.copy()
    # Merge p-values with out_df
    p_value_df = p_value_df[[p_value_column]]
    out_df = out_df.merge(p_value_df, left_index = True, right_index = True, how = 'left')
    out_df = out_df.rename(columns = {p_value_column: 'Discovery Significance'})
    out_df['abs(Discovery Significance)'] = np.abs(out_df['Discovery Significance'])
    # Add score and clean
    out_df['InverseRanking'] = out_df['abs(Discovery Significance)'].rank(ascending = False, method = 'min').astype(int)
    if show_indiv_scores:
        out_df['Score-Discovery Significance'] = (out_df['InverseRanking'] / len(out_df))
    out_df['Score'] += out_df['InverseRanking'] / len(out_df)
    out_df = out_df.drop(columns = ['InverseRanking', 'abs(Discovery Significance)'])
    return out_df

def _annotate_gold_standard(df: pd.DataFrame, gold_standard: list, show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with gold standard information and calculates prioritization scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    gold_standard (list): A list of indices that are considered as the gold standard.
    show_indiv_scores (bool): Whether to show individual scores for gold standard. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with added prioritization scores.

    Raises:
    ValueError: If gold_standard is not a list.
    """
    if not isinstance(gold_standard, list):
        raise ValueError("gold_standard value must be a list")
    hold_out = pd.DataFrame(index = df.index)
    hold_out['Gold Standard'] = np.nan
    hold_out.loc[hold_out.index.isin(gold_standard), 'Gold Standard'] = 1
    #hold_out['Gold Standard'] = hold_out.index.apply(lambda x: 1 if str(x) in gold_standard else np.nan)
    out_df = pd.merge(df, hold_out, left_index = True, right_index = True, how = 'left')

    # Score
    for idx, vals in out_df.iterrows():
        score = vals['Score']
        for j in hold_out.columns:
            hold_val = vals[j]
            if hold_val == 1:
                score += 1
                if show_indiv_scores:
                    out_df.loc[idx, 'Score-Gold Standard'] = 1
            else:
                out_df.loc[idx, j] = np.nan
                if show_indiv_scores:
                    out_df.loc[idx, 'Score-Gold Standard'] = 0
        out_df.loc[idx, 'Score'] = score
    return out_df

def _annotate_colocalization(df: pd.DataFrame, colocalization: pd.DataFrame, score_method:str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with colocalization information.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    colocalization (pd.DataFrame): A DataFrame containing colocalization information with genes as the index.
    show_indiv_scores (bool): If True, individual colocalization scores will be added to the DataFrame. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with colocalization information.

    Notes:
    - The `colocalization` DataFrame must have a column named 'SNPs' which will be renamed to 'GWAS-CoLoc'.
    - The function merges the `colocalization` DataFrame with the input `df` on the index.
    - It counts the number of colocalizations and creates an inverse ranking.
    - If `show_indiv_scores` is True, it adds a column 'Score-GWAS-Colocalization' with individual scores.
    - The overall score is updated by adding the inverse ranking score.
    - Temporary columns 'GWAS-Count' and 'InverseRanking' are dropped before returning the annotated DataFrame.
    """
    out_df = df.copy()
    colocalization = colocalization.rename(columns = {'SNPs': 'GWAS-CoLoc'})
    out_df = out_df.merge(colocalization, left_index = True, right_index = True, how = 'left')
    out_df = out_df.fillna('[]')
    out_df['GWAS-CoLoc'] = out_df['GWAS-CoLoc'].apply(lambda x: '[]' if str(x) == '[]' else x)
    # Count the number of colocalizations and create ranking
    out_df['GWAS-Count'] = out_df['GWAS-CoLoc'].apply(lambda x: len(str(x).split(',')) if str(x) != '[]' else 0)
    # Score
    if score_method == 'binary':
        out_df['Score-GWAS-Colocalization'] = out_df['GWAS-Count'].apply(lambda x: 1 if x > 0 else 0)
        out_df['Score'] += out_df['Score-GWAS-Colocalization']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['GWAS-Count'])
        else:
            out_df = out_df.drop(columns = ['GWAS-Count', 'Score-GWAS-Colocalization'])
    elif score_method == 'rank':
        out_df['InverseRanking'] = out_df['GWAS-Count'].rank(method = 'min', ascending = True).astype(int)
        out_df.loc[out_df['GWAS-Count'] == 0, 'InverseRanking'] = 0
        out_df['Score-GWAS-Colocalization'] = out_df['InverseRanking'] / len(out_df)
        out_df['Score'] += out_df['InverseRanking'] / len(out_df)
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['GWAS-Count', 'InverseRanking'])
        else:
            out_df = out_df.drop(columns = ['GWAS-Count', 'InverseRanking', 'Score-GWAS-Colocalization'])
    else:
        raise ValueError("Invalid score method. Please choose 'binary' or 'rank'")
    return out_df

def _annotate_interconnections(df: pd.DataFrame, interconnections: pd.DataFrame, gold_standards:list, score_method:str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with interconnection information and calculates prioritization scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    interconnections (pd.DataFrame): A DataFrame containing interconnection data with 'node1' and 'node2' columns.
    gold_standards (list): A list of indices that are considered as the gold standard.
    show_indiv_scores (bool): Whether to show individual scores for interconnections. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with added prioritization scores.
    """
    out_df = df.copy()
    int_df = pd.DataFrame(index = out_df.index)

    # Get the interconnections
    for idx in int_df.index:
        # Get connections where query gene is located in node 1
        df_idx_node1 = interconnections[
            (interconnections['node1'] == idx) &
            (interconnections['node1'].isin(out_df.index)) &
            (~interconnections['node1'].isin(gold_standards))
        ]
        idx_node1_conn = df_idx_node1['node2'].tolist()
        # Get connections where query gene is located in node 2
        df_idx_node2 = interconnections[
            (interconnections['node2'] == idx) &
            (interconnections['node2'].isin(out_df.index)) &
            (~interconnections['node2'].isin(gold_standards))
        ]
        idx_node2_conn = df_idx_node2['node1'].tolist()
        # Combine
        idx_node1_conn.extend(idx_node2_conn)
        int_df.loc[idx, 'Interconnections'] = ";".join(idx_node1_conn)
    # Add connections to whole dataframe
    out_df = out_df.merge(int_df, left_index = True, right_index = True, how = 'left')

    # Count number of connections
    out_df['Connect-Count'] = out_df['Interconnections'].apply(lambda x: len(str(x).split(';')) if x and pd.notna(x) else 0)

    # Score
    if score_method == 'binary':
        out_df['Score-Interconnections'] = out_df['Connect-Count'].apply(lambda x: 1 if x > 0 else 0)
        out_df['Score'] += out_df['Score-Interconnections']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['Connect-Count'])
        else:
            out_df = out_df.drop(columns = ['Connect-Count', 'Score-Interconnections'])
    elif score_method == 'rank':
        out_df['InverseRanking'] = out_df['Connect-Count'].rank(method = 'min', ascending = True).astype(int)
        out_df.loc[out_df['Connect-Count'] == 0, 'InverseRanking'] = 0
        out_df['Score-Interconnections'] = out_df['InverseRanking'] / len(out_df)
        out_df['Score'] += out_df['Score-Interconnections']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['Connect-Count', 'InverseRanking'])
        else:
            out_df = out_df.drop(columns = ['Connect-Count', 'InverseRanking', 'Score-Interconnections'])
    else:
        raise ValueError("Invalid score method. Please choose 'binary' or 'rank'")
    return out_df

def _annotate_clusters(df: pd.DataFrame, clusters: pd.DataFrame, cluster_enrichment: dict, gold_standards: list, score_method: str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with cluster information and calculates prioritization scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    clusters (pd.DataFrame): A DataFrame containing cluster data with a 'cluster' column.
    cluster_enrichment (dict): A dictionary containing enrichment data for each cluster.
    gold_standards (list): A list of indices that are considered as the gold standard.
    score_method (str): The method to use for scoring ('binary' or 'rank'). Default is 'rank'.
    show_indiv_scores (bool): Whether to show individual scores for clusters. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with added prioritization scores.
    """
    out_df = df.copy()
    clusters = clusters.copy()
    clusters = clusters.rename(columns = {'cluster': 'Cluster #'})
    for idx, vals in clusters.iterrows():
        # Check if Gold Standard in cluster
        cluster_number = vals['Cluster #']
        hold_cluster = clusters.index[clusters['Cluster #'] == cluster_number]
        gs_in_cluster = [x for x in hold_cluster if x in gold_standards and x != idx]
        clusters.loc[idx, 'n_GSinCluster'] = len(gs_in_cluster)
        clusters.loc[idx, 'GSinCluster'] = ','.join(gs_in_cluster)
        # Load and clean cluster enrichment
        enrichment_hold = cluster_enrichment[cluster_number]['combo']
        enrichment_hold = enrichment_hold[enrichment_hold['qval'] <= 0.01]
        sig_pathways = ";".join(enrichment_hold['pathway'].tolist())
        clusters.loc[idx, 'n_SigPathways'] = len(enrichment_hold)
        clusters.loc[idx, 'ClusterPathways'] = sig_pathways
        clusters = clusters.sort_values(by = 'Cluster #', ascending = True)
    # Drop duplicates if they exist, prioritize the lower cluster number
    for idx, vals in clusters.iterrows():
        idx_sub = clusters[clusters.index == idx]
        if len(idx_sub) > 1:
            # Remove the rest of the rows from the dataframe
            clusters = clusters.drop(idx_sub.index[1:])
    # Merge with input df
    out_df = out_df.merge(clusters, left_index = True, right_index = True, how = 'left')
    # Fix values after merging
    out_df['n_GSinCluster'] = out_df['n_GSinCluster'].fillna(0)
    out_df['n_SigPathways'] = out_df['n_SigPathways'].fillna(0)
    # Score
    if score_method == 'binary':
        out_df['Score-GoldStandardInCluster'] = out_df['n_GSinCluster'].apply(lambda x: 1 if x > 0 else 0)
        out_df['Score-ClusterPathways'] = out_df['n_SigPathways'].apply(lambda x: 1 if x else 0)
        out_df['Score'] += out_df['Score-GoldStandardInCluster'] + out_df['Score-ClusterPathways']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['source', 'n_GSinCluster', 'n_SigPathways'])
        else:
            out_df = out_df.drop(columns = ['source', 'n_GSinCluster', 'n_SigPathways', 'Score-GoldStandardInCluster', 'Score-ClusterPathways'])
    elif score_method == 'rank':
        out_df['GS-InverseRanking'] = out_df['n_GSinCluster'].rank(ascending = True, method = 'min').astype(int)
        out_df.loc[out_df['n_GSinCluster'] == 0, 'GS-InverseRanking'] = 0
        out_df['Score-GoldStandardInCluster'] = out_df['GS-InverseRanking'] / len(out_df)

        out_df['Pathway-InverseRanking'] = out_df['n_SigPathways'].rank(ascending = True, method = 'min').astype(int)
        out_df.loc[out_df['n_SigPathways'] == 0, 'Pathway-InverseRanking'] = 0
        out_df['Score-ClusterPathways'] = out_df['Pathway-InverseRanking'] / len(out_df)

        out_df['Score'] += out_df['Score-GoldStandardInCluster'] + out_df['Score-ClusterPathways']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['source', 'GS-InverseRanking', 'Pathway-InverseRanking', 'n_GSinCluster', 'n_SigPathways'])
        else:
            out_df = out_df.drop(columns = ['source', 'GS-InverseRanking', 'Pathway-InverseRanking', 'n_GSinCluster', 'n_SigPathways', 'Score-GoldStandardInCluster', 'Score-ClusterPathways'])
    else:
        raise ValueError("Invalid score method. Please choose 'binary' or 'rank'")
    return out_df

def _annotate_odds(df:pd.DataFrame, odds_df:pd.DataFrame, q_value_threshold:float = 0.1, score_method:str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with odds ratio information and computes scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    odds_df (pd.DataFrame): A DataFrame containing odds ratio information.
    show_indiv_scores (bool): Whether to include individual scores in the output DataFrame. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with odds ratio information and computed scores.
    """
    out_df = df.copy()
    odds_df = odds_df[['genomic_object', 'gene', 'OR', 'LowerCI', 'UpperCI', 'qvalue']]
    odds_df = odds_df[odds_df['qvalue'] <= q_value_threshold]
    odds_df = odds_df.reset_index(drop = True)
    hold_out = pd.DataFrame(index = out_df.index)
    hold_out['Odds-Ratios'] = np.nan
    hold_out['n_Odds-Ratios'] = np.nan

    # Loop through priority index and test if sig. odds
    for idx in hold_out.index:
        # Check for sig odds
        odds_df_hold = odds_df[odds_df['gene'] == idx]
        # Count the number of significant odds
        n_odds = len(odds_df_hold)
        hold_out.loc[idx, 'n_Odds-Ratios'] = n_odds
        # Format recording for Odds-Ratios
        if n_odds > 0:
            holder_string = []
            for idx2 in odds_df_hold.index:
                holder_string.append(
                    f"{odds_df_hold.loc[idx2, 'genomic_object']}-{round(odds_df_hold.loc[idx2, 'OR'], 3)} ({odds_df_hold.loc[idx2, 'LowerCI']}-{odds_df_hold.loc[idx2, 'UpperCI']})")
            hold_out.loc[idx, 'Odds-Ratios'] = ";".join(holder_string)
    # Add to out_df
    out_df = out_df.merge(hold_out, left_index = True, right_index = True, how = 'left')
    # Score
    if score_method == 'binary':
        out_df['Score-Odds-Ratios'] = out_df['n_Odds-Ratios'].apply(lambda x: 1 if x > 0 else 0)
        out_df['Score'] += out_df['Score-Odds-Ratios']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['n_Odds-Ratios'])
        else:
            out_df = out_df.drop(columns = ['n_Odds-Ratios', 'Score-Odds-Ratios'])
    elif score_method == 'rank':
        out_df['InverseRanking'] = out_df['n_Odds-Ratios'].rank(ascending = True, method = 'min').astype(int)
        out_df.loc[out_df['n_Odds-Ratios'] == 0, 'InverseRanking'] = 0
        out_df['Score-Odds-Ratios'] = out_df['InverseRanking'] / len(out_df)
        out_df['Score'] += out_df['Score-Odds-Ratios']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['n_Odds-Ratios', 'InverseRanking'])
        else:
            out_df = out_df.drop(columns = ['n_Odds-Ratios', 'InverseRanking', 'Score-Odds-Ratios'])
    else:
        raise ValueError("Invalid score method. Please choose 'binary' or 'rank'")

    return out_df

def _annotate_risk_prediction(df:pd.DataFrame, risk_weight_df:pd.DataFrame, show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates the input DataFrame with risk prediction feature weights and rankings.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    risk_weight_df (pd.DataFrame): A DataFrame containing risk weights to merge with the input DataFrame.
    show_indiv_scores (bool, optional): If True, individual risk prediction scores will be shown in the output DataFrame. Defaults to True.

    Returns:
    pd.DataFrame: The annotated DataFrame with risk prediction feature weights and rankings.
    """
    out_df = df.copy()
    risk_weight_df['abs-Feature Weight'] = np.abs(risk_weight_df['Feature Weight'])
    out_df = out_df.merge(risk_weight_df, left_index = True, right_index = True, how = 'left')
    out_df['Feature Weight'] = out_df['Feature Weight'].fillna(0)
    out_df['abs-Feature Weight'] = out_df['abs-Feature Weight'].fillna(0)
    out_df = out_df.rename(columns = {'Feature Weight': 'Risk Prediction Feature Weight'})

    # Score
    out_df['InverseRanking'] = out_df['abs-Feature Weight'].rank(ascending = True, method = 'min').astype(int)
    out_df.loc[out_df['abs-Feature Weight'] == 0, 'InverseRanking'] = 0
    if show_indiv_scores:
        out_df['Score-Risk Prediction'] = out_df['InverseRanking'] / len(out_df)
    out_df['Score'] += out_df['InverseRanking'] / len(out_df)
    out_df = out_df.drop(columns = ['InverseRanking', 'abs-Feature Weight'])
    return out_df

def _validate_mgi_groups(mgi_groups: list) -> list:
    """
    Validates a list of MGI (Mouse Genome Informatics) groups against a predefined set of valid groups.

    Parameters:
    mgi_groups (list): A list of MGI groups to be validated. If the list contains the string 'all', all valid MGI groups are returned.

    Returns:
    list: A list of valid MGI groups. If 'all' is provided as input, the entire list of valid MGI groups is returned.

    Raises:
    warnings.warn: Issues a warning if any of the provided MGI groups are not in the list of valid MGI groups.
    """
    valid_mgi_groups = ['hematopoietic system', 'homeostasis/metabolism', 'cardiovascular system','embryo', 'cellular', 'growth/size/body region', 'nervous system', 'renal/urinary system', 'liver/biliary system', 'craniofacial', 'digestive/alimentary', 'hearing/vestibular/ear', 'limbs/digits/tail', 'skeleton', 'behavior/neurological', 'mortality/aging', 'reproductive system', 'neoplasm', 'vision/eye', 'respiratory system', 'normal', 'endocrine/exocrine gland', 'integument', 'adipose tissue', 'muscle', 'taste/olfaction', 'pigmentation phenotype']
    if mgi_groups == ['all']:
        return valid_mgi_groups
    else:
        issues = []
        good = []
        for val in mgi_groups:
            if val not in valid_mgi_groups:
                issues.append(val)
            else:
                good.append(val)
        if issues:
            warnings.warn(f"Invalid MGI groups: {issues}. Valid MGI groups are: {valid_mgi_groups}")
        return good

def _annotate_mgi(df: pd.DataFrame, mgi_data:dict, mgi_groups: list, score_method: str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates a DataFrame with MGI (Mouse Genome Informatics) phenotype information and scores.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    mgi_df (pd.DataFrame): The MGI DataFrame containing phenotype information.
    mgi_groups (list): A list of MGI groups to filter the MGI DataFrame. If 'all' is provided, all valid MGI groups are used.
    score_method (str): The method to score the annotations. Options are 'binary' or 'rank'. Default is 'rank'.
    show_indiv_scores (bool): Whether to show individual scores in the output DataFrame. Default is True.

    Returns:
    pd.DataFrame: The annotated DataFrame with MGI phenotype information and scores.
    """
    out_df = df.copy()
    query_mgi_pheno = mgi_data['query_enrichment']
    query_mgi_pheno = query_mgi_pheno[query_mgi_pheno['fdr'] <= 0.05]
    receiver_rankings = mgi_data['receiver_rankings']
    overlap_column = [x for x in query_mgi_pheno.columns if "pheno_overlap" in x][0]
    # Loop through the genes and annotate with nDiffusion ranking and if recovered
    hold_df = pd.DataFrame(index = out_df.index, columns = ['MGI-nDiffusionRank', 'MGI-True Phenotypes Count', 'MGI-True Phenotypes'])
    for idx, vals in out_df.iterrows():
        gene_phenotypes_values = []
        gene_pheno_overlaps = []
        gene_pheno_overlaps_count = 0
        # Grab MGI phenotypes with the gene
        gene_phenotypes = query_mgi_pheno[query_mgi_pheno['geneMappings'].str.contains(idx, case = False, na = False)]
        gene_phenotypes_values = gene_phenotypes['modelPhenotypeLabel'].tolist()
        # Look at overlapping phenotypes
        gene_pheno_overlaps = gene_phenotypes['modelPhenotypeLabel'][gene_phenotypes[overlap_column] == 1].tolist()
        gene_pheno_overlaps_count = len(gene_pheno_overlaps)
        # grab max nDiffusion rank
        if len(gene_phenotypes_values) > 0:
            hold_nDiffusion_rank = receiver_rankings[receiver_rankings['Gene'].isin(gene_phenotypes_values)]
            max_nDiffusion_rank = hold_nDiffusion_rank['Diffusion score (Ranking)'].max()
        else:
            max_nDiffusion_rank = 0
        
        hold_df.loc[idx,:] = [max_nDiffusion_rank, gene_pheno_overlaps_count, gene_pheno_overlaps]
    hold_df = hold_df.sort_values(by = 'MGI-nDiffusionRank', ascending = False)
    out_df = out_df.merge(hold_df, left_index = True, right_index = True, how = 'left')

    # Score overlaps
    out_df['Score-MGI-Overlap'] = out_df['MGI-True Phenotypes Count'].apply(lambda x: 1 if x > 0 else 0)
    out_df['Score'] += out_df['Score-MGI-Overlap']
    # Score diffusion
    #out_df.loc[out_df['Score-MGI-Overlap'] == 1, 'MGI-nDiffusionRank'] = 0 #######
    out_df['InverseRanking'] = out_df['MGI-nDiffusionRank'].rank(ascending = True, method = 'min').astype(int)
    out_df.loc[out_df['MGI-nDiffusionRank'] == 0, 'InverseRanking'] = 0
    out_df['Score-MGI-Diffusion'] = out_df['InverseRanking'] / len(out_df)
    out_df['Score'] += out_df['Score-MGI-Diffusion']
    # Remove the columns
    if show_indiv_scores:
        out_df = out_df.drop(columns = ['MGI-True Phenotypes Count', 'InverseRanking'])
    else:
        out_df = out_df.drop(columns = ['MGI-True Phenotypes Count', 'Score-MGI-Overlap', 'Score-MGI-Diffusion', 'InverseRanking'])

    return out_df

def _annotate_go_term_enrichment(df: pd.DataFrame, go_term_enrichment_df:pd.DataFrame, score_method:str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    out_df = df.copy()
    out_df = out_df.merge(go_term_enrichment_df, left_index = True, right_index = True, how = 'left')
    # Fill na with 0 across Flag_1_0, Percentile_Rank, and Diffusion score (Ranking)
    out_df['GO Terms - True GO Term Overlap'] = out_df['GO Terms - True GO Term Overlap'].fillna(0)
    out_df['GO Terms - Diffusion Score (Percentile Ranking)'] = out_df['GO Terms - Diffusion Score (Percentile Ranking)'].fillna(0)
    out_df['GO Terms - Diffusion Score'] = out_df['GO Terms - Diffusion Score'].fillna(0)
    
    # Annotate the genes with GO Terms overlap
    out_df['Score-GO Term Overlap'] = out_df['GO Terms - True GO Term Overlap'].apply(lambda x: 1 if x > 0 else 0)
    out_df['Score'] += out_df['Score-GO Term Overlap']
    # Annotate GO Term diffusion
    #out_df.loc[out_df['Score-GO Term Overlap'] == 1, 'GO Terms - Diffusion Score (Percentile Ranking)'] = 0 #######

    #out_df['InverseRanking'] = out_df['GO Terms - Diffusion Score (Percentile Ranking)'].rank(ascending = True, method = 'min').astype(int)
    out_df['InverseRanking'] = out_df['GO Terms - Diffusion Score'].rank(ascending = True, method = 'min').astype(int)
    #out_df.loc[out_df['GO Terms - Diffusion Score (Percentile Ranking)'] == 0, 'InverseRanking'] = 0
    out_df.loc[out_df['GO Terms - Diffusion Score'] == 0, 'InverseRanking'] = 0
    out_df['Score-GO Term Diffusion'] = out_df['InverseRanking'] / len(out_df)
    out_df['Score'] += out_df['Score-GO Term Diffusion']
    # Remove the columns
    if show_indiv_scores:
        #out_df = out_df.drop(columns = ['GO Terms - True GO Term Overlap', 'GO Terms - Diffusion Score'])
        out_df = out_df.drop(columns = ['GO Terms - True GO Term Overlap', 'GO Terms - Diffusion Score (Percentile Ranking)'])
    else:
        out_df = out_df.drop(columns = ['GO Terms - True GO Term Overlap', 'GO Terms - Diffusion Score (Percentile Ranking)', 'GO Terms - Diffusion Score', 'Score-GO Term Diffusion'])
    return out_df

def _annotate_ndiffusion(df: pd.DataFrame, ndiffusion_df:pd.DataFrame, score_method:str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    out_df = df.copy()
    out_df = out_df.merge(ndiffusion_df, left_index = True, right_index = True, how = 'left')
    out_df['Diffusion score (Ranking)'] = out_df['Diffusion score (Ranking)'].fillna(0)
    # Score
    if score_method == 'binary':
        out_df['Score-nDiffusion'] = out_df['Diffusion score (Ranking)'].apply(lambda x: 1 if x > 0 else 0)
        out_df['Score'] += out_df['Score-nDiffusion']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['Diffusion score (Ranking)'])
        else:
            out_df = out_df.drop(columns = ['Diffusion score (Ranking)', 'Score-nDiffusion'])
    elif score_method == 'rank':
        out_df['InverseRanking'] = out_df['Diffusion score (Ranking)'].rank(ascending = True, method = 'min').astype(int)
        out_df.loc[out_df['Diffusion score (Ranking)'] == 0, 'InverseRanking'] = 0
        out_df['Score-nDiffusion'] = out_df['InverseRanking'] / len(out_df)
        out_df['Score'] += out_df['Score-nDiffusion']
        #out_df = out_df.drop(columns = ['Unnamed: 0', 'Is the gene in Set_1Excl? (1=yes)', 'Percentile_Rank','InverseRanking'])
        out_df = out_df.drop(columns = ['Unnamed: 0', 'Is the gene in Set_1? (1=yes)', 'Percentile_Rank','InverseRanking'])
        out_df = out_df.rename(columns = {'Diffusion score (Ranking)': 'nDiffusion Signal Received'})
    else:
        raise ValueError("Invalid score method. Please choose 'binary' or 'rank'")
    return out_df

def _annotate_drugs(df: pd.DataFrame, drug_df:pd.DataFrame, score_method:str = 'rank', show_indiv_scores:bool = True) -> pd.DataFrame:
    """doc"""
    def _remove_word_suffixes(string:str, suffixes:list) -> str:
        """doc"""
        try:
            words = string.split()
        except AttributeError:
            return string
        filtered_words = [word for word in words if not any(word.endswith(suffix) for suffix in suffixes)]
        return ' '.join(filtered_words)

    out_df = df.copy()
    out_df['DrugInteractions'] = np.nan
    # Load drug df
    suffixes = ['tate', 'ide', 'ous', 'ium', '-', 'ile', ',', 'sine', 'ecan']
    suffixes = [x.capitalize() for x in suffixes]
    drug_df[drug_df.columns[0]] = drug_df[drug_df.columns[0]].apply(lambda x: _remove_word_suffixes(x, suffixes))

    # Annotate drugs
    for idx, vals in out_df.iterrows():
        hold_drugs = drug_df['drug'][drug_df.index.str.contains(idx, case = False, na = False)].tolist()
        hold_drugs = [x for x in hold_drugs if str(x) != '' and str(x) != 'nan']
        hold_drugs = list(set(hold_drugs))
        out_df.loc[idx, 'n_DrugInteractions'] = len(hold_drugs)
        out_df.loc[idx, 'DrugInteractions'] = ";".join(hold_drugs)

    # Score
    if score_method == 'binary':
        out_df['Score-DrugInteractions'] = out_df['n_DrugInteractions'].apply(lambda x: 1 if x > 0 else 0)
        out_df['Score'] += out_df['Score-DrugInteractions']
        if show_indiv_scores:
            out_df = out_df.drop(columns = ['n_DrugInteractions'])
        else:
            out_df = out_df.drop(columns = ['n_DrugInteractions', 'Score-DrugInteractions'])

    elif score_method == 'rank':
        out_df['InverseRanking'] = out_df['n_DrugInteractions'].rank(ascending = True, method = 'min').astype(int)
        out_df.loc[out_df['n_DrugInteractions'] == 0, 'InverseRanking'] = 0
        if show_indiv_scores:
            out_df['Score-DrugInteractions'] = out_df['InverseRanking'] / len(out_df)
        out_df['Score'] += out_df['Score-DrugInteractions']
        out_df = out_df.drop(columns = ['n_DrugInteractions', 'InverseRanking'])
    else:
        raise ValueError("Invalid score method. Please choose 'binary' or 'rank'")
    return out_df

def _annotate_depmap(df: pd.DataFrame, depmap_scores: pd.DataFrame, ranking_method:str = 'tumor_suppressor', show_indiv_scores:bool = True) -> pd.DataFrame:
    """
    Annotates the input DataFrame with DepMap scores and rankings.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    depmap_scores (pd.DataFrame): A DataFrame containing DepMap scores to merge with the input DataFrame.
    show_indiv_scores (bool, optional): If True, individual DepMap scores will be shown in the output DataFrame. Defaults to True.

    Returns:
    pd.DataFrame: The annotated DataFrame with DepMap scores and rankings.
    """
    out_df = df.copy()
    out_df = out_df.merge(depmap_scores, left_index = True, right_index = True, how = 'left')

    # Score
    if ranking_method == 'tumor_suppressor':
        ascending = False
    elif ranking_method == 'oncogene':
        ascending = True
    else:
        raise ValueError("Invalid ranking method. Please choose 'tumor_suppressor' or 'oncogene'")
    out_df['InverseRanking'] = out_df['DepMap Score'].fillna(0).rank(ascending = ascending, method = 'min').astype(int)
    if show_indiv_scores:
        out_df['Score-DepMap'] = out_df['InverseRanking'] / len(out_df)
    out_df['Score'] += out_df['InverseRanking'] / len(out_df)
    out_df = out_df.drop(columns = ['InverseRanking'])
    return out_df

def _annotate_comentions(df: pd.DataFrame, comention_df:pd.DataFrame) -> pd.DataFrame:
    """
    Annotates a DataFrame with comention information.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be annotated.
    comention_df (pd.DataFrame): A DataFrame containing comention information to merge with the input DataFrame.

    Returns:
    pd.DataFrame: The annotated DataFrame with comention information merged.
    """
    out_df = df.copy()
    comention_df = comention_df.drop(columns = [x for x in comention_df.columns if f"PubMed_CoMentions-" not in x])
    out_df = out_df.merge(comention_df, left_index = True, right_index = True, how = 'left')
    return out_df

def prioritize_genes(query: list, result_dict:dict = {}, result_path:str = "", result_experiments:list = [], score_method:str = 'rank', p_value_col:str = 'fdr', or_directories: list = [], or_threshold:float = 0.1, mgi_groups:list = ['all'], depmap_ranking:str = 'tumor_suppressor', drug_source:str = 'all', show_indiv_scores:bool = True, savepath:str = None) -> pd.DataFrame:
    """
    Annotates and ranks a list of queries based on various experimental results.

    Parameters:
    - query (list): List of queries to be annotated and ranked.
    - result_dict (dict, optional): Dictionary containing experimental results. Defaults to an empty dictionary.
    - result_path (str, optional): Path to load experimental results from. Defaults to an empty string.
    - result_experiments (list, optional): List of experiments to consider for annotation. Defaults to an empty list.
    - score_method (str, optional): Method to use for scoring. Defaults to 'rank'. Options include 'rank' or 'binary'.
    - p_value_col (str, optional): Column name for p-values. Defaults to 'fdr'.
    - or_directories (list, optional): List of directories for odds ratios. Defaults to an empty list.
    - or_threshold (float, optional): Threshold for odds ratios. Defaults to 0.1.
    - mgi_groups (list, optional): List of MGI groups. Defaults to ['all'].
    - depmap_ranking (str, optional): Method for DepMap ranking. Defaults to 'tumor_suppressor'. Options include 'tumor_suppressor' or 'oncogene'.
    - drug_source (str, optional): Source for drug-gene interactions. Defaults to 'all'.
    - show_indiv_scores (bool, optional): If True, individual scores will be shown in the output DataFrame. Defaults to True.
    - savepath (str, optional): Path to save the annotated and ranked DataFrame. Defaults to None.

    Returns:
    pd.DataFrame: The annotated and ranked DataFrame.
    """
    # Define the valide arguments for inputs
    valid_keys_and_types = {'p_value': pd.DataFrame, 'consensus': pd.DataFrame, 'goldstandard_overlap': list, 'gwas_catalog_colocalization': pd.DataFrame, 'ndiffusion': pd.DataFrame, 'interconnectivity': pd.DataFrame, 'functional_clustering': pd.DataFrame, 'functional_clustering_enrichment':dict, 'pubmed_comentions': pd.DataFrame, 'depmap_enrichment': pd.DataFrame, 'risk_prediction': pd.DataFrame, 'odds_ratios': pd.DataFrame, 'mouse_phenotype_enrichment': pd.DataFrame, 'drug_gene_interactions': pd.DataFrame, 'go_term_enrichment': pd.DataFrame}

    # Validate result_dict
    if result_dict != {}:
        _validate_dict_keys(result_dict, valid_keys_and_types)
        result_experiments = list(result_dict.keys())
    # Load result path
    elif result_path != "":
        result_dict = _load_results(result_path, result_experiments, valid_keys_and_types, or_directories, drug_source = drug_source, valid_cluster_threshold = 2)
    else:
        raise ValueError("No input provided")
    # Annotate prioritization scores
    main_df = pd.DataFrame(index = query)
    main_df['Score'] = 0
    # Annotate p-values
    if 'p_value' in result_experiments:
        main_df = _annotate_p_value(main_df, result_dict['p_value'], p_value_col, show_indiv_scores = show_indiv_scores)
    # Annotate consensus
    if 'consensus' in result_experiments:
        main_df = _annotate_consensus(main_df, result_dict['consensus'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate Down-Sampling
    # Annotate Gold Standards
    if 'goldstandard_overlap' in result_experiments:
        main_df = _annotate_gold_standard(main_df, result_dict['goldstandard_overlap'], show_indiv_scores = show_indiv_scores)
    # Annotate co-localization
    if 'gwas_catalog_colocalization' in result_experiments:
        main_df = _annotate_colocalization(main_df, result_dict['gwas_catalog_colocalization'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate GO Term Enrichment
    if 'go_term_enrichment' in result_experiments:
        main_df = _annotate_go_term_enrichment(main_df, result_dict['go_term_enrichment'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate nDiffusion
    if 'ndiffusion' in result_experiments:
        main_df = _annotate_ndiffusion(main_df, result_dict['ndiffusion'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate interactions
    if 'interconnectivity' in result_experiments:
        main_df = _annotate_interconnections(main_df, result_dict['interconnectivity'], result_dict['goldstandard_overlap'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate clusters
    if 'functional_clustering' in result_experiments:
        main_df = _annotate_clusters(main_df, result_dict['functional_clustering'], result_dict['functional_clustering_enrichment'], result_dict['goldstandard_overlap'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate odds
    if 'odds_ratios' in result_experiments:
        main_df = _annotate_odds(main_df, result_dict['odds_ratios'], q_value_threshold = or_threshold, score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate risk prediction
    if 'risk_prediction' in result_experiments:
        main_df = _annotate_risk_prediction(main_df, result_dict['risk_prediction'], show_indiv_scores = show_indiv_scores)
    # Annotate MGI
    if 'mouse_phenotype_enrichment' in result_experiments:
        main_df = _annotate_mgi(main_df, result_dict['mouse_phenotype_enrichment'], mgi_groups, score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate tissues
    # Annotate drugs
    if 'drug_gene_interactions' in result_experiments:
        main_df = _annotate_drugs(main_df, result_dict['drug_gene_interactions'], score_method = score_method, show_indiv_scores = show_indiv_scores)
    # Annotate depmap
    if 'depmap_enrichment' in result_experiments:
        main_df = _annotate_depmap(main_df, result_dict['depmap_enrichment'], ranking_method = depmap_ranking, show_indiv_scores = show_indiv_scores)
    # Annotate comentions
    if 'pubmed_comentions' in result_experiments:
        main_df = _annotate_comentions(main_df, result_dict['pubmed_comentions'])
    # Plot heatmap table
    main_df = main_df.sort_values(by = 'Score', ascending = False)
    #heatmap_table = _plot_score_table(main_df)
    # Save
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = savepath + "/PrioritizationTable/"
        os.makedirs(new_savepath, exist_ok = True)
        if show_indiv_scores:
            main_df.to_csv(new_savepath + "PrioritizationTable_IndividualScores.csv", index = True, sep = ',')
        else:
            main_df.to_csv(new_savepath + "PrioritizationTable.csv", index = True, sep = ',')

    return main_df

#endregion
