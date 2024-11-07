# pyCfS
Version 0.1.2 <br>
The aggregation of Lichtarge Lab genotype-phenotype validation experiments<br>

## Installation
### Install Git-LFS (in terminal)
Mac (Homebrew) - `brew install git-lfs` <br>
Windows - Follow instructions here: https://gitforwindows.org/ <br>
Ubuntu/Debian - `sudo apt-get install git-lfs` <br>
Fedora/CentOS - `sudo yum install git-lfs`<br>
<br>
Activate git-lfs - `git lfs install` <br>

### Create an anaconda environment and install non-pip packages
conda create -n pyCfS python=3.8.18 <br>
conda activate pyCfS<br>
conda install -c conda-forge r-base r-ggplot2 r-deldir r-rcppeigen r-interp rpy2 rasterio r-tzdb r-vroom r-readr r-cowplot r-tidyverse <br>

### Install pyCfS (in anaconda environment)
pip install git+https://github.com/kevwilhelm95/pyCfS.git <br>
(Ensure pip is pointing to anaconda environment, if it is not, use anaconda environment pip: /path/to/env/../bin/pip install git+...) <br>

#### Examples
See "example.ipynb" for help

#### Notes
- Parallelized functions require the user to run function under blocking guard (i.e. if `__name__ == "__main__"`:) <br>
- Save path should be a parent directory (e.g. /path/to/folder) as the functions will create experiment-specific folders automatically (e.g. /path/to/folder/experiment) <br>
- STRING-defined custom_background (e.g. "string_v12.0") will use all genes with interactions above low threshold (> 0.15)

#### Available Methods
- `pyCFS.Combine`
    - `consensus`
    - `functional_clustering`
    - `statistical_combination`
- `pyCFS.GoldStandards`
    - `string_enrichment`
    - `goldstandard_overlap`
    - `ndiffusion`
    - `interconnectivity`
    - `gwas_catalog_colocalization`
    - `pubmed_comentions`
- `pyCFS.Clinical`
    - `mouse_phenotype_enrichment`
    - `protein_family_enrichment`
    - `drug_gene_interactions`
    - `depmap_enrichment`
- `pyCFS.Association`
    - `variants_by_sample`
    - `risk_prediction`
    - `odds_ratios`
    - `ea_distributions`
- `pyCFS.Structure`
    - `lollipop_plot`
    - `protein_structures`
- `pyCFS.Summarize`
    - `prioritize_genes`

# Modules

## pyCFS.Combine

### `consensus()`
Combines multiple lists of genes by counting the number of times a gene mets significance in each method. This function will return a table with genes, the number of times it reaches significance, and where they came from, if list_names is assigned. We suggest prioritizing genes that reach significance in at least 2 methods, where increased stringency can come from reaching significance in more than 2 methods.
#### Parameters:
- **Optional**:
    - `gene_dict` (dict): Dict of gene lists (as alternative to below). {Name: [gene1, gene2, ...]} format
    - `genes_1` (list): List of genes.
    - `genes_2` (list): List of genes.
    - `genes_3` (list): Additional list of genes.
    - `genes_4` (list): Additional list of genes.
    - `genes_5` (list): Additional list of genes.
    - `genes_6` (list): Additional list of genes.
    - `list_names` (list): Names of your lists (Default set to "set_x").
    - `plot_fontface` (str): Font for upset plot (Default = Avenir).
    - `plot_fontsize` (int): Font size for upset plot (Default = 14).
    - `savepath` (str): Path to save files. If no path provided, files are not saved.
    #### Returns:
- `pd.DataFrame`: A dataframe with columns 'gene', 'occurrences', and 'lists', detailing each unique gene, the number of its occurrences, and the lists it appeared in.
- `Image`: Upset plot showing overlap between input genelists.

### `functional_clustering()`
`Parallelized` <br>
Clusters genes from multiple sources in STRING network. Can be used as a "functional consensus" to prioritize genes across methods or to cluster prioritized genes with gold standards, uncovering functional biology related to known genes. When using, we suggest you threshold the clusters at those with 2 or more genes in a cluster
#### Parameters:
- **Optional**:
    - `genes_1` (list): list of genes
    - `genes_2` (list): list of genes
    - `genes_3` (list): Additional list of genes.
    - `genes_4` (list): Additional list of genes.
    - `genes_5` (list): Additional list of genes.
    - `source_names` (list): Gene list names.
    - `gene_dict` (dict): Dict of gene lists (as alternative to above). {Name: [gene1, gene2, ...]} format
    - `string_version` (str): Version of STRING to use. Choose "v10.0", "v11.0", "v11.5", "v12.0". Default = "v11.0"
    - `evidences` (list): Evidences to compute edge weight. Options include ['neighborhood', 'fusion', 'coocurence', 'coexpression', 'experimental', 'database', 'textmining'] (Default = ['all']).
    - `edge_confidence` (str): Minimum edge weight for network. Options include 'all' (weight > 0), 'low' (weight > 0.2), 'medium' (weight > 0.4), 'high' (weight > 0.7), 'highest' (weight > 0.9). (Default = 'highest').
    - `custom_background` (str OR list): Background gene set for optimal inflation parameter and pathway enrichment. Options include 'string', 'ensembl', 'reactome' or user defined list (Default = 'string').
    - `random_iter` (int): # of random iterations to perform (Default = 100).
    - `inflation` (float): Set inflation parameter. If not set, algorithm determines optimal inflation from 1.5-3.0.
    - `pathways_min_group_size` (int): Minimum group size for pathway enrichment (Default = 5).
    - `pathways_max_group_size` (int): Maximum group size for pathway enrichment (Default = 100).
    - `cores` (int): # of cores for parallelization (Default = 1).
    - `savepath` (str): File path. If not provided, no files are saved
    - `verbose` (int): Verbosity argument. Default = 0 (No verbose)
#### Returns:
- `pd.DataFrame` : Pairwide edges from true connection network. Formatted as STRING network.
- `pd.DataFrame` : Table of query genes, their true clusters, and gene sources.
- `dict` : Dictionary of dataframes by cluster contained pathway enrichment results.


### `statistical_combination()`
Statistical p-value combination methods, including Cauchy, MCM, CMC, Minimum p, and P multiplication. We suggest using a p-value threshold of 1e-4 for genes scored by all methods, and a p-value threshold of 5e-8 for those scored by fewer methods.
#### Parameters
- **Optional**:
    - `df_1` (pd.DataFrame): Two-column df with genes (col1) and p-value (col2). No specific header format. Do not fill NAs with 1
    - `df_2` (pd.DataFrame): Two-column df with genes (col1) and p-value (col2).
    - `df_3` (pd.DataFrame): Additional two-column df.
    - `df_4` (pd.DataFrame): Additional two-column df.
    - `df_5` (pd.DataFrame): Additional two-column df.
    - `df_6` (pd.DataFrame): Additional two-column df.
    - `list_names` (Any): Names of the dataframes to be included as column identifiers. Defaults to False.
    - `gene_df` (pd.DataFrame): Dataframe of p-values if more than 6. Index = gene names. Column 1-x = 'p_{list name}' (p-values). Do not fill NAs with 1s.
    - `savepath` (str): File path. No files saved if not provided.
#### Returns:
- `pd.DataFrame` : Dataframe containing genes, their original p-values, and p-value combinations for Cauchy, MinP, CMC, MCM, and multiplied p-values.





## pyCFS.GoldStandards
### `string_enrichment()`
Assess gene set network connectivity and functional enrichment using the STRING API. Returns the same results as if you were using the web-browser website (string-db.org).
#### Parameters:
- `query` (list): List of genes
- **Optional**:
    - `string_version` (str): Version of STRING to use. Choose "v10.0", "v11.0", "v11.5", "v12.0". Default = "v11.0"
    - `edge_confidence` (str): Minimum edge weight for network. Options include 'all' (weight > 0), 'low' (weight > 0.15), 'medium' (weight > 0.4), 'high' (weight > 0.7), 'highest' (weight > 0.9). (Default = 'medium').
    - `species` (int): Species code from STRING (Default = 9606 (human))
    - `plot_fontsize` (int): Default = 14.
    - `plot_fontface` (str): Default = Avenir.
    - `savepath` (str): Parent path for saving.
    - `verbose` (int): Verbosity toggle. Default = 0
#### Returns:
- `pd.DataFrame`: Table of network edges 
- `float` : P-value of PPI enrichment
- `Image` : STRING network image
- `pd.DataFrame` : Table of functional enrichment for gene set
- `Dictionary` : Plots of significant functional enrichment for each pathway set


### `goldstandard_overlap()`
Assess the overlap with a reference gene set, tested using a hypergeometric test.
#### Parameters:
- `query` (list): List of query genes
- `goldstandard` (list): List of gold standard genes
- **Optional**:
    - `custom_background` (str OR list): Background gene set. Options include 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'ensembl').
    - `plot_query_color` (str): Color of query venn diagram (Default = red).
    - `plot_goldstandard_color` (str): Color of goldstandard venn diagram (Default = gray)
    - `plot_show_gene_pval` (bool) : Default = True. Toggle showing the overlapping gene names and p-value on plot image
    - `plot_fontsize` (int): Fontsize for venn diagram (Default = 14).
    - `plot_fontface` (str): Fontface for venn diagram (Default = Avenir).
    - `savepath` (str): File path. If not provided, no files saved.
#### Returns:
- `list` : List of overlapping genes
- `float` : P-value of hypergeometric overlap.
- `Image` : Venn diagram of overlap


### `ndiffusion()`
`Parallelized` <br>
Assess the broad network connectivity between two gene sets in the STRING network. (Can take approx. 40 minutes for 'all' edge confidence with 5 cores and 100 random iterations). Significance is computed using z-test against the AUROC of 100 random background sets.
#### Parameters:
- `set_1` (list): List of genes
- `set_2` (list): List of genes
- **Optional**:
    - `set_1_name` (str): Name of set 1 for plotting & saving (Default = Set_1)
    - `set_2_name` (str): Name of set 2 for plotting & saving (Default = Set_2)
    - `string_version` (str): Version of STRING to use. Choose "v10.0", "v11.0", "v11.5", "v12.0". Default = "v11.0"
    - `evidences` (list): Evidences to compute edge weight. Options include ['neighborhood', 'fusion', 'coocurence', 'coexpression', 'experimental', 'database', 'textmining'] (Default = ['all']).
    - `edge_confidence` (str): Minimum edge weight for network. Options include 'all' (weight > 0), 'low' (weight > 0.15), 'medium' (weight > 0.4), 'high' (weight > 0.7), 'highest' (weight > 0.9). (Default = 'all').
    - `custom_background` (str OR list): Background gene set. Options include 'string' (matching above version), 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'string').
    - `n_iter` (int): # of randomizations to perform (Default = 100).
    - `cores` (int): # of cores for parallelization (Default = 1).
    - `savepath` (str): Path for saving.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `Image`: AUROC plot for show_1 ("from Set1 Exclusive to Set2", if no overlap then "from Set1 to Set2")
- `float`: Z-score for show_1 AUROC (randomized set1, degree-matched)
- `Image`: AUROC plot for show_2 ("from Set2 Exclusive to Set1", if no overlap then "from Set1 to Set2")
- `float`: Z-score for show_2 AUROC (randomized set2, degree-matched)


### `interconnectivity()`
`Parallelized` <br>
Assess the level of direct connections with reference gene set in the STRING network. Significance is determined by z-test against the number of true connections for 100 random degree-matched gene sets.
#### Parameters:
- `set_1` (list): List of genes.
- `set_2` (list): List of genes. 
- **Optional**:
    - `set_3` (list): List of genes.
    - `set_4` (list): List of genes.
    - `set_5` (list): List of genes.
    - `string_version` (str): Version of STRING to use. Choose "v10.0", "v11.0", "v11.5", "v12.0". Default = "v11.0"
    - `evidences` (list): Evidences to compute edge weight. Options include ['neighborhood', 'fusion', 'coocurence', 'coexpression', 'experimental', 'database', 'textmining'] (Default = ['all']).
    - `edge_confidence` (str): Minimum edge weight for network. Options include 'all' (weight > 0), 'low' (weight > 0.15), 'medium' (weight > 0.4), 'high' (weight > 0.7), 'highest' (weight > 0.9). (Default = 'highest').
    - `custom_background` (str OR list): Background gene set. Options include 'string' (matching above string version),'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'string').
    - `num_iterations` (int): # of iterations for background connections (Default = 250). 
    - `cores` (int): For parallelization (Default = 1).
    - `plot_fontface` (str): (Default = Avenir).
    - `plot_fontsize` (int): (Default = 14).
    - `plot_query_color` (str): Line color for enrichment plot (Default = red).
    - `plot_background_color` (str): Distribution color for enrichment plot (Default = gray).
    - `savepath` (str): File path.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `Image` : Venn diagram of list overlap.
- `Image` : Random connection histogram with true query connections.
- `list` : List of random connections.
- `pd.DataFrame` : Unique gene network.
- `dict` : Gene sources


### `gwas_catalog_colocalization()`
`Parallelized` <br>
Assess the enrichment for co-localization within X Mbp of genome-wide significant loci. Enrichment is determined using a Fisher's exact test.
#### Parameters:
- `query` (list): List of genes:
- **Optional**:
    - `mondo_id` (str): Disease ID from GWAS Catalog. If used, calls API for disease.
    - `gwas_summary_path` (str): Path to GWAS Catalog downloaded summary statistics. If used, API is not called.
    - `gwas_p_thresh` (float): Filter for genome-wide significance to compare to (Default = 5e-8).
    - `distance_mbp` (float): Distance threshold for colocalization in Mbp (Default = 0.5 Mbp).
    - `custom_background` (str OR list): Background gene set. Options include 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'ensembl').
    - `cores` (int): # of cores for parallelization (Default = 1).
    - `savepath` (str): Path to save. If not used, no files are saved.
    - `save_summary_statistics` (bool): True to save downloaded summary stats if savepath is also defined.
    - `verbose` (int): Verbosity toggle. Default = 1
#### Returns:
- `pd.DataFrame` : Two-column table of SNPs that colocalize with query gene.
- `float` : Fisher's exact test p-value


### `pubmed_comentions()`
`Parallelized` <br>
Assess the enrichment for co-mentions with specific keywords in PubMed. Significance is determined by a z-test against 100 random gene sets of equal size to query.
#### Parameters:
- `query` (list): List of genes.
- **Optional**:
    - `field` (str): Toggle for stringency of associations. Default = 'all'. Options = ['all', 'title/abstract', 'title'].
    - `keyword` (str): Keyword to search co-mentions for. Default search query is '("{gene}") AND ("{keyword"}) AND (("gene") OR ("protein"))'.
    - `custom_terms` (str): A custom search query for PubMed. Function will add and "AND" to the end of the gene and then add your custom term after. For example, the custom_term = '(("adipose") OR ("diabetes")) AND (("gene") OR ("protein"))' would produce a search of '("{gene}") AND (("adipose") OR ("diabetes")) AND (("gene") OR ("protein"))'
    - `custom_background` (str OR list): Background gene set. Options include 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'ensembl').
    - `email` (str): Email of a PubMed account. (Default = my email).
    - `api_key` (str): API key of PubMed account. (Default = my key).
    - `enrichment_trials` (int): # of randomization trials (Default = 100).
    - `run_enrichment` (bool): Toggle use of randomization. If False, no enrichment performed (Default = True).
    - `workers` (int): # of workers for threaded API calls (Default = 15).
    - `enrichment_cutoffs` (list): Breaks for enrichment of co-mentions (>1st number, <=2nd number; Default = [[-1,0], [0,5], [5,15], [15,50], [50,10000]])
    - `plot_background_color` (str): Color for background distribution (Default = gray).
    - `plot_query_color` (str): Color for query line (Default = red).
    - `plot_fontface` (str): Default = Avenir.
    - `plot_fontsize` (int): Default = 14.
    - `savepath` (str): File path.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `pd.DataFrame` : Dataframe of genes, co-mention counts, and PMIDs.
- `dict` : Keys = enrichment_cutoffs values; Values = (# of query genes, z score).
- `dict` : Keys = enrichment_cutoffs values; Values = Image of enrichment distribution.





## pyCFS.Clinical
### `mouse_phenotype_enrichment()`
`Parallelized` <br>
Assess abnormal mouse phenotype enrichments from Mouse Genome Informatics (Data parsed and downloaded from OpenTargets). Enrichment is assessed using z-test against 5,000 random representation-matched genes (the number of models for a gene in MGI). 
#### Parameters:
- `query` (list): List of genes
- **Optional**:
    - `custom_background` (str OR list): Background gene set. Options include 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'ensembl').
    - `random_iter` (int): Iterations for background run (Default = 5000).
    - `plot_sig_color` (str): Color for sig. phenotypes in strip plot (Default = red).
    - `plot_q_threshold` (float): Significance threshold for strip plot (Default = 0.05).
    - `plot_show_labels` (bool): If true, plot labels provided in `plot_labels_to_show` (Default = False).
    - `plot_labels_to_show` (list): Phenotype labels to plot. Use labels in "modelPhenotypeLabel" of output dataframe.
    - `plot_fontface` (str)
    - `cores` (int): Default = 1.
    - `savepath` (str): File path.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `pd.DataFrame` : Enrichment summary.
- `Image` : Strip plot of enrichment.
- `Image` : Histogram of FDR values.


### `protein_family_enrichment()`
`Parallelized` <br>
Assess enrichment of protein family type from OpenTargets data. Enrichment is determined with a z-test against 5000 random gene sets.
#### Parameters:
- `query` (list): List of genes
- **Optional**:
    - `custom_background` (str OR list): Background gene set. Options include 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'ensembl').
    - `level` (list): Levels to test. Options include "all", "level1", "level2", "level3", "level4", "level5" (Default = ["all"])
    - `random_iter` (int): Number of background iterations (Default = 5000).
    - `plot_q_cut` (float): Plot significance threshold (Default = 0.05).
    - `plot_sig_dot_color` (str): Color of significant dots in plot (Default = red).
    - `plot_fontface` (str): Default = Avenir.
    - `plot_fontsize` (int): Default = 14.
    - `cores` (int): For parallelization (Default = 1).
    - `savepath` (str): File path.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `pd.DataFrame` : Enrichment dataframe for protein families.
- `Image` : Horizontal strip plot of enrichment.


### `tissue_expression_enrichment()`

### `depmap_enrichment()`
Assess enrichment for cancer-dependent genes. Significance is determined by a Mann Whitney U test against background scores.
#### Parameters:
- `query` (list) : List of genes
- `cancer_type` (list) : List of cancer cell types (Available cancer cell types can be found at - https://depmap.org/portal/ > Tools > Cell Line Selector > Create custom list (Broad types = Lineage, Most specific = Lineage Sub-subtype))
- **Optional**:
    - `custom_background` (str OR list): Background gene set. Options include 'depmap', 'ensembl', 'reactome', 'string_v10.0', 'string_v11.0', 'string_v11.5', 'string_v12.0', or user defined list (Default = 'depmap').
    - `plot_fontface` (str) : Defualt = Avenir.
    - `plot_fontsize` (int) : Default = 14.
    - `plot_query_color` (str) : Default = red.
    - `plot_background_color` (str) : Default = gray.
    - `savepath` (str) : File path
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns
- `pd.DataFrame` : Dataframe of query genes and their cancer dependency scores
- `float` : P-value of Mann Whitney U test
- `Image` : Histogram of Chronos DepMap scores for query and background genes


### `drug_gene_interactions()`
Pulls drug-gene interactions in order to find potential repurposable therapies.
#### Parameters:
- `query` (list): List of genes
- **Optional**:
    - `drug_source` (list) : Resource to pull from. Options = 'OpenTargets', 'DGIdb'. Default = ['OpenTargets']
    - `dgidb_min_citations` (int) : DGIdb-specific. Minimum number of citations noting drug-gene interaction. Default = 1
    - `approved` (bool) : Filter for FDA-approved or not drugs. Default = True.
    - `savepath` (str) : File path
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `dict` : Resource name and drug interactors (e.g. {'DGIdb': pd.DataFrame})




## pyCFS.Association
### `variants_by_sample()`
`Parallelized` <br>
Parses the input VCF to create a .csv of samples and their individual variants in the queried genes.
#### Parameters:
- `query` (list): List of genes.
- `vcf_path` (str): Path to .gz vcf file. Index (.tbi) should be in the same directory.
- `samples` (pd.DataFrame): Two-column df containing samples (name = "SampleID") and their case(1)/control(0) mappings (name = "CaseControl")
- **Optional**:
    - `transcript` (str): Method for parsing transcripts. Default = 'canonical'. Options = 'canonical', 'max', 'mean'
    - `cores` (int): Cores for parallelization. Default = 1
    - `savepath` (str): Path to save dataframe.
#### Returns:
- `pd.DataFrame` : Dataframe of parsed variants for each individual.


### `risk_prediction`
`Parallelized` <br>
Takes an input feature matrix (rows = samples, columns = features) and trains, using recursive feature elimination and Bayesian hyperparameter optimization, a machine learning model to predict who is a case or a control in a left-out sample. Recursive feature elimination uses a 10-fold cross-validation to evaluate model performance. Bayesian optimization uses a 5 by 5-fold stratified cross-validation to evaluate hyperparameters. <br>
Notes: If multiple models are input (e.g. ['RF', 'LR', 'GB']), the three models will be evaluated using cross-validation on the training samples. The best performing model of those three will be used to predict on the test samples.
#### Parameters:
- `feature_matrix` (pd.DataFrame): A matrix of features where samples are the rows and the features are the columns.
- `train_samples` (pd.DataFrame): A two-column table with SampleIDs as column one and CaseControl as the second column. CaseControl should be formatted as 1 for Case and 0 for Control. These samples will be used to train the models.
- `test_samples` (pd.DataFrame): A two-column table with SampleIDs as column one and CaseControl as the second column. CaseControl should be formatted as 1 for Case and 0 for Control. These samples will be used to evaluate the model.
- **Optional**:
    - `models` (list or str): Abbreviations for models to evaluate. LR = Logistic Regression. SVC = Support Vector Classifier. RF = Random Forest. GB = Gradient Boosting. XGB = Extreme Gradient Boosting. Can define using either ['RF', 'LR'] or "RF, LR". Default = RF.
    - `rfe` (bool): Toggle to perform recursive feature elimination. Default = False
    - `rfe_min_feature_ratio` (float): Ratio to set the minimum number of features to keep (e.g. 0.5 represents at minimum, keep 50% of the features). Default = 0.5.
    - `optimize_hyperparameters` (bool): Toggle to optimize hyperparameters using Bayesian optimization. If False, random hyperparameters will be chosen and only the last model will be analyzed. Default = True.
    - `cores` (int): Number of cores for parallel rfe and hyperparameter optimization. Default = 1
    - `savepath` (str): Path to save the output files.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `Image`: Distribution of evaluation/test samples with calculated odds ratios
- `Image`: Evaluation/test sample AUROC curve
- `pd.DataFrame`: Table showing probability of case/control for evaluation/test samples.
- `pd.DataFrame`: Table of feature importance scores for each feature.

### `odds_ratios`
`Parallelized` <br>
Takes the variants_by_sample output and performs odds ratio calculations based on allelic counts between two sets of patients. We recommend performing variant-based calculations on variants with an allele frequency > 1%, as more rare variants will be underpowered. We recommend performing gene-based and domain-based calculations on variants with allele frequency < 1%, as common variants can confound the signal. For a detailed description of how the "method", "model", and "level" arguments affect the odds ratios, please see "resource/odds_ratio_calculations.pdf".
#### Parameters:
- `variants_by_sample` (pd.DataFrame): Output from the variants_by_sample function.
- `samples` (pd.DataFrame): Two-column dataframe for samples to calculate odds ratios on. First column = sample_ids. Second column = CaseControl (1/0).
- `query` (list): List of genes to analyze
- **Optional**:
    - `method` (str): Method for determining counts in contingency table. Options = ['sample', 'allelic']. Default = 'sample'
    - `model` (str): Model to calculate odds ratios. Options = ['dominant', 'recessive']. Dominant model includes both heterozygous and homozygous variants. Recessive model only analyzes homozygous variants. Default = 'dominant'
    - `level` (str): Level of analysis. Options = ['variant', 'gene', 'domain']. Variant analyzes variant-by-variant. Gene analyzes gene-collapsed odds ratios. Domain analyzes protein-domains annotated by Evidence and Conclusion Ontology (ECO). Default = 'variant'.
    - `consequence` (str): Regex pattern to filter variants for their consequences. Options = ['stop_gained', 'frameshift_variant', 'stop_lost', 'missense_variant', 'splice_region_variant', 'splice_donor_variant', 'splice_acceptor_variant', 'start_lost'] Default = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost".
    - `ea_lower` (int): Minimum EA score for variants to include. Default = 0.
    - `ea_upper` (int): Maximum EA score for variants to include. Default = 100.
    - `min_af` (float): Minimum allele frequency to include. Default = 0.0.
    - `max_af` (float): Maximum allele frequency to include. Default = 1.0.
    - `significance_level` (float): FDR threshold to be considered "significant" for results plot. Default = 0.1
    - `show_plot_labels` (bool): Toggle to show result plot labels for significant findings. Default = True.
    - `cores` (int): Number of cores for parallelization. Default = 1.
    - `savepath` (str): Path for saving.
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `pd.DataFrame`: Table of resulting odds ratios calculated, their p-values, and FDR corrections.
- `Image`: Odds ratio scatter plot of analyzed genomic objects.


### `ea_distributions`
Takes the odds_ratio output dataframe and tests EA score distribution differences within a gene for two sets of patients, using the Kolmogorov-Smirnov test. !! To maximize usability, run odds_ratios for all allele frequencies and ea scores and use that exact_test_file for ea_distributions
#### Parameters:
- `variants_by_sample` (pd.DataFrame): Output from variants_by_sample
- `genes` (list): List of genes to analyze EA distributions.
- **Optional**:
    - `min_vars` (int): Minimum number of variants needed to analyze distributions.
    - `distribution` (str): Method of counting EA scores. Options = ['non_degenerate', 'degenerate']. Non_degenerate expands EA scores by the allele count in cases and controls. Degenerate only records one EA score if any sample has the variant. Default = 'non_degenerate'.
    - `consequence` (str): Regex pattern to filter variants for their consequences. Options = ['stop_gained', 'frameshift_variant', 'stop_lost', 'missense_variant', 'splice_region_variant', 'splice_donor_variant', 'splice_acceptor_variant', 'start_lost'] Default = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost".
    - `min_af` (float): Minimum allele frequency for variants. Default = 0.0.
    - `max_af` (float): Maximum allele frequency for variants. Default = 1.0.
    - `savepath` (str): Path for saving. Default = None.
#### Returns:
- `pd.DataFrame`: Table of p-values for KS test results
- `dict`: Dictionary of resulting EA score distribution plots. Can be accessed using out_dict[gene_name].





## pyCFS.Structure
### `lollipop_plot()`
Generates a lollipop plot given case and control variants and tests odds ratios (for 'both' only) using a Fisher's exact test calculated using sample-based model (see "resource/odds_ratio_calculations.pdf").
#### Parameters:
- `variants` (pd.DataFrame): Dataframe of variants by sample. Can get from "variants_by_sample()"
- `gene` (str): Gene name that you wish to plot ("PDGFRB")
- **Optional**:
    - `group` (str): Which side (case/control) to plot. 'both' plots case on top and controls on bottom. 'case'/'control' plots only one or the other. Default = 'both'.
    - `case_pop` (int): The number of cases in your population. If not defined, it is calculated from the input variants dataframe.
    - `cont_pop` (int): The number of controls in your population. If not defined, it is calculated from the input variants dataframe.
    - `max_af` (float): Max allele frequency to display in the plot. Default = 1.0
    - `ea_lower` (float): Minimum EA score to include. Default = 0
    - `ea_upper` (float): Maximum EA score to include. Default = 100
    - `consequence` (str): Regex pattern to filter variants for their consequences. Options = ['stop_gained', 'frameshift_variant', 'stop_lost', 'missense_variant', 'splice_region_variant', 'splice_donor_variant', 'splice_acceptor_variant', 'start_lost'] Default = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost".
    - `show_domains` (bool): Toggle plotting of domains in structure. Defaut = True
    - `ac_scale` (str): Change the scale of the y-axis. Default = 'linear'. Options = ['linear', 'log']
    - `ea_color` (str): Color scale for Lollipop and Linear structure. Default = 'prismatic'. Options = ['prismatic', 'gray_scale', 'EA_bin', 'black']
    - `domain_min_dist` (int): The minimum distance that separates protein domains. Default = 20
    - `savepath` (str): Directory to save
    - `verbose` (int): Verbosity argument. Default = no verbose (0).
#### Returns:
- `Image`: Lollipop plot
- `float`: P-value of Odds Ratio
- `float`: Odds Ratio
- `float`: Lower Odds Ratio Confidence Interval
- `float`: Upper Odds Ratio Confidence Interval


### `protein_structures()`
`Parallelized` <br>
Generates a Pymol script to visualize variant location in 3D protein structure (AlphaFold) colored with Evolutionary Trace. Variants mapped to protein will also be tested for structural clustering using the SCW method [PMID: 12875851] using all variants, case variants, and control variants. 
#### Parameters:
- `variants` (pd.DataFrame): Dataframe of variants by sample. Can get from "variants_by_sample()"
- `gene` (str): Gene name that you wish to analyze ("PDGFRB")
- **Optional**:
    - `run_scw` (bool): Toggle running SCW analysis. (Default = True)
    - `scw_chain` (str): Chain ID to use for AlphaFold structure. Default = "A"
    - `scw_plddt_cutoff` (int): Minimum pLDDT confidence score to include for background analysis. Default = 50
    - `scw_min_dist_cutoff` (int): Minimum distance to analyze clustering enrichment for. Default = 4
    - `scw_max_dist_cutoff` (int): Maximum distance to analyze clustering enrichment for. Default = 12
    - `max_af` (float): Max allele frequency to include. Default = 1.0
    - `min_af` (float): Minimum allele frequency to include. Default = 0
    - `ea_upper` (int): Maximum EA score to include. Default = 100
    - `ea_lower` (int): Minimum EA score to include. Default = 0
    - `consequence` (str): Regex pattern to filter variants for their consequences. Options = ['stop_gained', 'frameshift_variant', 'stop_lost', 'missense_variant', 'splice_region_variant', 'splice_donor_variant', 'splice_acceptor_variant', 'start_lost'] Default = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost".
    - `cores` (int): Number of cores to include. SCW is memory intensive and may run out of memory locally if more than a few cores.
    - `savepath` (str): Path to save. If not provided, PyMol model is not created, but SCW will run
#### Returns: 
- `pd.DataFrame` : Clustering enrichment for all variants meeting criteria (both cases & controls)
- `pd.DataFrame` : Clustering enrichment for case variants meeting criteria
- `pd.DataFrame` : Clustering enrichment for control variants meeting criteria
- `Image` : Clustering enrichment plot for all variants meeting criteria (both cases & controls)
- `Image` : Clustering enrichment plot for case variants meeting criteria
- `Image` : Clustering enrichment plot for control variants meeting criteria



## pyCFS.Structure
### `prioritize_genes()`
An evidence-based gene prioritization schema, ranking genes based on levels of evidence in previous validation criteria. Results can either be loaded using "result_dict" for in-script files or "result_path", which automatically load the outputs declared by "result_experiments".
#### Parameters:
- `query` (list): List of query genes
- **Optional**:
    - `result_dict` (dict): Dictionary of output files annotated with their experiment. Valid experiments and the required file types include: {'p_value': pd.DataFrame, 'consensus': pd.DataFrame, 'goldstandard_overlap': list, 'gwas_catalog_colocalization': pd.DataFrame, 'interconnectivity': pd.DataFrame, 'functional_clustering': pd.DataFrame, 'functional_clustering_enrichment':dict, 'pubmed_comentions': pd.DataFrame, 'depmap_enrichment': pd.DataFrame, 'risk_prediction': pd.DataFrame, 'odds_ratios': pd.DataFrame, 'mouse_phenotype_enrichment': pd.DataFrame, 'drug_gene_interactions': pd.DataFrame}
    - `result_path` (str): Path to parent directory of outputs from previous experiments
    - `result_experiments` (list): List of experiments that should be loaded from the result_path.
    - `score_method` (str): Options include "rank" (default) or "binary". For rank, evidence scores will be ranked based on number of significant findings then normalized to a range of 1 (highest) to 0 (lowest). For binary, each gene will be tested for positive criteria and will be given 1 if so and 0 aif not.
    - `p_value_col` (str): Column name of the p-value/q-value/fdr column to use when loading "p_value". Default = 'fdr'.
    - `or_directories` (list): List of directory names for odds ratios results (i.e. domain_dominant_EA-0-100_AF-0.0-0.01_missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost)
    - `or_threshold` (float): FDR threshold for considering significant odds ratios. Default = 0.1.
    - `mgi_groups` (list): List of high-level mouse phenotypes to filter for significant findings in ranking. Default = 'all'. Options include: ['hematopoietic system', 'homeostasis/metabolism', 'cardiovascular system','embryo', 'cellular', 'growth/size/body region', 'nervous system', 'renal/urinary system', 'liver/biliary system', 'craniofacial', 'digestive/alimentary', 'hearing/vestibular/ear', 'limbs/digits/tail', 'skeleton', 'behavior/neurological', 'mortality/aging', 'reproductive system', 'neoplasm', 'vision/eye', 'respiratory system', 'normal', 'endocrine/exocrine gland', 'integument', 'adipose tissue', 'muscle', 'taste/olfaction', 'pigmentation phenotype']
    - `depmap_ranking` (str): Method to rank depmap scores for query genes. Default = 'tumor_suppressor', in which more negative scores receive higher ranking. For 'oncogene', more positive scores receive a higher ranking.
    - `drug_source` (str): Selection for "result_path" to select drug source. Options include "all" (default), "DGIdb" and "OpenTargets".
    - `show_indiv_scores` (bool): Toggle to output each individual experiments scores in the output dataframe.
    - `savepath` (str): Directory to save
#### Returns: 
- `pd.DataFrame` : The annotated and ranked dataframe