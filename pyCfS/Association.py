"""
Functions to assess variant/gene associations.

Functions:
- variants_by_sample
"""

import pandas as pd
from typing import Any
from pysam import VariantFile
from joblib import Parallel, delayed
from tqdm import tqdm
# Risk Prediction
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import DeltaYStopper
from sklearn.metrics import balanced_accuracy_score, roc_curve, RocCurveDisplay, auc
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV

import numpy as np
import statsmodels.api as sm
from scipy.stats import ks_2samp, fisher_exact
import math
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import io
from adjustText import adjust_text
import joblib
import multiprocessing as mp
import re
from PIL import Image
import os
from .utils import _load_grch38_background, _fix_savepath, _load_pdb_et_mapping, _validate_ea_thresh, _validate_af_thresh

# Ignore machine learning warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


#region VariantsBySample
def _fetch_anno(anno:Any) -> Any:
    """
    Fetches the annotation value from the given input.

    Args:
        anno (Any): The input annotation.

    Returns:
        Any: The fetched annotation value.
    """
    if isinstance(anno, tuple) and len(anno) == 1:
        return anno[0]
    else:
        return anno

def _validate_ea(ea:Any) -> float:
    """
    Checks for valid EA score
    Args:
        ea (str/float/None): EA score as string
    Returns:
        float: EA score between 0-100 if valid, otherwise returns NaN
    """
    try:
        ea = float(ea)
    except ValueError:
        if isinstance(ea, str) and (ea == 'fs-indel' or 'STOP' in ea):
            ea = 100
        else:
            ea = np.nan
    except TypeError:
        ea = np.nan
    return ea

def _fetch_ea_vep(ea:tuple, canon_ensp:str, all_ensp:tuple, csq:str, ea_parser:str) -> Any:
    """
    Fetches the EA VEP (Variant Effect Predictor) score based on the given parameters.

    Args:
        ea (list): List of EA scores.
        canon_ensp (str): Canonical Ensembl protein ID.
        all_ensp (list): List of all Ensembl protein IDs.
        csq (str): Variant consequence.
        ea_parser (str): EA parser type ('canonical', 'mean', 'max', or 'all').

    Returns:
        Any: The EA VEP score based on the given parameters.
    """
    if 'stop_gained' in csq or 'frameshift_variant' in csq or 'stop_lost' in csq:
        return 100
    if ea_parser == 'canonical':
        try:
            canon_idx = all_ensp.index(canon_ensp)
        except ValueError:
            return np.nan
        else:
            return _validate_ea(ea[canon_idx])
    else:
        new_ea = []
        for score in ea:
            new_ea.append(_validate_ea(score))
        if np.isnan(new_ea).all():
            return np.nan
        elif ea_parser == 'mean':
            return np.nanmean(new_ea)
        elif ea_parser == 'max':
            return np.nanmax(new_ea)
        else:
            return new_ea

def _convert_zygo(genotype:tuple) -> (int, int): # type: ignore
    """
    Convert a genotype tuple to a zygosity integer
    Args:
        genotype (tuple): The genotype of a variant for a sample
    Returns:
        int: The zygosity of the variant (0/1/2)
    """
    if genotype in [(1, 0), (0, 1)]:
        zygo = 1
        an = 2
    elif genotype in [(1, None), (None, 1)]:
        zygo = 1
        an = 1
    elif genotype == (1, 1):
        zygo = 2
        an = 2
    elif genotype == (0,0):
        zygo = 0
        an = 2
    elif genotype in [(0, None), (None, 0)]:
        zygo = 0
        an = 1
    elif genotype == (None, None):
        zygo = 0
        an = 0
    else:
        zygo = 0
        an = 0
    return zygo, an

def _parse_vep(vcf_fn:str, gene:str, gene_ref:pd.DataFrame, samples:list, ea_parser:str) -> pd.DataFrame:
    """
    Parse the Variant Effect Predictor (VEP) data from a VCF file.

    Args:
        vcf (VariantFile): The VCF file object.
        gene_ref (pd.DataFrame): The gene reference data.
        contig_prefix (str): The prefix for the contig.
        samples (list): The list of samples to process.
        ea_parser (str): The EA parser.

    Returns:
        pd.DataFrame: The parsed VEP data.

    """
    # Get the vcf
    vcf = VariantFile(vcf_fn)
    # Parse for the samples only
    vcf.subset_samples(samples)
    # Get the contig type
    contig_prefix = 'chr' if 'chr' in next(vcf).chrom else ''
    contig = contig_prefix + gene_ref.chrom
    rows = []
    for rec in vcf.fetch(contig=contig, start=gene_ref.start, stop=gene_ref.end):
        total_an = 0
        sample_rows = []
        for sample in samples:
            zyg, an = _convert_zygo(rec.samples[sample]['GT'])
            total_an += an
            rec_gene = _fetch_anno(rec.info['SYMBOL'])
            if (zyg!=0) and (rec_gene == gene):
                all_ea = rec.info.get('EA', (None,))
                all_ensp = rec.info.get('Ensembl_proteinid', (rec.info['ENSP'][0],))
                canon_ensp = _fetch_anno(rec.info['ENSP'])
                rec_hgvsp = _fetch_anno(rec.info['HGVSp'])
                csq = _fetch_anno(rec.info['Consequence'])
                ea = _fetch_ea_vep(all_ea, canon_ensp, all_ensp, csq, ea_parser=ea_parser)
                if not np.isnan(ea):
                    sample_rows.append(
                        [
                            canon_ensp,
                            rec.chrom,
                            rec.pos,
                            rec.ref,
                            rec.alts[0],
                            rec_hgvsp,
                            csq,
                            ea,
                            rec_gene,
                            sample,
                            zyg,
                            rec.info['AF'][0]
                        ]
                    )
        # Append the AN to all samples
        for row in sample_rows:
            row.append(total_an)
            rows.append(row)
    cols = ['ENSP', 'chr','pos','ref','alt', 'HGVSp', 'Consequence', 'EA','gene','sample','zyg','AF', 'AN_Cohort']
    col_type = {'chr': str, 'pos': str, 'ref': str, 'alt': str, 'sample':str, 'EA':float, 'zyg':int, 'AF':float}
    df = pd.DataFrame(rows, columns = cols)
    df = df.astype(col_type)
    return df

def variants_by_sample(query:list, vcf_path:str, samples:pd.DataFrame, transcript: str = 'canonical', cores:int = 1, savepath:str = False) -> pd.DataFrame:
    """
    Retrieves variants from a VCF file for a given list of genes and sample IDs.

    Args:
        genes (list): List of genes to retrieve variants for.
        vcf_path (str): Path to the VCF file. .tbi index file must be present in the same directory.
        samples (pd.DataFrame): DataFrame containing sample IDs.
        transcript (str, optional): Transcript type to use for parsing VEP annotations. Defaults to 'canonical'.
        cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the design matrix of variants for the specified genes and samples.
    """
    # Get the sample IDs you are interested in
    samples[[samples.columns[0]]] = samples[[samples.columns[0]]].astype(str)
    cases = samples.loc[samples.iloc[:,1] == 1]
    controls = samples.loc[samples.iloc[:,1] == 0]
    case_ids = cases.iloc[:, 0].astype(str).tolist()
    control_ids = controls.iloc[:, 0].astype(str).tolist()
    # Get the gene positions
    gene_positions = _load_grch38_background(just_genes = False)
    gene_positions = gene_positions.loc[gene_positions.index.isin(query)]

    # Parse the VCF for variants of interest
    case_gene_dfs = Parallel(n_jobs=cores)(delayed(_parse_vep)(
         vcf_fn=vcf_path,
         gene = gene,
         gene_ref = gene_positions.loc[gene],
         samples=case_ids,
         ea_parser=transcript
    ) for gene in tqdm(gene_positions.index.unique()))
    control_gene_dfs = Parallel(n_jobs=cores)(delayed(_parse_vep)(
            vcf_fn=vcf_path,
            gene = gene,
            gene_ref = gene_positions.loc[gene],
            samples=control_ids,
            ea_parser=transcript
        ) for gene in tqdm(gene_positions.index.unique()))
    case_design = pd.concat(case_gene_dfs, axis=0)
    control_design = pd.concat(control_gene_dfs, axis=0)
    design_matrix = pd.concat([case_design, control_design], axis=0)

    # Map the Sample IDs to CaseControl
    design_matrix = design_matrix.merge(samples, left_on='sample', right_on='SampleID', how='left')
    design_matrix = design_matrix.drop(columns = 'SampleID')

    # Save the design matrix
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'VariantsBySample/')
        os.makedirs(new_savepath, exist_ok=True)
        design_matrix.to_csv(new_savepath + 'Variants.csv', index=False)

    return design_matrix
#endregion


#region risk_prediction
def _validate_sample_dfs(sample_df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """
    Validate the sample DataFrames for the risk prediction function. Sample DataFrames must have sample_ids as the index and column names as 'SampleID' and 'CaseControl'.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame.

    Returns:
        pd.DataFrame: The validated sample DataFrame.
    """
    # Check number of columns
    if len(sample_df.columns) != 2:
        raise ValueError(f"{sample_type} DataFrame must have 2 columns. First column is sample IDs and second column is CaseControln (1 = Case, 0 = Control).")
    # Check for valid 'CaseControl' values
    if not sample_df.iloc[:, 1].isin([0, 1]).all():
        raise ValueError(f"{sample_type} - CaseControl values must be 0 or 1.")
    # Change the column values for later accession
    sample_df.columns = ['SampleID', 'CaseControl']
    return sample_df

def _validate_feature_matrix(feature_matrix: pd.DataFrame, train_samples: pd.DataFrame, test_samples: pd.DataFrame) -> None:
    """
    Validate the feature matrix for the risk prediction function. Feature matrix must have sample_ids as the index and column names as gene names or feature names.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix.

    Returns:
        pd.DataFrame: The validated feature matrix.
    """
    # Check train samples are in feature matrix index
    if not train_samples.iloc[:, 0].isin(feature_matrix.index).all():
        missing_samples = train_samples[~train_samples.iloc[:, 0].isin(feature_matrix.index)].iloc[:, 0].tolist()
        raise ValueError(f"Train samples are not in feature matrix index. {len(missing_samples)} samples in test_samples not found in feature_matrix: {missing_samples}")
    # Check test samples are in feature matrix index
    if not test_samples.iloc[:, 0].isin(feature_matrix.index).all():
        # Which sample IDs in test_samples are not in feature matrix
        missing_samples = test_samples[~test_samples.iloc[:, 0].isin(feature_matrix.index)].iloc[:, 0].tolist()
        raise ValueError(f"Test samples are not in feature matrix index. {len(missing_samples)} samples in test_samples not found in feature_matrix: {missing_samples}")

def _generate_x_y_matrices(feature_matrix: pd.DataFrame, sample_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Generate the X and y matrices for the risk prediction function.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix.
        sample_df (pd.DataFrame): The sample DataFrame.

    Returns:
        tuple: A tuple containing the X and y matrices.
    """
    # Merge the dataframes
    sub_df = sample_df.merge(feature_matrix, left_on='SampleID', right_index=True, how = 'inner')
    sub_df.index = sub_df.SampleID
    # Get the X matrix
    x = sub_df.drop(columns = ['SampleID', 'CaseControl'])
    # Get the y matrix
    y = sub_df.CaseControl
    return x, y

def _validate_separate_input_models(input_models: Any) -> list:
    """
    Validates the input models and returns a list of valid models.

    Args:
        input_models (Union[str, list]): The input models to be validated. It can be a string or a list of strings.

    Returns:
        list: A list of valid models.

    Raises:
        ValueError: If the input models are neither a string nor a list of strings.

    """
    if isinstance(input_models, str):
        # Check if the input has comma separated values
        if ', ' in input_models:
            input_models = input_models.split(', ')
        elif "," in input_models:
            input_models = input_models.split(',')
        else:
            input_models = [input_models]
    elif isinstance(input_models, list):
        pass
    else:
        raise ValueError("Input models must be a string or list of strings")

    # Check against valid models
    valid_models = ['LR', 'SVC', 'GB', 'RF', 'XGB']
    for model in input_models:
        if model not in valid_models:
            warnings.warn(f"Invalid model: {model}. Skipping this model. Valid models are: {valid_models}.")
            input_models.remove(model)
    return input_models

def _get_hyperparameter_search_space(model: str, n_features:int) -> (dict, Any): # type: ignore
    """
    Get the hyperparameter search space for the given model and number of features.

    Args:
        model (str): The model name. Possible values: 'LR', 'SVC', 'GB', 'RF', 'XGB'.
        n_features (int): The number of features.

    Returns:
        tuple: A tuple containing the hyperparameter search space dictionary and the corresponding estimator object.
    """

    params = dict()
    # Logistic Regression
    if model == 'LR':
        params['penalty'] = Categorical(['l1', 'l2', 'elasticnet'])
        # 'newton-cg', 'lbfgs', 'liblinear', 'sag'
        params['solver'] = Categorical(['saga'])
        params['max_iter'] = Categorical([5000, 10000])
        params['warm_start'] = Categorical([True, False])
        params['l1_ratio'] = Real(0.0001, 0.5, prior='log-uniform')
        params['tol'] = Real(0.0001, 100.0, prior='log-uniform')
        params['C'] = Real(0.0001, 100.0, prior='log-uniform')
        params['random_state'] = [n_features]
        estimator = LogisticRegression()
    # Support Vector Machine
    if model == 'SVC':
        # Lower bound - 1e-6
        params['C'] = Real(0.0001, 10.0, prior='log-uniform')
        # Lower bound - 1e-6
        params['gamma'] = Real(0.0001, 1.0, prior='log-uniform')
        params['degree'] = Integer(1.0, 5.0)
        params['kernel'] = Categorical(['linear', 'poly', 'rbf'])  # 'sigmoid'
        params['probability'] = Categorical([True])
        params['random_state'] = [n_features]
        estimator = SVC()
    # Gradient Boosting
    if model == 'GB':
        params['n_estimators'] = Integer(100, 10000, prior='log-uniform')
        params['loss'] = Categorical(['log_loss','exponential'])
        params['learning_rate'] = Real(0.0001, 1.0, prior='log-uniform')
        params['subsample'] = Real(0.0001, 0.5, prior='log-uniform')
        params['criterion'] = Categorical(['friedman_mse', 'squared_error'])
        params['max_depth'] = Integer(1, round(np.sqrt(n_features), 0), prior = 'log-uniform')
        params['warm_start'] = Categorical([True, False])
        params['tol'] = Real(0.0001, 100.0, prior='log-uniform')
        params['ccp_alpha'] = Real(0.0001, 100.0, prior='log-uniform')
        params['random_state'] = [n_features]
        estimator = GradientBoostingClassifier()
    if model == 'RF':
        params['n_estimators'] = Integer(100, 10000, prior='log-uniform')
        params['criterion'] = Categorical(['gini', 'entropy'])
        params['bootstrap'] = Categorical([True])
        params['warm_start'] = Categorical([True, False])
        params['ccp_alpha'] = Real(0.0001, 100.0, prior='log-uniform')
        params['oob_score'] = Categorical([True, False])
        params['min_samples_split'] = Real(0.001, 0.5, prior='log-uniform')
        params['min_samples_leaf'] = Integer(1, 10)
        params['max_depth'] = Integer(1.0, round(np.sqrt(n_features), 0), prior = 'log-uniform')
        params['max_features'] = Categorical(['sqrt', 'log2'])
        params['random_state'] = [n_features]
        estimator = RandomForestClassifier()
    if model == 'XGB':
        params['n_estimators'] = Integer(10, 5000, prior='log-uniform')
        params['subsample'] = Real(0.1, 0.5, prior = 'log-uniform')
        params['max_depth'] = Integer(2, round(np.sqrt(n_features),0))
        params['eta'] = Real(0.01, 0.5, prior='log-uniform')
        #params['gamma'] = Real(1e-4, 0.5, prior = 'log-uniform') -- Warning
        params['reg_alpha'] = Real(1e-4, 0.5, prior='log-uniform')
        params['reg_lambda'] = Real(1e-4, 0.9, prior='log-uniform')
        params['booster'] = Categorical(['dart', 'gbtree'])
        params['tree_method'] = Categorical(['hist'])
        params['random_state'] = [n_features]
        estimator = xgb.XGBClassifier()
    return params, estimator

def _aggregate_feature_selection(cv_obj: Any, estimator: Any, x_train:pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate feature selection results from cross-validation and the final estimator.

    Args:
        cv_obj (Any): The cross-validation object used for feature selection.
        estimator (Any): The final estimator used for feature selection.
        x_train (pd.DataFrame): The training data.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated feature selection results.

    """
    # Get feature importances from the final estimator
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Feature_Importance': importances})
    elif hasattr(estimator, 'coef_'):
        importances = estimator.coef_[0]
        feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Feature_Importance': importances})
    # Trying out creating a saving object
    rfecv_results = pd.DataFrame({
        'Feature': cv_obj.feature_names_in_,
        'Support': cv_obj.support_,
        'Ranking': cv_obj.ranking_
    })
    # Merge the results
    feature_importances = feature_importances.merge(rfecv_results, left_on='Feature', right_on='Feature', how='left')
    feature_importances = feature_importances.sort_values(by=['Feature_Importance', 'Ranking'], ascending=False).reset_index(drop=True)
    return feature_importances

def _plot_rfe_cv(rfe_cv_results: pd.DataFrame) -> Image:
    """
    Plot the results of the Recursive Feature Elimination Cross-Validation.

    Args:
        rfe_cv_results (pd.DataFrame): The results of the RFE-CV.

    Returns:
        PIL.Image.Image: The image of the plot.

    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        rfe_cv_results.n_features,
        rfe_cv_results.mean_roc_auc,
        yerr=rfe_cv_results.std_roc_auc,
        label='ROC-AUC',
        color='r',
        linewidth=2,
        ecolor = 'gray',
        elinewidth = 1,
        capsize = 3)
    plt.xlabel('Number of Features')
    plt.ylabel('ROC-AUC')
    plt.title('Recursive Feature Elimination Cross-Validation')
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _rfe_cv(estimator: Any, x_train: pd.DataFrame, y_train: pd.DataFrame, min_feature_ratio:float, cores: int) -> (list, pd.DataFrame, pd.DataFrame, Image): # type: ignore
    """
    Perform recursive feature elimination with cross-validation (RFECV) to select the optimal features for a given estimator.

    Parameters:
        - estimator (Any): The estimator object implementing the 'fit' method.
        - x_train (pd.DataFrame): The input features for training.
        - y_train (pd.DataFrame): The target variable for training.
        - min_feature_ratio (float): The minimum ratio of features to select.
        - cores (int): The number of CPU cores to use for parallel processing.

    Returns:
        - selected_features (list): The list of selected features.
        - feature_importances (pd.DataFrame): The feature importances.
        - rfe_cv_results (pd.DataFrame): The results of RFECV.
        - rfe_cv_plot (matplotlib.pyplot): The plot of RFECV results.
    """
    # Initialize the RFECV object
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=len(x_train.columns))
    min_features = math.floor(min_feature_ratio * x_train.shape[1])
    rfecv = RFECV(
        estimator,
        step = 1,
        cv = cv,
        scoring = 'roc_auc',
        min_features_to_select = min_features,
        verbose = 0,
        n_jobs = cores,
        importance_getter = 'auto'
    )
    # Fit the RFECV object
    rfecv = rfecv.fit(x_train, y_train)
    estimator.fit(x_train, y_train)
    # Get the feature importances
    feature_importances = _aggregate_feature_selection(rfecv, estimator, x_train)
    selected_features = feature_importances.loc[feature_importances.Support == True, 'Feature'].tolist()

    # Collect RFE results
    rfe_cv_results = pd.DataFrame({
        'n_features': range(min_features, x_train.shape[1] + 1),
        'mean_roc_auc': rfecv.cv_results_['mean_test_score'],
        'std_roc_auc': rfecv.cv_results_['std_test_score']
    })
    rfe_cv_plot = _plot_rfe_cv(rfe_cv_results)

    return selected_features, feature_importances, rfe_cv_results, rfe_cv_plot

def _bayesian_optimization_auroc_progression(df: pd.DataFrame) -> Image:
    """
    Plot the progression of AUROC through the hyperparameter search space.

    Args:
        df (pd.DataFrame): The DataFrame containing the AUROC scores.

    Returns:
        PIL.Image.Image: The image of the plot.

    """
    plt.plot(df.index, df.mean_test_score)
    plt.xlabel("HyperParameter Search Index", size=12)
    plt.ylabel("AUROC", size=12)
    plt.title("Progression of AUROC through hyperparameter search space", size=14)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _optimize_hyperparameters(estimator: Any, params: dict, x_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, n_splits: int, n_repeats: int, cores: int, verbose:int = 0) -> (Any, pd.DataFrame, Image): #type: ignore
    """
    Optimize hyperparameters using Bayesian optimization.

    This function performs hyperparameter optimization using Bayesian optimization
    with a custom stopper callback. It uses the BayesSearchCV class from the scikit-optimize
    library to perform the optimization.

    Parameters
    ----------
    estimator : Any
        The estimator object to be optimized.

    params : dict
        The search space for the hyperparameters.

    x_train : pd.DataFrame
        The input training data.

    y_train : pd.DataFrame
        The target training data.

    n_iter : int
        The number of iterations for the optimization.

    n_splits : int
        The number of splits for cross-validation.

    n_repeats : int
        The number of repeats for cross-validation.

    cores : int
        The number of CPU cores to use for parallel processing.

    Returns
    -------
    tuple
        A tuple containing the best score, best parameters, cross-validation results,
        and a plot of the optimization progression.

    """
    # Define our custom stopper for the BayesSearchCV
    class CustomDeltaYStopper(DeltaYStopper):
        """
        Custom callback for the BayesSearchCV function to stop the search early when
        the best scores have converged below a specified threshold.

        Parameters
        ----------
        delta : float, required
            The threshold value for the difference between the best scores. If the
            difference between the best n_best scores is less than delta for
            patience consecutive iterations, the search will stop early.

        n_best : int, optional, default: 5
            The number of best scores to consider when checking for convergence.
            The difference between the best n_best scores will be compared to the
            delta value.

        patience : int, optional, default: 5
            The number of consecutive iterations for which the difference between
            the best n_best scores must be less than delta to stop the search early.

        Attributes
        ----------
        _count : int
            A counter to track the number of consecutive iterations for which the
            difference between the best n_best scores is less than delta.

        Methods
        -------
        _criterion(result) -> bool
            A helper function that checks if the search should be stopped early
            based on the provided convergence criteria.

        """
        def __init__(self, delta:float, n_best:int=5, patience:int=5, min_iters:int = 20, verbose:int = 0) -> None:
            super().__init__(delta)
            self.n_best = n_best
            self.patience = patience
            self._count = 0
            self.min_iters = min_iters
            self.verbose = verbose

        def _criterion(self, result: Any) -> bool:
            """
            Check if the criterion for early stopping is met.

            Args:
                result (object): The result object containing function values.

            Returns:
                bool: True if the criterion is met and early stopping should occur, False otherwise.
            """
            if len(result.func_vals) < self.min_iters:
                return False
            if len(result.func_vals) < self.n_best:
                return False
            best_scores = sorted(result.func_vals)[-self.n_best:]
            delta = best_scores[-1] - best_scores[0]
            if delta < self.delta:
                self._count += 1
                if self._count >= self.patience:
                    if self.verbose > 0:
                        print(f"Early stopping occurred at iteration: {len(result.func_vals)}")
                    return True
            else:
                self._count = 0

            return False

    # Initialize search objects
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=len(x_train.columns))
    delta_y_callback = CustomDeltaYStopper(delta=0.0001, n_best = 5, patience = 5, min_iters = 20, verbose = verbose)
    search = BayesSearchCV(
        estimator = estimator,
        search_spaces = params,
        n_jobs = cores,
        n_iter = n_iter,
        cv = cv,
        scoring = 'roc_auc',
        verbose = 0,
        random_state = len(x_train.columns)
    )
    search.fit(x_train, y_train, callback = delta_y_callback)
    cv_result = pd.DataFrame(search.cv_results_)
    cv_result_plot = _bayesian_optimization_auroc_progression(cv_result)

    return search.best_score_, search.best_params_, cv_result, cv_result_plot

def _set_hyperparameters(estimator: Any, params: dict) -> Any:
    """
    Set the hyperparameters for the given model.

    Args:
        model (Any): The model object.
        params (dict): The hyperparameters.

    Returns:
        Any: The model object with the set hyperparameters.
    """
    estimator.set_params(**params)
    return estimator

def _optimize_features_hyperparameters(input_models: Any, rfe:bool, x_train:pd.DataFrame, y_train:pd.DataFrame, max_feature_ratio:float, cores:int) -> (Any, pd.DataFrame, dict, dict, dict, dict, dict): # type: ignore
    """
    Optimize features and hyperparameters for each model.

    Args:
        input_models (Any): The input models to test.
        rfe (bool): Flag indicating whether to perform Recursive Feature Elimination (RFE).
        x_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data labels.
        max_feature_ratio (float): The maximum feature ratio for RFE.
        cores (int): The number of cores to use for parallel processing.

    Returns:
        tuple: A tuple containing the final estimator, best results dataframe, RFE importances, RFE results, RFE results plot, Bayesian results, and Bayesian results plot.
    """
    # Get the estimators to test
    models_to_test = _validate_separate_input_models(input_models)
    # Set holder values
    best_results = dict()
    best_params_dict = dict()
    rfe_importances, rfe_results, rfe_results_plot = dict(), dict(), dict()
    bayes_results, bayes_results_plot = dict(), dict()
    # Optimize Hyperparameters for each model
    for model in models_to_test:
        ### Need to understand how to save everything for each model
        params, estimator = _get_hyperparameter_search_space(model, x_train.shape[1])
        # Perform rfe if needed
        if rfe:
            selected_features, feature_importances, rfe_cv_results, rfe_cv_plot = _rfe_cv(estimator, x_train, y_train, max_feature_ratio, cores)
            rfe_importances[model] = feature_importances
            rfe_results[model] = rfe_cv_results
            rfe_results_plot[model] = rfe_cv_plot
            x_train = x_train[selected_features]
        else:
            rfe_importances[model] = None
            rfe_results[model] = None
            rfe_results_plot[model] = None
            selected_features = x_train.columns.to_list()

        # Perform Bayesian hyperparameter optimization
        params, estimator = _get_hyperparameter_search_space(model, len(selected_features))
        best_score, best_params, cv_result, cv_plot = _optimize_hyperparameters(estimator, params, x_train, y_train, n_iter=50, n_splits=5, n_repeats=5, cores=cores)
        best_results[model] = best_score
        best_params_dict[model] = best_params
        bayes_results[model] = cv_result
        bayes_results_plot[model] = cv_plot
    # Merge best results and params for saving
    best_result_df = pd.DataFrame({'Model': list(best_results.keys()), 'Best_AUROC': list(best_results.values())})
    best_result_df = best_result_df.sort_values(by='Best_AUROC', ascending=False).reset_index(drop=True)
    best_params_df = pd.DataFrame.from_dict(best_params_dict, orient='index').reset_index().rename(columns={'index': 'Model'})
    best_df = best_result_df.merge(best_params_df, left_on='Model', right_on='Model', how='left')

    # Set the best model
    best_model = best_result_df.loc[0, 'Model']
    _, best_model_estimator = _get_hyperparameter_search_space(best_model, len(selected_features))
    best_model_params = best_params_dict[best_model]
    final_estimator = _set_hyperparameters(best_model_estimator, best_model_params)
    final_estimator.fit(x_train, y_train)

    return final_estimator, best_model, selected_features, best_df, rfe_importances, rfe_results, rfe_results_plot, bayes_results, bayes_results_plot, models_to_test

def _save_feature_weights(estimator: Any, model: str, feature_matrix: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
    """
    Save the feature weights of a given model.

    Parameters:
    - estimator (Any): The trained model estimator.
    - model (str): The type of model used. Possible values are 'LR', 'SVC', 'GB', 'RF', 'XGB'.
    - feature_matrix (pd.DataFrame): The feature matrix used for training the model.

    Returns:
    - pd.DataFrame: A DataFrame containing the feature weights sorted in descending order.

    Note:
    - For models 'LR' and 'SVC', the function retrieves the coefficients of the model and maps them to the feature names.
    - For models 'GB', 'RF', and 'XGB', the function retrieves the feature importances directly from the model.
    - If the model is not one of the specified types, the function returns None.
    """
    if model in ['LR', 'SVC']:
        try:
            coefficients = estimator.coef_[0]
            feature_weights = dict(zip(feature_matrix.columns, coefficients))
        except:
            if verbose > 0:
                print("Model does not have coefficients")
            feature_weights = None
    elif model in ['GB', 'RF', 'XGB']:
        importances = estimator.feature_importances_
        feature_weights = dict(zip(feature_matrix.columns, importances))
    else:
        feature_weights = None
        return feature_weights
    
    if feature_weights:
        feature_weights_df = pd.DataFrame.from_dict(feature_weights, orient='index', columns=['Feature Weight'])
        feature_weights_df['abs(Feature Weight)'] = feature_weights_df['Feature Weight'].abs()
        feature_weights_df = feature_weights_df.sort_values(by='abs(Feature Weight)', ascending=False)
        feature_weights_df.drop(columns = 'abs(Feature Weight)', inplace=True)
        return feature_weights_df

def _plot_auroc(fpr: pd.Series, tpr: pd.Series, roc_auc:float) -> Image:
    """
    Plots the Area Under the Receiver Operating Characteristic (AUROC) curve for a given estimator.

    Parameters:
        estimator (Any): The trained estimator object.
        x_test (pd.DataFrame): The input features for testing.
        y_test (pd.DataFrame): The target labels for testing.

    Returns:
        Image: The AUROC curve plot as an Image object.
    """
    plt.clf()
    plt.close('all')
    _ = plt.figure(figsize=(5, 5))
    display = RocCurveDisplay(fpr = fpr, tpr=tpr, roc_auc = roc_auc, estimator_name=' ')
    display.plot()
    plt.legend(loc='upper left')
    plt.xlabel('')
    plt.ylabel('')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _plot_histo_w_or(predictions_gene: pd.DataFrame, bins:int = 50, xlim_lower:float = None, xlim_upper:float = None, ylim:tuple = (None, None), ylim_or: tuple = (None, None)) -> (Image, pd.DataFrame): # type: ignore
    """
    Plots a histogram of the probability of being a case or control, along with the odds ratio (OR) values.

    Args:
        predictions_gene (pd.DataFrame): A DataFrame containing the predictions for each sample, including the 'case' column indicating whether the sample is a case or control.
        bins (int, optional): The number of bins for the histogram. Defaults to 50.
        xlim_lower (float, optional): The lower limit of the x-axis range for the histogram. Defaults to None.
        xlim_upper (float, optional): The upper limit of the x-axis range for the histogram. Defaults to None.
        ylim (tuple, optional): The y-axis limits for the histogram. Defaults to (None, None).
        ylim_or (tuple, optional): The y-axis limits for the odds ratio (OR) plot. Defaults to (None, None).

    Returns:
        tuple: A tuple containing the histogram image (Image) and a DataFrame (pd.DataFrame) with the odds ratio (OR) values for different thresholds.

    Raises:
        None

    """
    def calculate_or_with_ci(tp:int, fp:int, fn:int, tn:int) -> tuple:
        """
        :param a: affected with mutation
        :param b: healthy with mutation
        :param c: affected w/o mutation
        :param d: healthy w/o mutation
        :return: odds ratio, p-value, and confidence interval
        """
        table = np.zeros((2, 2))
        table[0][0] = tp
        table[0][1] = fp
        table[1][0] = fn
        table[1][1] = tn
        or_table = sm.stats.Table2x2(table)
        odds_ratio = or_table.oddsratio
        p_value = or_table.oddsratio_pvalue()
        confidence_interval = list(or_table.oddsratio_confint())
        '''used for cleaned-up version'''
        confidence_interval = [float(f'{x:.2f}') for x in confidence_interval]
        return odds_ratio, p_value, confidence_interval

    max_thr = predictions_gene['Case'].max()
    thrlist = np.arange(0.5, max_thr, 0.05).tolist()
    if len(thrlist) < 4:
        thrlist = np.arange(0.5, max_thr, 0.01).tolist()
    or_df = pd.DataFrame(columns=['Thr', 'OR', 'CI', 'pval',
                                  'Dis w exp', 'HC w exp', 'Dis w/o exp', 'HC w/o exp'])
    for i, thr in enumerate(thrlist):
        high_thr = thr
        low_thr = 1-thr
        # high = gene_probs[gene_probs['Prob']>high_thr].copy()
        # low = gene_probs[gene_probs['Prob']<low_thr].copy()
        if thr >=0.5:
            high = predictions_gene[predictions_gene['Case'] > high_thr].copy()
            low = predictions_gene[predictions_gene['Case'] < low_thr].copy()
        elif thr < 0.5:
            high = predictions_gene[predictions_gene['Control'] > high_thr].copy()
            low = predictions_gene[predictions_gene['Control'] < low_thr].copy()
        TP = np.sum(high['True_Class'])
        FP = high.shape[0]-TP
        FN = np.sum(low['True_Class'])
        TN = low.shape[0]-FN

        odr, p, ci = calculate_or_with_ci(TP, FP, FN, TN)
        or_df.loc[i, 'Thr'] = thr
        or_df.loc[i, 'OR'] = odr
        or_df.loc[i, 'CI'] = ci
        or_df.loc[i, 'pval'] = p
        or_df.loc[i, 'Dis w exp'] = TP
        or_df.loc[i, 'HC w exp'] = FP
        or_df.loc[i, 'Dis w/o exp'] = FN
        or_df.loc[i, 'HC w/o exp'] = TN

    no_gene = []
    for th in thrlist:
        try:
            TP = np.where(
                (predictions_gene['True_Class'] == 1) &
                (predictions_gene['Case'] > th)
            )
            TN = np.where(
                (predictions_gene['True_Class'] == 0) &
                (predictions_gene['Case'] < (1-th))
            )
            P = np.where(predictions_gene['Case'] > th)
            N = np.where(predictions_gene['Case'] < (1-th))
            # Correct for testing split size
            no_gene += [((len(P[0])+len(N[0])) /
                         len(predictions_gene['True_Class']))*100]
        except: no_gene += [0]

    # Append Sampling percent to or_df
    or_df['SamplingPercent'] = no_gene

    # Calculate Histogram
    prob_case = predictions_gene[predictions_gene['True_Class'] == 1]
    prob_cont = predictions_gene[predictions_gene['True_Class'] == 0]
    _ = plt.hist(prob_case['Case'], bins=bins)
    _ = plt.hist(prob_cont['Case'], bins=bins)
    # Plot figure
    _, ax = plt.subplots(figsize = (10,10))
    plt.grid(False)
    if (xlim_lower != None) & (xlim_upper != None):
        bin_range = (xlim_lower, xlim_upper)
        ax.set_xlim(bin_range)
    else:
        bin_range = (
            pd.concat([prob_case['Case'], prob_cont['Case']], ignore_index = True).min(),
            pd.concat([prob_case['Case'], prob_cont['Case']], ignore_index = True).max()
        )
        ax.set_xlim(bin_range)

    ax.hist(prob_case['Case'], bins = bins, range = bin_range, color = 'red', alpha = 0.5, label = 'Cases')
    ax.hist(prob_cont['Case'], bins = bins, range = bin_range, color = 'blue', alpha = 0.5, label = 'Controls')
    if ylim[0] != None and ylim[1] != None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("Probability of being Case", size = 16)
    ax.set_ylabel("# of Samples", size = 16)
    plt.legend(loc = 'upper left', frameon = False, labelspacing = 0.01)
    ax2 = ax.twinx()
    plt.grid(False)
    ax2.plot(thrlist, or_df.OR, color = 'orange')
    ax2.set_ylabel("OR", color = 'orange', fontsize = 16)
    if ylim_or[0] != None and ylim_or[1] != None:
        ax2.set_ylim(ylim_or[0], ylim_or[1])
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)

    # Get the KS Test p-value
    cases = predictions_gene[predictions_gene['True_Class'] == 1]
    controls = predictions_gene[predictions_gene['True_Class'] == 0]
    _, ks_p = ks_2samp(cases['Case'], controls['Case'], alternative = 'two-sided')

    return image, or_df, ks_p

def risk_prediction(feature_matrix: pd.DataFrame, train_samples: pd.DataFrame, test_samples: pd.DataFrame, models: Any = 'RF', rfe:bool = False, rfe_min_feature_ratio:float = 0.5, cores:int = 1, savepath: str = None, verbose:int = 0) -> (Image, Image, pd.DataFrame): # type: ignore
    """
    Perform risk prediction using machine learning models.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix containing the input features.
        train_samples (pd.DataFrame): The training samples.
        test_samples (pd.DataFrame): The testing samples.
        models (Any, optional): The machine learning models to use. Defaults to 'RF'.
        rfe (bool, optional): Whether to perform Recursive Feature Elimination (RFE). Defaults to False.
        rfe_min_feature_ratio (float, optional): The minimum feature ratio for RFE. Defaults to 0.5.
        cores (int, optional): The number of CPU cores to use. Defaults to 1.
        savepath (str, optional): The path to save the results. Defaults to None.

    Returns:
        eval_dist_plot (pd.DataFrame): The evaluation distribution plot.
        val_auroc_curve (pd.DataFrame): The validation AUROC curve.
        predictions_gene_df (pd.DataFrame): The predicted gene dataframe.
    """
    # Ensure feature matrix and train/test samples are in correct format
    train_samples = _validate_sample_dfs(train_samples, "Training")
    test_samples = _validate_sample_dfs(test_samples, "Testing")
    _validate_feature_matrix(feature_matrix, train_samples, test_samples)
    # Define the training and testing variables
    x_train, y_train = _generate_x_y_matrices(feature_matrix, train_samples)
    x_test, y_test = _generate_x_y_matrices(feature_matrix, test_samples)

    # Optimize the hyperparameters
    final_estimator, best_model, selected_features, best_optimization_df, rfe_importances, rfe_results, rfe_results_plot, bayes_results, bayes_results_plot, tested_models =  _optimize_features_hyperparameters(models, rfe, x_train, y_train, rfe_min_feature_ratio, cores)

    ## Get intra-sample model performance metrics
    # Cross-Validation
    cv_score = cross_val_score(final_estimator, x_train, y_train, cv=10, scoring='roc_auc')
    # Get Feature Weights
    feature_weights = _save_feature_weights(final_estimator, best_model, x_train, verbose = verbose)

    ## Test the final model
    x_test = x_test.loc[:,selected_features]
    y_pred = final_estimator.predict(x_test)
    acc_score = balanced_accuracy_score(y_test, y_pred)
    predictions_gene = final_estimator.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, predictions_gene[:, 1])
    rocauc_score = auc(fpr, tpr)
    auc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    val_auroc_curve = _plot_auroc(fpr, tpr, rocauc_score)

    # Evaluate the true state v. predicted state
    predictions_gene_df = pd.DataFrame(predictions_gene, columns = ['Control', 'Case'])
    predictions_gene_df['True_Class'] = list(y_test)
    predictions_gene_df.index = x_test.index
    eval_dist_plot, or_df, ks_p = _plot_histo_w_or(predictions_gene_df, bins = 50)

    # Save the feature matrix
    if savepath:
        # Clean Savepath
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'RiskPrediction/')
        os.makedirs(new_savepath, exist_ok=True)

        # Save the bayesian optimization results
        best_optimization_df.to_csv(new_savepath + 'best_optimization_results.csv', index=False)

        # Save training and testing intermediate split files
        input_file_savepath = new_savepath + 'IntermediateFiles/'
        os.makedirs(input_file_savepath, exist_ok=True)
        for df, name in zip([x_train, y_train, x_test, y_test], ['X_train', 'y_train', 'X_test', 'y_test']):
            df.to_csv(input_file_savepath + name + '.csv', index=False)

        # Save final files according to models tested
        for model in tested_models:
            # Create model-specific savepath
            model_savepath = new_savepath + model + '/'
            os.makedirs(model_savepath, exist_ok=True)
            # Save the rfe results
            rfe_savepath = model_savepath + 'RFE/'
            os.makedirs(rfe_savepath, exist_ok=True)
            if rfe_results[model] is not None:
                rfe_results[model].to_csv(rfe_savepath + 'rfe_results.csv', index=False)
            if rfe_importances[model] is not None:
                rfe_importances[model].to_csv(rfe_savepath + 'rfe_importances.csv', index=False)
            if rfe_results_plot[model] is not None:
                rfe_results_plot[model].save(rfe_savepath + 'rfe_results_plot.png')
            # Save the bayesian results
            bayes_savepath = model_savepath + 'BayesianOptimization/'
            os.makedirs(bayes_savepath, exist_ok=True)
            bayes_results[model].to_csv(bayes_savepath + 'bayes_optimization_results.csv', index=False)
            bayes_results_plot[model].save(bayes_savepath + 'bayes_optimization_results_plot.png')
            # Save the best model results
            if model == best_model:
                testing_savepath = model_savepath + 'TestingSamples/'
                os.makedirs(testing_savepath, exist_ok=True)
                # Save the feature weights
                if feature_weights is not None:
                    feature_weights.to_csv(testing_savepath + 'feature_weights.csv', index=True)
                # Save the final model
                with open(testing_savepath + 'final_model.pkl', 'wb') as f:
                    joblib.dump(final_estimator, f)
                # Save the final model predictions
                predictions_gene_df.to_csv(testing_savepath + 'predictions_gene.csv', index=True)
                auc_df.to_csv(testing_savepath + 'auc_df.csv', index=False)
                # Save the final model evaluation
                or_df.to_csv(testing_savepath + 'or_df.csv', index=False)
                eval_dist_plot.save(testing_savepath + 'eval_distribution_plot.png')
                val_auroc_curve.save(testing_savepath + 'eval_auroc_curve.png')
                # Save the final model performance metrics
                with open(testing_savepath + 'model_performance_metrics.txt', 'w') as f:
                    f.write(f'Intra-Training Sample 10x Cross-Validation ROC-AUC: {cv_score.mean()}\n')
                    f.write("\n")
                    f.write(f'Left-Out Sample Balanced Accuracy Score: {acc_score}\n')
                    f.write(f'Left-Out Sample ROC-AUC Score: {rocauc_score}\n')
                    f.write(f'Left-Out KS Test p-value b/w the two distributions: {ks_p}')

    return eval_dist_plot, val_auroc_curve, predictions_gene_df, feature_weights
#endregion


#region odds_ratios
def _validate_or_level(level: str) -> None:
    """
    Validates the input method for association analysis.

    Args:
        method (str): The method to be validated.

    Raises:
        ValueError: If the method is not one of ['variant', 'domain', 'gene'].

    Returns:
        None
    """
    if level not in ['variant', 'domain', 'gene']:
        raise ValueError("Invalid level. Please choose from 'variant', 'domain', or 'gene'.")

def _validate_or_method(method: str) -> None:
    """
    Validates the input method for association analysis.

    Args:
        method (str): The method to be validated.

    Raises:
        ValueError: If the method is not 'sample' or 'allelic'.

    Returns:
        None
    """
    if method not in ['sample', 'allelic']:
        raise ValueError("Invalid method. Please choose from 'sample' or 'allelic'.")

def _validate_model(model: str) -> None:
    """
    Validates the model parameter.

    Args:
        model (str): The model to be validated.

    Raises:
        ValueError: If the model is not 'dominant' or 'recessive'.
    """
    if model not in ['dominant', 'recessive']:
        raise ValueError("Invalid model. Please choose from 'dominant' or 'recessive'.")

def _check_af_filters(af_min: float, af_max:float, method:str) -> None:
    """
    Checks the allele frequency filters for the exact test.

    Args:
        exact_test (pd.DataFrame): The DataFrame containing the exact test results.
        af_min (float): The minimum allele frequency.
        af_max (float): The maximum allele frequency.
        method (str): The method used for association analysis.

    Raises:
        ValueError: If the allele frequency filters are invalid.
    """
    if method == 'variant' and af_min < 0.01:
        warnings.warn("WARNING: Minimum allele frequency is less than 1%. This may result in many underpowered variants being tested, leading to a stringent FDR.")
    elif method in ['domain', 'gene'] and af_max > 0.01:
        warnings.warn("WARNING: Maximum allele frequency is greater than 1%. This may result in an Exact Test error due to allele number calculations. We suggest lowering the max_af to 0.01.")

def _parse_domains(row: pd.Series) -> list:
    """
    Parses the domain information from a given row in a pandas Series object.

    Args:
        row (pd.Series): The row containing the domain information.

    Returns:
        list: A list of dictionaries, where each dictionary represents a parsed domain.
              Each dictionary contains the following keys:
              - 'ENSP': The protein ID.
              - 'domain_start': The start position of the domain.
              - 'domain_end': The end position of the domain.
              - 'domain_name': The name of the domain.
    """
    ensp = row.prot_id.split('.')[0]
    if "ENSP" not in ensp:
        return []
    domain_info = row.domain
    if not isinstance(domain_info, str):
        return []
    # Adjusted regular expression to match domain annotations
    domain_pattern = re.compile(r'DOMAIN (\d+)\.\.(\d+); /note=(".*?"|""".*?""");')
    # Find all matches
    matches = domain_pattern.findall(domain_info)
    parsed_domains = []
    for match in matches:
        domain_start, domain_end, domain_name = match
        domain_name = domain_name.strip('"')
        parsed_domains.append({
            'ENSP': ensp,
            'domain_start': int(domain_start),
            'domain_end': int(domain_end),
            'domain_name': domain_name
        })

    return parsed_domains

def _extract_aa_pos_value(hgvsp: str) -> Any:
    """
    Extracts the amino acid position value from the given HGVSp string.

    Args:
        hgvsp (str): The HGVSp string from which to extract the amino acid position value.

    Returns:
        int or None: The extracted amino acid position value if found, otherwise None.
    """
    match = re.search(r'[A-Za-z]{3}(\d+)[A-Za-z]{3}', hgvsp)
    if match:
        return int(match.group(1))
    return None

def _annotate_domain_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates the given DataFrame with domain information based on variant positions.

    Args:
        df (pd.DataFrame): The input DataFrame containing variant information.

    Returns:
        pd.DataFrame: The annotated DataFrame with domain information.

    """
    # Load domain information
    domain_info = _load_pdb_et_mapping()
    # Create a Protein position column
    df['aa_pos'] = df['HGVSp'].apply(_extract_aa_pos_value)
    # Parse the domain info
    parsed_domains = [domain for index, row in domain_info.iterrows() for domain in _parse_domains(row)]
    parsed_domains_df = pd.DataFrame(parsed_domains)
    # Annotate the dataframe with domains if the variant is in the correct position
    merged_df = pd.merge(df, parsed_domains_df, on='ENSP', how='left')
    # Determine if aa_pos is within the domain range
    merged_df['within_domain'] = merged_df.apply(lambda row: row['domain_start'] <= row['aa_pos'] <= row['domain_end'] if pd.notnull(row['domain_start']) else False, axis=1)
    # Annotate domain names, and keep only the required annotations
    merged_df['domain_name'] = merged_df.apply(lambda row: row['domain_name'] if row['within_domain'] else None, axis=1)
    # Annotated domain region
    merged_df['domain_annotation'] = merged_df.apply(
        lambda row: f"{row['gene']}_{row['domain_name']} (aa{int(row['domain_start'])} - {int(row['domain_end'])})" if row['within_domain'] else None, axis=1)
    # Prioritize rows with domain annotations
    merged_df = merged_df.sort_values(by='domain_annotation', na_position='last')
    # Drop unnecessary columns
    final_df = merged_df.drop(columns=['domain_start', 'domain_end', 'within_domain', 'domain_name']).drop_duplicates(subset=['chr', 'pos', 'ref', 'alt', 'gene', 'ENSP', 'aa_pos'])
    return final_df

def _transform_vbysample_to_exact_test(variants_by_sample: pd.DataFrame, level:str) -> pd.DataFrame:
    """
    Transforms the variants_by_sample DataFrame to perform exact tests based on the specified level.
    - variants_by_sample (pd.DataFrame): The input DataFrame containing variant information.
    - level (str): The level at which the exact tests should be performed. Valid options are 'variant', 'domain', or 'gene'.
    - pd.DataFrame: The transformed DataFrame containing aggregated results based on the specified level.
    - list: The list of unique values in the specified level column.
    - str: The name of the column used for aggregation.
    Raises:
    - ValueError: If an invalid level is provided.
    """
    # Remove non-HGVSp (non-coding) variants
    variants_by_sample = variants_by_sample[variants_by_sample['HGVSp'] != '.']
    # Define allele count aggregation functions
    def max_case_an(x: pd.Series) -> int:
        """
        Returns the maximum value of 'AN_Cohort' from the 'variants_by_sample' DataFrame
        for the indices where 'x' is equal to 1.

        Parameters:
        - x: pandas Series or DataFrame
            The input series or dataframe containing values equal to 1.

        Returns:
        - int or float
            The maximum value of 'AN_Cohort' for the indices where 'x' is equal to 1.
        """
        return variants_by_sample.loc[x.index, 'AN_Cohort'][x == 1].max()

    def max_control_an(x: pd.Series) -> int:
        """
        Returns the maximum value of 'AN_Cohort' from the 'variants_by_sample' DataFrame
        for the indices where the input series 'x' is equal to 0.

        Parameters:
        x (pd.Series): Input series to filter the 'AN_Cohort' values.

        Returns:
        int: Maximum value of 'AN_Cohort' for the filtered indices.
        """
        return variants_by_sample.loc[x.index, 'AN_Cohort'][x == 0].max()

    # Define your aggregation functions
    aggregation_functions = {
        'Case_All_AC': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 1)).sum() + ((x == 1) & (variants_by_sample['zyg'] == 2)).sum() * 2
        ),
        'Case_All_Samples': (
            'CaseControl', lambda x: variants_by_sample.loc[x[x == 1].index, 'sample'].nunique()
        ),
        'Case_Homozygote_AC': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 2)).sum()*2
        ),
        'Case_Homozygote_Samples': (
            'CaseControl', lambda x: variants_by_sample.loc[(x == 1) & (variants_by_sample['zyg'] == 2), 'sample'].nunique()
        ),
        'Case_Heterozygote_AC': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 1)).sum()
        ),
        'Case_Heterozygote_Samples': (
            'CaseControl', lambda x: variants_by_sample.loc[(x == 1) & (variants_by_sample['zyg'] == 1), 'sample'].nunique()
        ),
        'Control_All_AC': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 1)).sum() + ((x == 0) & (variants_by_sample['zyg'] == 2)).sum() * 2
        ),
        'Control_All_Samples': (
            'CaseControl', lambda x: variants_by_sample.loc[x[x == 0].index, 'sample'].nunique()
        ),
        'Control_Homozygote_AC': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 2)).sum()*2
        ),
        'Control_Homozygote_Samples': (
            'CaseControl', lambda x: variants_by_sample.loc[(x == 0) & (variants_by_sample['zyg'] == 2), 'sample'].nunique()
        ),
        'Control_Heterozygote_AC': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 1)).sum()
        ),
        'Control_Heterozygote_Samples': (
            'CaseControl', lambda x: variants_by_sample.loc[(x == 0) & (variants_by_sample['zyg'] == 1), 'sample'].nunique()
        ),
        'Case_AN': (
            'CaseControl', max_case_an
        ),
        'Control_AN': (
            'CaseControl', max_control_an
        )
    }
    if level == 'variant':
        column = 'HGVSp'
        agg_df = variants_by_sample.groupby(column).agg(**aggregation_functions).reset_index()
        # Clean up the agg_df
        variants_sub = variants_by_sample.drop(columns = ['sample', 'CaseControl', 'zyg']).drop_duplicates()
        agg_df = pd.merge(agg_df, variants_sub, on=column, how='left')
        agg_df = agg_df[['chr', 'pos', 'ref', 'alt', 'gene', 'ENSP', 'Consequence', 'HGVSp', 'EA', 'AF', 'Case_All_Samples', 'Case_All_AC', 'Case_Homozygote_Samples', 'Case_Homozygote_AC', 'Case_Heterozygote_Samples', 'Case_Heterozygote_AC', 'Control_All_Samples', 'Control_All_AC', 'Control_Homozygote_Samples', 'Control_Homozygote_AC', 'Control_Heterozygote_Samples', 'Control_Heterozygote_AC', 'Case_AN', 'Control_AN']]
        agg_df = agg_df.drop_duplicates()
        iterable = agg_df.HGVSp.unique().tolist()

    elif level == 'gene':
        column = 'gene'
        agg_df = variants_by_sample.groupby(column).agg(**aggregation_functions).reset_index()
        # Clean up the agg_df
        variants_sub = variants_by_sample.drop(columns = ['sample', 'CaseControl', 'zyg']).drop_duplicates()
        agg_df = pd.merge(agg_df, variants_sub, on=column, how='left')
        # Select columns of interest
        agg_df = agg_df[['gene', 'ENSP', 'Case_All_Samples', 'Case_All_AC', 'Case_Homozygote_Samples', 'Case_Homozygote_AC', 'Case_Heterozygote_Samples', 'Case_Heterozygote_AC', 'Control_All_Samples', 'Control_All_AC', 'Control_Homozygote_Samples', 'Control_Homozygote_AC', 'Control_Heterozygote_Samples', 'Control_Heterozygote_AC', 'Case_AN', 'Control_AN']]
        agg_df = agg_df.drop_duplicates()
        iterable = agg_df.gene.unique().tolist()

    elif level == 'domain':
        column = 'domain_annotation'
        variants_by_sample = _annotate_domain_region(variants_by_sample)
        agg_df = variants_by_sample.groupby(column).agg(**aggregation_functions).reset_index()
        # Clean up the agg_df
        variants_sub = variants_by_sample.drop(columns = ['sample', 'CaseControl', 'zyg']).drop_duplicates()
        agg_df = pd.merge(agg_df, variants_sub, on = column, how='left')
        # Select columns of interest
        agg_df = agg_df[['domain_annotation', 'Case_All_Samples', 'Case_All_AC', 'Case_Homozygote_Samples', 'Case_Homozygote_AC', 'Case_Heterozygote_Samples', 'Case_Heterozygote_AC', 'Control_All_Samples', 'Control_All_AC', 'Control_Homozygote_Samples', 'Control_Homozygote_AC', 'Control_Heterozygote_Samples', 'Control_Heterozygote_AC', 'Case_AN', 'Control_AN']]
        agg_df = agg_df.drop_duplicates()
        iterable = agg_df.domain_annotation.unique().tolist()
    else:
        raise ValueError("Invalid level. Please choose from 'variant', 'domain', or 'gene'.")
    # Clean the dataframe
    return agg_df, iterable, column

def _transform_vbysample_to_exact_test_old(variants_by_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the variants by sample DataFrame to perform an exact test.

    Args:
        variants_by_sample (pd.DataFrame): DataFrame containing variants by sample information.

    Returns:
        pd.DataFrame: Transformed DataFrame with aggregated variant counts.

    """
    # Remove non-HGVSp (non-coding) variants
    variants_by_sample = variants_by_sample[variants_by_sample['HGVSp'] != '.']
    # Define allele count aggregation functions
    def max_case_an(x: pd.Series) -> int:
        """
        Returns the maximum value of 'AN_Cohort' from the 'variants_by_sample' DataFrame
        for the indices where 'x' is equal to 1.

        Parameters:
        - x: pandas Series or DataFrame
            The input series or dataframe containing values equal to 1.

        Returns:
        - int or float
            The maximum value of 'AN_Cohort' for the indices where 'x' is equal to 1.
        """
        return variants_by_sample.loc[x.index, 'AN_Cohort'][x == 1].max()

    def max_control_an(x: pd.Series) -> int:
        """
        Returns the maximum value of 'AN_Cohort' from the 'variants_by_sample' DataFrame
        for the indices where the input series 'x' is equal to 0.

        Parameters:
        x (pd.Series): Input series to filter the 'AN_Cohort' values.

        Returns:
        int: Maximum value of 'AN_Cohort' for the filtered indices.
        """
        return variants_by_sample.loc[x.index, 'AN_Cohort'][x == 0].max()

    aggregation_functions = {
        'Case_All_AC': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 1)).sum() + ((x == 1) & (variants_by_sample['zyg'] == 2)).sum() * 2
        ),
        'Case_All_Samples': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 1)).sum() + ((x == 1) & (variants_by_sample['zyg'] == 2)).sum()
        ),
        'Case_Homozygote_AC': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 2)).sum()*2
        ),
        'Case_Homozygote_Samples': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 2)).sum()
        ),
        'Case_Heterozygote_AC': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 1)).sum()
        ),
        'Case_Heterozygote_Samples': (
            'CaseControl', lambda x: ((x == 1) & (variants_by_sample['zyg'] == 1)).sum()
        ),
        'Control_All_AC': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 1)).sum() + ((x == 0) & (variants_by_sample['zyg'] == 2)).sum() * 2
        ),
        'Control_All_Samples': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 1)).sum() + ((x == 0) & (variants_by_sample['zyg'] == 2)).sum()
        ),
        'Control_Homozygote_AC': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 2)).sum()*2
        ),
        'Control_Homozygote_Samples': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 2)).sum()
        ),
        'Control_Heterozygote_AC': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 1)).sum()
        ),
        'Control_Heterozygote_Samples': (
            'CaseControl', lambda x: ((x == 0) & (variants_by_sample['zyg'] == 1)).sum()
        ),
        'Case_AN': (
            'CaseControl', max_case_an
        ),
        'Control_AN': (
            'CaseControl', max_control_an
        )
    }
    # Aggregate by sample counts into variant counts
    agg_df = variants_by_sample.groupby('HGVSp').agg(**aggregation_functions).reset_index()
    # Merge back with original data
    variants_sub = variants_by_sample.drop(columns = ['sample', 'CaseControl', 'zyg']).drop_duplicates()
    agg_df = pd.merge(agg_df, variants_sub, on='HGVSp', how='left')
    # Select columns of interest
    agg_df = agg_df[['chr', 'pos', 'ref', 'alt', 'gene', 'ENSP', 'Consequence', 'HGVSp', 'EA', 'AF', 'Case_All_Samples', 'Case_All_AC', 'Case_Homozygote_Samples', 'Case_Homozygote_AC', 'Case_Heterozygote_Samples', 'Case_Heterozygote_AC', 'Control_All_Samples', 'Control_All_AC', 'Control_Homozygote_Samples', 'Control_Homozygote_AC', 'Control_Heterozygote_Samples', 'Control_Heterozygote_AC', 'Case_AN', 'Control_AN']]
    agg_df = agg_df.drop_duplicates()
    return agg_df

def _or_fdr(or_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the False Discovery Rate (FDR) correction to the p-values in the input DataFrame.

    Args:
        or_matrix (pd.DataFrame): A DataFrame containing the odds ratio (OR) values and p-values.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'qvalue' representing the FDR-corrected p-values.

    """
    or_matrix = or_matrix.sort_values(by=['pvalue'], ascending=True)
    pvals = list(or_matrix['pvalue'])
    if len(pvals) != 0:
        qvals = multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=True)[1]
        or_matrix['qvalue'] = qvals
    return or_matrix

def _perform_or(exact_test_df: pd.DataFrame, iterable: str, it_column:str, model_column:str, n_case:int, n_control:int, verbose: int = 0) -> pd.DataFrame:
    """
    Perform odds ratio calculation and confidence interval estimation for a given iterable.

    Args:
        exact_test_df (pd.DataFrame): DataFrame containing the exact test data.
        iterable (str): The iterable to perform the odds ratio calculation for.
        it_column (str): The column in `exact_test_df` that contains the iterable values.
        model_column (str): The column in `exact_test_df` that contains the model values.

    Returns:
        pd.DataFrame: DataFrame containing the odds ratio, confidence intervals, p-value, and counts.

    """
    # Get the iterable
    sub_df = exact_test_df[exact_test_df[it_column] == iterable].reset_index(drop=True)
    # Get allele counts in cases and controls
    case_with = int(sum(sub_df[f'Case{model_column}']))
    control_with = int(sum(sub_df[f'Control{model_column}']))
    # Perform odds ratio for allelic testing
    if 'AC' in model_column:
        it_column_prefix = 'AC'
        try:
            case_without = math.floor(np.mean(sub_df['Case_AN']) - case_with)
            control_without = math.floor(np.mean(sub_df['Control_AN']) - control_with)
            oddsratio, pvalue = fisher_exact(
                [[case_with, control_with],
                [case_without, control_without]]
            )
        except ValueError as e:
            e = str(e)
            if "nonnegative" in e:
                if verbose > 0:
                    warnings.warn("WARNING: Unable to calculate odds ratio due to negative allele number values. This is due to allele frequency issues. Please decrease the max_af threshold to 0.01.")
            elif "zero" in e:
                if verbose > 0:
                    warnings.warn("WARNING: Unable to calculate odds ratio due to zero allele number values.")
            else:
                if verbose > 0:
                    warnings.warn("WARNING: Unable to calculate odds ratio.")
            output = pd.DataFrame({
                'genomic_object': [iterable],
                'OR': [1],
                'LowerCI': [1],
                'UpperCI': [1],
                f'case_with_{it_column_prefix}': [np.nan],
                f'control_with_{it_column_prefix}': [np.nan],
                f'case_without_{it_column_prefix}': [np.nan],
                f'control_without_{it_column_prefix}': [np.nan],
                'pvalue': [1]
            })
            return output
    # Perform odds ratio for sample-wise testing
    else:
        it_column_prefix = 'samples'
        case_without = n_case - case_with
        control_without = n_control - control_with
        try:
            oddsratio, pvalue = fisher_exact(
                [[case_with, control_with],
                [case_without, control_without]]
            )
        except ValueError as e:
            e = str(e)
            if "nonnegative" in e:
                if verbose > 0:
                    warnings.warn("WARNING: Unable to calculate odds ratio due to negative allele number values. This is due to allele frequency issues. Please decrease the max_af threshold to 0.01.")
            elif "zero" in e:
                if verbose > 0:
                    warnings.warn("WARNING: Unable to calculate odds ratio due to zero allele number values.")
            else:
                if verbose > 0:
                    warnings.warn("WARNING: Unable to calculate odds ratio.")
            output = pd.DataFrame({
                'genomic_object': [iterable],
                'OR': [1],
                'LowerCI': [1],
                'UpperCI': [1],
                f'case_with_{it_column_prefix}': [np.nan],
                f'control_with_{it_column_prefix}': [np.nan],
                f'case_without_{it_column_prefix}': [np.nan],
                f'control_without_{it_column_prefix}': [np.nan],
                'pvalue': [1]
            })
            return output

    # Invert OR if AF > 0.5 because its actually not the variant but the reference allele
    if (it_column == 'HGVSp') and (sub_df.loc[0, 'AF'] >= 0.5):
        try:
            oddsratio = 1/oddsratio
            a = control_with
            b = case_with
            c = control_without
            d = case_without
        except ZeroDivisionError: oddsratio = np.nan
    else:
        a = case_with
        b = control_with
        c = case_without
        d = control_without
     # Calculate CIs
    if np.sum(case_with) == 0 or np.sum(control_with) == 0 or case_without == 0 or control_without == 0:
        upper_ci = np.nan
        lower_ci = np.nan
    else:
        upper_ci = np.exp(
            np.log(oddsratio) + 1.96*np.sqrt((1/np.sum(a)) + (1/np.sum(b)) + (1/c) + (1/d)))
        lower_ci = np.exp(
            np.log(oddsratio) - 1.96*np.sqrt((1/np.sum(a)) + (1/np.sum(b)) + (1/c) + (1/d)))
        upper_ci = round(upper_ci, 3)
        lower_ci = round(lower_ci, 3)
    # Create output matrix
    output = pd.DataFrame({
        'genomic_object': [iterable],
        'OR': [oddsratio],
        'LowerCI': [lower_ci],
        'UpperCI': [upper_ci],
        f'case_with_{it_column_prefix}': [case_with],
        f'control_with_{it_column_prefix}': [control_with],
        f'case_without_{it_column_prefix}': [case_without],
        f'control_without_{it_column_prefix}': [control_without],
        'pvalue': [pvalue]
    })
    return output

def _parallel_or(exact_test_df: pd.DataFrame, iterable: list, it_column:str, model_column: str, n_case:int, n_control:int, cores: int, verbose: int = 0) -> pd.DataFrame:
    """
    Perform parallelized OR calculation for each item in the iterable.

    Args:
        exact_test_df (pd.DataFrame): The DataFrame containing the exact test results.
        iterable (list): The list of items to iterate over.
        it_column (str): The column name in the DataFrame representing the iterable.
        model_column (str): The column name in the DataFrame representing the model.
        cores (int): The number of CPU cores to use for parallelization.

    Returns:
        pd.DataFrame: The DataFrame containing the calculated OR values.

    """
    # Set up the parallelization
    args_ = tuple(zip(
        [exact_test_df]*len(iterable),
        iterable,
        [it_column] * len(iterable),
        [model_column]*len(iterable),
        [n_case]*len(iterable),
        [n_control]*len(iterable),
        [verbose]*len(iterable)
    ))
    pool = mp.Pool(processes = cores)
    output = pool.starmap(_perform_or, args_)
    pool.close()
    pool.join()
    output = pd.concat(output)
    output = _or_fdr(output)
    return output

def _plot_or(or_df: pd.DataFrame, sig_level: float, show_plot_labels: bool, text_col:str) -> Image:
    """
    Plot Odds Ratio data.

    Args:
        or_df (pd.DataFrame): DataFrame containing Odds Ratio data.
        sig_level (float): Significance level.
        show_plot_labels (bool): Whether to show text labels for points of interest.
        text_col (str): Column name in the DataFrame to be used as text labels.

    Returns:
        Image: The plotted image.

    """
    # Create plotting matrices
    if text_col == 'HGVSp':
        or_df[['ENSP', 'variant']] = or_df['genomic_object'].str.split(':', expand=True)
        or_df['genomic_object'] = or_df['gene'] + ':' + or_df['variant']
    or_df_sort = or_df.sort_values(by='OR', ascending=True).reset_index(drop=True)
    sig_pos_matrix = or_df_sort[(or_df_sort['OR'] > 1) & (or_df_sort['qvalue'] <= sig_level)]
    sig_neg_matrix = or_df_sort[(or_df_sort['OR'] < 1) & (or_df_sort['qvalue'] <= sig_level)]
    non_sig_matrix = or_df_sort[(or_df_sort['qvalue'] > sig_level)]

    # Plotting sorted Odds Ratio data
    _, ax = plt.subplots(figsize=(5,5))
    # Plot scatter plots
    ax.scatter(non_sig_matrix.index, non_sig_matrix.OR, color='gray')
    ax.scatter(sig_pos_matrix.index, sig_pos_matrix.OR, color = 'red', label='OR > 1, q < 0.1')
    ax.scatter(sig_neg_matrix.index, sig_neg_matrix.OR, color = 'blue', label = 'OR < 1, q < 0.1')
    ax.fill_between(sig_pos_matrix.index, (sig_pos_matrix.UpperCI), (sig_pos_matrix.LowerCI), color='r', alpha=0.2)
    ax.fill_between(sig_neg_matrix.index, (sig_neg_matrix.LowerCI), (sig_neg_matrix.UpperCI), color='b', alpha=0.2)

    # Adjust plot settings
    ax.tick_params(labelbottom = False, bottom = False)
    ax.tick_params(axis='y', labelsize=16)
    ax.axhline(y=1, color='black', linestyle='--')
    ax.legend(fontsize = 12, loc = 'upper left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('Odds Ratio', fontsize=16)

    # Defining and adjusting the text labels for points of interest
    if show_plot_labels:
        texts = []
        for i, row in sig_pos_matrix.iterrows():
            texts.append(ax.text(row.name, row.OR, row.genomic_object, fontsize=8, color='red'))
        for i, row in sig_neg_matrix.iterrows():
            texts.append(ax.text(row.name, row.OR, row.genomic_object, fontsize=8, color='blue'))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def odds_ratios(variants_by_sample: pd.DataFrame, samples: pd.DataFrame, query: list = [], method: str = 'sample', model:str = 'dominant', level: str = 'variant', consequence:str = "missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost", ea_lower:int = 0, ea_upper:int = 100, min_af:float = 0.0, max_af:float = 1.0, significance_level: float = 0.1, show_plot_labels:bool = True, cores: int = 1, savepath: str = None, verbose:int = 0) -> (pd.DataFrame, pd.DataFrame, Image): #type: ignore
    """
    Calculate odds ratios for genetic variants based on different association methods.

    Parameters:
    - variants_by_sample (pd.DataFrame): DataFrame containing variant information for each sample.
    - samples (pd.DataFrame): DataFrame containing sample information.
    - genes (list): List of genes to include in the analysis. Default is an empty list.
    - model (str): The genetic model to use for odds ratio calculations. Default is 'dominant'.
    - method (str): The association method to use. Default is 'variant'.
    - ea_lower (int): The lower threshold for effect size (EA). Default is 0.
    - ea_upper (int): The upper threshold for effect size (EA). Default is 100.
    - min_af (float): The minimum allele frequency (AF) threshold. Default is 0.0.
    - max_af (float): The maximum allele frequency (AF) threshold. Default is 1.0.
    - significance_level (float): The significance level for plotting. Default is 0.1.
    - show_plot_labels (bool): Whether to show labels on the odds ratio plot. Default is True.
    - cores (int): The number of CPU cores to use for parallel processing. Default is 1.
    - savepath (str): The path to save the results. Default is None.

    Returns:
    - exact_test_format (pd.DataFrame): DataFrame containing the filtered and transformed variant data.
    - or_df (pd.DataFrame): DataFrame containing the odds ratio results.
    - or_plot (Image): Image object of the odds ratio plot.
    """
    # Validate the inputs
    _validate_or_level(level)
    _validate_or_method(method)
    _validate_ea_thresh(ea_lower, ea_upper)
    _validate_af_thresh(min_af, max_af)
    _validate_model(model)
    _check_af_filters(min_af, max_af, level)

    # Filter variants_by_sample
    variants_by_sample = variants_by_sample[
        (variants_by_sample['EA'] >= ea_lower) &
        (variants_by_sample['EA'] <= ea_upper) &
        (variants_by_sample['AF'] >= min_af) &
        (variants_by_sample['AF'] <= max_af) &
        (variants_by_sample['Consequence'].str.contains(consequence, na=False)) &
        (variants_by_sample['sample'].isin(samples.iloc[:, 0])) &
        (variants_by_sample['gene'].isin(query) if query else True)
    ]
    n_case = samples[samples.iloc[:, 1] == 1].shape[0]
    n_control = samples[samples.iloc[:, 1] == 0].shape[0]

    # Transform the variants by sample data
    exact_test_format, iterable, column = _transform_vbysample_to_exact_test(variants_by_sample, level)

    # Check the model
    if model == 'dominant' and method == 'sample':
        model_column = '_All_Samples'
    elif method == 'allelic':
        model_column = '_All_AC'
    elif model == 'recessive' and method == 'sample':
        model_column = '_Homozygote_Samples'
    else:
        raise ValueError("Invalid model for the given method.")

    # Perform the odds ratio calculations
    or_df = _parallel_or(exact_test_format, iterable, column, model_column, n_case, n_control, cores, verbose = verbose)
    # Clean up the resulting dataframe
    if level == 'variant':
        or_df = pd.merge(exact_test_format[['HGVSp', 'gene', 'EA', 'AF']], or_df, left_on = 'HGVSp', right_on = 'genomic_object', how = 'right')
        or_df = or_df.drop(columns = ['genomic_object'])
        or_df = or_df.rename(columns = {'HGVSp': 'genomic_object'})
    elif level == 'domain':
        or_df['gene'] = or_df['genomic_object'].str.split('_').str[0]
    else:
        or_df['gene'] = or_df['genomic_object']
    or_plot = _plot_or(or_df, sig_level = significance_level, show_plot_labels = show_plot_labels, text_col = column)

    if savepath:
        savepath = _fix_savepath(savepath)
        if method == 'sample':
            new_savepath = os.path.join(savepath, f'OddsRatios/{level}_{method}_{model}_EA-{ea_lower}-{ea_upper}_AF-{float(min_af)}-{float(max_af)}_{consequence}/')
        else:
            new_savepath = os.path.join(savepath, f'OddsRatios/{level}_{method}_EA-{ea_lower}-{ea_upper}_AF-{float(min_af)}-{float(max_af)}_{consequence}/')
        os.makedirs(new_savepath, exist_ok=True)
        # Save the exact test format
        exact_test_format.to_csv(new_savepath + 'exact_test_format.csv', index=False)
        # Save the odds ratio results
        or_df.to_csv(new_savepath + 'odds_ratio_results.csv', index=False)
        # Save the odds ratio plot
        or_plot.save(new_savepath + 'odds_ratio_plot.png', bbox_inches='tight')
    return or_df, or_plot
#endregion


#region ea_distributions
def _validate_distribution(distribution:str) -> None:
    """
    Validates the distribution parameter.

    Parameters:
    distribution (str): The distribution to be validated.

    Raises:
    ValueError: If the distribution is not 'degenerate' or 'uniform'.

    Returns:
    None
    """
    if distribution not in ['non_degenerate', 'degenerate']:
        raise ValueError("Distribution must be either 'degenerate' or 'non_degenerate'.")

def _fetch_ea_by_distribution(exact_test_format: pd.DataFrame, gene:str, min_vars:int, distribution: str, min_af: float, max_af: float, consequence: str) -> (np.array, np.array): # type: ignore
    """
    Fetches the effect sizes based on the distribution.

    Parameters:
    exact_test_format (pd.DataFrame): The DataFrame containing the exact test results.
    distribution (str): The distribution to fetch the effect sizes from.
    min_af (float): The minimum allele frequency threshold.
    max_af (float): The maximum allele frequency threshold.
    ea_lower (int): The lower threshold for effect size.
    ea_upper (int): The upper threshold for effect size.

    Returns:
    pd.DataFrame: The DataFrame containing the effect sizes based on the distribution.
    """
    exact_test_sub = exact_test_format[
            (exact_test_format['AF'] >= min_af) &
            (exact_test_format['AF'] <= max_af) &
            (exact_test_format['gene'] == gene) &
            (exact_test_format['Consequence'].str.contains(consequence, na=False))
        ]
    if exact_test_sub.shape[0] < min_vars:
        return np.array([]), np.array([])

    # Grab EA scores based on distribution
    if distribution == 'non_degenerate':
        case_ea_het = exact_test_sub['EA'][(exact_test_sub['CaseControl'] == 1) & (exact_test_sub['zyg'] == 1)]
        case_ea_hom = exact_test_sub['EA'][(exact_test_sub['CaseControl'] == 1) & (exact_test_sub['zyg'] == 2)]
        case_ea = pd.concat([case_ea_het, case_ea_hom, case_ea_hom])
        control_ea_het = exact_test_sub['EA'][(exact_test_sub['CaseControl'] == 0) & (exact_test_sub['zyg'] == 1)]
        control_ea_hom = exact_test_sub['EA'][(exact_test_sub['CaseControl'] == 0) & (exact_test_sub['zyg'] == 2)]
        control_ea = pd.concat([control_ea_het, control_ea_hom, control_ea_hom])
    else:
        case_ea = exact_test_sub[exact_test_sub['CaseControl'] == 1].drop_duplicates(subset = ['EA', 'HGVSp'])['EA']
        control_ea = exact_test_sub[exact_test_sub['CaseControl'] == 0].drop_duplicates(subset = ['EA', 'HGVSp'])['EA']
    return case_ea, control_ea

def _plot_ea_distribution(case_ea: np.array, control_ea: np.array, bins: int, xlim: tuple, gene:str, distribution:str, ks_p:float) -> Image:
    """
    Plot the effect size distribution.

    Parameters:
    case_ea (np.array): The effect sizes for the case samples.
    control_ea (np.array): The effect sizes for the control samples.
    bins (int): The number of bins for the histogram.
    xlim (tuple): The x-axis limits.
    ylim (tuple): The y-axis limits.
    gene (str): The gene name.

    Returns:
    Image: The plotted image.
    """
    _, ax = plt.subplots(figsize=(5,5))
    range_hist = (0,100)
    case_weights = np.ones_like(case_ea) / len(case_ea)
    ax.hist(case_ea, bins = bins, range = range_hist, weights = case_weights, alpha = 0.5, color = 'red', label = 'Case')
    control_weights = np.ones_like(control_ea) / len(control_ea)
    ax.hist(control_ea, bins = bins, range = range_hist, weights = control_weights, alpha = 0.5, color = 'blue', label = 'Control')
    ax.set_xlim(xlim)
    ax.set_xlabel('EA Score', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.legend(fontsize=12)
    # Add title
    distribution_fix = "Degenerate" if distribution == "degenerate" else "Non-Degenerate"
    ax.set_title(f'{gene} - {distribution_fix}', fontsize=16)
    # Add KS tes
    # t p-value to top right corner
    ax.text(0.75, 0.8, f'KS Test p-value: {round(ks_p, 3)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
    plt.tight_layout()
    # Save the plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def ea_distributions(variants_by_sample: pd.DataFrame, genes:list, min_vars: int = 1, distribution: str = "non_degenerate", consequence: str = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost', min_af: float = 0.0, max_af: float = 1.0, savepath: str = None) -> (pd.DataFrame, dict): # type: ignore
    """
    Perform association analysis by comparing effect size distributions between case and control groups for multiple genes.

    Parameters:
    - variants_agg (pd.DataFrame): The input dataframe containing the exact test results.
    - genes (list): The list of genes to analyze.
    - min_vars (int, optional): The minimum number of variants required for a gene to be included in the analysis. Default is 1.
    - distribution (str, optional): The type of distribution to use for the analysis. Default is "non_degenerate".
    - consequence (str, optional): The consequence type of variants to consider. Default is 'missense_variant|frameshift_variant|stop_gained'.
    - min_af (float, optional): The minimum allele frequency threshold for variants to be included in the analysis. Default is 0.0.
    - max_af (float, optional): The maximum allele frequency threshold for variants to be included in the analysis. Default is 1.0.
    - savepath (str, optional): The path to save the results. Default is None.

    Returns:
    - p_values_df (pd.DataFrame): A dataframe containing the p-values and FDR-corrected p-values for each gene.
    - plots (dict): A dictionary containing the plots of effect size distributions for each gene.

    """
    # Validate the inputs
    _validate_distribution(distribution)

    # Get the distributions
    p_values = {}
    plots = {}
    for gene in genes:
        case_ea, control_ea = _fetch_ea_by_distribution(variants_by_sample, gene, min_vars, distribution, min_af, max_af, consequence)
        # Test distributions by 2 sample KS test
        try: _, ks_p = ks_2samp(case_ea, control_ea, alternative = 'two-sided')
        except ValueError: _, ks_p = np.nan, np.nan
        # Plot the distributions
        plot = _plot_ea_distribution(case_ea, control_ea, bins = 20, xlim = (0, 100), gene = gene, distribution = distribution, ks_p = ks_p)
        # Save the results
        p_values[gene] = ks_p
        plots[gene] = plot

    # Create a dataframe and FDR correct the p-values
    p_values_df = pd.DataFrame(p_values.items(), columns = ['Gene', 'KS_p'])
    p_values_df['FDR'] = multipletests(p_values_df['KS_p'], alpha = 0.05, method = 'fdr_bh')[1]

    # Save values
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'EA_Distributions/{distribution}_AF-{min_af}-{max_af}_{consequence}/')
        os.makedirs(new_savepath, exist_ok=True)
        # Save the p-values
        p_values_df.to_csv(new_savepath + 'p_values.csv', index=False)
        # Save the plots
        for gene, plot in plots.items():
            plot_savepath = os.path.join(new_savepath, "Plots/")
            os.makedirs(plot_savepath, exist_ok=True)
            plot.save(plot_savepath + f'{gene}_EA_Distribution.png')

    return p_values_df, plots
#endregion
