"""
Structure.py
This file contains the functions for creating lollipop plots and analyzing the structures of proteins.
"""

import os
conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')
r_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R')
r_libs_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R', 'library')
os.environ['R_HOME'] = r_path

import tempfile
from IPython.display import Image
import rpy2 # type: ignore
from rpy2.robjects.packages import importr, isinstalled # type: ignore
import rpy2.robjects as robjects # type: ignore
from rpy2.robjects import r, pandas2ri, globalenv # type: ignore
from rpy2.robjects.conversion import localconverter # type: ignore
from rpy2.robjects.vectors import StrVector # type: ignore
import rpy2.rinterface_lib.callbacks # type: ignore
import logging
import pandas as pd
import re
import urllib.request
import numpy as np
import time
import statsmodels.api as sm
import warnings
import pkg_resources
import multiprocessing as mp
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from .utils import _fix_savepath, _filter_variants, _clean_variant_formats, _load_pdb_et_mapping, _check_ensp_len

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in sqrt")
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)


#region Lollipop Plots
def _fishers_exact_test(case_vars: pd.DataFrame, cont_vars: pd.DataFrame, case_pop:int, cont_pop:int) -> (float, float, float, float): # type: ignore
    """
    Perform Fisher's exact test to calculate odds ratio, confidence interval, and p-value.

    Args:
        case_vars (pd.DataFrame): DataFrame containing case variables.
        cont_vars (pd.DataFrame): DataFrame containing control variables.
        case_pop (int): Total population of cases.
        cont_pop (int): Total population of controls.

    Returns:
        tuple: A tuple containing the odds ratio, lower confidence interval, upper confidence interval, and p-value.
    """
    # Create contingency table based on sample-wise counts
    obs_case_count = len(case_vars['sample'][case_vars['zyg'] != 0].unique())
    obs_cont_count = len(cont_vars['sample'][cont_vars['zyg'] != 0].unique())
    exp_case_count = case_pop - obs_case_count
    exp_cont_count = cont_pop - obs_cont_count
    contingency_table = [[obs_case_count, obs_cont_count], [exp_case_count, exp_cont_count]]
    # Perform Fisher's exact test
    oddsratio_data = sm.stats.Table2x2(contingency_table)
    odds_ratio = oddsratio_data.oddsratio
    lower_ci, upper_ci = oddsratio_data.oddsratio_confint()
    pval = oddsratio_data.oddsratio_pvalue()
    return odds_ratio, lower_ci, upper_ci, pval

def _r_install_package(package_name:str) -> None:
    """
    Attempts to install an R package and its dependencies.

    Parameters:
    - package_name: The name of the package to install.
    """
    # Define the CRAN repository URL
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    os.environ['R_REMOTES_NO_ERRORS_FROM_WARNINGS'] = 'true'
    os.environ['R_REMOTES_UPGRADE'] = "never"

    # Install the package if it is not already installed
    if not isinstalled(package_name):
        try:
            if package_name == 'EvoTrace':
                remote = importr('remotes')
                remote.install_github('LichtargeLab/EvoTrace', build_vignettes=False)
            else:
                utils.install_packages(StrVector([package_name]))
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            warnings.warn(f"An error occurred while installing {package_name}: {e}")
            # Attempt to install system dependencies if the package is known to require compilation
            if package_name in ['interp']:
                try:
                    utils.install_packages(StrVector([package_name]))
                except Exception as e:
                    raise ValueError(f"Failed to install {package_name} after attempting to resolve dependencies: {e}")

def _install_evotrace() -> None:
    """
    Installs EvoTrace package.
    """
    # Set CRAN mirror
    cran_mirror = "https://cran.rstudio.com"
    robjects.r.options(repos=cran_mirror)
    # Install R packages
    _r_install_package('remotes')
    _r_install_package('EvoTrace')

def _check_pdb_id(protein_id:str) -> str:
    """
    Checks if the given protein ID exists in the PDB mapping dataframe.

    Parameters:
    protein_id (str): The protein ID to be checked.

    Returns:
    str: The validated protein ID.

    Raises:
    ValueError: If the protein ID is not found in the PDB mapping.
    """
    # Load pdb mapping dataframe
    pdb_df = _load_pdb_et_mapping()
    # Check if protein ID in dataframe
    if protein_id not in pdb_df['prot_id'].values:
        protein_id = protein_id.split('.')[0]
        pdb_sub = pdb_df[pdb_df['prot_id'].str.contains(protein_id)]
        if pdb_sub.shape[0] == 0:
            raise ValueError("Protein ID not found in PDB mapping.")
        protein_id = pdb_sub['prot_id'].values[0]
    return protein_id

def _r_lollipop_plot2(case_vars: pd.DataFrame, cont_vars: pd.DataFrame, gene: str, plot_domain:bool, ac_scale:str, ea_color:str, domain_min_dist:int, verbose:int = 0) -> PILImage.Image:
    """
    Generate a lollipop plot using R's EvoTrace package.

    Args:
        case_vars (pd.DataFrame): DataFrame containing case variants.
        cont_vars (pd.DataFrame): DataFrame containing control variants.
        plot_domain (bool): Flag indicating whether to plot the protein domain.
        ac_scale (str): Scale for the allele count axis. Must be one of 'linear' or 'log'.
        ea_color (str): Color scheme for the effect allele. Must be one of 'prismatic', 'gray_scale', 'EA_bin', or 'black'.
        domain_min_dist (int): Minimum distance between two domains.

    Returns:
        PIL.Image.Image: The generated lollipop plot as a PIL Image object.
    """
    # Install EvoTrace package
    _install_evotrace()
    evotrace = importr('EvoTrace')
    # Convert local DataFrames to R DataFrames
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_case_vars = robjects.conversion.py2rpy(case_vars)
        r_cont_vars = robjects.conversion.py2rpy(cont_vars)
    # Check options
    ensp_hold = _check_ensp_len(case_vars.ENSP.unique().tolist(), verbose = verbose)
    if ea_color not in ['prismatic', 'gray_scale', 'EA_bin', 'black']:
        raise ValueError("ea_color must be one of 'prismatic', 'gray_scale', 'EA_bin', or 'black'")
    if ac_scale not in ['linear', 'log']:
        raise ValueError("ac_scale must be one of 'linear' or 'log'")
    # Create temporary directory
    temp_file = tempfile.NamedTemporaryFile(delete = False, suffix = ".png")
    plot_path = temp_file.name

    # Create the lollipop plot
    for pos in ensp_hold:
        prot_id = _check_pdb_id(pos)
        try:
            globalenv['plot'] = evotrace.LollipopPlot2(
                variants_case = r_case_vars,
                variants_ctrl = r_cont_vars,
                prot_id = prot_id,
                plot_domain = plot_domain,
                AC_scale = ac_scale,
                show_EA_bin = True,
                fix_scale = True,
                EA_color = ea_color,
                domain_min_dist = domain_min_dist
            )
            ran = True
            continue
        except Exception as e:
            if "missing value where TRUE/FALSE needed" in str(e):
                print(f'{prot_id} -- {gene} does not match length of input variants')
            elif "unmatched residues" in str(e):
                print(f"{prot_id} -- {gene} has mismatching residues: {e}")
            else:
                print(f"Error: {e}")
            ran = False
    if ran == False:
        warnings.warn(f"No ENSP IDs match")
        return PILImage.new('RGB', (1, 1))
    # Save the plot
    r(f"""ggsave("{plot_path}", plot, device = "png", width = 10, height = 5, dpi = 300)""")
    # Display the plot
    lollipop_plot_plot = Image(filename=plot_path)
    os.unlink(plot_path)
    temp_file.close()
    image_buffer = BytesIO(lollipop_plot_plot.data)
    lollipop_plot_plot = PILImage.open(image_buffer)
    return lollipop_plot_plot

def _r_lollipop_plot1(input_vars: pd.DataFrame, gene:str, plot_domain:bool, ac_scale:str, ea_color:str, domain_min_dist:int, verbose:int = 0) -> PILImage.Image:
    """
    Generate a lollipop plot using the EvoTrace package.

    Args:
        input_vars (pd.DataFrame): A DataFrame containing input variables.
        plot_domain (bool): Whether to plot the domain.
        ac_scale (str): The scale for the AC (Allele Count) axis. Must be one of 'linear' or 'log'.
        ea_color (str): The color scheme for the EA (Effect Allele) axis. Must be one of 'prismatic', 'gray_scale', 'EA_bin', or 'black'.
        domain_min_dist (int): The minimum distance between two domains.

    Returns:
        PILImage.Image: The generated lollipop plot as a PIL Image object.
    """
    # Install EvoTrace package
    _install_evotrace()
    evotrace = importr('EvoTrace')
    # Convert local DataFrames to R DataFrames
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_input_vars = robjects.conversion.py2rpy(input_vars)
    # Check options
    ensp_hold = _check_ensp_len(input_vars.ENSP.unique(), verbose = verbose)
    # Check values
    if ea_color not in ['prismatic', 'gray_scale', 'EA_bin', 'black']:
        raise ValueError("ea_color must be one of 'prismatic', 'gray_scale', 'EA_bin', or 'black'")
    if ac_scale not in ['linear', 'log']:
        raise ValueError("ac_scale must be one of 'linear' or 'log'")

    # Create temporary directory
    temp_file = tempfile.NamedTemporaryFile(delete = False, suffix = ".png")
    plot_path = temp_file.name

    ran = False
    for pos in ensp_hold:
        prot_id = _check_pdb_id(pos)
        # Create the lollipop plot
        try:
            globalenv['plot'] = evotrace.LollipopPlot(
                variants = r_input_vars,
                prot_id = prot_id,
                plot_domain = plot_domain,
                AC_scale = ac_scale,
                show_EA_bin = True,
                fix_scale = True,
                EA_color = ea_color,
                domain_min_dist = domain_min_dist
            )
            ran = True
            continue
        except Exception as e:
            if "missing value where TRUE/FALSE needed" in str(e):
                print(f'{prot_id} -- {gene} does not match length of input variants')
            elif "unmatched residues" in str(e):
                print(f"{prot_id} -- {gene} has mismatching residues: {e}")
            else:
                print(f"Error: {e}")
            ran = False
    if ran == False:
        warnings.warn(f"No ENSP IDs match")
        return PILImage.new('RGB', (1, 1))
    # Save the plot
    r(f"""ggsave("{plot_path}", plot, device = "png", width = 10, height = 5, dpi = 300)""")
    # Display the plot
    lollipop_plot_plot = Image(filename=plot_path)
    os.unlink(plot_path)
    temp_file.close()
    image_buffer = BytesIO(lollipop_plot_plot.data)
    lollipop_plot_plot = PILImage.open(image_buffer)
    return lollipop_plot_plot

def lollipop_plot(variants: pd.DataFrame, gene: str, group:str = 'both', case_pop:int=0, cont_pop:int=0, max_af:float = 1.0, min_af:float = 0.0, ea_lower:float = 0.0, ea_upper:float = 100.0, consequence: str = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost', show_domains:bool = True, ac_scale:str = 'linear', ea_color:str = 'prismatic', domain_min_dist:int = 20, savepath:str = False, verbose: int = 0) -> (Image, float, float, float, float): # type: ignore
    """
    Generate a lollipop plot for a given gene based on variant data.

    Parameters:
    - variants (pd.DataFrame): DataFrame containing variant data.
    - gene (str): Gene of interest.
    - case_pop (int, optional): Number of cases in the variant file. If not provided, it will be calculated based on the data.
    - cont_pop (int, optional): Number of controls in the variant file. If not provided, it will be calculated based on the data.
    - max_af (float, optional): Maximum allele frequency threshold for filtering variants. Default is 1.0.
    - ea_lower (float, optional): Lower bound for effect size threshold. Default is 0.0.
    - ea_upper (float, optional): Upper bound for effect size threshold. Default is 100.0.
    - show_domains (bool, optional): Whether to show domain annotations on the lollipop plot. Default is True.
    - ac_scale (str, optional): Scale for the allele count axis. Default is 'linear'.
    - ea_color (str, optional): Color scheme for effect size. Default is 'prismatic'.
    - domain_min_dist (int, optional): Minimum distance between domain annotations. Default is 20.
    - savepath (str, optional): Path to save the lollipop plot image. If not provided, the plot will not be saved.

    Returns:
    - plot (Image): Lollipop plot image.
    - pval (float): p-value from Fisher's exact test.
    - odds_ratio (float): odds ratio from Fisher's exact test.
    - lower_ci (float): lower confidence interval from Fisher's exact test.
    - upper_ci (float): upper confidence interval from Fisher's exact test.
    """
    # Check the number of samples
    if (case_pop == 0) & (cont_pop == 0):
        case_pop = variants['sample'][variants['CaseControl'] == 1].nunique()
        cont_pop = variants['sample'][variants['CaseControl'] == 0].nunique()
    if verbose >= 1:
        print(f"Number of cases in variant file: {case_pop}")
        print(f"Number of controls in variant file: {cont_pop}")
    # Filter dataframe for gene and split by case and control
    case_vars, cont_vars = _filter_variants(variants, gene, max_af, min_af, ea_lower, ea_upper, consequence)
    if case_vars.empty:
        warnings.warn(f"No variants found for {gene} in case group. Using controls only")
        group = 'control'
        cont_vars_collapsed = _clean_variant_formats(cont_vars)
    elif cont_vars.empty:
        warnings.warn(f"No variants found for {gene} in control group. Using cases only")
        group = 'case'
        case_vars_collapsed = _clean_variant_formats(case_vars)
    else:
        # Clean variant annotations
        case_vars_collapsed, cont_vars_collapsed = _clean_variant_formats(case_vars), _clean_variant_formats(cont_vars)

    # Run lollipop_plot2 for both groups
    if group == 'both':
        # Calculate fisher's exact test
        odds_ratio, lower_ci, upper_ci, pval = _fishers_exact_test(case_vars, cont_vars, case_pop, cont_pop)
        # Create the lollipop plot
        plot = _r_lollipop_plot2(case_vars_collapsed, cont_vars_collapsed, gene, plot_domain = show_domains, ac_scale = ac_scale, ea_color = ea_color, domain_min_dist = domain_min_dist, verbose = verbose)

    # Run lollipop_plot1 for single group
    elif group == 'case' or group == 'control':
        input_vars = case_vars_collapsed if group == 'case' else cont_vars_collapsed
        # Create lollipop plot
        plot = _r_lollipop_plot1(input_vars, gene, plot_domain = show_domains, ac_scale = ac_scale, ea_color = ea_color, domain_min_dist = domain_min_dist, verbose = verbose)
        pval, odds_ratio, lower_ci, upper_ci = None, None, None, None

    # Save data
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'LollipopPlots/{gene}/')
        os.makedirs(new_savepath, exist_ok=True)
        plot.save(new_savepath + f'{gene}_{group}_EA{ea_lower}-{ea_upper}_AF{max_af}-{min_af}_lollipop_plot.png')
        # Save the p-values
        with open(new_savepath + f'{gene}_{group}_EA{ea_lower}-{ea_upper}_AF{max_af}-{min_af}_pvalues.txt', 'w') as f:
            f.write(f"Parameters: gene={gene}, n_case={case_pop}, n_cont={cont_pop}, EA={ea_lower}-{ea_upper}, AF={max_af}-{min_af}\n")
            f.write(f'p-value: {pval}\n')
            f.write(f'Odds Ratio: {odds_ratio}\n')
            f.write(f'Lower CI: {lower_ci}\n')
            f.write(f'Upper CI: {upper_ci}\n')
    return plot, pval, odds_ratio, lower_ci, upper_ci
#endregion


#region Protein Structure Visualization
def _r_alphafold_structure(case_vars: pd.DataFrame, cont_vars: str, savepath: str) -> None:
    """
    Calls the AlphaFold structure prediction method using R and saves the output in a specified path.

    Args:
        case_vars (pd.DataFrame): A DataFrame containing case variables.
        cont_vars (str): A string representing control variables.
        savepath (str): The path where the output will be saved.

    Returns:
        None
    """
    # Set CRAN mirror
    cran_mirror = "https://cran.rstudio.com"
    robjects.r.options(repos=cran_mirror)
    # Install R packages
    _r_install_package('remotes')
    _r_install_package('EvoTrace')
    _r_install_package('curl')
    evotrace = importr('EvoTrace')
    if case_vars.empty:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_cont_vars = robjects.conversion.py2rpy(cont_vars)
        prot_id = cont_vars.loc[0, 'ENSP']
        # Call the structure
        evotrace.Color_Variants_AlphaFold(
            variants_ctrl=r_cont_vars,
            prot_id=prot_id,
            pml_output=savepath
        )
    if cont_vars.empty:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_case_vars = robjects.conversion.py2rpy(case_vars)
        prot_id = case_vars.loc[0, 'ENSP']
        # Call the structure
        evotrace.Color_Variants_AlphaFold(
            variants_case=r_case_vars,
            prot_id=prot_id,
            pml_output=savepath
        )
    else:
        # Convert local DataFrames to R DataFrames
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_case_vars = robjects.conversion.py2rpy(case_vars)
            r_cont_vars = robjects.conversion.py2rpy(cont_vars)
        prot_id = case_vars.loc[0, 'ENSP']
        # Call the structure
        evotrace.Color_Variants_AlphaFold(
            variants_case=r_case_vars,
            variants_ctrl=r_cont_vars,
            prot_id=prot_id,
            pml_output=savepath
        )

def _prepare_resi_df(variants: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame containing unique residues from the given variants DataFrame.

    Args:
        variants (pd.DataFrame): A DataFrame containing variant data.

    Returns:
        pd.DataFrame: A DataFrame containing unique residues extracted from the variants DataFrame.
    """
    residues = variants['SUB'].unique().tolist()
    residues = [re.search(r'\d+', residue).group() for residue in residues]
    vars_residues = pd.DataFrame({'residues': residues})
    return vars_residues

def _test_cutoff(min_cutoff:int, max_cutoff:int) -> None:
    """
    Check if the minimum cutoff is greater than the maximum cutoff.

    Parameters:
    min_cutoff (int): The minimum cutoff value.
    max_cutoff (int): The maximum cutoff value.

    Raises:
    ValueError: If the minimum cutoff is greater than the maximum cutoff.

    Returns:
    None
    """
    if min_cutoff > max_cutoff:
        raise ValueError("Minimum cutoff cannot be greater than maximum cutoff.")

def _download_af_model(url:str) -> pd.DataFrame:
    """
    Downloads a file from the given URL and returns its contents as a pandas DataFrame.

    Parameters:
    url (str): The URL of the file to be downloaded.

    Returns:
    pd.DataFrame: The contents of the downloaded file as a pandas DataFrame.
    """
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
            lines = text.splitlines()
    except urllib.error.URLError as e:
        warnings.warn(f"An error occurred while downloading the file: {e}, waiting another 60s")
        time.sleep(60)
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
                lines = text.splitlines()
        except urllib.error.URLError as e:
            warnings.warn(f"An error occurred while downloading the file: {e}, skipping")
            return pd.DataFrame()
    #df = pd.DataFrame([line.split('\t') for line in lines[1:]], columns=lines[0].split('\t'))
    return lines

def _clean_af_model(lines: list, chain:str, plddt_cutoff:int) -> pd.DataFrame:
    """
    Cleans and filters the atomic model data based on specified criteria.

    Args:
        lines (list): List of lines containing atomic model data.
        chain (str): Chain identifier to filter the data by.
        plddt_cutoff (int): Minimum pLDDT value to filter the data by.

    Returns:
        pd.DataFrame: Filtered and cleaned atomic model data.

    """
    # Filter lines that start with 'ATOM' or 'HETATM'
    atom_lines = [line for line in lines if line[:6].strip() in ['ATOM', 'HETATM']]
    # Define column widths as per PDB format specifications
    col_specs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26), (26, 27),
                 (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78), (78, 80)]
    col_names = [
        'record', 'atom_number', 'atom_type', 'blank_1', 'residue_name', 'chain', 'residue_seq_number', 'blank_3', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'element', 'charge'
    ]
    # Create DataFrame from the filtered lines
    df = pd.read_fwf(pd.io.common.StringIO('\n'.join(atom_lines)), colspecs=col_specs, names=col_names, header=None)
    # Filter DataFrame by chain and residue type
    df = df[df['record'].isin(['ATOM', 'HETATM'])]
    df = df[df['residue_name'].isin(
        [
            'HIS', 'PRO', 'GLU', 'THR', 'LEU', 'VAL', 'LYS', 'ASP', 'ALA',
            'GLN', 'GLY', 'ARG', 'TYR', 'ILE', 'ASN', 'SER', 'PHE', 'MET',
            'CYS', 'TRP', 'MSE'
        ]
    )]
    df = df[df['atom_type'].str.strip() == 'CA']
    df = df[df['chain'].str.strip() == chain]
    # Select and rename the columns
    df = df[['chain', 'residue_seq_number', 'temp_factor']]
    df.columns = ['chain', 'POS', 'pLDDT']
    df = df.sort_values(by=['chain', 'POS'])
    df = df[df['pLDDT'] >= plddt_cutoff]
    return df

def _get_plddt(prot_id:str, gene:str, chain:str, plddt_cutoff:int, savepath:str) -> pd.DataFrame:
    """
    Retrieves the pLDDT scores for a given protein and chain from the AlphaFold model.

    Args:
        prot_id (str): The protein identifier.
        chain (str): The chain identifier.
        plddt_cutoff (int): The pLDDT cutoff value.

    Returns:
        af_model_clean (DataFrame): The cleaned dataframe containing the pLDDT scores.
    """
    # Load the pdb mapping dataframe
    pdb_df = _load_pdb_et_mapping()
    af_id = pdb_df['AF_id_rep'][pdb_df['prot_id'].str.contains(prot_id)].values[0]
    # Create the save path
    if savepath:
        savepath = _fix_savepath(savepath)
        half_pdb_path = os.path.join(savepath, f'ProteinStructures/{gene}/')
        os.makedirs(half_pdb_path, exist_ok=True)
        pdb_path = os.path.join(savepath, f'ProteinStructures/{gene}/{gene}_AF-{af_id}.pdb')
    if not savepath:
        temp_file = tempfile.NamedTemporaryFile(delete = False, suffix = ".pdb")
        pdb_path = temp_file.name
    # Filter for the given gene name
    try: af_url = pdb_df['AF_url'][pdb_df['prot_id'].str.contains(prot_id)].values[0]
    except:
        warnings.warn("Gene not found in PDB-AF ID map.")
        return pd.DataFrame()
    # Download the AlphaFold model
    af_model = _download_af_model(af_url)
    with open(pdb_path, 'w') as f:
        f.write('\n'.join(af_model))
    if len(af_model) == 0:
        return pd.DataFrame()
    # Parse the model into cleaned dataframe
    af_model_clean = _clean_af_model(af_model, chain, plddt_cutoff)
    background_resi = af_model_clean['POS'].unique().tolist()
    return background_resi, pdb_path

def _scw_analysis(pdb_path:str, background_resi:list, residues:pd.DataFrame, chain:str, dist_cutoff:int) -> pd.DataFrame:
    """
    Perform SCW analysis on a protein structure.

    Args:
        pdb_path (str): Path to the PDB file.
        background_resi (list): List of background residue positions.
        residues (pd.DataFrame): DataFrame containing residue information.
        chain (str): Chain identifier.
        dist_cutoff (int): Distance cutoff for SCW analysis.

    Returns:
        pd.DataFrame: DataFrame containing the computed Z-scores.

    """
    # Install EvoTrace
    _r_install_package('EvoTrace')
    evotrace = importr('EvoTrace')

    if residues.empty:
        return pd.DataFrame()
    else:
        # Convert Vars
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_vector = robjects.conversion.py2rpy(residues['residues'].unique().tolist())
            background_resi = robjects.conversion.py2rpy(background_resi)

        # Get the background distribution
        globalenv['background'] = evotrace.GetSCWBackgound(
            pdb_file = pdb_path,
            chain = chain,
            dist_cutoff = dist_cutoff,
            resi = background_resi
        )
        # Compute the Z-score
        globalenv['z_score'] = evotrace.ComputeSCWzscore(
            globalenv['background'],
            resi = r_vector,
            output_df = True
        )
        # Convert globalenv['z_score'] to a pandas DataFrame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            z_score = robjects.conversion.rpy2py(globalenv['z_score'])
            z_score['dist'] = dist_cutoff
        return z_score

def _scw_analysis_task(pdb_path:str, background_resi:pd.DataFrame, residue:pd.DataFrame, scw_chain:str, dist_cutoff:int, annotate_background:str) -> pd.DataFrame:
    """
    Wrapper function for SCW analysis to be used in multiprocessing.

    Args:
        pdb_path (str): Path to the PDB file.
        background_resi (list): Background residues.
        residue (list): Residue list.
        scw_chain (str): SCW chain identifier.
        dist_cutoff (int): Distance cutoff.
        scw_plddt_cutoff (float): pLDDT cutoff value.

    Returns:
        DataFrame: Resulting DataFrame with analysis results.
    """
    result = _scw_analysis(pdb_path, background_resi, residue, scw_chain, dist_cutoff)
    result['type'] = annotate_background
    return result

def _parallel_scw_analysis(pdb_path:str, all_residues:pd.DataFrame, case_vars_residues:pd.DataFrame, cont_vars_residues:pd.DataFrame, all_background_resi: pd.DataFrame, plddt_background_resi: pd.DataFrame, scw_chain:str, scw_min_dist_cutoff: int, scw_max_dist_cutoff:int, scw_plddt_cutoff: int, cores:int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Parallelizes the SCW analysis across multiple residues, background residues, and distance cutoffs using multiprocessing.

    Args:
        pdb_path (str): Path to the PDB file.
        all_residues (list): All residues.
        case_vars_residues (list): Case variable residues.
        cont_vars_residues (list): Control variable residues.
        all_background_resi (list): All background residues.
        plddt_background_resi (list): pLDDT background residues.
        scw_chain (str): SCW chain identifier.
        scw_min_dist_cutoff (int): Minimum distance cutoff.
        scw_max_dist_cutoff (int): Maximum distance cutoff.
        scw_plddt_cutoff (float): pLDDT cutoff value.

    Returns:
        tuple: Concatenated DataFrames for all_output_df, case_output_df, and cont_output_df.
    """
    residues = [all_residues, case_vars_residues, cont_vars_residues]
    backgrounds = [all_background_resi, plddt_background_resi]

    tasks = [(pdb_path, background_resi, residue, scw_chain, dist_cutoff,
                'all' if background_resi == all_background_resi else f'plddt >{scw_plddt_cutoff}')
                for residue in residues
                for background_resi in backgrounds
                for dist_cutoff in range(scw_min_dist_cutoff, scw_max_dist_cutoff + 1)]

    with mp.Pool(processes=cores) as pool:
        results = pool.starmap(_scw_analysis_task, tasks)

    # Group results back to the structure (all_output_df, case_output_df, cont_output_df)
    results_per_residue = len(backgrounds) * (scw_max_dist_cutoff - scw_min_dist_cutoff + 1)
    grouped_results = [results[i:i + results_per_residue] for i in range(0, len(results), results_per_residue)]

    all_output_df = pd.concat(grouped_results[0])
    case_output_df = pd.concat(grouped_results[1])
    cont_output_df = pd.concat(grouped_results[2])

    return all_output_df, case_output_df, cont_output_df

def _plot_scw_z(output_df:pd.DataFrame, scw_plddt_cutoff: int) -> PILImage.Image:
    """
    Create a plot similar to the provided image using the given dataframe.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data to be plotted.
    """
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Create a figure and a set of subplots
    _, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Filter data for each type
    all_data = output_df[output_df['type'] == 'all']
    plddt_data = output_df[output_df['type'] == f'plddt >{scw_plddt_cutoff}']

    # Plot the data for 'all'
    sns.lineplot(ax=axes[0], data=all_data, x='dist', y='bias', marker='o', label='bias', color='green')
    sns.lineplot(ax=axes[0], data=all_data, x='dist', y='unbias', marker='o', label='unbias', color='blue')
    sns.lineplot(ax=axes[0], data=all_data, x='dist', y='adj_dist', marker='o', label='adj_dist', color='red')
    axes[0].axhline(2, ls='--', color='red')
    axes[0].set_ylabel('Z_score')
    axes[0].set_title('all_resi')

    # Plot the data for 'plddt >50'
    sns.lineplot(ax=axes[1], data=plddt_data, x='dist', y='bias', marker='o', label='bias', color='green')
    sns.lineplot(ax=axes[1], data=plddt_data, x='dist', y='unbias', marker='o', label='unbias', color='blue')
    sns.lineplot(ax=axes[1], data=plddt_data, x='dist', y='adj_dist', marker='o', label='adj_dist', color='red')
    axes[1].axhline(2, ls='--', color='red')
    axes[1].set_ylabel('Z_score')
    axes[1].set_xlabel('dist_cutoff')
    axes[1].set_title(f'pLDDT > {scw_plddt_cutoff}')

    # Adjust layout and show plot
    plt.tight_layout()
    # Save plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi = 300)
    buffer.seek(0)
    image = PILImage.open(buffer)
    plt.close()
    return image

def protein_structures(variants: pd.DataFrame, gene: str, run_scw:bool = True, max_af:float = 1.0, min_af:float = 0.0, ea_lower:int = 0, ea_upper:int = 100, consequence: str = 'missense_variant|frameshift_variant|stop_gained|stop_lost|start_lost', scw_chain:str = 'A', scw_plddt_cutoff: int = 50, scw_min_dist_cutoff:int = 4, scw_max_dist_cutoff:int = 12, cores:int = 1, savepath: str=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, PILImage, PILImage, PILImage): # type: ignore
    """
    Generate AlphaFold protein structures for the given variants and save them to a specified path.

    Args:
        variants (pd.DataFrame): DataFrame containing variant information.
        gene (str): Gene name for which protein structures are generated.
        savepath (str): Path to save the generated protein structures.
        max_af (float, optional): Maximum allele frequency threshold for variant filtering. Defaults to 1.0.
        min_af (float, optional): Minimum allele frequency threshold for variant filtering. Defaults to 0.0.
        ea_lower (int, optional): Lower bound of the effect allele count threshold for variant filtering. Defaults to 0.
        ea_upper (int, optional): Upper bound of the effect allele count threshold for variant filtering. Defaults to 100.

    Returns:
        None
    """
    # Separate into case and control variants
    case_vars, cont_vars = _filter_variants(variants, gene, max_af, min_af, ea_lower, ea_upper, consequence)
    if case_vars.empty:
        warnings.warn(f"No variants found for {gene} in case group. Using controls only")
        cont_vars_collapsed = _clean_variant_formats(cont_vars)
        cont_vars_residues = _prepare_resi_df(cont_vars_collapsed)
        all_residues = pd.DataFrame()
        case_vars_collapsed = pd.DataFrame()
        case_vars_residues = pd.DataFrame()
        # Get the Protein ID
        prot_id = cont_vars.loc[0, 'ENSP']
        prot_id = _check_pdb_id(prot_id)
        gene = cont_vars.loc[0, 'gene']
    elif cont_vars.empty:
        warnings.warn(f"No variants found for {gene} in control group. Using cases only")
        case_vars_collapsed = _clean_variant_formats(case_vars)
        case_vars_residues = _prepare_resi_df(case_vars_collapsed)
        all_residues = pd.DataFrame()
        cont_vars_collapsed = pd.DataFrame()
        cont_vars_residues = pd.DataFrame()
        # Get the Protein ID
        prot_id = case_vars.loc[0, 'ENSP']
        prot_id = _check_pdb_id(prot_id)
        gene = case_vars.loc[0, 'gene']
    else:
        # Clean variant annotations
        case_vars_collapsed, cont_vars_collapsed = _clean_variant_formats(case_vars), _clean_variant_formats(cont_vars)
        case_vars_residues, cont_vars_residues = _prepare_resi_df(case_vars_collapsed), _prepare_resi_df(cont_vars_collapsed)
        all_residues = pd.concat([case_vars_residues, cont_vars_residues])
        # Get the Protein ID
        prot_id = case_vars.loc[0, 'ENSP']
        prot_id = _check_pdb_id(prot_id)
        gene = case_vars.loc[0, 'gene']


    # Run SCW analysis
    if run_scw:
        # Get the protein file and save it to a temp directory
        plddt_background_resi, pdb_path = _get_plddt(prot_id, gene, scw_chain, scw_plddt_cutoff, savepath)
        all_background_resi, pdb_path = _get_plddt(prot_id, gene, scw_chain, 0, savepath)

        # Run analysis
        all_output_df, case_output_df, cont_output_df = _parallel_scw_analysis(
            pdb_path, all_residues, case_vars_residues, cont_vars_residues,
            all_background_resi, plddt_background_resi, scw_chain,
            scw_min_dist_cutoff, scw_max_dist_cutoff, scw_plddt_cutoff, cores
        )
        # Plot results
        if not all_output_df.empty:
            all_plot = _plot_scw_z(all_output_df, scw_plddt_cutoff)
        else:
            all_plot = PILImage.new('RGB', (1, 1))
        if not case_output_df.empty:
            case_plot = _plot_scw_z(case_output_df, scw_plddt_cutoff)
        else:
            case_plot = PILImage.new('RGB', (1, 1))
        if not cont_output_df.empty:
            cont_plot = _plot_scw_z(cont_output_df, scw_plddt_cutoff)
        else:
            cont_plot = PILImage.new('RGB', (1, 1))
    else:
        all_output_df = pd.DataFrame()
        case_output_df = pd.DataFrame()
        cont_output_df = pd.DataFrame()
        all_plot = PILImage.new('RGB', (1, 1))
        case_plot = PILImage.new('RGB', (1, 1))
        cont_plot = PILImage.new('RGB', (1, 1))

    # Run if savepath is definedf
    if savepath:
        # Change the save paths
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'ProteinStructures/{gene}/')
        os.makedirs(new_savepath, exist_ok=True)
        # Call the protein visualization
        new_savefile = os.path.join(new_savepath, f'{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_AlphaFold.pml')
        # Call the protein visualizations
        _r_alphafold_structure(case_vars_collapsed, cont_vars_collapsed, new_savefile)
        # Write case residues to output file
        if not case_vars_residues.empty:
            with open(os.path.join(new_savepath, f'Cases_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_residues.txt'), 'a') as f:
                for residue in case_vars_residues['residues']:
                    residue = ''.join(filter(str.isdigit, residue))
                    f.write(f'{residue}\n')
                f.close()
        # Write control residues to output file
        if not cont_vars_residues.empty:
            with open(os.path.join(new_savepath, f'Controls_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_residues.txt'), 'a') as f:
                for residue in cont_vars_residues['residues']:
                    residue = ''.join(filter(str.isdigit, residue))
                    f.write(f'{residue}\n')
                f.close()
        # Save the SCW analysis output
        all_output_df.to_csv(os.path.join(new_savepath, f'All-Variants_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_SCW_analysis.csv'), index=False)
        case_output_df.to_csv(os.path.join(new_savepath, f'Cases_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_SCW_analysis.csv'), index=False)
        cont_output_df.to_csv(os.path.join(new_savepath, f'Controls_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_SCW_analysis.csv'), index=False)
        # Save the PIL images
        if not all_output_df.empty:
            all_plot.save(os.path.join(new_savepath, f'All-Variants_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_SCW_analysis.png'))
        if not case_output_df.empty:
            case_plot.save(os.path.join(new_savepath, f'Cases_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_SCW_analysis.png'))
        if not cont_output_df.empty:
            cont_plot.save(os.path.join(new_savepath, f'Controls_{gene}_AF{min_af}-{max_af}_EA{ea_lower}-{ea_upper}_SCW_analysis.png'))

    if not savepath:
        warnings.warn("No savepath provided. Skipping structure visualization.")

    return all_output_df, case_output_df, cont_output_df, all_plot, case_plot, cont_plot


#endregion
