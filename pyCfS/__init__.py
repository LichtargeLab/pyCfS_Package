"""
init file for pyCFS
"""
from .Combine import consensus, functional_clustering, statistical_combination
from .GoldStandards import string_enrichment, goldstandard_overlap, ndiffusion, interconnectivity, gwas_catalog_colocalization, pubmed_comentions
from .Clinical import mouse_phenotype_enrichment, protein_family_enrichment, drug_gene_interactions, depmap_enrichment
from .Association import variants_by_sample, risk_prediction, odds_ratios, ea_distributions
from .Summarize import prioritize_genes

# Lazy loading for Structure module
def __getattr__(name):
    if name in ['lollipop_plot', 'protein_structures']:
        try:
            from .Structure import lollipop_plot, protein_structures
            globals()[name] = locals()[name]
            return globals()[name]
        except ImportError:
            raise ImportError(
                f"The {name} function requires additional dependencies. "
                "These dependencies are optional and not installed by default. "
                "Please install them using conda: conda install -c bioconda -c conda-forge biopython prody"
            )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .utils import _hypergeo_overlap, _load_grch38_background, _load_string, _load_reactome, _get_open_targets_gene_mapping, _define_background_list, _clean_genelists, _format_scientific, _fix_savepath, _select_evidences, _get_evidence_types, _get_combined_score, _get_edge_weight, _load_clean_string_network, _validate_ea_thresh, _validate_af_thresh

__version__ = '0.1.7.2'
__author__ = 'Kevin Wilhelm, Jenn Asmussen'
