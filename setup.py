"""
Setup script for pyCfS, a package for gene list validation experiments
"""
import setuptools

VERSION = '0.1.7'

setuptools.setup(
    name = 'pyCfS',
    version = VERSION,
    url = "",
    author = "Kevin Wilhelm, Jenn Asmussen, Andrew Bigler",
    author_email = "kevin.wilhelm@bcm.edu, jennifer.asmussen@bcm.edu, andrew.bigler@bcm.edu",
    description = "Gene list validation experiments",
    long_description = open('DESCRIPTION.rst').read(),
    packages = setuptools.find_packages(),
    install_requires = [
        'requests>=2.31.0',
        'pandas>=2.0.3',
        'numpy>=1.24.4',
        'matplotlib>=3.7.3,<3.9.0',
        'matplotlib_venn>=0.11.9',
        'Pillow>=10.1.0',
        'venn>=0.1.3',
        'scipy>=1.10.1',
        'networkx>=3.1',
        'biopython>=1.81',
        'upsetplot>=0.8.0',
        'markov_clustering>=0.0.6.dev0',
        'statsmodels>=0.14.0',
        'pyarrow>=14.0.1',
        'adjustText',
        'seaborn>=0.13.0',
        'tqdm>=4.66.1',
        'scipy>=1.10.1',
        'scikit-learn>=1.3.2',
        'pysam>=0.22.0',
        'xgboost>=2.1.0',
        'scikit-learn>=1.3.2',
        'scikit-optimize>=0.10.2'
    ],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    include_package_data = True,
    package_data = {'':[
        'data/*.feather',
        'data/*.txt',
        'data/*.gmt',
        'data/*.csv',
        'data/*.parquet',
        'data/mousePhenotypes/*.parquet',
        'data/targets/*.parquet',
        'data/*.graphml',
        'data/*.rpt'
    ]}
)
