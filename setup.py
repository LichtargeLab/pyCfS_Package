"""
Setup script for pyCfS, a package for gene list validation experiments
"""
import setuptools

VERSION = '0.2'

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
        'python>=3.12',
        'requests>=2.32.0',
        'pandas>=2.3.0',
        'numpy>=2.3.1',
        'matplotlib>=3.10.3',
        'matplotlib_venn>=1.1.2',
        'Pillow>=11.3.0',
        'venn>=0.1.3',
        'scipy>=1.16.0',
        'networkx>=3.5',
        'biopython>=1.85',
        'upsetplot>=0.9.0',
        'markov_clustering>=0.0.6.dev0',
        'statsmodels>=0.14.4',
        'pyarrow>=20.0.0',
        'adjustText>=1.3.0',
        'seaborn>=0.13.2',
        'tqdm>=4.67.1',
        'scikit-learn>=1.7.0',
        'pysam>=0.23.3',
        'xgboost>=3.0.2',
        'scikit-optimize>=0.10.2',
        'IPython>=9.4.0',
        'setuptools>=80.9.0'
    ],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
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
