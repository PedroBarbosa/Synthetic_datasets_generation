# Generation of synthetic genomics datasets

This repository contains the analysis steps associated with the paper "Semantically Rich Local Dataset Generation for Explainable AI in Genomics", which introduces a new method for generating local synthetic genomic datasets based on grammars of sequence perturbations.

All datasets generated for this study are available on Zenodo. Simply download the `datasets.tar.gz` and uncompress the repository to access them. To reproduce the analysis, there are README files within each section with steps employed. It is important to note that our evolutionary searches were conducted on a specific GPU model (NVIDIA GeForce RTX 3090) within a local server environment. Given this hardware specificity and the time constraint of 10 minutes per experiment, the exact replication of results may vary when using different hardware setups. To reproduce the figures presented in the paper, simply run the `plots.ipynb` notebook.
