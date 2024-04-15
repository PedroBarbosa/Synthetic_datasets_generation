[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10955718.svg)](https://doi.org/10.5281/zenodo.10955718)

# Generation of synthetic genomics datasets

This repository contains the analysis steps associated with the paper ["Semantically Rich Local Dataset Generation for Explainable AI in Genomics"](https://doi.org/10.1145/3638529.3653990), accepted at [GECCO 2024](https://gecco-2024.sigevo.org/HomePage), which introduces a new approach for generating local synthetic genomic datasets based on grammars of sequence perturbations.

The code used to generate the figures of the manuscript is available in the `figures.ipynb` notebook. 

To reproduce the analysis, there are README files within each section with the steps employed. It is important to note that our evolutionary searches were conducted on a specific GPU model (NVIDIA GeForce RTX 3090) within a local server environment. Given this hardware specificity and the time constraint of 5 minutes per experiment, the exact replication of results may vary when using different hardware setups.

A full copy of this repository and the generated datasets are available at [Zenodo](https://zenodo.org/records/10955718).
