### Run evolutions using both GGGP, the baseline and SQUID

```
sbatch 1_run_evolutions.sbatch listYaml.txt 
sbatch 1_run_evolutions.sbatch listYaml_randomSearch.txt
srun papermill -k python3 2_run_squid.ipynb squid_out.ipynb > papermill_squid.log 2>&1
```

To run motif scanning on the GGGP and RS generated datasets, we converted DRESS output format from v0.0.1 to v0.1.0 version.

```
srun papermill -k python3 3_run_motif_scanning.ipynb motifs_out.ipynb > papermill_motifs.log 2>&1
```
