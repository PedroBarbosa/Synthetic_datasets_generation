### Create new config files for all ablation experiments and run the corresponding evolutions
```
cp ../2_performanceComparison/1_bestConfiguration30Seeds/bin_filler_1_trial428_seed30.yaml .
./0_createNewConfigs.sh bin_filler_1_trial428_seed30.yaml
find $(pwd -P) -mindepth 2 -name "*yaml" > listYaml.txt
sbatch run_evolutions.sbatch listYaml.txt
```
