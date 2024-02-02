### Create config files to run 30 seeds on the best configuration of each strategy
```
cat ../1_hyperparameter_search/3_bestConfigurationPerStrategy.txt | parallel cp ../1_hyperparameter_search/1_bestTrialsMultipleSeeds/{}_seed0.yaml 1_bestConfiguration30seeds
./0_createNewConfigs.sh
cd 1_bestConfiguration30Seeds/ && find $(pwd -P) -name "*yaml" | grep -v "args_used.yaml" > listYaml.txt && cd ../
sbatch run_evolutions.sbatch 1_bestConfiguration30Seeds/listYaml.txt
```

### Run extra analysis for fitness function eval
`sbatch run_evolutions.sbatch 1_bestConfiguration30Seeds/with_IAD_and_best_BF_params/listYamls.txt`
