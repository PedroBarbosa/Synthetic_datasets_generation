#!/bin/bash

for file in 1_bestConfiguration30Seeds/*seed0.yaml; do
    for (( N=30; N<60; N++ )); do
        new_file=$(echo $file | sed "s/seed0/seed$N/")
        sed "s/seed: 0/seed: $N/" "$file" | sed "s/1_bestTrialsMultipleSeeds/1_bestConfiguration30Seeds/g"  > "$new_file"
    done
done
rm 1_bestConfiguration30Seeds/*seed0.yaml
