#!/bin/bash

file=$(readlink -f $1)

# Lexicase selection
for (( N=30; N<60; N++ )); do
    new_file=$(echo $(basename $file) | sed "s/seed30/seed$N/")
    sed "s/seed: 30/seed: $N/g" "$file" | \
    sed "s/1_bestConfiguration30Seeds/1_lexicase/g" | \
    sed "s/selection_method: tournament/selection_method: lexicase/g" > "1_lexicase/$new_file"
done

# Custom mutation operator 
#weight 0
for (( N=30; N<60; N++ )); do
    new_file=$(echo $(basename $file) | sed "s/seed30/seed$N/")
    sed "s/seed: 30/seed: $N/g" "$file" | \
    sed "s/1_bestConfiguration30Seeds/2_customMutationOperator\/weight_0/g" | \
    sed "s/custom_mutation_operator: true/custom_mutation_operator: false/g" | \
    sed "s/custom_mutation_operator_weight: 1.0/custom_mutation_operator_weight: 0/g" > "2_customMutationOperator/weight_0/$new_file"
done

weights=(0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9)

for weight in ${weights[@]}; do
    mkdir -p "2_customMutationOperator/weight_$weight"

    # Loop over each seed
    for (( N=30; N<60; N++ )); do
        new_file=$(echo $(basename $file) | sed "s/seed30/seed$N/")
        sed "s/seed: 30/seed: $N/g" "$file" | \
        sed "s/1_bestConfiguration30Seeds/2_customMutationOperator\/weight_$weight/g" | \
        sed "s/custom_mutation_operator_weight: 1.0/custom_mutation_operator_weight: $weight/g" > "2_customMutationOperator/weight_$weight/$new_file"
    done
done
  
# No deletions
for (( N=30; N<60; N++ )); do
    new_file=$(echo $(basename $file) | sed "s/seed30/seed$N/")
    sed "s/seed: 30/seed: $N/g" "$file" | \
    sed "s/1_bestConfiguration30Seeds/3_grammarNodeTypes\/noDeletions/g" | \
    sed "s/deletion_weight: 0.2/deletion_weight: 0/g" > "3_grammarNodeTypes/noDeletions/$new_file"
done

# No insertions
for (( N=30; N<60; N++ )); do
    new_file=$(echo $(basename $file) | sed "s/seed30/seed$N/")
    sed "s/seed: 30/seed: $N/g" "$file" | \
    sed "s/1_bestConfiguration30Seeds/3_grammarNodeTypes\/noInsertions/g" | \
    sed "s/insertion_weight: 0.45/insertion_weight: 0/g" > "3_grammarNodeTypes/noInsertions/$new_file"
done

# No deletions and insertions
for (( N=30; N<60; N++ )); do
    new_file=$(echo $(basename $file)| sed "s/seed30/seed$N/")
    sed "s/seed: 30/seed: $N/g" "$file" | \
    sed "s/1_bestConfiguration30Seeds/3_grammarNodeTypes\/noDeletionsAndInsertions/g" | \
    sed "s/deletion_weight: 0.2/deletion_weight: 0/g" | \
    sed "s/insertion_weight: 0.45/insertion_weight: 0/g" > "3_grammarNodeTypes/noDeletionsAndInsertions/$new_file"
done
