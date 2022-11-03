#!/bin/bash
# use the base conda env.
species="$1"
dataset_location="$2"
feature_path="$3"
kover_location="$4"

readarray -t Anti_List <  ${feature_path}/${species}/anti_list

for anti in ${Anti_List[@]};do


    mkdir -p ${feature_path}/${species}/${anti}_temp
    echo "$anti"

    ${kover_location}kover dataset create from-contigs \
    --genomic-data ${feature_path}/${species}/${anti}_data \
    --phenotype-metadata ${feature_path}/${species}/${anti}_pheno \
    --output ${feature_path}/${species}/${anti}_koverdataset_0 \
    --kmer-size 31 \
    --n-cpu 1 \
    --temp-dir ${feature_path}/${species}/${anti}_temp/ \
    --phenotype-description 'No description.' \
    -x


    for j in {1..9};do
      cp ${feature_path}/${species}/${anti}_koverdataset_0 ${feature_path}/${species}/${anti}_koverdataset_${j}

    done



    for j in {0..9};do

        echo "CV ${j}"
        ${kover_location}kover dataset split --dataset ${feature_path}/${species}/${anti}_koverdataset_${j} \
        --id ${feature_path}/${species}/${anti}_id \
        --train-ids ${feature_path}/${species}/${anti}_Train_${j}_id \
        --test-ids ${feature_path}/${species}/${anti}_Test_${j}_id \
        --random-seed 42 \
        -x
    done


    wait
done
