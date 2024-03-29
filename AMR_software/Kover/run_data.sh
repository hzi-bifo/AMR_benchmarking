#!/bin/bash
# use the base conda env.
species="$1"
feature_path="$2"
kover_location="$3"

readarray -t Anti_List <  ${feature_path}/${species}/anti_list

for anti in ${Anti_List[@]};do


    mkdir -p ${feature_path}/${species}/${anti}_temp
    echo "$anti"


    ./AMR_software/Kover/bin/kover dataset create from-contigs \
    --genomic-data ${feature_path}/${species}/${anti}_data \
    --phenotype-metadata ${feature_path}/${species}/${anti}_pheno \
    --output ${feature_path}/${species}/${anti}_koverdataset_0 \
    --kmer-size 31 \
    --n-cpu 1 \
    --temp-dir ${feature_path}/${species}/${anti}_temp/ \
    --phenotype-description 'No description.' \
    -x

    echo " finish dataset creating"

    for j in {1..9};do
      cp ${feature_path}/${species}/${anti}_koverdataset_0 ${feature_path}/${species}/${anti}_koverdataset_${j}

    done



    for j in {0..9};do

        echo "CV ${j}"
        ./AMR_software/Kover/bin/kover dataset split --dataset ${feature_path}/${species}/${anti}_koverdataset_${j} \
        --id ${feature_path}/${species}/${anti}_id \
        --train-ids ${feature_path}/${species}/${anti}_Train_${j}_id \
        --test-ids ${feature_path}/${species}/${anti}_Test_${j}_id \
        --random-seed 42 \
        -x
    done


    wait
done
