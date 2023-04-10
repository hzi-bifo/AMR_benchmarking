#!/bin/bash
# use the base conda env.
species="$1"
feature_path="$2"
n_jobs="$3"

readarray -t Anti_List <  ${feature_path}/${species}/anti_list

for anti in ${Anti_List[@]};do

    mkdir -p ${feature_path}/${species}/${anti}_temp
    echo "$anti"

#    ./AMR_software/Kover/bin/kover dataset create from-contigs \ #todo change back when uploadig git
    /vol/projects/khu/amr/kover/bin/kover dataset create from-contigs \
    --genomic-data ${feature_path}/${species}/${anti}_data \
    --phenotype-metadata ${feature_path}/${species}/${anti}_pheno \
    --output ${feature_path}/${species}/${anti}_koverdataset \
    --kmer-size 31 \
    --n-cpu ${n_jobs} \
    --temp-dir ${feature_path}/${species}/${anti}_temp/ \
    --phenotype-description 'No description.' \
    -x

    echo " finish dataset creating"

    wait
done
