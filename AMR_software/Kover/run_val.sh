#!/bin/bash
# use the base conda env.
species="$1"
feature_path="$2"
n_jobs="$3"

readarray -t Anti_List <  ${feature_path}/${species}/anti_list

for anti in ${Anti_List[@]};do


    mkdir -p ${feature_path}/${species}/${anti}_temp
    echo "$anti"

#
    ./AMR_software/Kover/bin/kover dataset create from-contigs \
    --genomic-data ${feature_path}/${species}/${anti}_data \
    --phenotype-metadata ${feature_path}/${species}/${anti}_pheno \
    --output ${feature_path}/${species}/${anti}_koverdataset_0 \
    --kmer-size 31 \
    --n-cpu 2 \
    --temp-dir ${feature_path}/${species}/${anti}_temp/ \
    --phenotype-description 'No description.' \
    -x
##
#    echo " finish dataset creating"
#
    for j in {0..9};do
      for i in  {0..8};do
      cp ${feature_path}/${species}/${anti}_koverdataset_0 ${feature_path}/${species}/${anti}_koverdataset_${j}_${i}
      wait
      done
    done


#
    for j in {0..9};do
        for i in  {0..8};do
        echo "CV ${j}"
        ./AMR_software/Kover/bin/kover dataset split --dataset ${feature_path}/${species}/${anti}_koverdataset_${j}_${i}
        --id ${feature_path}/${species}/${anti}_id \
        --train-ids ${feature_path}/${species}/${anti}_Train_outer_${j}_inner_${i}_id \
        --test-ids ${feature_path}/${species}/${anti}_Test_${j}_inner_${i}_id \
        --random-seed 42 \
        -x
    done
    done

    wait

    for j in {0..9};do
        for i in  {0..8};do
     ./AMR_software/Kover/bin/kover learn scm --dataset ${feature_path}/${species}/${anti}_koverdataset_${j}_${i}
         --split ${feature_path}/${species}/${anti}_id \
         --hp-choice bound \
         --model-type conjunction disjunction \
         --p 0.1 0.178 0.316 0.562 1.0 1.778 3.162 5.623 10.0 999999.0 \
         --max-rules 10 \
         --output-dir ${feature_path}/${species}/${anti}_temp/scm_b_${j}_${i}\
         --n-cpu ${n_jobs} \
         --progress
#         --bound-max-genome-size #By default number of k-mers in the dataset is used.

         ./AMR_software/Kover/bin/kover learn tree --dataset ${feature_path}/${species}/${anti}_koverdataset_${j}_${i}
         --split ${feature_path}/${species}/${anti}_id \
         --hp-choice bound \
         --criterion gini \
         --max-depth 20 \
         --min-samples-split 2 \
         --class-importance 0.25 0.5 0.75 1.0 \
         --n-cpu ${n_jobs} \
         --output-dir ${feature_path}/${species}/${anti}_temp/tree_b_${j}_${i} \
         --progress \
#         --bound-max-genome-size #By default number of k-mers in the dataset is used.

      done
      done




done
