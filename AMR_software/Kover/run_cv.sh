#!/bin/bash
# use the base conda env.
species="$1"
dataset_location="$2"
feature_path="$3"
n_jobs="$4"
kover_location="$5"


readarray -t Anti_List <  ${feature_path}/${species}/anti_list

for anti in ${Anti_List[@]};do


    mkdir -p ${feature_path}/${species}/${anti}_temp
    echo "$anti"


    for j in {0..9};do
#        rm -rf ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}
#        rm -rf ./log/temp/loose/${species}/${anti}_temp/CV_te${j}
        echo "CV ${j}"

        ${kover_location}kover learn scm --dataset ${feature_path}/${species}/${anti}_koverdataset_${j} \
         --split ${feature_path}/${species}/${anti}_id \
         --hp-choice bound \
         --model-type conjunction disjunction \
         --p 0.1 0.178 0.316 0.562 1.0 1.778 3.162 5.623 10.0 999999.0 \
         --max-rules 10 \
         --output-dir ${feature_path}/${species}/${anti}_temp/scm_b_${j}\
         --n-cpu ${n_jobs} \
         --progress
#         --bound-max-genome-size #By default number of k-mers in the dataset is used.

         ${kover_location}kover learn tree --dataset ${feature_path}/${species}/${anti}_koverdataset_${j} \
         --split ${feature_path}/${species}/${anti}_id \
         --hp-choice bound \
         --criterion gini \
         --max-depth 20 \
         --min-samples-split 2 \
         --class-importance 0.25 0.5 0.75 1.0 \
         --n-cpu ${n_jobs} \
         --output-dir ${feature_path}/${species}/${anti}_temp/tree_b_${j} \
         --progress \
#         --bound-max-genome-size #By default number of k-mers in the dataset is used.


    done
    wait
done
