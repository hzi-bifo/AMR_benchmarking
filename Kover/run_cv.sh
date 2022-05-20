#!/bin/bash
# use the base conda env.
species="$1"
path="/vol/projects/BIFO/patric_genome/"
readarray -t Anti_List <  ./log/temp/loose/${species}/anti_list

for anti in ${Anti_List[@]};do


    mkdir -p ./log/temp/loose/${species}/${anti}_temp
    echo "$anti"




    for j in {0..9};do
#        rm -rf ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}
#        rm -rf ./log/temp/loose/${species}/${anti}_temp/CV_te${j}
        (echo "CV ${j}"
            /vol/projects/khu/amr/kover/bin/kover dataset create from-contigs \
        --genomic-data ./log/temp/loose/${species}/${anti}_data \
        --phenotype-metadata ./log/temp/loose/${species}/${anti}_pheno \
        --output ./log/temp/loose/${species}/${anti}_koverdataset_${j} \
        --kmer-size 31 \
        --n-cpu 5 \
        --temp-dir ./log/temp/loose/${species}/${anti}_temp/ \
        --phenotype-description 'No description.' \
        -x )&

#        /vol/projects/khu/amr/kover/bin/kover dataset split --dataset ./log/temp/loose/${species}/${anti}_koverdataset_${j} \
#        --id ./log/temp/loose/${species}/${anti}_id \
#        --train-ids ./log/temp/loose/${species}/${anti}_Train_${j}_id \
#        --test-ids ./log/temp/loose/${species}/${anti}_Test_${j}_id \
#        --random-seed 42 \
#        -x
#
#        /vol/projects/khu/amr/kover/bin/kover learn scm --dataset ./log/temp/loose/${species}/${anti}_koverdataset_${j} \
#         --split ./log/temp/loose/${species}/${anti}_id \
#         --hp-choice bound \
#         --model-type conjunction disjunction \
#         --p 0.1 0.178 0.316 0.562 1.0 1.778 3.162 5.623 10.0 999999.0 \
#         --max-rules 10 --hp-choice bound  \
#         --output-dir ./log/temp/loose/${species}/${anti}_temp/scm_b_${j}\
#         --progress
##         --bound-max-genome-size #By default number of k-mers in the dataset is used.
#
#         /vol/projects/khu/amr/kover/bin/kover learn tree --dataset ./log/temp/loose/${species}/${anti}_koverdataset_${j} \
#         --split ./log/temp/loose/${species}/${anti}_id \
#         --hp-choice bound \
#         --criterion gini \
#         --max-depth 20 \
#         --min-samples-split 2 \
#         --class-importance 0.25 0.5 0.75 1.0 \
#         --n-cpu 5 \
#         --output-dir ./log/temp/loose/${species}/${anti}_temp/tree_b_${j} \
#         --progress \
##         --bound-max-genome-size #By default number of k-mers in the dataset is used.


    done
    wait
done
