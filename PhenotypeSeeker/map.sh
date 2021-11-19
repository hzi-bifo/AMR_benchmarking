#!/bin/bash
#2. generate uinions for the specific species and antibiotic---------------------------------------
#3. mapping samples to feature vector space--------------------------------------------------------
species="$1"
path="/vol/projects/BIFO/patric_genome/"
readarray -t Anti_List <  ./log/temp/loose/${species}/anti_list
for anti in ${Anti_List[@]};do


    mkdir -p ./log/temp/loose/${species}/${anti}_temp
    echo "$anti"


    for j in {0..9};do

        echo "CV ${j}"
        # 1).training-------------------
        mkdir -p ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}
        readarray -t id_list <  ./log/temp/loose/${species}/${anti}_Train_${j}_id
#        echo "glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/${anti}_feature_vector ${id_list[@]}"
        glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector ${id_list[@]}
        cat ./log/temp/loose/${species}/${anti}_Train_${j}_id2|
        while read sample; do #note: will rewrite every time.
#            echo "./log/temp/loose/${species}/${anti}_temp/${sample}_mapped"
#            echo " glistquery ${sample}  -l ./log/temp/loose/${species}/${anti}_temp/${anti}_feature_vector_13_union.list "
            #1. get kmer results w.r.t. feature space for each sample
            glistquery  K-mer_lists/${sample}_0_13.list  -l ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_13_union.list  >  ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${sample}_mapped #mapping samples to feature vector space
            #2. get_mash_sketches for each sample
            mash sketch -r "${path}${i}.fna"  -o  ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${sample}

        done
        #3.get_mash_distances
        mash paste ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/reference.msh ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/*.msh
        rm -f ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/mash_distances.mat
        mash dist ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/reference.msh ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/reference.msh >./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/mash_distances.mat
        python get_weights.py -s ${species} -cv ${j}
        #4. perform chi-square test
        python filter.py


        # 2).testing------------------
        # no need to get weights




    done
done