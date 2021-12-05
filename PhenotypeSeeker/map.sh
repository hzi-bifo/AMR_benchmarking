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

        rm -rf ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}
        rm -rf ./log/temp/loose/${species}/${anti}_temp/CV_te${j}
        echo "CV ${j}"
        # 1).training-------------------
        mkdir -p ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}
        readarray -t id_list <  ./log/temp/loose/${species}/${anti}_Train_${j}_id
#       echo "glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/${anti}_feature_vector ${id_list[@]}"


        length=${#id_list[@]}
        if [[ $length -gt 500 ]]
        then
            g=500

            for((i=0; i < ${#id_list[@]}; i+=g))
            do
              part=( "${id_list[@]:i:g}" )
#              echo "Elements in this group: ${part[*]}"
              glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_${i}  ${part[*]}
            done
            #todo: need to check closely
#            echo "glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector  ${anti}_feature_vector_*_13_union.list"
            glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector  ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_*_13_union.list

        else
          glistcompare -u -o ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector ${id_list[@]}
        fi

        cat ./log/temp/loose/${species}/${anti}_Train_${j}_id2|
        while read sample; do #note: will rewrite every time.
#            echo "./log/temp/loose/${species}/${anti}_temp/${sample}_mapped"
#            echo " glistquery ${sample}  -l ./log/temp/loose/${species}/${anti}_temp/${anti}_feature_vector_13_union.list "
            #1. get kmer results w.r.t. feature space for each sample
            glistquery  K-mer_lists/${sample}_0_13.list  -l ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_13_union.list  >  ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${sample}_mapped #mapping samples to feature vector space
            #2. get_mash_sketches for each sample
            mash sketch -r "${path}${sample}.fna"  -o  ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${sample}

        done
#        3.get_mash_distances. only for train.
        mash paste ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/reference.msh ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/*.msh
        rm -f ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/mash_distances.mat
        mash dist ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/reference.msh ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/reference.msh >./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/mash_distances.mat


        # For [Test] set
        mkdir -p ./log/temp/loose/${species}/${anti}_temp/CV_te${j}

#       #use the ${anti}_feature_vector_13_union.list from traning CV

        cat ./log/temp/loose/${species}/${anti}_Test_${j}_id2|
        while read sample; do #note: will rewrite every time.
#            echo "./log/temp/loose/${species}/${anti}_temp/${sample}_mapped"
#            echo " glistquery ${sample}  -l ./log/temp/loose/${species}/${anti}_temp/${anti}_feature_vector_13_union.list "
            #1. get kmer results w.r.t. feature space for each sample
            glistquery  K-mer_lists/${sample}_0_13.list  -l ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_13_union.list  >  ./log/temp/loose/${species}/${anti}_temp/CV_te${j}/${sample}_mapped #mapping samples to feature vector space


        done

        python get_weights.py -s ${species} -cv ${j} -anti ${anti}
        #4. perform chi-square test,and filter the kmers that :N_samples_w_kmer <  2 or N_samples_wo_kmer < 2
        python chisquare.py -s ${species} -cv ${j} -anti ${anti}
        #5. filter the kmers according to the p-value
        python filter.py -s ${species} -cv ${j} -anti ${anti}

        # 2).testing------------------
        # no need to get weights

        python filter.py -s ${species} -cv ${j} -f_test -anti ${anti}


        rm -r ./log/temp/loose/${species}/${anti}_temp/CV_tr${j}/
        rm -r ./log/temp/loose/${species}/${anti}_temp/CV_te${j}/

    done

done



