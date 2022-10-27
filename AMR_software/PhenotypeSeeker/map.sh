#!/bin/bash
#2. generate uinions for the specific species and antibiotic---------------------------------------
#3. mapping samples to feature vector space--------------------------------------------------------
species="$1"
kmer_path="$2"
feature_path="$3"
data_path="$4"
log_path="$5"
f_folds="$6"

readarray -t Anti_List <  ${feature_path}/${species}/anti_list
for anti in ${Anti_List[@]};do


    mkdir -p  ${feature_path}/${species}/${anti}_temp
    echo "$anti"


    for j in {0..9};do

        rm -rf  ${feature_path}/${species}/${anti}_temp/CV_tr${j}
        rm -rf  ${feature_path}/${species}/${anti}_temp/CV_te${j}
        echo "CV ${j}"
        # 1).training-------------------
        mkdir -p  ${feature_path}/${species}/${anti}_temp/CV_tr${j}
        readarray -t id_list <   ${feature_path}/${species}/${anti}_Train_${j}_id



        length=${#id_list[@]}

        if [[ $length -gt 1024 ]]
        then
            g=1024

            for((i=0; i < ${#id_list[@]}; i+=g))
            do
              part=( "${id_list[@]:i:g}" )
#              echo "Elements in this group: ${part[*]}"
              glistcompare -u -o  ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_${i}  ${part[*]}
            done

            glistcompare -u -o  ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector   ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_*_13_union.list

        else

          glistcompare -u -o  ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector ${id_list[@]}
        fi

        cat  ${feature_path}/${species}/${anti}_Train_${j}_id2|
        while read sample; do #note: will rewrite every time.
#            echo "${feature_path}/${species}/${anti}_temp/${sample}_mapped"
#            echo " glistquery ${sample}  -l ${feature_path}/${species}/${anti}_temp/${anti}_feature_vector_13_union.list "
            #1. get kmer results w.r.t. feature space for each sample
            glistquery   ${kmer_path}/K-mer_lists/${sample}_0_13.list  -l  ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_13_union.list  >  ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${sample}_mapped #mapping samples to feature vector space
            #2. get_mash_sketches for each sample
            mash sketch -r "${data_path}/${sample}.fna"  -o  ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${sample}

        done
#        3.get_mash_distances. only for train.
        mash paste ${feature_path}/${species}/${anti}_temp/CV_tr${j}/reference.msh ${feature_path}/${species}/${anti}_temp/CV_tr${j}/*.msh
        rm -f ${feature_path}/${species}/${anti}_temp/CV_tr${j}/mash_distances.mat
        mash dist ${feature_path}/${species}/${anti}_temp/CV_tr${j}/reference.msh ${feature_path}/${species}/${anti}_temp/CV_tr${j}/reference.msh >${feature_path}/${species}/${anti}_temp/CV_tr${j}/mash_distances.mat


        # For [Test] set
        mkdir -p ${feature_path}/${species}/${anti}_temp/CV_te${j}

#       #use the ${anti}_feature_vector_13_union.list from traning CV

        cat ${feature_path}/${species}/${anti}_Test_${j}_id2|
        while read sample; do #note: will rewrite every time.
#            echo "${feature_path}/${species}/${anti}_temp/${sample}_mapped"
#            echo " glistquery ${sample}  -l ${feature_path}/${species}/${anti}_temp/${anti}_feature_vector_13_union.list "
            #1. get kmer results w.r.t. feature space for each sample
            glistquery  ${kmer_path}/K-mer_lists/${sample}_0_13.list  -l ${feature_path}/${species}/${anti}_temp/CV_tr${j}/${anti}_feature_vector_13_union.list  >  ${feature_path}/${species}/${anti}_temp/CV_te${j}/${sample}_mapped #mapping samples to feature vector space


        done

        python ./AMR_software/PhenotypeSeeker/get_weights.py -s ${species} -cv ${j} -anti ${anti} -temp ${log_path} ${f_folds}
        #4. perform chi-square test,and filter the kmers that :N_samples_w_kmer <  2 or N_samples_wo_kmer < 2
        python ./AMR_software/PhenotypeSeeker/chisquare.py -s ${species} -cv ${j} -anti ${anti} -temp ${log_path} ${f_folds}
        #5. filter the kmers according to the p-value
        python ./AMR_software/PhenotypeSeeker/filter.py -s ${species} -cv ${j} -anti ${anti} -temp ${log_path} ${f_folds}

        # 2).testing------------------
        # no need to get weights

        python ./AMR_software/PhenotypeSeeker/filter.py -s ${species} -cv ${j} -f_test -anti ${anti} -temp ${log_path} ${f_folds}


        rm -r ${feature_path}/${species}/${anti}_temp/CV_tr${j}/
        rm -r ${feature_path}/${species}/${anti}_temp/CV_te${j}/

    done

done



