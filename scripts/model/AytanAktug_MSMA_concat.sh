#!/bin/bash



### Multi-species multi-antibiotics discrete databases model
### Note: this scripts should be run after AytanAktug_MSMA_discrete.sh, from whom some intermediate files will be used.
### For user defining species combinations, please replace -f_all in this script with -s "${species[@]}".

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml Config.yaml)

export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
#source activate ${multi_env_name}
source activate ${multi_torch_env_name}

IFS=', ' read -ra species_list_temp <<< "$species_list_multi_species"
species=( "${species_list_temp[@]//_/ }" )

python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_pre_meta -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA concatenated prepare metadata. Exit ."; exit; }
echo "Finished: initialize."




##########################################################################
## Folds are provide in ./data/PATRIC/cv_folds
# if you want to generate the CV folds again, uncomment following block:
##########################################################################
python ./AMR_software/AytanAktug/main_MSMA_concat.py  -f_kma -f_cluster_folds -cv 5 -temp ${log_path} -f_all -l ${QC_criteria}|| { echo "Errors in Aytan-Aktug MSMA concatenated folds prepare. Exit ."; exit; }

##########################################################################
# Prepare concatenated reference database for Point-/ResFinder
# We have already prepared the database, if you would like to re-build it, uncomment the following comments
##########################################################################
#cd ./AMR_software/resfinder
#python merge_database.py
#cd $BASEDIR
###########################################################################

##### index the merged database with KMA
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd ./AMR_software/resfinder/db_pointfinder
python3 INSTALL.py ${BASEDIR}/AMR_software/resfinder/cge/kma/kma non_interactive/
echo $SCRIPTPATH
cd $SCRIPTPATH
cd ../..

####running resfinder, based on our merged database
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_run_res -n_jobs ${n_jobs} -f_all -path_sequence ${dataset_location} -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA concatenated prepare metadata. Exit ."; exit; }
echo "Finished: use Point-/ResFinder to extract SNPs and genes."

#### analyze resfinder results.
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_res -f_all -temp ${log_path} -l ${QC_criteria}
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_matching_io -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA concatenated feature building. Exit ."; exit; }
echo "Finished: feature."

################################################
###1. Concatenated databases leave-one(-species)-out multi-species model
### version 1 , you can parallel of evaluating/testing on several species separately by manually setting i in the below before submitting to the GPU node.
## As the evaluating is very time consuming, you are advised to use version 2 instead of version 1.
################################################
for i in $(seq 0 ${#species[@]});do
echo ${i}
python ./AMR_software/AytanAktug/main_MSMA_concat_version1.py -f_nn -i_CV ${i} -cv 5 -f_all -f_kma -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug leave-one(specie)-out MSMA concatenated databases model. Exit ."; exit; }
done

echo "Finished: leave-one(specie)-out MSMA concatenated databases model."

################################################
###1. Concatenated databases leave-one(-species)-out multi-species model
### version 2 , you can parallel of both evaluating/testing on several species and CV in each evaluation by  manually setting i anf j below.
### i represents one of 9 species, and j represents one of CV loops.
### As the evaluating is very time consuming, you are advised to set i and j each time to a species number instead of a range before submitting to your GPU.
################################################
## Log: new Sep 27th
###i=0-8;j=0-4
for i in {0..8};do
echo ${i}
for j in {0..4};do
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_nn -i_CV ${i} -j_CV ${j} -cv 5 -f_all -f_kma -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug leave-one(specie)-out MSMA concatenated databases model. Exit ."; exit; }
done
done

for i in {0..8};do
echo ${i}
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_nn -f_nn_score -i_CV ${i} -cv 5 -f_all -f_kma -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug leave-one(specie)-out MSMA concatenated databases model. Exit ."; exit; }
done

################################################
##2. Concatenated databases mixed(-species) multi-species model
################################################
echo "check CV number, should be 6."
echo "${cv_number_multiS}"
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_nn_all_io -f_all -f_nn_all -f_kma -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug mixed-species MSMA concatenated databases model feature preparation. Exit ."; exit; }

#####
cv_number_mix=$((cv_number_multiS - 2))
echo ${cv_number_mix} #should be 4
for i in $(seq 0 ${cv_number_mix});do
###for i in {4..4};do
echo ${i}
python ./AMR_software/AytanAktug/main_MSMA_concat.py  -f_all -f_nn_all -f_kma -i_CV ${i} -cv ${cv_number_multiS}  -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria}
done
python ./AMR_software/AytanAktug/main_MSMA_concat.py -f_all -f_nn_all -f_kma -f_nn_score -cv ${cv_number_multiS} -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug mixed-species MSMA concatenated databases model testing. Exit ."; exit; }


conda deactivate
source activate ${amr_env_name}
wait
### CV score generation.
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}

python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_negative'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_positive'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'accuracy'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'precision_neg'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conMix -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'recall_neg'

#
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_conLOO  -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}


conda deactivate
echo "Finished: mixed-species MSMA concatenated databases model."
