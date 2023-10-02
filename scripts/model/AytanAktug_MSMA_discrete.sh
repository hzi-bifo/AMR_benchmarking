#!/bin/bash


# Multi-species multi-antibiotics discrete databases model
#For user defining species combinations, please replace -f_all in this script with -s "${species[@]}".

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




python ./AMR_software/AytanAktug/main_MSMA_discrete.py -f_pre_meta -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA discrete prepare metadata. Exit ."; exit; }
echo "Finished: initialize."

##########################################################################
## Folds are provide in ./data/PATRIC/cv_folds
# if you want to generate the CV folds again, uncomment following block:
##########################################################################
##KMA-based folds
#python ./AMR_software/AytanAktug/main_MSMA_discrete.py -f_pre_cluster -path_sequence ${dataset_location} -temp ${log_path} -f_all -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA discrete prepare clustering files. Exit ."; exit; }

#for s in "${species_list_temp[@]}"; \
#do bash ./AMR_software/AytanAktug/cluster_preparation/cluster_MSMA.sh ${s} ${log_path} ${merge_name}; done

#python ./AMR_software/AytanAktug/main_MSMA_discrete.py  -f_kma -f_cluster_folds -cv ${cv_number_multiS} -temp ${log_path} -f_all -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA discrete folds generating. Exit ."; exit; }
#echo "Finished: clustering."
##########################################################################


#### Feature preparing
python ./AMR_software/AytanAktug/main_MSMA_discrete.py -f_res -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA discrete feature building. Exit ."; exit; }
python ./AMR_software/AytanAktug/main_MSMA_discrete.py -f_merge_mution_gene -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA discrete feature building. Exit ."; exit; }
python ./AMR_software/AytanAktug/main_MSMA_discrete.py  -f_matching_io -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA discrete feature building. Exit ."; exit; }



#### CV
#cv_number_multiS=6
cv_iter=$((cv_number_multiS - 2))
echo ${cv_iter} #should be 4

for i in $(seq 0 cv_iter);do
###for i in {4..4};do
echo ${i}
python ./AMR_software/AytanAktug/main_MSMA_discrete.py  -i_CV ${i} -cv ${cv_number_multiS} -f_kma -f_nn -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA CV. Exit ."; exit; }
done

python ./AMR_software/AytanAktug/main_MSMA_discrete.py  -f_kma -f_nn -cv ${cv_number_multiS} -f_nn_score -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_all -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug MSMA CV. Exit ."; exit; }





conda deactivate
source activate ${amr_env_name}
wait
### CV score generation.
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}

python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_macro'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_negative'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_positive'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'accuracy'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'precision_negative'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_split_species -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'recall_negative'
#
#
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_macro'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_negative'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'f1_positive'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'accuracy'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'clinical_f1_negative'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'clinical_precision_negative'
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_MSMA_discrete -f_match_single -f_kma -f_all \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria} -fscore 'clinical_recall_negative'

conda deactivate
echo "Aytan-Aktug multi-species multi-antibiotics discrete databases model running finished successfully."
