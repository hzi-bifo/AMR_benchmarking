#!/bin/bash


# Single-species multi-antibiotics model


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

IFS=', ' read -ra species_list_temp <<< "$species_list_multi_antibiotics"
species=( "${species_list_temp[@]//_/ }" )




#### Initialization.
#python ./AMR_software/AytanAktug/main_SSMA.py -f_pre_meta -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA prepare metadata. Exit ."; exit; }
#echo "Finished: initialize."
##################################################################################
##### Folds are provide in ./data/PATRIC/cv_folds
#### if you want to generate the CV folds again, uncomment following block:
#### KMA-based folds
###-----------------------------------------------------------------------------------
##python ./AMR_software/AytanAktug/main_SSMA.py -f_pre_cluster -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
##for s in "${species_list_temp[@]}"; \
##do bash ./AMR_software/AytanAktug/cluster_preparation/cluster_SSMA.sh ${s} ${log_path};done
#
##python ./AMR_software/AytanAktug/main_SSMA.py  -f_kma -f_cluster_folds -cv ${cv_number} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA folds generating. Exit ."; exit; }
##echo "Finished: clustering."
##################################################################################
#
#### Feature preparing.
#python ./AMR_software/AytanAktug/main_SSMA.py -f_res  -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA feature building. Exit ."; exit; }
#python ./AMR_software/AytanAktug/main_SSMA.py -f_merge_mution_gene -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA feature building. Exit ."; exit; }
#python ./AMR_software/AytanAktug/main_SSMA.py -f_matching_io -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA feature building. Exit ."; exit; }
#echo "Finished: features."
#
#
#### Nested CV
#for i in {3..3};do
##for i in {0..9};do
#python ./AMR_software/AytanAktug/main_SSMA.py -cv ${cv_number} -i_CV ${i} -f_kma -f_nn  -s "${species[@]}" -f_fixed_threshold -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA NN. Exit ."; exit; }
#done
##### To tear CV evaluation into smaller running jobs, you can set a smaller range for i (e.g. for i in {0..0};do ) each time when running above loop commands before proceeding to the rest.
##
##python ./AMR_software/AytanAktug/main_SSMA.py -cv ${cv_number} -f_kma -f_nn -f_nn_score  -s "${species[@]}" -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA NN scores generating. Exit ."; exit; }



conda deactivate
source activate ${amr_env_name}
wait
### CV score generation.
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_SSMA -cv ${cv_number} -f_kma -s "${species[@]}" -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}


conda deactivate
echo "Aytan-Aktug single-species multi-antibiotics model running finished successfully."


