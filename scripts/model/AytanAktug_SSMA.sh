#!/bin/bash

#SBATCH --job-name=45SSMA  # Name of job
#SBATCH --output=/vol/cluster-data/khu/sge_stdout_logs/%x_%j.out  # stdout
#SBATCH --error=/vol/cluster-data/khu/sge_stdout_logs/%x_%j.err  # stderr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1 # Number of tasks | Alternative: --ntasks-per-node
#SBATCH --threads-per-core=1 # Ensure we only get one logical CPU per core
#SBATCH --cpus-per-task=1 # Number of cores per task
#SBATCH --mem=40G # Memory per node | Alternative: --mem-per-cpu
#SBATCH --time=48:00:00 # wall time limit (HH:MM:SS)
#SBATCH --clusters=bioinf

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
#export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
#export PYTHONPATH=$PWD
##source activate ${multi_env_name}
#source activate ${multi_torch_env_name}

IFS=', ' read -ra species_list_temp <<< "$species_list_multi_antibiotics"
species=( "${species_list_temp[@]//_/ }" )




#
#python ./AMR_software/AytanAktug/main_SSMA.py -f_pre_meta -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA prepare metadata. Exit ."; exit; }
#echo "Finished: initialize."
#################################################################################
#### Folds are provide in ./data/PATRIC/cv_folds
# if you want to generate the CV folds again, uncomment following block:
##KMA-based folds
##-----------------------------------------------------------------------------------
#python ./AMR_software/AytanAktug/main_SSMA.py -f_pre_cluster -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
#for s in "${species_list_temp[@]}"; \
#do bash ./AMR_software/AytanAktug/cluster_preparation/cluster_SSMA.sh ${s} ${log_path};done

#python ./AMR_software/AytanAktug/main_SSMA.py  -f_kma -f_cluster_folds -cv ${cv_number} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA folds generating. Exit ."; exit; }
#echo "Finished: clustering."
#################################################################################

#python ./AMR_software/AytanAktug/main_SSMA.py -f_res  -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA feature building. Exit ."; exit; }
#python ./AMR_software/AytanAktug/main_SSMA.py -f_merge_mution_gene -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA feature building. Exit ."; exit; }
#python ./AMR_software/AytanAktug/main_SSMA.py -f_matching_io -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA feature building. Exit ."; exit; }
echo "Finished: features."

#for i in {4..5};do
#python ./AMR_software/AytanAktug/main_SSMA.py -cv ${cv_number} -i_CV ${i} -f_kma -f_nn  -s "${species[@]}" -f_fixed_threshold -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA NN. Exit ."; exit; }
#done

#python ./AMR_software/AytanAktug/main_SSMA.py -cv ${cv_number} -f_kma -f_nn -f_nn_score  -s "${species[@]}" -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSMA NN scores generating. Exit ."; exit; }



#python ./src/analysis_utility/result_analysis_AytanAktug.py -f_SSMA -cv ${cv_number} -f_kma -s "${species[@]}" -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}




echo "Aytan-Aktug single-species multi-antibiotics model running finished successfully."


