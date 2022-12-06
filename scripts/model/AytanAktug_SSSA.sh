#!/bin/bash

# Single-species-antibiotic model

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
export PATH=~/miniconda2/bin:$PATH
export PYTHONPATH=$PWD
source activate ${multi_env_name}
#source activate ${multi_torch_env_name}
wait
echo $CONDA_DEFAULT_ENV
IFS=', ' read -ra species_list_temp <<< "$species_list"
species=( "${species_list_temp[@]//_/ }" )

IFS=', ' read -ra species_list_temp_tree <<< "$species_list_phylotree"
species_tree=( "${species_list_temp_tree[@]//_/ }" )

#### Initialization
python ./AMR_software/AytanAktug/main_SSSA.py -f_initialize -temp ${log_path} --n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSSA initialization. Exit ."; exit; }
echo "Finished: initialize."


###########################################################################
# ## Folds are provide in ./data/PATRIC/cv_folds
### if you want to generate the CV folds again, uncomment following block:
############################################################################
###install kma_clustering
#SCRIPT=$(realpath "$0")
#SCRIPTPATH=$(dirname "$SCRIPT")
#cd ./AMR_software/AytanAktug/cluster_preparation
#gcc -O3 -o kma_clustering kma_clustering.c -lm -lz
#mv kma_clustering ~/bin/
#cd SCRIPTPATH
#cd ../..
#
#python ./AMR_software/AytanAktug/main_SSSA.py -f_pre_cluster -path_sequence ${dataset_location} -temp ${log_path} -n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug pre-clustering. Exit ."; exit; }
#echo "Finished: merge_scaffold."
##
###
#for s in "${species_list_temp[@]}"; \
#do bash ./AMR_software/AytanAktug/cluster_preparation/cluster_SSSA.sh ${s} ${log_path};done || { echo "Errors in Aytan-Aktug KMA clustering. Exit ."; exit; }
#echo "Finished: clustering."
#
##
#python ./src/cv_folds/generate_random_folds.py -s 'Mycobacterium tuberculosis' -l ${QC_criteria} -cv ${cv_number}|| { echo "Errors in folds regenerating. Exit ."; exit; }
#python ./src/cv_folds/prepare_folds.py -f_kma -s "${species[@]}" -l ${QC_criteria} -cv ${cv_number} -temp ${log_path}|| { echo "Errors in folds regenerating. Exit ."; exit; }
#echo "Folds regenerated."
############################################################################


### Feature preparing.
python ./AMR_software/AytanAktug/main_SSSA.py  -f_res -temp ${log_path} -n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSSA feature building 1. Exit ."; exit; }
python ./AMR_software/AytanAktug/main_SSSA.py  -f_merge_mution_gene -temp ${log_path} -n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSSA feature building 2. Exit ."; exit; }
python ./AMR_software/AytanAktug/main_SSSA.py  -f_matching_io -temp ${log_path} -n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in Aytan-Aktug SSSA feature building 3. Exit ."; exit; }
echo "Finished: features."



### nested CV
if [ "$gpu_on" = True ]
then
  #### we modified their codes by adding in early stop mechanism (patience 200), dropout (0, 0.2) hyperparameters, and a hyperparameter optimization procedure
  python ./AMR_software/AytanAktug/main_SSSA.py  -f_phylotree -f_nn -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria} -learning 0.0 -e 0 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py  -f_kma -f_nn -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.0 -e 0 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py  -f_nn -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.0 -e 0 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }

  ### Aytan-Aktug oroginal version.
  python ./AMR_software/AytanAktug/main_SSSA.py  -f_phylotree -f_nn_base -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria} -learning 0.001 -e 1000 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py  -f_kma -f_nn_base -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.001 -e 1000 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py  -f_nn_base -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.001 -e 1000 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }


else #parallel on CPUs
  ### we modified their codes by adding in early stop mechanism (patience 200), dropout (0, 0.2) hyperparameters, and a hyperparameter optimization procedure
  python ./AMR_software/AytanAktug/main_SSSA.py -cv ${cv_number} -n_jobs ${n_jobs} -f_cpu -f_phylotree -f_nn -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria} -learning 0.0 -e 0 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py -cv ${cv_number} -n_jobs ${n_jobs} -f_cpu -f_kma -f_nn -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.0 -e 0 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py -cv ${cv_number} -n_jobs ${n_jobs} -f_cpu -f_nn -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.0 -e 0 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }

  ##Aytan-Aktug oroginal version.
  python ./AMR_software/AytanAktug/main_SSSA.py -cv ${cv_number} -n_jobs ${n_jobs} -f_cpu -f_phylotree -f_nn_base -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria} -learning 0.001 -e 1000 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py -cv ${cv_number} -n_jobs ${n_jobs} -f_cpu -f_kma -f_nn_base -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.001 -e 1000 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }
  python ./AMR_software/AytanAktug/main_SSSA.py -cv ${cv_number} -n_jobs ${n_jobs} -f_cpu -f_nn_base -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} -learning 0.001 -e 1000 -f_fixed_threshold -f_optimize_score 'f1_macro' || { echo "Errors in Aytan-Aktug SSSA NN. Exit ."; exit; }


fi

conda deactivate
source activate ${amr_env_name}
wait
### CV score generation.
python ./src/analysis_utility/result_analysis_AytanAktug.py -f_SSSA -cv ${cv_number} -f_phylotree -s "${species_tree[@]}" \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}

python ./src/analysis_utility/result_analysis_AytanAktug.py -f_SSSA -cv ${cv_number} -f_kma -s "${species[@]}" \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}

python ./src/analysis_utility/result_analysis_AytanAktug.py -f_SSSA -cv ${cv_number} -s "${species[@]}" \
-f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -temp ${log_path} -o ${output_path} -l ${QC_criteria}


conda deactivate
echo "Aytan-Aktug single-species-antibiotic model running finished successfully."
