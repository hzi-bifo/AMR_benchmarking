#!/bin/bash
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


IFS=', ' read -ra species_list_temp <<< "$species_list"
species=( "${species_list_temp[@]//_/ }" )

IFS=', ' read -ra species_list_temp_tree <<< "$species_list_phylotree"
species_tree=( "${species_list_temp_tree[@]//_/ }" )
export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD

# Initialization
source activate ${amr_env_name}
python ./AMR_software/Kover/main_kover.py -f_phylotree -cv ${cv_number} -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in kover initializing. Exit ."; exit; }
python ./AMR_software/Kover/main_kover.py -f_kma -cv ${cv_number} -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in kover initializing. Exit ."; exit; }
python ./AMR_software/Kover/main_kover.py  -cv ${cv_number} -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in kover initializing. Exit ."; exit; }
conda deactivate
wait

####################################
### 1.  Running Kover pipeline
####################################
source activate ${kover_env_name}
wait
#### Prepare data sets
for s in "${species_list_temp_tree[@]}"; do
bash ./AMR_software/Kover/run_data.sh ${s} ${log_path}log/software/kover/software_output/phylotree;done

for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/run_data.sh ${s} ${log_path}log/software/kover/software_output/kma;done

for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/run_data.sh ${s} ${log_path}log/software/kover/software_output/random ;done



#### Running bound selection CV
for s in "${species_list_temp_tree[@]}"; do
bash ./AMR_software/Kover/run_bs.sh ${s} ${log_path}log/software/kover/software_output/phylotree ${n_jobs} ;done
#
for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/run_bs.sh ${s} ${log_path}log/software/kover/software_output/kma ${n_jobs};done

for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/run_bs.sh ${s} ${log_path}log/software/kover/software_output/random ${n_jobs} ;done
#
conda deactivate
wait

####################################
### 2. CV : classifier selection Initialization
####################################
source activate ${amr_env_name}
python ./AMR_software/Kover/main_kover.py -f_phylotree -cv ${cv_number} -f_prepare_meta_val -path_sequence ${dataset_location} -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria} || { echo "Errors in kover initializing. Exit ."; exit; }
python ./AMR_software/Kover/main_kover.py -f_kma -cv ${cv_number} -f_prepare_meta_val -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in kover initializing. Exit ."; exit; }
python ./AMR_software/Kover/main_kover.py  -cv ${cv_number} -f_prepare_meta_val -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in kover initializing. Exit ."; exit; }
conda deactivate


##### Running Kover pipeline
source activate ${kover_env_name}
wait
##### Prepare data sets
for s in "${species_list_temp_tree[@]}"; do
bash ./AMR_software/Kover/run_val.sh ${s} ${log_path}log/software/kover/software_output_val/phylotree;done

for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/run_val.sh  ${s} ${log_path}log/software/kover/software_output_val/kma;done
#
for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/run_val.sh  ${s} ${log_path}log/software/kover/software_output_val/random  ;done
conda deactivate
####################################
### 3. CV score generation.
####################################
source activate ${amr_env_name}
wait
python ./AMR_software/Kover/kover_analyse.py -f_phylotree -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}

python ./AMR_software/Kover/kover_analyse.py -f_kma -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

python ./AMR_software/Kover/kover_analyse.py -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

conda deactivate
