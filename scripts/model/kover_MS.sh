#!/bin/bash
### Leave-one-species-out evaluation of Kover

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


IFS=', ' read -ra species_list_temp <<< "$species_list_multi_species"
species=( "${species_list_temp[@]//_/ }" )

export PATH=$( dirname $( dirname $( /usr/bin/which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD

# Initialization
source activate ${amr_env_name}
python ./AMR_software/Kover/multiSpecies/kover_multiSpecies.py  -f_all -cv ${cv_number} -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path}  -l ${QC_criteria}
conda deactivate
wait
source activate ${kover_env_name}

# Running Kover pipeline
for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/multiSpecies/run_data_multi.sh ${s} ${log_path}log/software/kover/software_output/MS ${n_jobs};done


### Running bound selection CV
for s in "${species_list_temp[@]}"; do
bash ./AMR_software/Kover/multiSpecies/run_cv_multi.sh ${s} ${log_path}log/software/kover/software_output/MS ${n_jobs};done

conda deactivate
wait

source activate ${amr_env_name}
### extract results to metric table.
python ./AMR_software/Kover/multiSpecies/kover_analyse_multi.py  -f_all  -temp ${log_path}  -l ${QC_criteria}

