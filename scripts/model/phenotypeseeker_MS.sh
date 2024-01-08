#!/bin/bash

### Leave-one-species-out evaluation of PhenotypeSeeker

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
python ./AMR_software/PhenotypeSeeker/multiSpecies/pts_multiSpecies.py -f_prepare_meta -temp ${log_path} -f_all -l ${QC_criteria}
conda deactivate


source activate ${PhenotypeSeeker_env_name}
### Prepare features #done
for s in "${species_list_temp[@]}"; \
do bash ./AMR_software/PhenotypeSeeker/kmer.sh ${dataset_location} ${log_path}  by_species_bq/id_${s};done

for s in "${species_list_multi_species[@]}"; \
do bash ./AMR_software/PhenotypeSeeker/multiSpecies/map_multi.sh ${s} ${log_path}log/software/phenotypeseeker/software_output \
${log_path}log/software/phenotypeseeker/software_output/MS ${dataset_location} ${log_path} ;done

conda deactivate
#
##### ML CV
source activate ${amr_env_name}
#wait
python ./AMR_software/PhenotypeSeeker/multiSpecies/pts_multiSpecies.py -cv ${cv_number} -f_ml -temp ${log_path} -f_all  -l ${QC_criteria} -n_jobs 40


#### CV score generation.
python ./AMR_software/PhenotypeSeeker/multiSpecies/pts_multianalysis.py -f_all -cl_list  'lr'   -temp ${log_path} -o ${output_path}



