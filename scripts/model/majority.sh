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
source activate ${amr_env_name}

#CV
python ./AMR_software/majority/main_majority.py -f_phylotree -cv ${cv_number} -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./AMR_software/majority/main_majority.py -f_kma -cv ${cv_number} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
python ./AMR_software/majority/main_majority.py  -cv ${cv_number} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}



###CV socres to table
python ./src/analysis_utility/result_analysis.py -software 'majority' -f_phylotree -fscore 'f1_macro' -cl_list 'majority' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'majority' -f_kma -fscore 'f1_macro' -cl_list 'majority'  -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'majority'  -fscore 'f1_macro' -cl_list 'majority' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
#conda deactivate
