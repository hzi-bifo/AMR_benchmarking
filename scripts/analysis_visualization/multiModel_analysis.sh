#!/bin/bash

######Finished Oct 17th 2022. Khu.

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
source activate ${amr_env_name}

wait
echo $CONDA_DEFAULT_ENV
IFS=', ' read -ra species_list_temp <<< "$species_list"
species=( "${species_list_temp[@]//_/ }" )





### 1) Generate intermediate files for plotting and winner analysis.
### Paired T test of F1-macro between default hyperparameters and hyperparameter optimization.
### Paired T test of F1-macro between two of Aytan-Aktug single- and multi- models
### note: Species list hard coded.
python ./src/benchmark_utility/lib/multiModel/excel_multi_analysis.py  -f_compare -f_Ttest -fscore 'f1_macro'

## Paired T test  F1-macro between Kover single-models and multi-species LOSO model
python ./src/benchmark_utility/lib/multiModel/excel_multi_analysis.py -f_kover  -f_Ttest -fscore 'f1_macro'
python ./src/benchmark_utility/lib/multiModel/excel_multi_analysis.py -f_pts  -f_Ttest -fscore 'f1_macro'


### 2) Generate tables of all AytanAktug-related results (F1-macro,F1-pos,F1-neg,accuracy) and clinical oriented metrics. Sup F7 & Sup F8
python ./src/benchmark_utility/lib/multiModel/excel_multi.py  -f_compare -s "${species[@]}"


### 3) Radar visualization of AytanAktug SSSA and SSMA
python ./src/benchmark_utility/lib/multiModel/vis_radar_SSSA_SSMA.py  -o ${output_path} -fscore 'f1_macro'


### 4) Radar plot of AytanAktug SSSA, MSMA_discrete, MSMA_concat_mixedS
### scores were based on samples from multiple species. Aytan-Aktug article version analysis
python ./src/benchmark_utility/lib/multiModel/vis_radar_SSSA_MSMA.py -f_all -f_fixed_threshold -o ${output_path}


### 5) Bar plot of AytanAktug SSSA,MSMA_discrete,MSMA_concat_mixedS , MSMA_concat_LOO and Kover, PhenotypeSeeker LOSO
python ./src/benchmark_utility/lib/multiModel/vis_bar_SSSA_MSMA.py -f_all -f_fixed_threshold  -o ${output_path}
python ./src/benchmark_utility/lib/multiModel/vis_bar_multi.py -f_all -f_fixed_threshold  -o ${output_path}
python ./src/benchmark_utility/lib/multiModel/vis_box.py -f_all -f_fixed_threshold  -o ${output_path}
python ./src/benchmark_utility/lib/multiModel/vis_box_multi.py -f_all -f_fixed_threshold  -o ${output_path}

conda deactivate
