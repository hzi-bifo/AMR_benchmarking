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





### 0) Generate intermediate files for plotting and winner analysis.
### Paired T test of F1-macro between default hyperparameters and hyperparameter optimization.
### Paired T test of F1-macro between any 2 of SSSA,MSMA_discrete,MSMA_concat_mixedS , MSMA_concat_LOO
### note: Species list hard coded.
python ./src/benchmark_utility/lib/AytanAktug/excel_multi_analysis.py  -f_compare -f_Ttest -fscore 'f1_macro'

### 1) Generate tables of all AytanAktug-related results (F1-macro,F1-pos,F1-neg,accuracy) and clinical oriented metrics. Sup F7 & Sup F8
python ./src/benchmark_utility/lib/AytanAktug/excel_multi.py  -f_compare -s "${species[@]}"


### 2) Fig. 7 Radar visualization of SSSA and SSMA
python ./src/benchmark_utility/lib/AytanAktug/vis_radar_SSSA_SSMA.py  -o ${output_path} -fscore 'f1_macro'


### 3) Radar plot of SSSA, MSMA_discrete, MSMA_concat_mixedS (supplement File 2. S12)
### scores were based on samples from multiple species. Aytan-Aktug article version analysis
python ./src/benchmark_utility/lib/AytanAktug/vis_radar_SSSA_MSMA.py -f_all -f_fixed_threshold -o ${output_path}


### 4)Fig. 8 Bar plot of  SSSA,MSMA_discrete,MSMA_concat_mixedS , MSMA_concat_LOO
python ./src/benchmark_utility/lib/AytanAktug/vis_bar_SSSA_MSMA.py -f_all -f_fixed_threshold  -o ${output_path}
python ./src/benchmark_utility/lib/AytanAktug/vis_box.py -f_all -f_fixed_threshold  -o ${output_path}


conda deactivate
