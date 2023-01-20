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
export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
source activate ${amr_env_name}

IFS=', ' read -ra species_list_temp <<< "$species_list"
species=( "${species_list_temp[@]//_/ }" )



###1. Fig.1 Data set overview. sample size. Further annotated through Drawio before the version in our article.
python ./src/benchmark_utility/benchmark.py -f_sample  -o ${output_path}

###2. Fig. 2 Draw by Drawio.



###4. Fig. 4.
## bar plot generated through ./scripts/analysis_visualization/compare_supplement.sh.

###5. Fig. 5. Software performance (F1-macro)  of 5 methods + ML baseline. E. coli  phylogeny-aware folds
### Radar plot was generated through ./scripts/analysis_visualization/compare_supplement.sh.

###6. Fig. 6.
### Paired box plot
python  ./src/benchmark_utility/benchmark.py -f_robust -fscore 'f1_macro' -f_all -o ${output_path}

###Fig. 7-8 generated through ./scripts/analysis_visualization/AytanAktug_analysis.sh


###3. Fig. 3  heatmap (comparison of 4 methods with 3 folds panels.)
conda deactivate
source activate ${amr_env_name2}
python  ./src/benchmark_utility/benchmark.py -f_hmap -fscore 'f1_macro' -f_all -o ${output_path}
conda deactivate



