#!/bin/bash

#This script generates tables and graphs for supplementary files.

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

##1. A paired t test was performed to compare KMA and BLAST version Point-/ResFinder.
####Output location: Results/supplement_figures_tables/Pvalue_resfinder_kma_blast.json. P-value=0.54 (oct 15 2022)
####echo "A paired t test to compare KMA and BLAST version Point-/ResFinder:"
python ./AMR_software/resfinder/extract_results.py -s "${species[@]}" -fscore 'f1_macro' -f_no_zip -f_com -o ${output_path} -temp ${log_path} || { echo "Errors in resfinder results summarize. Exit ."; exit; }
#
#
#
####2. Supplemental File 1. Performance(F1-macro, negative F1-score, positive F1-score, accuracy) of five methods alongside with the baseline method (Majority)
##w.r.t. random folds, phylogeny-aware folds, and homology-aware folds, in the 10-fold nested cross-validation.
####Output location: Results/supplement_figures_tables/S1_cv_results.xlsx
#### Supplemental File 5. Performance of KMA-based Point-/ResFinder and BLAST-based Point-/ResFinder. Not evaluated with folds.
python ./src/benchmark_utility/benchmark.py -f_table -f_all -o ${output_path}

####3. Supplemental File 6. & Fig 3. heatmap.
### Generate Three pieces of software lists, each corresponded to evaluation under random folds, phylogeny-aware folds, and homology-aware folds.
## Generate tables for further analysis (ML comparison with ResFinder, ML baseline)
python  ./src/benchmark_utility/benchmark.py -f_table_analysis -fscore 'f1_macro' -f_all -o ${output_path}


####4.  Fig. 4 & Supplemental File 2 Fig. S5-7 Error bar plot.
python  ./src/benchmark_utility/benchmark.py -f_anti -fscore 'f1_macro' -f_all -o ${output_path}


####5.  Fig. 5 & Supplemental File 2 Fig. S3.  Radar plot.
python  ./src/benchmark_utility/benchmark.py -f_species -fscore 'f1_macro' -f_all -o ${output_path}

#### 6.  Fig. 6 & Supplemental File 2 Fig. S8-11.   Paired box plot
#### were generated through ./scripts/analysis_visualization/compare.sh


#### 7. Clinical-oriented performance analysis
python  ./src/benchmark_utility/benchmark.py -f_clinical_analysis -fscore 'clinical_f1_negative' -f_all -o ${output_path}
python  ./src/benchmark_utility/benchmark.py -f_clinical_analysis -fscore 'clinical_precision_neg' -f_all -o ${output_path}
python  ./src/benchmark_utility/benchmark.py -f_clinical_analysis -fscore 'clinical_precision_neg' -f_all -o ${output_path}


conda deactivate
