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


### Initialization
export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
source activate ${amr_env_name}
python ./AMR_software/PhenotypeSeeker/main_pts.py -f_phylotree -f_prepare_meta -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./AMR_software/PhenotypeSeeker/main_pts.py -f_kma -f_prepare_meta -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
python ./AMR_software/PhenotypeSeeker/main_pts.py -f_prepare_meta -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
conda deactivate


source activate ${PhenotypeSeeker_env_name}
#### Prepare features
for s in "${species_list_temp[@]}"; \
do bash ./AMR_software/PhenotypeSeeker/kmer.sh ${dataset_location} ${log_path}  by_species_bq/id_${s};done


for s in "${species_list_temp_tree[@]}"; \
do bash ./AMR_software/PhenotypeSeeker/map.sh ${s} ${log_path}log/software/phenotypeseeker/software_output \
${log_path}log/software/phenotypeseeker/software_output/phylotree ${dataset_location} ${log_path} -f_phylotree;done
#
for s in "${species_list_temp[@]}"; \
do bash ./AMR_software/PhenotypeSeeker/map.sh ${s} ${log_path}log/software/phenotypeseeker/software_output  \
${log_path}log/software/phenotypeseeker/software_output/kma ${dataset_location} ${log_path} -f_kma;done

for s in "${species_list_temp[@]}"; \
do bash ./AMR_software/PhenotypeSeeker/map.sh ${s} ${log_path}log/software/phenotypeseeker/software_output  \
${log_path}log/software/phenotypeseeker/software_output/random ${dataset_location} ${log_path};done

conda deactivate

#### ML CV
export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
source activate ${amr_env_name}
python ./AMR_software/PhenotypeSeeker/main_pts.py -f_phylotree -cv ${cv_number} -n_jobs ${n_jobs} -f_ml -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./AMR_software/PhenotypeSeeker/main_pts.py -f_kma -cv ${cv_number} -n_jobs ${n_jobs} -f_ml -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
python ./AMR_software/PhenotypeSeeker/main_pts.py -cv ${cv_number} -n_jobs ${n_jobs} -f_ml -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}


### CV score generation.
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_phylotree -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_phylotree -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_phylotree -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_phylotree -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}


python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_kma -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_kma -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_kma -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker' -f_kma -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker'  -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker'  -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf'  -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker'  -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'phenotypeseeker'  -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

conda deactivate




