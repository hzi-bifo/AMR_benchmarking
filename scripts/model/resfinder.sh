#!/bin/bash
#SBATCH --job-name=res  # Name of job
#SBATCH --output=/vol/cluster-data/khu/sge_stdout_logs/%x_%j.out  # stdout
#SBATCH --error=/vol/cluster-data/khu/sge_stdout_logs/%x_%j.err  # stderr
#SBATCH --partition=cpu  # partition to use (check with sinfo)
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1 # Number of tasks | Alternative: --ntasks-per-node
#SBATCH --threads-per-core=1 # Ensure we only get one logical CPU per core
#SBATCH --cpus-per-task=1 # Number of cores per task
#SBATCH --mem=20G # Memory per node | Alternative: --mem-per-cpu
#SBATCH --time=48:00:00 # wall time limit (HH:MM:SS)
#SBATCH --clusters=bioinf

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
source activate ${multi_env_name}

IFS=', ' read -ra species_list_temp <<< "$species_list"
species=( "${species_list_temp[@]//_/ }" )
IFS=', ' read -ra species_list_temp_tree <<< "$species_list_phylotree"
species_tree=( "${species_list_temp_tree[@]//_/ }" )






echo "Point-/Resfinder blastn version:"
#python ./AMR_software/resfinder/main_run_blastn.py -path_sequence ${dataset_location} -temp ${log_path} --n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in resfinder running. Exit ."; exit; }
echo "Point-/Resfinder KMA version:"
#python ./AMR_software/resfinder/main_run_kma.py -path_sequence ${dataset_location} -temp ${log_path} --n_jobs ${n_jobs} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in resfinder running. Exit ."; exit; }
echo "Point-/Resfinder extract results:"
#python ./AMR_software/resfinder/extract_results.py -s "${species[@]}" -f_no_zip -o ${output_path} -temp ${log_path} || { echo "Errors in resfinder results summarize. Exit ."; exit; }
###
###evaluate resfinder under CV folds
#echo "Evaluate Point-/Resfinder under CV folds:"
#python ./AMR_software/resfinder/main_resfinder_folds.py -f_phylotree -cv ${cv_number} -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria}  -f_no_zip|| { echo "Errors in resfinder running. Exit ."; exit; }
#python ./AMR_software/resfinder/main_resfinder_folds.py -f_kma -cv ${cv_number} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}  -f_no_zip|| { echo "Errors in resfinder running. Exit ."; exit; }
#python ./AMR_software/resfinder/main_resfinder_folds.py -cv ${cv_number} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}  -f_no_zip|| { echo "Errors in resfinder running. Exit ."; exit; }
#

###CV socres to table
python ./src/analysis_utility/result_analysis.py -software 'resfinder_folds' -f_phylotree -fscore 'f1_macro' -cl_list 'resfinder' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'resfinder_folds' -f_kma -fscore 'f1_macro' -cl_list 'resfinder'  -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'resfinder_folds'  -fscore 'f1_macro' -cl_list 'resfinder' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
#conda deactivate



#conda deactivate
#echo "Point-/Resfinder finished successfully."
