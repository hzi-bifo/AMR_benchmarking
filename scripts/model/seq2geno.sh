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
#source activate ${multi_env_name}
source activate ${amr_env_name}

### Initialization
python ./AMR_software/seq2geno/main_s2p.py -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria} || { echo "Errors in S2G initializing. Exit ."; exit; }
conda deactivate


### Runnning seq2geno
for s in "${species_list_temp_tree[@]}"; \
do bash ./AMR_software/seq2geno/run_s2g.sh ${s} ${log_path} ;done
echo "Finished: seg2geno."



### Generating phylo-trees, based on which phylogeny-aware folds were generated.
export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
source activate ${phylo_r_name}
for s in "${species_list_temp_tree[@]}"; \
do  Rscript --vanilla ./src/cv_folders/phylo_tree.r -f ${log_path}log/software/seq2geno/software_output/${species}/results/denovo/roary/core_gene_alignment_renamed.aln -o d+ ${log_path}log/software/seq2geno/software_output/${species}/results/denovo/roary/nj_tree.newick ;done
echo "Finished: phylo-trees ."
conda deactivate

### Generate 6-mer matrix for all speceis samples.
export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
source activate ${kmer_env_name}
bash ./AMR_software/seq2geno/kmc.sh ${dataset_location} ${log_path}
python ./AMR_software/seq2geno/k_mer.py -c -temp ${log_path} -l ${QC_criteria} -k 6 -s "${speciesPhylotree[@]}" -n_jobs ${n_jobs}|| { echo "Errors in kmer generating. Exit ."; exit; }
conda deactivate
echo "Seg2Geno model finished successfully, you need to use Geno2Pheno via https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno."


#########################################################################################################
###  To run Geno2Pheno, please refer to https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno
#########################################################################################################

### CV score generation.
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}

python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species[@]}" -l ${QC_criteria}
conda deactivate
