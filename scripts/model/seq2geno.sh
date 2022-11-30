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





#source activate ${amr_env_name}
#### Initialization
#python ./AMR_software/seq2geno/main_s2p.py -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}  -n_jobs ${n_jobs}|| { echo "Errors in S2G initializing. Exit ."; exit; }
#conda deactivate



### Runnning seq2geno
check_conda_channels () {
	## ensure conda channels
	echo '+check conda channels...'
	for c in hzi-bifo conda-forge/label/broken bioconda conda-forge defaults; do
		echo '+'$c
		if [ $(conda config --get channels | grep $c | wc -l) -eq 0 ]; then
			conda config --add channels $c
		fi
	done
}
###check_conda_channels ||{ echo "Errors in setting conda channels"; exit; }
#source activate ${se2ge_env_name}
#for s in "${species_list_temp_tree[@]}"; \
#do
#  bash ./AMR_software/seq2geno/run_s2g.sh ${s} ${log_path} >  "${log_path}log/software/seq2geno/software_output/${s}/log.txt" ;done
#echo "Finished: seg2geno."
#conda deactivate

#
#### Generating phylo-trees, based on which phylogeny-aware folds were generated.
#source activate ${phylo_name}
#for s in "${species_list_temp_tree[@]}"; \
#do echo "${log_path}log/software/seq2geno/software_output/${s}/results/denovo/roary/core_gene_alignment.aln"
#  Rscript --vanilla ./src/cv_folds/phylo_tree.r \
#  --file ${log_path}log/software/seq2geno/software_output/${s}/results/denovo/roary/core_gene_alignment.aln \
#   -o d+ ${log_path}log/software/seq2geno/software_output/${species}/results/denovo/roary/nj_tree.newick ;done
#echo "Finished: phylo-trees."
#conda deactivate
###
#### Generate 6-mer matrix for all speceis samples.
#export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
#export PYTHONPATH=$PWD
#source activate ${kmer_env_name}
#bash ./AMR_software/seq2geno/kmc.sh ${dataset_location} ${log_path}
#python ./AMR_software/seq2geno/k_mer.py -c -temp ${log_path} -l ${QC_criteria} -k 6 -s "${speciesPhylotree[@]}" -n_jobs ${n_jobs}|| { echo "Errors in kmer generating. Exit ."; exit; }
#conda deactivate
#echo "Seg2Geno model finished successfully, you need to use Geno2Pheno via https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno."


#########################################################################################################
###  To run Geno2Pheno, please refer to https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno
#########################################################################################################

####todo: rm the following when finished.
source activate ${amr_env_name}
#python ./AMR_software/seq2geno/main_s2p.py  -f_phylotree -cv ${cv_number} -n_jobs ${n_jobs} -f_ml -temp ${log_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./AMR_software/seq2geno/main_s2p.py  -f_kma -cv ${cv_number} -n_jobs ${n_jobs} -f_ml -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
#python ./AMR_software/seq2geno/main_s2p.py  -cv ${cv_number} -n_jobs ${n_jobs} -f_ml -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}
#
#
#
#
#
#
### CV score generation.

#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'clinical_f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'clinical_precision_neg' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_phylotree -fscore 'clinical_recall_neg' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}


#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'clinical_f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'clinical_precision_neg' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno' -f_kma -fscore 'clinical_recall_neg' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}





#python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'f1_macro' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'f1_positive' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
##python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'accuracy' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'clinical_f1_negative' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'clinical_precision_neg' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}
#python ./src/analysis_utility/result_analysis.py -software 'seq2geno'  -fscore 'clinical_recall_neg' -cl_list 'svm' 'lr' 'rf' 'lsvm' -cv ${cv_number} -temp ${log_path} -o ${output_path} -s "${species_tree[@]}" -l ${QC_criteria}


conda deactivate
