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



### Initialization
source activate ${amr_env_name}
python ./AMR_software/seq2geno/main_s2p.py -f_prepare_meta -path_sequence ${dataset_location} -temp ${log_path} -s "${species[@]}" -l ${QC_criteria}  -n_jobs ${n_jobs}|| { echo "Errors in S2G initializing. Exit ."; exit; }
conda deactivate



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
check_conda_channels ||{ echo "Errors in setting conda channels"; exit; }
source activate ${se2ge_env_name}
for s in "${species_list_temp_tree[@]}"; \
do
  bash ./AMR_software/seq2geno/run_s2g.sh ${s} ${log_path} >  "${log_path}log/software/seq2geno/software_output/${s}/log.txt" ;done
echo "Finished: seg2geno."
conda deactivate

##if above process does not generate core_gene_alignment.aln in ${log_path}log/software/seq2geno/software_output/Campylobacter_jejuni/results/denovo/roary folder. In our experience, this may happen in different operating system, due to Perl dependencies issues.
### Then you need to generate it using a more recent Roary version (tested using Dec 31 2022 version) instead of the one in Seq2Geno
### Please re-download and re-install our AMR_benchmarking git repository at a new location,
### and manually copy all the contents in ./AMR_software/seq2geno_3 to ./AMR_software/seq2geno_2, and then rename the se2ge_env_name item in ./Config.yaml
### Then run the Seq2Geno as it specified in ./main.sh as usual.
### You will find core_gene_alignment.aln in roary folder
### This is the input file of the next step. Through nest step, we generate a phylogeny tree for each species' samples.
# Based on the tree, Geno2Pheno generate phylogeny-awareness folds.


export PATH=$( dirname $( dirname $( which conda ) ) )/bin:$PATH
export PYTHONPATH=$PWD
source activate ${phylo_name}
for s in "${species_list_temp_tree[@]}"; \
do
  Rscript --vanilla ./src/cv_folds/phylo_tree.r \
  --file ${log_path}log/software/seq2geno/software_output/${s}/results/denovo/roary2/core_gene_alignment.aln \
   --out  ${log_path}log/software/seq2geno/software_output/${s}/results/denovo/roary2/nj_tree.newick ;done
echo "Finished: phylo-trees."
conda deactivate
##
## Generate 6-mer matrix for all speceis samples.
source activate ${kmer_env_name}
bash ./AMR_software/seq2geno/kmc.sh ${dataset_location} ${log_path}
python ./AMR_software/seq2geno/k_mer.py -c -temp ${log_path} -l ${QC_criteria} -k 6 -s "${speciesPhylotree[@]}" -n_jobs ${n_jobs}|| { echo "Errors in kmer generating. Exit ."; exit; }
conda deactivate
echo "Seg2Geno model finished successfully, you need to use Geno2Pheno via https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno."


#########################################################################################################
###  To run Geno2Pheno, please refer to https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno
#########################################################################################################

