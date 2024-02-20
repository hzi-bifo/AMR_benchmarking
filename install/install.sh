#!/bin/bash


check_conda_channels () {
	## ensure conda channels
	echo '+check conda channels...'
	for c in hzi-bifo conda-forge/label/broken bioconda conda-forge defaults r pytorch anaconda ; do
		echo '+'$c
		if [ $(conda config --get channels | grep $c | wc -l) -eq 0 ]; then
			conda config --add channels $c
		fi
	done
}
check_conda_channels ||{ echo "Errors in setting conda channels. Please set it by hand yourself."; exit; }


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


###1.AMR benchmarking general use
conda env create -n ${amr_env_name}  -f ./install/amr_env.yml  || { echo "Errors in installing env."; exit; }
conda env create -n ${amr_env_name2}  -f ./install/amr_env2.yml python=3.8 || { echo "Errors in installing env."; exit; }
####2.Point-/ResFinder.
conda env create -n ${resfinder_env}  -f ./install/res_env.yml || { echo "Errors in installing env."; exit; }

####3.PhenotypeSeeker.
conda env create -n ${PhenotypeSeeker_env_name}  -f ./install/PhenotypeSeeker_env.yml python=3.7 || { echo "Errors in installing env."; exit; }



##4.multi_bench.
conda create -n ${multi_env_name} -y python=3.6
source activate ${multi_env_name}|| { echo "Errors in activate env."; exit; }
echo $CONDA_DEFAULT_ENV
pip install -r install/multi_env.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu|| { echo "Errors."; exit; }
conda deactivate
echo " ${multi_env_name} created successfully."
#
##Please install pytorch manually here.
## To install pytorch compatible with your CUDA version, please fellow this instruction: https://pytorch.org/get-started/locally/.
## Our code was tested with pytorch v1.7.1, with CUDA Version 10.1 and 11.0 .
#######################################################################################################################
conda create -n ${multi_torch_env_name} -y  python=3.6
conda init
source activate ${multi_torch_env_name}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
pip install -r install/multi_env.txt
conda deactivate
echo " To install pytorch compatible with your CUDA version, please fellow this instruction: https://pytorch.org/get-started/locally/."



##5.Kover #Tested Dec 6. successful only after updating Kover version.
###conda env create -n ${kover_env_name}  -f ./install/kover_env.yml
conda create -n ${kover_env_name} -y  python=2.7
source activate ${kover_env_name}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
pip install -r install/kover_env.txt
conda deactivate
echo " ${kover_env_name} created successfully."
#

##6.phylo_r for generating phylo-tree.
###conda create -n ${perl_name}  -f ./install/perl5_22_env.yml python=2.7 || { echo "Errors."; exit; }
conda env create -n ${phylo_name} -y  python=3.9
source activate ${phylo_name}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
conda install -y -c conda-forge r-base=4.1.0|| { echo "Errors."; exit; }
conda deactivate
echo " ${phylo_name} created successfully."

#
###7.  ${kmer_env_name}: generating kmer matrix. KMC.
conda create -n ${kmer_env_name} -y  python=3.6
source activate ${kmer_env_name}|| { echo "Errors in activate env."; exit; }
echo $CONDA_DEFAULT_ENV
conda install -c bioconda  -y kmc=3.1|| { echo "Errors."; exit; } # we used 3.1.2rc1 version, but now it is not available anymore.
conda install -y numpy=1.19.1|| { echo "Errors."; exit; }
conda install -y pandas=1.1.3|| { echo "Errors."; exit; }
conda install -y tables==3.6.1|| { echo "Errors."; exit; }
conda deactivate


###8. #for visualization of misclassified genomes on trees.
conda create -n ${phylo_name2} -y  python=3.9
source activate ${phylo_name2}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
conda install -y -c conda-forge r-base=4.2.0|| { echo "Errors."; exit; }
conda deactivate





