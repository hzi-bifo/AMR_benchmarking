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

config_file=options.yaml
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

##todo after version control saving then delete those not needed here.
####1.AMR benchmarking general use
### F1: tested working onw week before Nov12 2022.
#conda env create -n ${amr_env_name}  -f ./install/amr_env.yml python=3.7 || { echo "Errors in installing env."; exit; }
###----------------
###F2:
###conda create -n ${amr_env_name} -y python=3.7
###conda init
###source activate ${amr_env_name} || { echo "Errors in activate env."; exit; }
###wait
###echo $CONDA_DEFAULT_ENV
###conda install -y biopython=1.78|| { echo "Errors."; exit; }
###conda install -y hdf5=1.10.4|| { echo "Errors."; exit; }
###conda install -y pytables=3.6.1|| { echo "Errors."; exit; }
###pip install matplotlib==3.4.2|| { echo "Errors."; exit; }
###pip install numpy==1.20.1|| { echo "Errors."; exit; }
###pip install openpyxl==3.0.9|| { echo "Errors."; exit; }
###pip install pandas==1.2.2|| { echo "Errors."; exit; }
###pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
###pip install scipy==1.6.0|| { echo "Errors."; exit; }
###pip install seaborn==0.11.1|| { echo "Errors."; exit; }
###pip install sklearn==0.0|| { echo "Errors."; exit; }
###pip install tqdm==4.62.3|| { echo "Errors."; exit; }
######pip install xgboost==1.4.2|| { echo "Errors."; exit; }
###conda deactivate
###echo " ${amr_env_name} created successfully."
#
#####2.Point-/ResFinder. Tested Nov12, 2022
#conda env create -n ${resfinder_env}  -f ./install/res_env.yml python=3.8 || { echo "Errors in installing env."; exit; }
#
##
##conda create -n ${resfinder_env} -y python=3.8
##conda init
##source activate ${resfinder_env} || { echo "Errors in activate env."; exit; }
##wait
##echo $CONDA_DEFAULT_ENV
#####conda install -y blast=2.13.0 #note : blast=2.5.0 not working
#####conda install -c bioconda blast
#####conda install -c anaconda biopython
#####conda install -c anaconda gitpython
#####conda install -c anaconda tabulate
#####conda install -c anaconda cgecore
#####conda install -c conda-forge python-dateutil
#####conda install -c anaconda -y pandas
#####conda install -c conda-forge tqdm
##conda install -y biopython=1.78|| { echo "Errors."; exit; }
##conda install -y gitpython=3.1.18 || { echo "Errors."; exit; }
##conda install -y tabulate==0.8.10|| { echo "Errors."; exit; }
##conda install -y cgecore=1.5.6|| { echo "Errors."; exit; }
##conda install -y python-dateutil= 2.8.2|| { echo "Errors."; exit; }
##conda install -y blast=2.13.0|| { echo "Errors."; exit; }
##conda install -y pandas=1.4.3 || { echo "Errors."; exit; }
##conda install -y tqdm=4.64.1 || { echo "Errors."; exit; }
##conda install -y ete3==3.1.1
##conda deactivate
#
#####3.PhenotypeSeeker. Now testing. todo seems good. just to be sure, test again.
#conda env create -n ${PhenotypeSeeker_env_name}  -f ./install/PhenotypeSeeker_env.yml python=3.7 || { echo "Errors in installing env."; exit; }
##conda create -n ${PhenotypeSeeker_env_name} -y python=3.7
##conda init
##source activate ${PhenotypeSeeker_env_name} || { echo "Errors in activate env."; exit; }
##wait
##echo $CONDA_DEFAULT_ENV
##conda install -y mash=2.3|| { echo "Errors."; exit; }
##conda install -y genometester4=4.0|| { echo "Errors."; exit; }
##conda install -y biopython=1.76|| { echo "Errors."; exit; }
##conda install -y numpy=1.18.1|| { echo "Errors."; exit; }
##conda install -y pandas=1.0.1|| { echo "Errors."; exit; }
##conda install -y pyparsing=2.4.7|| { echo "Errors."; exit; }
##pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
##pip install scipy==1.4.1|| { echo "Errors."; exit; }
##conda deactivate
##echo " ${PhenotypeSeeker_env_name} created successfully."


###4.multi_bench. Tested Dec 6 2022
#conda create -n ${multi_env_name} -y python=3.6
#source activate ${multi_env_name}|| { echo "Errors in activate env."; exit; }
#echo $CONDA_DEFAULT_ENV
#pip install -r install/multi_env.txt
##pip install numpy==1.19.2|| { echo "Errors."; exit; }
##pip install pandas==1.1.3|| { echo "Errors."; exit; }
##pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
##pip install scipy==1.5|| { echo "Errors."; exit; } ##previous1.5.4
##pip install biopython==1.78|| { echo "Errors."; exit; }
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu|| { echo "Errors."; exit; }
#conda deactivate
#echo " ${multi_env_name} created successfully."
##
###you need to install pytorch manually here.
### To install pytorch compatible with your CUDA version, please fellow this instruction: https://pytorch.org/get-started/locally/.
### Our code was tested with pytorch v1.7.1, with CUDA Version 10.1 and 11.0 .
########################################################################################################################
#conda create -n ${multi_torch_env_name} -y  python=3.6
#conda init
#source activate ${multi_torch_env_name}|| { echo "Errors in activate env."; exit; }
#wait
#echo $CONDA_DEFAULT_ENV
#pip install numpy==1.19.2|| { echo "Errors."; exit; }
#pip install pandas==1.1.3|| { echo "Errors."; exit; }
#pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
#pip install scipy==1.5.4|| { echo "Errors."; exit; }
#pip install biopython==1.78|| { echo "Errors."; exit; }
#conda deactivate
#echo " ${multi_torch_env_name} created successfully."
#


##5.Kover #todo need retest.
conda create -n ${kover_env_name} -y python=2.7
source activate ${kover_env_name}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
pip install -r install/kover_env.txt
##pip install cython==0.29.24|| { echo "Errors."; exit; }
##pip install h5py==2.10.0|| { echo "Errors."; exit; }
##pip install numpy==1.16.6|| { echo "Errors."; exit; }
##pip install pandas==0.24.2|| { echo "Errors."; exit; }
##pip install progressbar==2.5|| { echo "Errors."; exit; }
##pip install scipy==1.2.3
conda deactivate
echo " ${kover_env_name} created successfully."
#

##6.phylo_r for generating phylo-tree. checkked on Dec 1st 2022
conda env create -n ${perl_name}  -f ./install/ perl5_22_env.yml python=2.7 || { echo "Errors."; exit; }
conda create -n ${phylo_name} -y  python=3.9
source activate ${phylo_name}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
conda install -y -c conda-forge r-base=4.1.0|| { echo "Errors."; exit; }
conda deactivate
echo " ${phylo_name} created successfully."

#
###7.  ${kmer_env_name}: generating kmer matrix. KMC. # trying at hzi. finished. Nov 17,2022
conda create -n ${kmer_env_name} -y  python=3.6 #tod now trying at hzi
source activate ${kmer_env_name}|| { echo "Errors in activate env."; exit; }
echo $CONDA_DEFAULT_ENV
conda install -c bioconda  -y kmc=3.1|| { echo "Errors."; exit; } # we used 3.1.2rc1 version, but now it is not available anymore.
conda install -y numpy=1.19.1|| { echo "Errors."; exit; }
conda install -y pandas=1.1.3|| { echo "Errors."; exit; }
conda install -y tables==3.6.1|| { echo "Errors."; exit; }
###pip install tables==3.6.1|| { echo "Errors."; exit; }
conda deactivate









