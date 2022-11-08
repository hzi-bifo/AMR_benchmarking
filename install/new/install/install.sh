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


###1.amr
#conda create -n ${amr_env_name} -y python=3.7
#conda init
#source activate ${amr_env_name} || { echo "Errors in activate env."; exit; }
#wait
#echo $CONDA_DEFAULT_ENV
#conda install -y biopython=1.78|| { echo "Errors."; exit; }
#conda install -y hdf5=1.10.4|| { echo "Errors."; exit; }
#conda install -y pytables=3.6.1|| { echo "Errors."; exit; }
#pip install matplotlib==3.4.2|| { echo "Errors."; exit; }
#pip install numpy==1.20.1|| { echo "Errors."; exit; }
#pip install openpyxl==3.0.9|| { echo "Errors."; exit; }
#pip install pandas==1.2.2|| { echo "Errors."; exit; }
#pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
#pip install scipy==1.6.0|| { echo "Errors."; exit; }
#pip install seaborn==0.11.1|| { echo "Errors."; exit; }
#pip install sklearn==0.0|| { echo "Errors."; exit; }
#pip install tqdm==4.62.3|| { echo "Errors."; exit; }
####pip install xgboost==1.4.2|| { echo "Errors."; exit; }
#conda deactivate
#echo " ${amr_env_name} created successfully."


####2.PhenotypeSeeker
#conda create -n ${PhenotypeSeeker_env_name} -y python=3.7
#conda init
#source activate ${PhenotypeSeeker_env_name} || { echo "Errors in activate env."; exit; }
#wait
#echo $CONDA_DEFAULT_ENV
#conda install -y mash=2.3|| { echo "Errors."; exit; }
#conda install -y genometester4=4.0|| { echo "Errors."; exit; }
#conda install -y biopython=1.76|| { echo "Errors."; exit; }
#conda install -y matplotlib=3.4.3|| { echo "Errors."; exit; }
#conda install -y numpy=1.18.1|| { echo "Errors."; exit; }
#conda install -y pandas=1.0.1|| { echo "Errors."; exit; }
#conda install -y pyparsing=2.4.7|| { echo "Errors."; exit; }
#pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
#pip install scipy==1.4.1|| { echo "Errors."; exit; }
#conda deactivate
#echo " ${PhenotypeSeeker_env_name} created successfully."


####3.multi_bench
#conda create -n ${multi_env_name} -y python=3.6
#conda init
#source activate ${multi_env_name}|| { echo "Errors in activate env."; exit; }
#wait
#echo $CONDA_DEFAULT_ENV
#pip install numpy==1.19.2|| { echo "Errors."; exit; }
#pip install pandas==1.1.3|| { echo "Errors."; exit; }
#pip install scikit-learn==0.24.1|| { echo "Errors."; exit; }
#pip install scipy==1.5.4|| { echo "Errors."; exit; }
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu|| { echo "Errors."; exit; }
#conda deactivate
#echo " ${multi_env_name} created successfully."
#
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
#conda deactivate
#echo " ${multi_torch_env_name} created successfully."
#


##5.Kover
#conda create -n ${kover_env_name} -y python=2.7
#conda init
#source activate ${kover_env_name}|| { echo "Errors in activate env."; exit; }
#wait
#echo $CONDA_DEFAULT_ENV
#pip install cython==0.29.24|| { echo "Errors."; exit; }
#pip install h5py==2.10.0|| { echo "Errors."; exit; }
#pip install numpy==1.16.6|| { echo "Errors."; exit; }
#pip install pandas==0.24.2|| { echo "Errors."; exit; }
#pip install progressbar==2.5|| { echo "Errors."; exit; }
#conda deactivate
#echo " ${kover_env_name} created successfully."


##6.phylo_r for generating phylo-tree
conda create -n ${phylo_name} -y  python=3.6
conda init
source activate ${phylo_name}|| { echo "Errors in activate env."; exit; }
wait
echo $CONDA_DEFAULT_ENV
#conda install -y -c conda-forge r-base=4.0.5|| { echo "Errors."; exit; }
conda install -c conda-forge r r=3.4.1 #todo try this!
conda deactivate
echo " ${phylo_name} created successfully."

#
###7.  ${kmer_env_name}: generating kmer matrix. KMC.
#conda create -n ${kmer_env_name} -y  python=3.6
#conda init
#source activate ${kmer_env_name}|| { echo "Errors in activate env."; exit; }
#wait
#echo $CONDA_DEFAULT_ENV
#conda install -c -y bioconda kmc=3.1|| { echo "Errors."; exit; } # we used 3.1.2rc1 version, but now it is not available anymore.
#conda install -y numpy=1.19.1|| { echo "Errors."; exit; }
#conda install -y pandas=1.1.3|| { echo "Errors."; exit; }
#pip install tables==3.6.1|| { echo "Errors."; exit; }
#conda deactivate









