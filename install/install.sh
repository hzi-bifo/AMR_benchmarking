#!/bin/bash


#check_conda_channels () {
#	## ensure conda channels
#	echo '+check conda channels...'
#	for c in hzi-bifo conda-forge/label/broken bioconda conda-forge defaults r pytorch anaconda ; do
#		echo '+'$c
#		if [ $(conda config --get channels | grep $c | wc -l) -eq 0 ]; then
#			conda config --add channels $c
#		fi
#	done
#}
#check_conda_channels ||{ echo "Errors in setting conda channels. Please set it by hand yourself."; exit; }

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

#---------------------------------------------------------------------------------------
##1.amr
conda create -n ${amr_env_name} -y python=3.7
conda init
source activate ${amr_env_name} || { echo "Errors in activate env."; exit; }
wait
conda install -y biopython=1.78
conda install -y hdf5=1.10.4
conda install -y pytables=3.6.1
conda install -y matplotlib=3.4.2
conda install -y numpy=1.20.1
conda install -y openpyxl=3.0.9
conda install -y pandas=1.2.2
conda install -y scikit-learn=0.24.1
conda install -y scipy=1.6.0
conda install -y seaborn=0.11.1
conda install -y sklearn=0.24.1
conda install -y tqdm=4.62.3
#conda install -y xgboost=1.4.2
conda deactivate
echo " ${amr_env_name} created successfully."



##2.PhenotypeSeeker
conda create -n ${PhenotypeSeeker_env_name} -y python=3.7
conda init
source activate ${PhenotypeSeeker_env_name} || { echo "Errors in activate env."; exit; }
wait
conda install -y mash=2.3
conda install -y genometester4=4.0
conda install -y biopython=1.76
conda install -y matplotlib=3.4.3
conda install -y numpy=1.18.1
conda install -y pandas=1.0.1
conda install -y pyparsing=2.4.7
conda install -y scikit-learn=0.24.1
conda install -y scipy=1.4.1
echo " ${PhenotypeSeeker_env_name} created successfully."


###3.multi_bench
conda create -n ${multi_env_name} -y python=3.6
conda init
conda activate ${multi_env_name}|| { echo "Errors in activate env."; exit; }
wait
conda install -y numpy=1.19.2
conda install -y pandas=1.1.3
conda install -y scikit-learn==0.24.1
conda install -y scipy==1.5.4
conda install -y pytorch=1.8.1
echo " ${multi_env_name} created successfully."

conda create -n ${multi_torch_env_name} -y  python=3.6
conda init
conda activate ${multi_torch_env_name}|| { echo "Errors in activate env."; exit; }
wait
conda install -y numpy=1.19.2
conda install -y pandas=1.1.3
conda install -y scikit-learn==0.24.1
conda install -y scipy==1.5.4
echo " ${multi_torch_env_name} created successfully."

##5.Kover
conda create -n ${kover_env_name} python=2.7
conda init
conda activate ${kover_env_name}|| { echo "Errors in activate env."; exit; }
wait
conda install -y pip 19.3.1
pip install cython==0.29.24
pip install h5py==2.10.0
pip install numpy==1.16.6
pip install pandas==0.24.2
pip install progressbar==2.5
echo " ${kover_env_name} created successfully."


##6.phylo_r for generating phylo-tree
conda create -n ${phylo_r_name} -y  python=3.6
conda init
conda activate ${phylo_r_name}|| { echo "Errors in activate env."; exit; }
wait
conda install -c conda-forge r=3.4.1
echo " ${phylo_r_name} created successfully."

#####################################################
##7.  ${kmer_env_name}: generating kmer matrix. KMC.
#####################################################
conda create -n ${kmer_env_name} -y  python=3.6
conda init
conda activate ${kmer_env_name}|| { echo "Errors in activate env."; exit; }
wait
conda install -c -y bioconda kmc=3.1 # we used 3.1.2rc1 version, but now it is not available anymore.
conda install -y numpy=1.19.1
conda install -y pandas=1.1.3
conda install -y pip=19.3.1
pip install tables==3.6.1
conda deactivate








