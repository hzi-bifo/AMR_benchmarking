#!/bin/bash

amr_env_name=$1
if [[ amr_env_name == '' ]]; then
  amr_env_name=amr_env

PhenotypeSeeker_env_name=$2
if [[ PhenotypeSeeker_env_name == '' ]]; then
  PhenotypeSeeker_env_name=amr_env

multi_env_name=$3
if [[ multi_env_name == '' ]]; then
  multi_env_name=amr_env

multi_torch_env_name=$4
if [[ multi_torch_env_name == '' ]]; then
  multi_torch_env_name=amr_env

kover_env_name=$5
if [[ kover_env_name == '' ]]; then
  kover_env_name=amr_env



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

#---------------------------------------------------------------------------------------
#1.amr
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/amr_env_name ]; then
  echo '-----'
  echo 'Naming conflict: an existing environment with same name found: '
  echo $( dirname $( dirname $( which conda ) ) )/envs/amr_env_name
  exit
else
  # start creating the environment
  conda env create -n ${amr_env_name} -f amr_env.yml python=3.7.9 || { echo "Errors in downloading the ${amr_env_name}"; exit; }
fi
echo " ${amr_env_name} created successfully."

#2.PhenotypeSeeker
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/PhenotypeSeeker_env ]; then
  echo '-----'
  echo 'Naming conflict: an existing environment with same name found: '
  echo $( dirname $( dirname $( which conda ) ) )/envs/PhenotypeSeeker_env
  exit
else
  # start creating the environment
  conda create -n ${PhenotypeSeeker_env_name} -f PhenotypeSeeker_env.yml python=3.8.2 || { echo "Errors in downloading the ${PhenotypeSeeker_env}"; exit; }
fi
echo " ${PhenotypeSeeker_env} created successfully."


#3.multi_bench
#for NN model [Aytan-Aktug et al.]
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/multi_env_name ]; then
  echo '-----'
  echo 'Naming conflict: an existing environment with same name found: '
  echo $( dirname $( dirname $( which conda ) ) )/envs/multi_env_name
  exit
else
  # start creating the environment
  conda create -n ${multi_env_name} -f multi_env.yml python=3.6.13 || { echo "Errors in downloading the ${multi_env_name}"; exit; }
fi
echo " ${multi_env_name} created successfully."


#4.multi_bench_torch
#for NN model [Aytan-Aktug et al.] with GPU version.
#pytorch : To install pytorch compatible with your CUDA version,
# please fellow this instruction: https://pytorch.org/get-started/locally/.
# Our code was tested with pytorch v1.7.1, with CUDA Version: 10.1 and 11.0 .
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/multi_torch_env_name ]; then
  echo '-----'
  echo 'Naming conflict: an existing environment with same name found: '
  echo $( dirname $( dirname $( which conda ) ) )/envs/multi_torch_env_name
  exit
else
  # start creating the environment
  conda create -n ${multi_torch_env_name} -f multi_torch_env.yml python=3.6.13 || { echo "Errors in downloading the ${multi_torch_env_name}"; exit; }
fi
echo " ${multi_torch_env_name} created successfully."


#5.Kover
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/kover_env_name ]; then
  echo '-----'
  echo 'Naming conflict: an existing environment with same name found: '
  echo $( dirname $( dirname $( which conda ) ) )/envs/kover_env_name
  exit
else
  # start creating the environment
  conda create -n ${kover_env_name} -f kover_env.yml python=2.7.17 || { echo "Errors in downloading the ${kover_env_name}"; exit; }
fi
echo " ${kover_env_name} created successfully."



