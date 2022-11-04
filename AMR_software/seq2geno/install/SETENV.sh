#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

export SEQ2GENO_HOME=$( dirname $( dirname $( realpath ${BASH_SOURCE[0]} ) ) )
export PATH=$SEQ2GENO_HOME:$SEQ2GENO_HOME/main:$PATH
echo 'SEQ2GENO_HOME is '$SEQ2GENO_HOME

check_conda_channels () {
	## ensure conda channels
	echo '+check conda channels...'
	for c in hzi-bifo conda-forge/label/broken bioconda conda-forge defaults; do 
		echo '+'$c
		if [ $(conda config --get channels | grep $c | wc -l) -eq 0 ]; then
			conda config --add channels $c
		fi
	done
	cd $SEQ2GENO_HOME
}

set_core_env_vars () {
	## set up environmental variables
	echo '+set up core environment'
	echo '+enter '$CONDA_PREFIX
	cd $CONDA_PREFIX
	mkdir -p ./etc/conda/activate.d
	mkdir -p ./etc/conda/deactivate.d
	export ACTIVATE_ENVVARS=./etc/conda/activate.d/env_vars.sh
	export DEACTIVATE_ENVVARS=./etc/conda/deactivate.d/env_vars.sh
	touch $ACTIVATE_ENVVARS
	touch $DEACTIVATE_ENVVARS

	echo 'export SEQ2GENO_HOME='$SEQ2GENO_HOME > $ACTIVATE_ENVVARS
	echo 'export PATH='$SEQ2GENO_HOME':'$SEQ2GENO_HOME'/main:$PATH' >> $ACTIVATE_ENVVARS

	echo 'unset SEQ2GENO_HOME' > $DEACTIVATE_ENVVARS
	cd $SEQ2GENO_HOME
}

create_core_env ()  {
	## create snakemake_env 
	echo '+enter install/'
	cd $SEQ2GENO_HOME/install
	core_env_name=$1
	if [[ $core_env_name == '' ]]; then
	  core_env_name=snakemake_env
	fi
	conda env create -n $core_env_name --file=snakemake_env.yml || return false
	cd $SEQ2GENO_HOME
}

# user's input
core_env_name=''
if [ "$#" -lt 1 ]; then
  core_env_name='snakemake_env'  
else
  core_env_name=$1
fi
echo 'Creating environment "'$core_env_name'"' 

# check conda and python versions
echo $( conda -V )
echo $( python -V )

# ensure the conda channels
check_conda_channels ||{ echo "Errors in setting conda channels"; exit; }
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/$core_env_name ]; then
  echo '-----'
  echo 'Naming conflict: an existing environment with same name found: '
  echo $( dirname $( dirname $( which conda ) ) )/envs/$core_env_name
  exit
else
  # start creating the environment
  create_core_env $core_env_name || { echo "Errors in downloading the core environment"; exit; }
fi
# activate the environment
source $( dirname $( dirname $( which conda ) ) )/etc/profile.d/conda.sh
conda activate $core_env_name || source activate $core_env_name
# make variables automatically set when the environment is activated 
set_core_env_vars || { echo "Errors in setting up the core environment"; exit; }

# Finalize
cat $SEQ2GENO_HOME/main/S2G| sed "s/_core_env/$core_env_name/g" \
	> $SEQ2GENO_HOME/S2G
chmod +x $SEQ2GENO_HOME/S2G
if [ -f $SEQ2GENO_HOME/S2G ]; then
	echo '-----'
	echo 'Environment set! The launcher "S2G" has been created in '$SEQ2GENO_HOME'. You might also want to: '
	echo '- copy '$SEQ2GENO_HOME'/S2G to a certain idirectory that is already included in your PATH variable '
	echo '- go to '$SEQ2GENO_HOME'/example_sg_dataset/ and try'
else
	echo 'Cannot create the launcher "S2G" to '$SEQ2GENO_HOME
	echo 'You still can use main/seq2geno and main/seq2geno_gui' 
fi
