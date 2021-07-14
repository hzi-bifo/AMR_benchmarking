#!/usr/bin/env bash
export SEQ2GENO_HOME=$( realpath ../ )
export PATH=$SEQ2GENO_HOME'/main:'$PATH
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
	echo 'export PATH='$SEQ2GENO_HOME'/main:$PATH' >> $ACTIVATE_ENVVARS

	echo 'unset SEQ2GENO_HOME' > $DEACTIVATE_ENVVARS
	cd $SEQ2GENO_HOME
}

set_roary_dependencies () {
	## Roary dependencies
	cd $SEQ2GENO_HOME/denovo/lib/Roary
	export PERL_MM_USE_DEFAULT=1
	export PERL5LIB=$( realpath . )/lib:$PERL5LIB
	./install_dependencies.sh
	cd $SEQ2GENO_HOME
}

download_proc_specific_env () {
	## decompress the example dataset to install the process-specific environments
  	echo $( realpath . )
	echo '+extract example dataset'
	tar -zxvf example_sg_dataset.tar.gz 
	cd $SEQ2GENO_HOME/example_sg_dataset/
	./CONFIG.sh
	echo '+install process-specific environments and dryrun the procedures for the example dataset'
	$SEQ2GENO_HOME/main/seq2geno -f ./seq2geno_inputs.yml || return false
}
create_core_env ()  {
	## create snakemake_env 
	echo '+enter install/'
	cd $SEQ2GENO_HOME/install
	conda env create -n snakemake_env --file=snakemake_env.yml || return false
	cd $SEQ2GENO_HOME
}

#>>>
check_conda_channels ||{ echo "Errors in setting conda channels"; exit; }
if [ -d $( dirname $( dirname $( which conda ) ) )/envs/snakemake_env ]; then
	echo '-----'
	echo 'Naming conflict: an existing environment is also called "snakemake_env".'
	echo 'Please remove it (with or without cloning it with the other name).'
	exit
else
	create_core_env || { echo "Errors in downloading the core environment"; exit; }
fi
## activate the environment
source $( dirname $( dirname $( which conda ) ) )/etc/profile.d/conda.sh
conda activate snakemake_env || source activate snakemake_env
set_core_env_vars || { echo "Errors in setting up the core environment"; exit; }
set_roary_dependencies || { echo "Errors in installation of Roary dependecies"; exit; }
download_proc_specific_env || { echo "Errors in installation of the process-specific environments failed"; exit; }
conda deactivate || source deactivate

## Finalize
export SEQ2GENO_HOME=$( realpath ../ )
export PATH=$SEQ2GENO_HOME/main:$PATH
mv $SEQ2GENO_HOME/main/S2G $SEQ2GENO_HOME
echo '-----'
echo 'Environments installed! The launcher "S2G" has been created in '$SEQ2GENO_HOME'. You might also want to: '
echo '- copy '$SEQ2GENO_HOME'/S2G to a certain idirectory that is already included in your PATH variable '
echo '- go to '$SEQ2GENO_HOME'/example_sg_dataset/ and try'
