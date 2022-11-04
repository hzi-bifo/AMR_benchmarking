<!--
SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo

SPDX-License-Identifier: GPL-3.0-or-later
-->

You can either use **SETENV.sh** (for setting up the environment and ensuring python apckages) 
and **TESTING.sh** (which will dryrun Seq2Geno with the example dataset) to automatlically set up 
everything. Alternatively, you can follow the instruction below:

#### Step 1. check if Conda is installed and the channels (as described in ../README.md) are included 
#### Step 2. check where seq2geno home folder is
It might look like `/YOUR/HOME/bin/seq2geno`

#### Step 3. install the core environment
create the environment with the commands:
```
export env_name=snakemake_env
export env_dir=$( dirname $( dirname $( which conda ) ) )"/envs/"$env_name
if [ -d $env_dir ]; then
  echo $env_name" already exists"
else
  conda env create -n $env_name --file=snakemake_env.yml
fi

export SEQ2GENO_HOME=/YOUR/HOME/bin/seq2geno
source activate $env_name
echo enter $CONDA_PREFIX
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
echo "export SEQ2GENO_HOME="$SEQ2GENO_HOME >> ./etc/conda/activate.d/env_vars.sh
echo "export PATH="$( realpath $SEQ2GENO_HOME )"/main:"$PATH >> ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
echo "unset "$SEQ2GENO_HOME >> ./etc/conda/deactivate.d/env_vars.sh
source deactivate
```
Yoou might need to replace `source activate` with `conda activate` 

#### Step 4. test the environment
```
source activate $env_name
seq2geno -h
source deactivate
```
That should show the usage about seq2geno.

#### Step 5. install dependencies of Raory 
Roary already has its own script for installing dependencies, so we can simply use it:
```
source activate snakemake_env
cd $SEQ2GENO_HOME/denovo/lib/Roary
./install_dependencies.sh
source deactivate
```
These softwares should now be available under `$SEQ2GENO_HOME/denovo/lib/Roary/build`.

#### Step 6. install process-specific environments and dryrun the procedures for the example dataset
```
cd $SEQ2GENO_HOME/examples
S2G -z example_input.zip
```

#### Step 7. 

Environment set and dependencies installed! The launcher "S2G" has been created in '$SEQ2GENO_HOME'. You might also want to: 
- for the convenience, copy '$SEQ2GENO_HOME'/S2G to a certain idirectory that is already included in your PATH variable; alternatively, add `$SEQ2GENO_HOME/main` to your PATH variable and edit either `~/.profile` or `~/.bashrc` accordingly
- go to '$SEQ2GENO_HOME'/examples/, read and modify the settings in `seq2geno_inputs.yml`, and try running seq2geno 
