#!/bin/bash
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
eval $(parse_yaml options.yaml)

#-------------------------------------------
#1. Install env
#-------------------------------------------
#note: if you use GPU for NN model, you need to install pytorch in the multi_torch_env env.
# To install pytorch compatible with your CUDA version, please fellow this instruction: https://pytorch.org/get-started/locally/.
# Our code was tested with pytorch v1.7.1, with CUDA Version: 10.1 and 11.0 .

bash ./install/install.sh ${amr_env_name} ${PhenotypeSeeker_env_name} ${multi_env_name} ${multi_torch_env_name} ${kover_env_name}

echo "Env created."
#-------------------------------------------
#2. Data
#-------------------------------------------
#2.1 PATRIC Data download
bash ./scripts/data_preprocess/retrive_PATRIC_data.sh ${dataset_location}

#2.2 todo Test data ?
#smaller-size dataset, for testing, faster.

#2.3 Quality control. (you can skip this step, as we provided the sample list after QC: ./data/PATRIC)
#uncomment  following, if you want to go through the QC, and re-generate the list.
#python preprocess.py ${QC_criteria}



#-------------------------------------------
#3.
#-------------------------------------------




#-------------------------------------------
#4.
#-------------------------------------------


