#!/bin/bash
species="$1"
log_path="$2"


echo ${log_path}log/software/seq2geno/software_output/${species}/seq2geno_inputs.yml
AMR_software/seq2geno/main/seq2geno -f ${log_path}log/software/seq2geno/software_output/${species}/seq2geno_inputs.yml -l ${log_path}log/software/seq2geno/software_output/${species}/seq2geno_log.txt
