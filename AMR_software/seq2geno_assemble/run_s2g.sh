#!/bin/bash
species="$1"
log_path="$2"



seq2geno -f ${log_path}log/software/seq2geno/software_output/${species}/seq2geno_inputs.yml -l ${log_path}log/software/seq2geno/software_output/${species}/seq2geno_log.txt
