#!/bin/bash

#script for kmer making
#tool:kmc
#output:kmer for each fna file
#the input for feature making k_mer.py
dna_path="$1"
log_path="$2"



#-----6mer 
cat ./data/PATRIC/meta/genome_list|
while read i; do
  echo
    "kmc -k6 -m24 -fm -ci0 -cs1677215 "${dna_path}/${i}.fna" ${log_path}log/software/seq2geno/software_output/cano6mer/temp/NA.res ${log_path}log/software/seq2geno/software_output/cano6mer
    kmc_dump -ci0 -cs1677215 ${log_path}log/software/seq2geno/software_output/cano6mer/temp/NA.res ${log_path}log/software/seq2geno/software_output/cano6mer/merge_6mers_${i}.txt "
done

