#!/bin/bash

#Generating the k-mer lists for every samples for each species

dataset_location="$1"
log_path="$2"
genome_list="$3"


cat ./data/PATRIC/meta/${genome_list}|
while read i; do
    glistmaker "${dataset_location}/${i}.fna"  -o ${log_path}log/software/phenotypeseeker/software_output/K-mer_lists/${i}_0 -w 13 -c 1
done




