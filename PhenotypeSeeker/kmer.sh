#!/bin/bash

#1. Generating the k-mer lists for every samples for each species-------------------------------------------------------
path="/vol/projects/BIFO/patric_genome/"
mkdir K-mer_lists
cat ./metadata/genome_list|
while read i; do
    glistmaker "${path}${i}.fna"  -o K-mer_lists/${i}_0 -w 13 -c 1
    #glistquery K-mer_lists/${i}_0_13.list  >  K-mer_lists/${i}_mapped

done




