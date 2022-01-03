#!/bin/bash
#script for kmer making
#tool:kmc
#output:kmer for each fna file
#the input for feature making k_mer.py



#------8mer
path="/vol/projects/BIFO/patric_genome/"
mkdir cano8mer
cat ../metadata/genome_list|
while read i; do
    kmc -k8 -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" NA.res cano8mer
    kmc_dump -ci0 -cs1677215 NA.res cano8mer/merge_8mers_${i}.txt
done


cat ../metadata/genome_list|
while read i; do
    kmc -k8 -b -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" NA.res non_cano8mer
    kmc_dump -ci0 -cs1677215 NA.res non_cano8mer/merge_8mers_${i}.txt
done

#-----6mer 
cat ../metadata/genome_list|
while read i; do
    kmc -k6 -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" NA.res cano6mer
    kmc_dump -ci0 -cs1677215 NA.res cano6mer/merge_6mers_${i}.txt
done

cat ../metadata/genome_list|
while read i; do
    kmc -k6 -b -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" NA.res non_cano6mer
    kmc_dump -ci0 -cs1677215 NA.res non_cano6mer/merge_6mers_${i}.txt
done


#----10mer
cat ../metadata/genome_list|
while read i; do
    kmc -k10 -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" NA.res cano10mer
    kmc_dump NA.res cano10mer/merge_10mers_${i}.txt
done
cat ../metadata/genome_list|
while read i; do
    kmc -k10 -b -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" NA.res non_cano10mer
    kmc_dump -ci0 -cs1677215 NA.res non_cano10mer/merge_10mers_${i}.txt
done

