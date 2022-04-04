#!/bin/bash
#$ -N kmc
#$ -l arch=linux-x64
#$ -pe multislot 10
#$ -b n
#$ -o /vol/cluster-data/khu/sge_stdout_logs/
#$ -e /vol/cluster-data/khu/sge_stdout_logs/
#$ -q all.q
#$ -cwd
#script for kmer making
#tool:kmc
#output:kmer for each fna file
#the input for feature making k_mer.py

export PATH=~/miniconda2/bin:$PATH
export PYTHONPATH=$PWD
source activate python36

#------8mer
path="/vol/projects/BIFO/patric_genome/"
#mkdir cano8mer
#mkdir non_cano8mer
#mkdir cano6mer
#mkdir non_cano6mer
#mkdir cano10mer
#mkdir non_cano10mer

(cat ../metadata/genome_list|
while read i; do
    kmc -k8 -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" cano8mer/NA1.res1 cano8mer
    kmc_dump -ci0 -cs1677215 cano8mer/NA1.res1 cano8mer/merge_8mers_${i}.txt
done)&

(cat ../metadata/genome_list|
while read i; do
    kmc -k8 -b -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" non_cano8mer/NA2.res2 non_cano8mer
    kmc_dump -ci0 -cs1677215 non_cano8mer/NA2.res2 non_cano8mer/merge_8mers_${i}.txt
done)&

#-----6mer 
(cat ../metadata/genome_list|
while read i; do
    kmc -k6 -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" cano6mer/NA3.res3 cano6mer
    kmc_dump -ci0 -cs1677215 cano6mer/NA3.res3 cano6mer/merge_6mers_${i}.txt
done)&

(cat ../metadata/genome_list|
while read i; do
    kmc -k6 -b -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" non_cano6mer/NA4.res4 non_cano6mer
    kmc_dump -ci0 -cs1677215 non_cano6mer/NA4.res4 non_cano6mer/merge_6mers_${i}.txt
done)&


#----10mer
(cat ../metadata/genome_list|
while read i; do
    kmc -k10 -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" cano10mer/NA5.res5 cano10mer
    kmc_dump cano10mer/NA5.res5 cano10mer/merge_10mers_${i}.txt
done)&

(cat ../metadata/genome_list|
while read i; do
    kmc -k10 -b -m24 -fm -ci0 -cs1677215 "${path}${i}.fna" non_cano10mer/NA6.res6 non_cano10mer
    kmc_dump -ci0 -cs1677215 non_cano10mer/NA6.res6 non_cano10mer/merge_10mers_${i}.txt
done)&

wait
