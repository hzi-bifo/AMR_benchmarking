#!/bin/bash

#preparing cluster bash files
python main_feature.py -l "loose" -f_cluster  -s 'Salmonella enterica' 'Staphylococcus aureus' 'Klebsiella pneumoniae'
python main_feature.py -l "loose" -f_cluster  -s 'Pseudomonas aeruginosa' 'Mycobacterium tuberculosis' 'Streptococcus pneumoniae'  'Neisseria gonorrhoeae'
python main_feature.py -l "loose" -f_cluster  -s 'Enterococcus faecium' 'Campylobacter jejuni' 'Acinetobacter baumannii'


#1. ===================================================================================
#python main_feature.py -l "loose" -f_pre_cluster --n_jobs 14 -s 'Escherichia coli'
#python main_feature.py -l "loose" -f_pre_cluster --n_jobs 9 -s 'Acinetobacter baumannii'

#wait
#bash ./cv_folders/loose/Escherichia_coli_kma.sh

bash ./cv_folders/loose/Acinetobacter_baumannii_kma.sh
#bash ./cv_folders/loose/Enterococcus_faecium_kma.sh
#bash ./cv_folders/loose/Klebsiella_pneumoniae_kma.sh
#bash ./cv_folders/loose/Neisseria_gonorrhoeae_kma.sh
#bash ./cv_folders/loose/Salmonella_enterica_kma.sh
#bash ./cv_folders/loose/Streptococcus_pneumoniae_kma.sh
#bash ./cv_folders/loose/Campylobacter_jejuni_kma.sh
#bash ./cv_folders/loose/Escherichia_coli_kma.sh
#bash ./cv_folders/loose/Mycobacterium_tuberculosis_kma.sh
#bash ./cv_folders/loose/Pseudomonas_aeruginosa_kma.sh
#bash ./cv_folders/loose/Staphylococcus_aureus_kma.sh


#python main_feature.py -l "loose" -f_pre_cluster --n_jobs 14 -s 'Escherichia coli'
#wait






python main_feature.py -l "loose" -f_res --n_jobs 14 -s 'Escherichia coli' -debug
wait
python main_feature.py -l "loose" -f_merge_mution_gene --n_jobs 1 -s 'Escherichia coli' -debug
wait
python main_feature.py -l "loose" -f_matching_io --n_jobs 14 -s 'Escherichia coli'



python main_feature.py -l "loose" -f_nn --n_jobs 14  -s 'Escherichia coli'
python main_feature.py -l "loose" -f_nn --n_jobs 14  -s 'Escherichia coli' -e 2000
python main_feature.py -l "loose" -f_nn --n_jobs 14 f_fixed_threshold -s 'Escherichia coli'
python main_feature.py -l "loose" -f_nn --n_jobs 14 f_fixed_threshold -s 'Escherichia coli' -e 2000

#2. ===================================================================================
#(python main_feature.py -l "loose" -f_pre_cluster --n_jobs 11 -s 'Salmonella enterica')&
#(python main_feature.py -l "loose" -f_pre_cluster --n_jobs 9 -s 'Staphylococcus aureus')&
#(python main_feature.py -l "loose" -f_pre_cluster --n_jobs 5 -s 'Pseudomonas aeruginosa' 'Acinetobacter baumannii')
#wait
#(python main_feature.py -l "loose" -f_cluster --n_jobs 11 -s 'Salmonella enterica')&
#(python main_feature.py -l "loose" -f_cluster --n_jobs 9 -s 'Staphylococcus aureus')&
#(python main_feature.py -l "loose" -f_cluster --n_jobs 5 -s 'Pseudomonas aeruginosa' 'Acinetobacter baumannii')
#
#wait

#python main_feature.py -l "loose" -f_res --n_jobs 5 -s 'Salmonella enterica' 'Staphylococcus aureus' 'Pseudomonas aeruginosa' 'Acinetobacter baumannii'
#wait
#python main_feature.py -l "loose" -f_merge_mution_gene --n_jobs 5 -s 'Salmonella enterica' 'Staphylococcus aureus' 'Pseudomonas aeruginosa' 'Acinetobacter baumannii'
#wait
#python main_feature.py -l "loose" -f_matching_io --n_jobs 5 -s 'Salmonella enterica' 'Staphylococcus aureus' 'Pseudomonas aeruginosa' 'Acinetobacter baumannii'



#python main_feature.py -l "loose" -f_nn --n_jobs 11  -s 'Salmonella enterica'
#python main_feature.py -l "loose" -f_nn --n_jobs 11  -s 'Salmonella enterica' -e 2000
#python main_feature.py -l "loose" -f_nn --n_jobs 11 f_fixed_threshold -s 'Salmonella enterica'
#python main_feature.py -l "loose" -f_nn --n_jobs 11 f_fixed_threshold -s 'Salmonella enterica' -e 2000












