#!/bin/bash

s=('Escherichia_coli' 'Staphylococcus_aureus' 'Salmonella_enterica' 'Klebsiella_pneumoniae' 'Pseudomonas_aeruginosa' 'Acinetobacter_baumannii' 'Streptococcus_pneumoniae' 'Mycobacterium_tuberculosis' 'Campylobacter_jejuni' 'Enterococcus_faecium' 'Neisseria_gonorrhoeae')
for species in ${s[@]}
do
    echo ${species}
#    cp ${species}_* /vol/projects/khu/amr/new/final20220506/log/temp/resfinder/loose/${species}
#    mkdir ${species}
#     cp ${species}.csv /vol/projects/khu/amr/new/final20220506/log/temp/resfinder/loose/
#    cp ${species}_* /vol/projects/khu/amr/new/final20220506/log/temp/resfinder_blast/loose/${species}
    cp ${species}.csv /vol/projects/khu/amr/new/final20220506/log/temp/resfinder_blast/loose/
done
