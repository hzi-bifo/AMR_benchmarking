cat ./data/PATRIC/meta/genome_list|
while read i; do
    p3-all-genomes --eq genome_id,${i} --attr sra_accession >> ./data/PATRIC/meta/sra.txt
done
