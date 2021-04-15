#!/bin/sh
for i in `cat genome_list`;do
    if [ ! -f "$i.fna" ]; then
	 printf 'Downloading (%s)\n' "$i.fna"
         wget -qN "ftp://ftp.patricbrc.org/genomes/$i/$i.fna"
    fi
done
