<!--
SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo

SPDX-License-Identifier: GPL-3.0-or-later
-->

For a zip file called _ABC.zip_, the structure in the file should include:

```
ABC/  # same as the zip filename
├── reads/  
│   ├── dna/  # paired-end DNA-seq reads
│   └── rna/  # single-end RNA-seq reads
├── functions/
│   └── functions  # a list of features to include; each one in a line 
├── reference/  # reference data
├── resources/
│   └── resources  # the resource allowance
└── phenotype/  # phenotype data
```

- reads/dna/:
This folder contains paired-end DNA-seq reads.
Each file represents one end of reads of a sample. The files should be named
after the sample names and have an extension of either **.1.fastq.gz**, **.1.fq.gz**,
**.2.fastq.gz**, or **.2.fq.gz**, in which **1** and **2** represent the ends of read pairs.

- reads/rna/: (optional)
This folder contains single-end RNA-seq reads.
Each file represents the reads of a sample. The files should be named
after the sample names and have an extension of either **.fastq.gz** or
**.fq.gz**.

- reference/:
This folder contains reference sequence and annotation files, of which the file extensions should be
 **.fna** (genome sequence; fasta format), **.gbk** (annotation genbank
file), and **gff** (annotation gff file). 

- functions/functions:
The file is adequate to the 'features' field of the yaml input file (see README.md). 
The file include the analyses to conduct. Each analyses are described in one row.

- resources/resources: (optional)
The file is adequate to the 'general -> cores' and 'general -> mem\_mb' fields of the yaml input file (see README.md). 
This file should include two columns separated by tab, of which the first is the
resource name (cores or mem\_mb) and the second includes the values. 

- phenotype/: (optional)
The phenotypes for differential expression analysis or model training
(Geno2Pheno). The phenotype table should be formatted as described in the README.md
and the filename should have an extension of **.mat**.
