#!/usr/bin/env Rscript
install.packages("remotes")
remotes::install_github("trevorld/r-optparse")
install.packages("phangorn")
install.packages("phytools")
install.packages('codetools')

library("optparse")
library(phytools)
library(phangorn)

option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL,
              help="dataset file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="out.txt",
              help="output file name [default= %default]", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


library(ape)
alignment_file<- opt$file
tree_out_file<- opt$out
aln<- read.phyDat(alignment_file, format = "fasta", type = "DNA")
dm <-  dist.hamming(aln)
tree <- NJ(dm)
write.tree(tree, file= tree_out_file)
