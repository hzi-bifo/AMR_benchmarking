#!/usr/bin/env Rscript

library("optparse")
library(phytools)
library(phangorn)
library(ape)


option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL,
              help="dataset file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="out.txt",
              help="output file name [default= %default]", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

alignment_file<- opt$file
tree_out_file<- opt$out
aln<- read.phyDat(alignment_file, format = "fasta", type = "DNA")
dm <-  dist.hamming(aln)
tree <- NJ(dm)
write.tree(tree, file= tree_out_file)
