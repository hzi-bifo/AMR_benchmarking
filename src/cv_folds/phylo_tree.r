#!/usr/bin/env Rscript

# install.packages("remotes")
# remotes::install_github("trevorld/r-optparse")
# # install.packages("optparse", repos="http://R-Forge.R-project.org")
# # packageurl <- "http://cran.r-project.org/src/contrib/Archive/phangorn/phangorn_2.7.0.tar.gz"
# # install.packages(packageurl, contriburl=NULL, type="source")
#
# install.packages("phangorn")
# install.packages("phytools")
# install.packages('codetools')

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
# opt <- parse_args(OptionParser(option_list=option_list))

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);



alignment_file<- opt$file
tree_out_file<- opt$out
aln<- read.phyDat(alignment_file, format = "fasta", type = "DNA")
dm <-  dist.hamming(aln)
tree <- NJ(dm)
write.tree(tree, file= tree_out_file)
