#!/usr/bin/env Rscript

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

.libPaths(paste(Sys.getenv('CONDA_PREFIX'), 'lib', 'R', 'library', sep='/'))

library(DESeq2)
library(BiocParallel)
library(apeglm)

#####
#Output:
#	full DESeq2 result
#	differentially expressed genes


####
## Updates:
## Generalized for seq2geno
## Note that the input matrix of DESeq2 should have strains in COLUMNS
script_dir<- Sys.getenv('TOOL_HOME')
source(paste(script_dir, 'seq2geno.MatIO.R', sep='/'))

expr_f<- snakemake@input[['expr_table']] # expression levels
phe_f<- snakemake@input[['phe_table']] # binary table of sample classes, which can include multiple columns but still follow the format of feature tables
#output_dir<- '' 
output_dir<- snakemake@output[['dif_xpr_out']]
cpu_num<- as.numeric(snakemake@threads[1])
alpha_cutoff<- as.numeric(snakemake@params[['alpha_cutoff']])
lfc_cutoff<- as.numeric(snakemake@params[['lfc_cutoff']])

#####
### The input matrix
### check if it in line with the feature table rules (refer to github issues and the scripts)
raw_rpg_mat<- read_seq2geno.tab(expr_f, string_values= F)
### remove zero-count genes
rpg_mat<- raw_rpg_mat[, (colSums(raw_rpg_mat) > 0)]
n_reduced<- ncol(raw_rpg_mat)- ncol(rpg_mat)
if (n_reduced > 0){
  write('WARNING: Genes not expressed in any sample were excluded from differential expression analysis', stderr())
}

#####
### classes
phe_df<- read_seq2geno.tab(phe_f, string_values= T)

#####
### detect target strains
target_strains<- rownames(phe_df)

#####
### set the output directory
dir.create(output_dir, showWarnings = FALSE)
output_suffix1<- 'deseq2.tsv'
output_suffix2<- 'deseq2.DifXpr.list'

#####
### start DESeq2
col_names<- colnames(phe_df)
for (target_col in col_names){
  print(target_col)
  ## ensure the target column is binary
  uq_all_phe<- unique(phe_df[,target_col])
  uq_all_phe<- uq_all_phe[grepl('\\w+', uq_all_phe)]
  if (length(uq_all_phe) != 2){
    stop('Differential expression analysis is only for two classes. Please turn off this function')
  }else{
    write("Detected groups: \n", stdout())
    write(paste(uq_all_phe, collapse= ','), stdout())
    ## exclude NA
    target_strains<- names(phe_df[phe_df[,target_col] %in% uq_all_phe,])
    target_strains<- target_strains[target_strains %in% rownames(rpg_mat)]

    ## create colData
    col_df<- data.frame(sample=target_strains, pheno= phe_df[target_strains, target_col])
    rownames(col_df)<- target_strains
    dds<-DESeqDataSetFromMatrix(countData = t(rpg_mat[target_strains,]),
				colData = col_df, 
				design = ~pheno) 

    # determine the reference class
    dds$pheno<- factor(dds$pheno, levels= uq_all_phe)

    # start the analysis
    register(MulticoreParam(cpu_num))
    dds <- DESeq(dds, parallel= T)
    resultsNames(dds)
    res <- results(dds)
    print(dim(res))
    print(head(res))
    res_filtered <- subset(res, (padj < alpha_cutoff) & (abs(log2FoldChange)>= lfc_cutoff))
    print(dim(res_filtered))
    print(head(res_filtered))

    out_f1<- file.path(output_dir, paste0(target_col, output_suffix1, collapse= '_'))
    write.table(res, out_f1, col.names=NA, sep= '\t', quote= F)
    out_f2<- file.path(output_dir, paste0(target_col, output_suffix2, collapse= '_'))
    write(rownames(res_filtered), out_f2)
  }
}
