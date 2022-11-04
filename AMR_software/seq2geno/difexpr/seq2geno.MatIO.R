# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

read_seq2geno.tab<-function(f, string_values= T){
    mat<- read.table(f, 
        sep= '\t', header= T, check.names= F, 
        stringsAsFactors= F, row.names= 1)

    strains<- rownames(mat)
    if (string_values){
      mat<- sapply(mat, function(x) as.character(x))
    }else{
      mat<- sapply(mat, function(x) as.numeric(x))
    }
    mat<- data.matrix(mat)
    rownames(mat)<- strains
    return(mat)
}

