# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Purpose:
# - Ancestral reconstruction for the expression levels along the tree topology
# Materials:
# - expression levels
# - tree
# Methods:
# - Reads mapped using BWA-MEM
# - Variant sites called with samtools mpileup
# Output:
# - the reconstructed values of nodes and the changes in eadges


import os
import yaml
from LoadFile import LoadFile
expr_config_f = config['expr_config']
phylo_config_f = config['phylo_config']
expr_config = yaml.load(LoadFile(expr_config_f))
phylo_config = yaml.load(LoadFile(phylo_config_f))
EXPR_OUT = os.path.join(os.path.dirname(expr_config_f),
    expr_config['out_table'])
TREE_OUT = os.path.join(os.path.dirname(phylo_config_f),
    phylo_config['tree_f'])
C_ANCREC_OUT = config['C_ANCREC_OUT']


rule all:
    input:
        output_dir = C_ANCREC_OUT


rule cont_anc_reconstruction:
    input:
        tree_f = TREE_OUT,
        data_f = EXPR_OUT
    output:
        output_dir = directory(C_ANCREC_OUT)
    conda: 'ar_env.yml'
    script: 'contAncRec.R'

