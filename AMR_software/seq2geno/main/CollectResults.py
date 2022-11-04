# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Purpose: Collect the key results made by the finished workflows
import os
import shutil
import yaml
from LoadFile import LoadFile


def collect_results(project_dir, config_files):
    results_newdir = os.path.join(project_dir, 'RESULTS')
    if os.path.isdir(results_newdir):
        shutil.rmtree(results_newdir)
    os.mkdir(results_newdir)
    denovo_config = config_files['denovo']
    snps_config = config_files['snps']
    expr_config = config_files['expr']
    phylo_config = config_files['phylo']
    diffexpr_config = config_files['de']

    # de novo assemblies
    d_config = yaml.safe_load(open(denovo_config, 'r'))
    assem_newdir = os.path.join(results_newdir, 'assemblies')
    assem_outdir = os.path.join(os.path.dirname(denovo_config),
                                d_config['out_spades_dir'])
    if os.path.isdir(assem_outdir):
        strains = os.listdir(assem_outdir)
        for s in strains:
            old = os.path.join(assem_outdir, s, 'contigs.fasta')
            new = os.path.join(assem_newdir, '{}.fasta'.format(s))
            if not os.path.isdir(assem_newdir):
                os.mkdir(assem_newdir)
            if os.path.isfile(old):
                os.symlink(os.path.abspath(old), new)
    # gpa, indel and snps tables
    bin_tab_newdir = os.path.join(results_newdir, 'bin_tables')
    os.mkdir(bin_tab_newdir)
    s_config = yaml.safe_load(open(snps_config, 'r'))
    bin_tables = [
        os.path.join(os.path.dirname(denovo_config),
                     d_config['out_gpa_f']),
        os.path.join(os.path.dirname(denovo_config),
                     d_config['out_indel_f']),
        os.path.join(os.path.dirname(snps_config),
                     s_config['snps_aa_bin_mat']),
        os.path.join(os.path.dirname(snps_config),
                     s_config['nonsyn_snps_aa_bin_mat'])
    ]
    n_rdnt_bin_tables = [f+'_NONRDNT' for f in bin_tables]
    for f in n_rdnt_bin_tables:
        new = os.path.join(bin_tab_newdir, os.path.basename(f))
        if os.path.isfile(f):
            os.symlink(os.path.abspath(f), new)
    grouping_bin_tab_newdir = os.path.join(results_newdir, 'bin_tables_groups')
    os.mkdir(grouping_bin_tab_newdir)
    grouping_bin_tables = [f+'_GROUPS' for f in bin_tables]
    for f in grouping_bin_tables:
        new = os.path.join(grouping_bin_tab_newdir, os.path.basename(f))
        if os.path.isfile(f):
            os.symlink(os.path.abspath(f), new)

    # phylogeny
    p_config = yaml.safe_load(open(phylo_config, 'r'))
    phy_newdir = os.path.join(results_newdir, 'phylogeny')
    os.mkdir(phy_newdir)
    phylo_nwk = os.path.join(os.path.dirname(phylo_config),
                             p_config['tree_f'])
    new = os.path.join(phy_newdir, 'tree.nwk')
    if os.path.isfile(phylo_nwk):
        os.symlink(os.path.abspath(phylo_nwk), new)

    # ###
    # optional
    # need to check existence before collecting them
    #
    # expr
    e_config = yaml.safe_load(open(expr_config, 'r'))
    expr_mat = os.path.join(os.path.dirname(expr_config),
                            e_config['out_log_table'])
    if os.path.isfile(expr_mat):
        num_tab_newdir = os.path.join(results_newdir, 'num_tables')
        os.mkdir(num_tab_newdir)
        new = os.path.join(num_tab_newdir, 'expr.log.mat')
        os.symlink(os.path.abspath(expr_mat), new)

    # ancestral reconstruction
    ar_outdir = os.path.join(os.path.dirname(expr_config), 'ancrec')
    if os.path.isdir(ar_outdir):
        ar_newdir = os.path.join(results_newdir,
                                 'expr_ancestral_reconstructions')
        if os.path.isdir(ar_outdir):
            os.symlink(os.path.abspath(ar_outdir), ar_newdir)

    # differential expression
    de_outdir = os.path.join(os.path.dirname(expr_config), 'dif/')
    if os.path.isdir(de_outdir):
        de_newdir = os.path.join(results_newdir, 'differential_expression')
        if os.path.isdir(de_outdir):
            os.symlink(os.path.abspath(de_outdir), de_newdir)

    # phenotype table
    de_config = yaml.safe_load(open(diffexpr_config, 'r'))
    phe_mat = de_config['pheno_tab']
    if os.path.isfile(phe_mat):
        phe_newdir = os.path.join(results_newdir, 'phenotype')
        os.mkdir(phe_newdir)
        new = os.path.join(phe_newdir, 'phenotypes.mat')
        if os.path.isfile(os.path.realpath(phe_mat)):
            os.symlink(os.path.abspath(os.path.realpath(phe_mat)), new)
