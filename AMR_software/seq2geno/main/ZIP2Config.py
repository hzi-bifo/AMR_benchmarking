# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import shutil
import re
import yaml

# Create a config file from a structured zip
#

def explore_resources_dir(resources_dir=''):
    # explore the resources folder and determine the resources to include
    possible_resources = {'cores': '1',
                          'mem_mb': '2000'}
    resources_file = os.path.join(resources_dir, 'resources')
    if resources_dir != '':
        with open(resources_file, 'r') as resources_fh:
            for line in resources_fh.readlines():
                res, val = line.strip().split('\t')
                if res in possible_resources:
                    possible_resources[res] = val
    return(possible_resources)


def explore_functions_dir(functions_dir): #modified by Khu for genomic data Nov 2022
    # explore the functions folder and determine the functions to include
    possible_functions = {'snps': 'N',
                          'denovo': 'Y',
                          'expr': 'N',
                          'phylo': 'N',
                          'de': 'N',
                          'ar': 'N',
                          'dryrun': 'Y'}

    functions_file = os.path.join(functions_dir, 'functions')
    assert os.path.isfile(functions_file), (
        'Functions file {} not found'.format(functions_file))
    # with open(functions_file, 'r') as functions_fh:
    #     for func in functions_fh.readlines():
    #         func = func.strip()
    #         if func in possible_functions:
    #             possible_functions[func] = 'Y'


    return(possible_functions)


def explore_dna_reads_dir(dna_reads_dir):
    # explore the dna reads folder and return a dictionary of the reads pairs
    # to the samples
    paired_dna_reads_pat = '([^\.]+)\.([12])\.(fastq|fq)\.gz$'
    dna_reads_dict = {}
    for f in os.listdir(dna_reads_dir):
        re_out = re.search(paired_dna_reads_pat, f)
        if re_out is None:
            continue
        sample = re_out.group(1)
        end = re_out.group(2)
        if not sample in dna_reads_dict:
            dna_reads_dict[sample] = {}

        dna_reads_dict[sample][end] = os.path.join(dna_reads_dir, f)
    assert len(dna_reads_dict) > 0, 'No dna reads found'
    assert all([len(dna_reads_dict[x]) == 2 for x in dna_reads_dict]), (
        'Unpaired reads found')
    return(dna_reads_dict)



def explore_reference_dir(reference_dir):
    # explore the reference folder and return a dictionary of the files
    # to each filetype
    reference_pats = {'fna': '\.fna$', 'gbk': '\.gbk$', 'gff': '\.gff$'}

    reference_dict = {}
    for pat_type in reference_pats:
        pat = reference_pats[pat_type]
        matched_files= [f for f in os.listdir(reference_dir)
                        if re.search(pat, f) is not None]
        # ensure a single reference of the filetype
        assert len(matched_files) == 1, 'Multiple {} files found'.format(
            pat)
        reference_dict[pat_type] = os.path.join(
            reference_dir, matched_files.pop())
    return(reference_dict)



def explore_rna_reads_dir(rna_reads_dir):
    # explore the rna reads folder and return a dictionary of the reads pairs
    # to the samples
    rna_reads_pat = '([^\.]+)\.(fastq|fq)\.gz$'
    rna_reads_dict = {}
    for f in os.listdir(rna_reads_dir):
        re_out = re.search(rna_reads_pat, f)
        if re_out is None:
            continue
        sample = re_out.group(1)
        if sample in rna_reads_dict:
            raise FileExistsError(rna_reads_dict[sample])
        rna_reads_dict[sample] = os.path.join(rna_reads_dir, f)
    assert len(rna_reads_dict) > 0, 'No rna reads found'
    return(rna_reads_dict)


def explore_phenotype_dir(phenotype_dir):
    # explore the phenotype folder and return a dictionary of the files
    # to each filetype
    phenotype_pats = {'mat': '\.mat$'}

    phenotype_dict = {}
    for pat_type in phenotype_pats:
        pat = phenotype_pats[pat_type]
        matched_files= [f for f in os.listdir(phenotype_dir)
                        if re.search(pat, f) is not None]
        # ensure a single phenotype of the filetype
        assert len(matched_files) == 1, 'Multiple or no {} files found'.format(
            pat)
        phenotype_dict[pat_type] = os.path.join(
            phenotype_dir, matched_files.pop())
    return(phenotype_dict)


def zip2config(input_zip):
    # use absolute path
    input_zip = os.path.abspath(input_zip)
    # unpack the file
    shutil.unpack_archive(filename=input_zip,
                         extract_dir=os.path.dirname(input_zip),
                         format='zip')
    input_dir = re.sub('\.zip$', '', input_zip)
    print(os.listdir(os.path.dirname(input_zip)))
    assert os.path.isdir(input_dir), '{} not found'.format(input_dir)

    # test the structure
    # required ones
    standard_rel_paths = {
        'functions': 'functions',
        'dna_reads': 'reads/dna/',
        'reference': 'reference'}
    standard_abs_paths = {x: os.path.join(
        input_dir, standard_rel_paths[x])
        for x in standard_rel_paths}
    for x in standard_abs_paths:
        if not os.path.isdir(standard_abs_paths[x]):
            raise NotADirectoryError(standard_abs_paths[x])
    # optional ones
    optional_rel_paths = {
        'rna_reads': 'reads/rna',
        'resources': 'resources',
        'phenotype': 'phenotype'}
    optional_abs_paths = {}
    for x in optional_rel_paths:
        dir_name = os.path.join(
            input_dir, optional_rel_paths[x])
        if os.path.isdir(dir_name):
            optional_abs_paths[x] = dir_name

    # the functions
    print('Checking functions')
    functions_dir = standard_abs_paths['functions']
    functions_dict = explore_functions_dir(functions_dir)


    # the reads file
    # dna
    print('Checking DNA-seq reads')
    dna_reads_dir = standard_abs_paths['dna_reads']
    dna_reads_dict = explore_dna_reads_dir(dna_reads_dir)
    dna_reads_list_f = os.path.join(input_dir, 'dna_list')
    with open(dna_reads_list_f, 'w') as dna_reads_list_fh:
        for sample in dna_reads_dict:
            dna_reads_list_fh.write('{}\t{},{}\n'.format(
                sample,
                dna_reads_dict[sample]['1'],
                dna_reads_dict[sample]['2']))


    # reference
    print('Checking reference data')
    reference_dict = explore_reference_dir(standard_abs_paths['reference'])

    # reference
    resources_dict = {}
    if 'resources' in optional_abs_paths:
        print('Checking resources data')
        resources_dict = explore_resources_dir(optional_abs_paths['resources'])
    else:
        resources_dict = explore_resources_dir()

    # rna
    rna_reads_dict = {}
    if 'rna_reads' in optional_abs_paths:
        print('Checking RNA-seq data')
        rna_reads_dir = optional_abs_paths['rna_reads']
        rna_reads_dict = explore_rna_reads_dir(rna_reads_dir)
        rna_reads_list_f = os.path.join(input_dir, 'rna_list')
        with open(rna_reads_list_f, 'w') as rna_reads_list_fh:
            for sample in rna_reads_dict:
                rna_reads_list_fh.write('{}\t{}\n'.format(
                    sample,
                    rna_reads_dict[sample]))

    # phenotype
    phenotype_dict = {}
    if 'phenotype' in optional_abs_paths:
        print('Checking phenotype data')
        phenotype_dir = optional_abs_paths['phenotype']
        phenotype_dict = explore_phenotype_dir(phenotype_dir)


    # write config yaml
    config_dict = {'features': functions_dict,
                   'general':{
                       'cores': resources_dict['cores'],
                       'mem_mb': resources_dict['mem_mb'],
                       'dna_reads': dna_reads_list_f,
                       'phe_table': phenotype_dict['mat'],
                       'ref_fa': reference_dict['fna'],
                       'ref_gff': reference_dict['gff'],
                       'ref_gbk': reference_dict['gbk'],
                       'rna_reads': rna_reads_list_f,
                       'wd': os.path.join(input_dir, 'results')}}

    print(config_dict)
    config_yml_f = os.path.join(input_dir, 'seq2geno_input.yml')
    with open(config_yml_f, 'w') as config_yml_fh:
        yaml.dump(config_dict, config_yml_fh, default_flow_style=False)
    assert os.path.isfile(config_yml_f)
    print(config_yml_f)
    return(config_yml_f)
