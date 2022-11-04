# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# To submit the data to Seq2Geno server, this script helps to
# pack the data, write new config file, and finally generate a zip file
import shutil
import yaml
import os
from tqdm import tqdm
import logging
import sys
import argparse
# SEQ2GENO_HOME has been added to the path in the main script
from main.UserOptions import main as AskArg
from main.UserOptions import arguments, make_parser
from main.UserOptions import parse_arg_yaml, check_primary_args
from LoadFile import LoadFile


def test_material(f):
    try:
        assert os.path.isfile(f)
    except AssertionError:
        raise FileNotFoundError(f)


def move_data(config_f, new_zip_prefix, new_dir,
              logger=''):
    ###
    # initiate the config parameters
    # by cloning the old ones
    # except for the filepaths below
    if not os.path.isfile(config_f):
        logger.error('{} not found')
        raise FileNotFoundError
    new_config_yaml_dict = {}
    with LoadFile(config_f) as config_fh:
        config_yaml_dict = yaml.safe_load(config_fh)
        new_config_yaml_dict = config_yaml_dict
    # fix the paths because the server will need to go editting them
    # immediately after unpacking the zip
    os.makedirs(os.path.join(new_dir, 'files'))
    new_config_yaml = os.path.join(new_dir, 'files', '_seq2geno_inputs.yml')
    new_list_f = os.path.join(new_dir, 'files', '_dna_list')
    new_rna_list_f = os.path.join(new_dir, 'files', '_rna_list')
    redirected_reads_dir = os.path.join(
        new_dir, 'reads', 'dna')
    redirected_rna_reads_dir = os.path.join(
        new_dir, 'reads', 'rna')
    # The paths below should be the 'configured' values; that is, after the
    # '__' are replaced with the exact path
    new_config_yaml_dict['general']['dna_reads'] = '__/dna_list'
    new_config_yaml_dict['general']['rna_reads'] = '__/rna_list'

    ###
    # edit the filenames
    # parse the arguments
    # in a similar way of UserOptions.py
    args_array = ['-f', config_f]
    parser = make_parser()
    primary_args = ''
    try:
        primary_args = parser.parse_args(args_array)
    except argparse.ArgumentError:
        sys.exit('Argument error')
    # check those primary arguments
    check_primary_args(primary_args)

    # details in the config file
    args = parse_arg_yaml(primary_args.yml_f)
    new_config_yaml_dict['general']['wd'] = '__/{}'.format(
        os.path.basename(args.wd))

    ###
    # move the files according to the configuration
    ###
    # dna reads
    # 1. copy the reads files
    # 2. generate a new list
    # 3. update data to generate a new config yaml
    if type(logger) is logging.Logger:
        logger.info('Moving DNA-seq data')
    list_f = os.path.abspath(args.dna_reads)
    if not os.path.isdir(redirected_reads_dir):
        os.makedirs(redirected_reads_dir)
    dna_reads = {}
    new_dna_reads = {}
    list_lines = LoadFile(list_f).readlines()
    for line in tqdm(list_lines):
        d = line.strip().split('\t')
        dna_reads[d[0]] = d[1].split(',')
        try:
            assert ((len(d) == 2) and (len(d[1].split(',')) == 2))
        except AssertionError:
            print('''ERROR: Incorrect format detected
                  in "{}"'''.format(line.strip()))
            raise AssertionError
        else:
            for f in dna_reads[d[0]]:
                test_material(f)
                shutil.copyfile(f,
                                os.path.join(redirected_reads_dir,
                                             os.path.basename(f)),
                                follow_symlinks=True)
            new_dna_reads[d[0]] = '__/{},__/{}'.format(
                os.path.join(os.path.relpath(redirected_reads_dir, new_dir),
                             os.path.basename(dna_reads[d[0]][0])),
                os.path.join(os.path.relpath(redirected_reads_dir, new_dir),
                             os.path.basename(dna_reads[d[0]][1])))
    with open(new_list_f, 'w') as new_list_fh:
        for s in new_dna_reads:
            new_list_fh.write('{}\t{}\n'.format(
                s, new_dna_reads[s]))

    ###
    # rna reads
    # 1. copy the reads files
    # 2. generate a new list
    # 3. update data to generate a new config yaml
    rna_list_f = os.path.abspath(args.rna_reads)
    if rna_list_f == '-':
        new_config_yaml_dict['general']['rna_reads'] = '-'
    else:
        if type(logger) is logging.Logger:
            logger.info('Moving RNA-seq data')
        if not os.path.isdir(redirected_rna_reads_dir):
            os.makedirs(redirected_rna_reads_dir)
        rna_reads = {}
        new_rna_reads = {}
        rna_list_lines = LoadFile(rna_list_f).readlines()
        for line in tqdm(rna_list_lines):
            d = line.strip().split('\t')
            rna_reads[d[0]] = d[1]
            f = d[1]
            try:
                assert len(d) == 2
            except AssertionError:
                print('ERROR: Incorrect format detected '
                      'in "{}"'.format(line.strip()))
                raise AssertionError
            else:
                test_material(f)
                shutil.copyfile(f,
                                os.path.join(redirected_rna_reads_dir,
                                             os.path.basename(f)),
                                follow_symlinks=True)
                new_rna_reads[d[0]] = '__/{}'.format(
                    os.path.join(os.path.relpath(redirected_rna_reads_dir,
                                                 new_dir),
                                 os.path.basename(rna_reads[d[0]])))
        with open(new_rna_list_f, 'w') as new_rna_list_fh:
            for s in new_rna_reads:
                new_rna_list_fh.write('{}\t{}\n'.format(
                    s, new_rna_reads[s]))

    ###
    # reference
    # 1. copy the files
    # 2. determine new filenames
    # 3. update data to generate a new config yaml
    if type(logger) is logging.Logger:
        logger.info('Moving reference data')
    ref_files_dict = {'ref_fa': args.ref_fa,
                      'ref_gbk': args.ref_gbk,
                      'ref_gff': args.ref_gff
                      }
    new_ref_files_dict = {}
    redirected_ref_dir = os.path.join(
        new_dir, 'reference')
    if not os.path.isdir(redirected_ref_dir):
        os.makedirs(redirected_ref_dir)
    for ft in ref_files_dict:
        f = ref_files_dict[ft]
        try:
            assert os.path.isfile(f)
        except AssertionError:
            raise FileNotFoundError(f)
        else:
            shutil.copyfile(f,
                            os.path.join(redirected_ref_dir,
                                         os.path.basename(f)),
                            follow_symlinks=True)
            new_ref_files_dict[ft] = os.path.join(redirected_ref_dir,
                                                  os.path.basename(f))

    for ft in new_ref_files_dict:
        new_config_yaml_dict['general'][ft] = '__/{}'.format(
            os.path.relpath(new_ref_files_dict[ft], new_dir))

    ###
    # adaptor
    # 1. copy the file
    # 2. determine new filename
    # 3. update data to generate a new config yaml
    # adaptor
    adaptor = args.adaptor
    redirected_adaptor_dir = os.path.join(
        new_dir, 'adaptor')
    new_adaptor = ''
    if adaptor == '-':
        new_config_yaml_dict['general']['adaptor'] = '-'
    else:
        if type(logger) is logging.Logger:
            logger.info('Moving daptor')
        if not os.path.isdir(redirected_adaptor_dir):
            os.makedirs(redirected_adaptor_dir)
        test_material(adaptor)
        new_adaptor = os.path.join(redirected_adaptor_dir,
                                   os.path.basename(adaptor))
        shutil.copyfile(adaptor, new_adaptor,
                        follow_symlinks=True)
        new_config_yaml_dict['general']['adaptor'] = '__/{}'.format(
            os.path.relpath(new_adaptor, new_dir))

    ###
    # phenotypes
    # 1. copy the file
    # 2. determine new filename
    # 3. update data to generate a new config yaml
    phe_table = args.phe_table
    new_phe_table = ''
    redirected_phe_table_dir = os.path.join(
        new_dir, 'phe_table')
    new_config_yaml_dict['general']['phe_table'] = '-'
    if phe_table != '-':
        if type(logger) is logging.Logger:
            logger.info('Moving phenotype data')
        if not os.path.isdir(redirected_phe_table_dir):
            os.makedirs(redirected_phe_table_dir)
        test_material(phe_table)
        new_phe_table = os.path.join(redirected_phe_table_dir,
                                     os.path.basename(phe_table))
        shutil.copyfile(phe_table, new_phe_table,
                        follow_symlinks=True)
        new_config_yaml_dict['general']['phe_table'] = '__/{}'.format(
            os.path.relpath(new_phe_table, new_dir))

    ###
    # Finally, the config yaml itself
    with open(new_config_yaml, 'w') as new_config_yaml_h:
        yaml.dump(new_config_yaml_dict,
                  new_config_yaml_h,
                  default_flow_style=False)
    # and create the zip
    shutil.make_archive(new_zip_prefix, 'zip',
                        os.path.dirname(new_dir),
                        os.path.basename(new_dir))
    if not os.path.isfile(new_zip_prefix+'.zip'):
        raise IOError('zip not created')
    else:
        shutil.rmtree(new_dir)
        return(True)


if __name__ == '__main__':
    ###
    # user args
    # input
    config_f = '../../example_sg_dataset/seq2geno_inputs.yml'
    # output
    new_zip_prefix = './test'
    new_dir = os.path.abspath(new_zip_prefix)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    move_data(config_f=config_f, new_zip_prefix=new_zip_prefix,
              new_dir=new_dir)
