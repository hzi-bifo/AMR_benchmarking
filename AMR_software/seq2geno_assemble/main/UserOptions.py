#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Parse the yaml file and pass the user's options to the package facade
import yaml
import os
from pprint import pprint
import sys
import argparse
import re
from LoadFile import LoadFile


class arguments:
    # The object of arguments, as argparse is replaced
    def add_opt(self, **entries):
        self.__dict__.update(entries)

    def print_args(self):
        pprint(vars(self))

    def check_args(self):
        import ArgsTest

        # default values of optional arguments
        optional_args = {'cores': 1, 'mem_mb': -1,
                         'adaptor': '-', 'rna_reads': '-',
                         'dryrun': 'Y', 'phe_table': '-',
                         'denovo': 'N', 'snps': 'N',
                         'expr': 'N', 'phylo': 'N'}
        for k in optional_args:
            if not hasattr(self, k):
                # blanck
                setattr(self, k, optional_args[k])
            elif len(str(getattr(self, k))) == 0:
                # empty
                setattr(self, k, optional_args[k])

        # obligatory arguments
        obligatory_args = ['dna_reads', 'ref_fa', 'wd']
        # ensure obligatory arguments included
        for k in obligatory_args:
            assert hasattr(self, k), '"{}" not properly set'.format(k)

        # check the reference genome
        ArgsTest.test_reference_seq(self.ref_fa)
        # check the dna-seq data
        ArgsTest.test_dna_reads_list(self.dna_reads)
        # check the rna-seq data (if set)

        if getattr(self, 'expr') == 'Y':
            if not re.search('\w', getattr(self, 'rna_reads')) is None:
                ArgsTest.test_rna_reads_list(self.rna_reads)


def parse_arg_yaml(yml_f):
    # Parse the yaml file where the parameters previously were
    # commandline options

    # read the arguments
    # flatten the structure
    opt_dict = {}
    with LoadFile(yml_f) as yml_fh:
        opt_dict = yaml.safe_load(yml_fh)
    # reuse the old config files
    if not ('old_config' in opt_dict['general']):
        opt_dict['general']['old_config'] = 'N'

    args = arguments()
    try:
        args.add_opt(**opt_dict['general'])
        args.add_opt(**opt_dict['features'])
    except KeyError as e:
        sys.exit('ERROR: {} not found in the input file'.format(str(e)))
    else:
        args.check_args()
        return(args)


def check_primary_args(primary_args):
    print(primary_args)
    # When to_gp is opted, the output must be 'g2p' zip file
    if primary_args.to_gp:
        primary_args.pack_output = 'g2p'
    # When the log file is used, merging the stdout and stderr
    if primary_args.log_f != '':
        sys.stdout = LoadFile(primary_args.log_f)
        sys.stderr = sys.stdout

    # the yml file must exist
    assert os.path.isfile(primary_args.yml_f), 'The yaml file not existing'
    print('#CONFIGFILE:{}'.format(primary_args.yml_f))


def make_parser():
    # Find the yaml file of arguments
    parser = argparse.ArgumentParser(
        prog='seq2geno',
        formatter_class=argparse.RawTextHelpFormatter,
        description='''
        Seq2Geno: the automatic tool for computing genomic features from
        the sequencing data\n''')

    parser.add_argument('-v', action='version',
                        version='v1.00001')
    parser.add_argument('-d', dest='dsply_args', action='store_true',
                        help='show the arguments described in '\
                        'the config file (yaml) and exit')
    parser.add_argument('-f', dest='yml_f', required=False, default='',
                        help='the yaml file where the arguments are listed')
    parser.add_argument('-z', dest='zip_f', required=False, default='',
                        help='the zip file of materials')
    parser.add_argument('-l', dest='log_f', required=False,
                        default='',
                        help='a non-existing filename for log')
    parser.add_argument('--to_gp', dest='to_gp', action='store_true',
                        help='''submit the result to Geno2Pheno server''')
    parser.add_argument('--outzip', dest='pack_output',
                        choices=['none', 'all', 'main', 'g2p'],
                        default='none',
                        help=(
                            "Pack the results into a zip file. Opt 'none' "
                            "for keeping the folder;\n'all' for packing "
                            "everything, including the intermediate data, and "
                            "delete the working directory;\n"
                            "'main' for packing the main results and deleting "
                            "the working directory;\n"
                            "'g2p' for packing only those needed by the "
                            "predictive package Geno2Pheno and deleting "
                            "the workign directory (automatically "
                            "opted in the to_gp mode)"))
    return(parser)


def main():
    parser = make_parser()
    primary_args = parser.parse_args()

    # check those primary arguments
    check_primary_args(primary_args)

    args = parse_arg_yaml(primary_args.yml_f)
    # display the primary arguments only
    if primary_args.dsply_args:
        args.print_args()
        sys.exit(0)

    return(args)
