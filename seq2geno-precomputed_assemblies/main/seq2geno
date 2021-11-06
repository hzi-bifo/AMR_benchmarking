#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Determine, initiate and launch the workflows based on user-defined
# arguments

import os
import sys
from tqdm import tqdm
import shutil
from SGProcesses import SGProcess
import UserOptions
from CollectResults import collect_results
import LogGenerator
import create_config

# ensure the core environment variable
assert 'SEQ2GENO_HOME' in os.environ, 'SEQ2GENO_HOME not available'
sys.path.append(os.environ['SEQ2GENO_HOME'])
sys.path.append(os.path.join(os.environ['SEQ2GENO_HOME'], 'main'))
#from Seq2GenoUtils import Warehouse
#from Seq2GenoUtils.Crane import Seq2Geno_Crane
#from PackOutput import SGOutputPacker
from ZIP2Config import *


def filter_procs(args, logger):
    # Determine which procedures to include
    config_files = {}
    # accept config files
    try:
        config_files = create_config.main(args, logger)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit()

    all_processes = []
    # >>>
    # initiate processes

    # expr
    if args.expr == 'Y':
        # ensure required data
        all_processes.append(SGProcess(
            args.wd, 'expr',
            config_f=config_files['expr'],
            dryrun=args.dryrun,
            mem_mb=int(args.mem_mb),
            max_cores=int(args.cores)))
    else:
        logger.info('Skip counting expression levels')

    # snps
    if args.snps == 'Y':
        # ensure required data
        all_processes.append(
            SGProcess(args.wd, 'snps',
                      config_f=config_files['snps'],
                      dryrun=args.dryrun,
                      mem_mb=int(args.mem_mb),
                      max_cores=int(args.cores)))
    else:
        logger.info('Skip calling single nucleotide variants')

    # denovo
    if args.denovo == 'Y':
        # ensure required data
        all_processes.append(
            SGProcess(args.wd, 'denovo',
                      config_f=config_files['denovo'],
                      dryrun=args.dryrun,
                      mem_mb=int(args.mem_mb),
                      max_cores=int(args.cores)))
    else:
        logger.info('Skip creating de novo assemblies')

    # phylo
    if args.phylo == 'Y':
        # ensure required data
        all_processes.append(
            SGProcess(args.wd, 'phylo',
                      config_f=config_files['phylo'],
                      dryrun=args.dryrun,
                      mem_mb=int(args.mem_mb),
                      max_cores=int(args.cores)))
    else:
        logger.info('Skip inferring phylogeny')

    # ancestral reconstruction
    if args.ar == 'Y':
        # ensure required data
        all_processes.append(
            SGProcess(args.wd, 'ar',
                      config_f=config_files['ar'],
                      dryrun=args.dryrun,
                      mem_mb=int(args.mem_mb),
                      max_cores=int(args.cores)))
    else:
        logger.info('Skip ancestral reconstruction')

    # differential expression
    if args.de == 'Y':
        all_processes.append(
            SGProcess(args.wd, 'de',
                      config_f=config_files['de'],
                      dryrun=args.dryrun,
                      mem_mb=int(args.mem_mb),
                      max_cores=int(args.cores)))
    else:
        logger.info('Skip differential expression analysis')

    return({'selected': all_processes, 'config_files': config_files})


def main(args, logger):
    determined_procs = filter_procs(args, logger)
    config_files = determined_procs['config_files']
    all_processes = determined_procs['selected']
    # >>>
    # start running processes
    processes_num = len(all_processes)+1
    pbar = tqdm(total=processes_num,
                desc="\nseq2geno")
    for p in all_processes:
        p.run_proc(logger)
        pbar.update(1)
    if args.dryrun != 'Y':
        collect_results(args.wd, config_files)
    logger.info('Working directory {} {}'.format(
        args.wd, 'updated' if args.dryrun != 'Y' else 'unchanged'))
    pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    parser = UserOptions.make_parser()
    primary_args = parser.parse_args()
    # ensure yml or zip specified
    if ((not os.path.isfile(primary_args.yml_f))
        and os.path.isfile(primary_args.zip_f)):
        primary_args.yml_f = zip2config(primary_args.zip_f)
    elif ((not os.path.isfile(primary_args.yml_f))
          and (not os.path.isfile(primary_args.zip_f))):
        raise AttributeError('Neither input yml nor zip found')
    # create logger
    logger = LogGenerator.make_logger(primary_args.log_f)
    # check those primary arguments
    logger.info('Parse arguments')
    args = UserOptions.parse_arg_yaml(primary_args.yml_f)
    args.print_args()
    # display the primary arguments only
    if primary_args.dsply_args:
        sys.exit(0)

    # run in local machine
    main(args, logger)
    # pack the results or not?
    '''
    if args.dryrun != 'Y':
        if primary_args.pack_output == 'none':
            logger.info('Not packing the results')
        else:
            output_zip = '{}.zip'.format(args.wd)
            packer = SGOutputPacker(seq2geno_outdir=args.wd,
                                    output_zip=output_zip)
            if primary_args.pack_output == 'all':
                logger.info('Packing all data')
                packer.pack_all_output()
            elif primary_args.pack_output == 'main':
                logger.info('Packing the main data')
                packer.pack_main_output()
            elif primary_args.pack_output == 'g2p':
                logger.info('Packing data needed by Geno2Pheno')
                gp_config_yml = '{}.yml'.format(args.wd)
                project_name = os.path.basename(args.wd)
                packer.make_gp_input_zip(gp_config=gp_config_yml,
                                         project_name=project_name,
                                         logger=logger)

            # ensure the zip file correctly generated
            if not os.path.isfile(output_zip):
                raise IOError('zip not created')

            # submit the data to Geno2Pheno server
#            if primary_args.pack_output == 'g2p' and primary_args.to_gp:
#                c = Seq2Geno_Crane(logger=logger)
#                c.choose_materials(output_zip)
#                c.launch()
            
 #       logger.info('DONE (local mode)')
    '''
