# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Define the class of workflow

import os
import psutil
import sys
import snakemake
import re
import pandas as pd


class SGProcess:
    # The class for each workflow
    # run_proc: execute the workflow
    # EditEnv: update the environment variables
    def __init__(self, wd, proc,
                 config_f, dryrun=True,
                 max_cores=1, mem_mb=-1):
        self.proc = proc
        self.dryrun = dryrun
        self.config_f = config_f
        # check and adjust the core number setting
        cpu_count = int(psutil.cpu_count())
        if (max_cores > cpu_count) or (max_cores < 1):
            print(('The number of cpu was {}; cores setting '
                  'adjusted').format(str(cpu_count)))
            self.max_cores = max(int(cpu_count-1), 1)
        else:
            self.max_cores = int(max_cores)

        # check and adjust the memory size setting
        freemem = psutil.virtual_memory().available/1e6
        if (mem_mb > freemem) or (mem_mb <= 0):
            print(('Currently free memory size was {}mb; memory setting '
                  'adjusted').format(str(freemem)))
            self.mem_mb = int(freemem * 0.8)
        else:
            self.mem_mb = int(mem_mb)

    def run_proc(self, logger):
        proc = self.proc
        dryrun = True if self.dryrun == 'Y' else False
        max_cores = self.max_cores
        config_f = self.config_f
        env_dict = self.EditEnv(proc, logger)
        logger.info(proc)

        os.environ['PATH'] = env_dict['PATH']

        # run the process
        success = snakemake.snakemake(
            snakefile=env_dict['SNAKEFILE'],
            lock=False,
            restart_times=3,
            cores=max_cores,
            resources={'mem_mb': self.mem_mb},
            configfiles=[config_f],
            force_incomplete=True,
            workdir=os.path.dirname(config_f),
            use_conda=True,
            conda_prefix=os.path.join(env_dict['TOOL_HOME'], 'env'),
            dryrun=dryrun,
            printshellcmds=False,
            notemp=False
            )

        if not success:
            logger.error('{} failed'.format(proc))
            sys.exit()

    def EditEnv(self, proc, logger):
        # Set up the enviornment variables accoring to the activated workflows
        script_dir = os.path.dirname(os.path.realpath(__file__))
        toolpaths_f = os.path.join(script_dir, 'ToolPaths.tsv')
        # read the env variables
        env_df = pd.read_csv(toolpaths_f, sep='\t', comment='#', index_col=0)
        env_series = pd.Series([])
        env_dict = {}
        try:
            # ensure the most important variable
            assert 'SEQ2GENO_HOME' in os.environ
        except AssertionError:
            logger.error('"SEQ2GENO_HOME" not properly set')
            sys.exit()

        try:
            # read the table of required paths
            env_series = env_df.loc[proc, :]
        except KeyError:
            # when unable to recognize the function
            logger.error('Unknown function {}'.format(proc))
            sys.exit()
        else:
            # start setting the environment variables
            try:
                os.environ['TOOL_HOME'] = os.path.join(
                    os.environ['SEQ2GENO_HOME'],
                    str(env_series['TOOL_HOME']))
                all_env_var = dict(os.environ)
                env_dict = {'TOOL_HOME': all_env_var['TOOL_HOME']}
                for env in env_series.index.values.tolist():
                    if env == 'TOOL_HOME':
                        continue
                    val = str(env_series[env])
                    included = list(set(re.findall('\$(\w+)', val)))
                    for included_env in included:
                        val = re.sub('\$'+included_env,
                                     all_env_var[included_env],
                                     val)
                    env_dict[env] = val
            except KeyError:
                logger.error(
                    'Unable to set environment for "{}"'.format(proc))
                sys.exit()
            else:
                logger.info(
                    'Environment variables for "{}" ready'.format(
                        proc))
                return(env_dict)
