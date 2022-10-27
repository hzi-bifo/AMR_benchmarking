# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Create the output zip files

import sys
import os
import shutil
from SGPUtils import create_genyml
import re
from lib.Geno2PhenoClient import validator


class SGOutputPacker:
    def __init__(self, seq2geno_outdir, output_zip):
        self.seq2geno_outdir = seq2geno_outdir
        self.output_zip = output_zip

    def pack_all_output(self):
        # pack everything from Seq2Geno
        if os.path.isfile(self.output_zip):
            raise FileExistsError('{} exists'.format(self.output_zip))
        shutil.make_archive(re.sub('.zip$', '', self.output_zip),
                            'zip', self.seq2geno_outdir)

    def pack_main_output(self):
        # pack the main results from Seq2Geno
        if os.path.isfile(self.output_zip):
            raise FileExistsError('{} exists'.format(self.output_zip))
        main_outdir = '{}/RESULTS'.format(self.seq2geno_outdir)
        shutil.make_archive(re.sub('.zip$', '', self.output_zip),
                            'zip', main_outdir)

    def make_gp_input_zip(self, gp_config,
                          project_name, logger):
        # pack those needed by Geno2Pheno
        if os.path.isfile(self.output_zip):
            raise FileExistsError('{} exists'.format(self.output_zip))
        genml_creator_args_array = [
            '--seq2geno', self.seq2geno_outdir,
            '--proj', 'SGP',
            '--yaml', gp_config]
        genml_creator_args_parser = create_genyml.make_parser()
        genml_creator_args = genml_creator_args_parser.parse_args(
            genml_creator_args_array)
        try:
            create_genyml.make_genyml(genml_creator_args)
        except IOError as e:
            print(e)
            print('Creation of {} failed'.format(gp_config))

        ####
        #  genml -> GP zip
        try:
            # run the validator
            log = '{}_log'.format(gp_config)
            val = validator.ValidateGenML(gp_config, log)
            val.create_zip_file(self.output_zip)
            if not os.path.isfile(self.output_zip):
                raise FileNotFoundError(self.output_zip)
        except IOError:
            logger.error('Validation or compression failed. Exit')
            sys.exit()
        except FileNotFoundError:
            logger.error('Failed in generating {}. Exit'.format(
                self.output_zip))
            sys.exit()
