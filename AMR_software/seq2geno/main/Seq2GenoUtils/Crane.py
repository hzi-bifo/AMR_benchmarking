import time
from bioblend.galaxy import GalaxyInstance
from bioblend.galaxy.tools.inputs import inputs
import sys
import logging
import re
import zipfile
import os
from .SetKey import SetKey


class Seq2Geno_Crane():
    def __init__(self, logger='', email='',
                 url='https://galaxy.bifo.helmholtz-hzi.de/galaxy/'):
        self.url = url
        self.email = ''
        if logger == '':
            logger_format = '%(asctime) %(message)s'
            logging.basicConfig(format=logger_format)
        self.logger = logger

    def _setup_key(self):
        self.key = SetKey()

    def _fetch_galaxy(self):
        '''upload the validated zip but not yet start the process'''
        assert hasattr(self, 'key'), 'Sumission key not set'
        self.gi = GalaxyInstance(url=self.url, key=self.key, verify=False)

    def _submit(self):
        histories = self.gi.histories.get_histories()
        self.history_id = ''
        if len(histories) > 0:
            self.history_id = histories[0]['id']
        else:
            hist = self.gi.histories.create_history()
            self.history_id = hist['id']
        job_obj = ''
        job_obj = self.gi.tools.upload_file(self.materials_zip,
                                            self.history_id)

        email_regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
        job_id = job_obj['jobs'][0]['id']
        self.myinputs = inputs().set_dataset_param("input", job_id, src="hda")
        if self.email != '' and \
           not (re.search(self.email, email_regex) is None):
            self.myinputs.set_param("email", self.email)
        else:
            print('Empty or invalid email address. Setup skipped')

    def choose_materials(self, materials_zip):
        self.materials_zip = materials_zip
        zipfile.ZipFile(self.materials_zip).testzip()
        self.wdir = os.path.dirname(materials_zip)

    def launch(self, download_outdir='geno2pheno_out',
               output_zip_f='seq2geno_results.zip',
               tool='genopheno'):

        ###
        #  SG zip submission to server
        self._setup_key()
        self._fetch_galaxy()
        self.logger.info('Submitting {} to the server'.format(
            self.materials_zip))
        self._submit()
        # ensuring that the submission is done
        history_obj = self.gi.histories.show_history(self.history_id,
                                                     contents=False)
        if self.logger is logging.Logger:
            self.logger.info(
                'Submission state {}'.format(history_obj['state']))

        while history_obj['state'] != 'ok':
            assert history_obj['state'] != 'error', sys.exit(
                'Data submission failed')
            time.sleep(100)
            history_obj = self.gi.histories.show_history(
                self.history_id, contents=False)
        self.logger.info('Data submission done')

        ####
        ##  SG run with submitted data
        self.gi.tools.run_tool(history_obj['id'],tool, self.myinputs)
        # check status and download the result once finished
        history_obj= self.gi.histories.show_history(self.history_id, contents=False)
        while history_obj['state'] != 'ok':
            # check it every 100s
            time.sleep(100)
            assert history_obj['state'] != 'error', (
                'Error in Geno2Pheno server')
            history_obj= self.gi.histories.show_history(self.history_id, contents=False)
        if self.logger is logging.Logger:
            self.logger.info('Genotyping finished')

        ####
        ## download the GP result
        data_name= 'GenoPheno output'
        sg_out= [d for d in self.gi.datasets.get_datasets()
                 if d['name'] == data_name].pop()
        download_outdir = os.path.join(self.wdir, download_outdir)
        if not os.path.isdir(download_outdir):
            os.makedirs(download_outdir)
        self.gi.datasets.download_dataset(sg_out['dataset_id'],
                                     file_path=os.path.join(download_outdir,output_zip_f),
                                     use_default_filename=False)

        self.logger.info('Data downloaded')
