from cerberus import Validator
import yaml
import newick
import codecs
from Bio import Phylo
from io import StringIO
import pandas as pd
import itertools
import os
import copy
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import fnmatch
import logging
from zipfile import ZipFile
import zipfile
import datetime
import time
import hashlib
import tqdm
from collections import  OrderedDict

class ValidateGenML(object):
    '''
        Developed by Ehsaneddin Asgari
        for the GenoPheno Server verification
    '''

    def __init__(self, config_file, logger_file):

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(logging.DEBUG)

        fileHandler = logging.FileHandler(logger_file)
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)

        self.nodes_in_tree = []
        self.nodes_in_phenotype  = []
        self.feature_types = []
        self.existing_validations = []
        self.genotype_tables = dict()
        self.sequence_dir = dict()
        self.__load_schema()
        self.config = self.__load_config(config_file)
        self.v = Validator()
        self.v.validate(self.config, self.schema_general)
        if self.v.errors:
            for er in self.v.errors:
                self.rootLogger.error(str(er))
        self.generate_zip = self.v.validate(self.config, self.schema_general)


    @staticmethod
    def exists(file_path):
        return os.path.exists(file_path)

    def check_tree(self, field, value, error):
        self.tree_path = value
        if not ValidateGenML.exists(self.tree_path):
            self.rootLogger.error("The phylogenetic tree file does not exist!")
            error(field, "The phylogenetic tree file does not exist!")
        try:
            self.newick = Phylo.read(codecs.open(self.tree_path,'rb','utf-8'), "newick")
            self.nodes_in_tree = [n.name for n in self.newick.get_terminals()]
        except:
            self.rootLogger.error("The phylogenetic tree does not follow standard newick format!")
            error(field, "The phylogenetic tree does not follow standard newick format!")
        return True

    def check_phenotype(self, field, value, error):
        self.phenotype_path = value
        if not ValidateGenML.exists(self.phenotype_path):
            self.rootLogger.error( "The phenotype file does not exist!")
            error(field, "The phenotype file does not exist!")
        try:
            self.phenotype_table = pd.read_table(self.phenotype_path, delimiter='\s+',index_col=0)
            self.phenotypes_list = self.phenotype_table.columns.tolist()

            if self.phenotype_table.index.name in self.phenotypes_list:
                self.phenotypes_list.remove(self.phenotype_table.index.name)

            self.nodes_in_phenotype = self.phenotype_table.index.tolist()
            temp = self.phenotype_table[self.phenotypes_list]
            self.phentype_unique_values = list(set(itertools.chain(*temp.values.tolist())))

            if len(set(self.nodes_in_phenotype).intersection(self.nodes_in_tree)) ==0:
                    self.rootLogger.error( "No overlap between phenotype instances and the instances in the tree")
                    error(field, "No overlap between phenotype instances and the instances in the tree")
            else:
                self.rootLogger.info(F"{len(set(self.nodes_in_phenotype).intersection(self.nodes_in_tree))} instances in common between the tree (#{len(self.nodes_in_tree)})  and phenotype table (#{len(self.nodes_in_phenotype)})")
        except:
            self.rootLogger.error( "The phenotype table does not follow the correct format")
            error(field, "The phenotype table does not follow the correct format")
        return True

    def parse_table(self, load_path, datatype, delimiter=None):
        if datatype == 'text':
            feature_lines = [' '.join(l.split()[1::]) if not delimiter else ' '.join(l.split(delimiter)[1::]) for l in load_list(load_path)]
            instances = [l.split()[0] if not delimiter else l.split(delimiter)[0] for l in load_list(load_path)]

            vectorizer = TfidfVectorizer(use_idf=False, analyzer='word', ngram_range=(1, 1),norm='l1',stop_words=[], tokenizer=str.split, lowercase=True, binary=False)
            matrix_representation = vectorizer.fit_transform(feature_lines)
            feature_names = vectorizer.get_feature_names()

        if datatype == 'numerical':
            df = pd.read_table(load_path, delimiter='\s+' if not delimiter else delimiter, index_col=0)
            feature_names = df.columns.tolist()
            instances = df.index.tolist()
            matrix_representation = sparse.csr_matrix(df[feature_names].values)

            if not matrix_representation.shape[0]==len(set(instances)):
                feature_names = df.columns.tolist()[1::]
                instances = df[df.columns.tolist()[0]].tolist()
                matrix_representation = sparse.csr_matrix(df[feature_names].values)

        return instances, feature_names, matrix_representation.shape

    def check_tables(self, field, value, error):
        vtemp = Validator()

        for idx, table in enumerate(value):
            if 'table' in table:
                table_name = table['table']['name']
                self.feature_types.append(table_name)
                res = vtemp.validate(table, self.schema_table)
                if vtemp.errors:
                    self.rootLogger.error( F"in Table {idx+1} {str(vtemp.errors)}")
                    error(field, F"in Table {idx+1} {str(vtemp.errors)}")
                else:
                    if not ValidateGenML.exists(table['table']['path']):
                        self.rootLogger.error( F"The path {table['table']['path']} does not exist!")
                        error(field, F"The path {table['table']['path']} does not exist!")
                    else:
                        self.genotype_tables[table['table']['name']] = table['table']['path']

                instances, feature_names, matrix_representation_shape = self.parse_table(table['table']['path'], table['table']['datatype'], table['table']['delimiter'] if 'delimiter' in table['table'] else ' ')
                if self.nodes_in_phenotype:
                    self.rootLogger.info(F"{table_name} has {len(instances)} instances with {len(feature_names)} features that {len(set(instances).intersection(self.nodes_in_phenotype))} of them have phenotypes")

            if 'sequence' in table:
                table_name = table['sequence']['name']
                self.feature_types.append(table_name)
                res = vtemp.validate(table, self.schema_sequences)
                if vtemp.errors:
                    self.rootLogger.error( F"in Sequence Table {idx+1} {str(vtemp.errors)}")
                    error(field, F"in Sequence Table {idx+1} {str(vtemp.errors)}")
                else:
                    if not ValidateGenML.exists(table['sequence']['path']):
                        self.rootLogger.error( F"The directory {table['sequence']['path']} does not exist!")
                        error(field, F"The directory {table['sequence']['path']} does not exist!")
                    else:
                        self.fasta_files = ValidateGenML.recursive_glob(table['sequence']['path'],'*.fasta')
                        fasta_instances = ['.'.join(fasta.split('/')[-1].split('.')[0:-1]) for fasta in self.fasta_files]
                        self.rootLogger.info(F"{table['sequence']['path']} contains {len(fasta_instances)} fasta files that {len(set(fasta_instances).intersection(self.nodes_in_phenotype))} of them have phenotypes")
                        self.sequence_dir[table_name] = table['sequence']['path']
        return True


    def __load_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as exception:
                raise exception

    def __load_schema(self):

        self.schema_meta = {
                'project': {
                    'required': True,
                    'type': 'string'
                },
                'phylogenetic_tree': {
                    'required': True,
                    'type': 'string',
                    'validator': self.check_tree
                },
                'phenotype_table': {
                    'required': True,
                    'type': 'string',
                    'validator': self.check_phenotype
                },
                'output_directory': {
                    'required': True,
                    'type': 'string'
                },
                'number_of_cores': {
                    'required': True,
                    'type': 'number',
                    'min': 1,
                    'max': 20
                }
            }

        self.schema_prediction = {
                                'prediction': {
                                'required': True,
                                'type': 'dict',
                                'schema': {
                                    'name':
                                            {

                                                'required': True,
                                                'type': 'string'
                                            },
                                    'label_mapping':
                                           {
                                                'required': True,
                                                'type': 'dict'
                                            },
                                    'optimized_for':
                                           {
                                                'required': True,
                                                'minlength': 1,
                                                'allowed': ['accuracy',  'auc_score_macro',  'auc_score_micro',  'f1_macro',  'f1_micro',  'f1_neg',  'f1_pos',  'p_neg',  'p_pos',  'precision_macro',  'precision_micro',  'r_neg',  'r_pos',  'recall_macro',  'recall_micro'],
                                                'type': 'string'
                                            },
                                    'reporting':
                                           {
                                                'required': True,
                                                'minlength': 1,
                                                'allowed': ['accuracy',  'auc_score_macro',  'auc_score_micro',  'f1_macro',  'f1_micro',  'f1_neg',  'f1_pos',  'p_neg',  'p_pos',  'precision_macro',  'precision_micro',  'r_neg',  'r_pos',  'recall_macro',  'recall_micro'],
                                                'type': 'list'
                                            },
                                    'features':
                                           {
                                                'required': True,
                                                'minlength': 1,
                                                'type': 'list',
                                            },
                                    'classifiers':
                                           {
                                                'required': True,
                                                'minlength': 1,
                                                'type': 'list',
                                                'allowed': ['svm','lsvm','rf','lr'],
                                            }}}}



        self.schema_general = {
                'metadata': {
                    'required': True,
                    'type': 'dict',
                    'schema': self.schema_meta
                },
                'genotype_tables': {
                    'required': True,
                    'type': 'dict',
                    'schema': {
                        'tables': {
                                    'required': True,
                                    'minlength': 1,
                                    'type': 'list',
                                    'validator': self.check_tables,
                                }
                              }
                },
                'predictions': {
                                    'required': True,
                                    'minlength': 1,
                                    'type': 'list',
                                    'validator': self.check_prediction_block,
                                }

            }
        self.schema_table = {
                            'table': {
                                'required': True,
                                'type': 'dict',
                                'schema': {
                            'name': {
                                'required': True,
                                'type': 'string',
                                },
                            'path': {
                                'required': True,
                                'type': 'string',
                                },
                             'preprocessing': {
                                'required': True,
                                'type': 'string',
                                'allowed': ['l1','l2','percent','zero2one','none','std','binary']
                                },
                             'delimiter': {
                                'required': True,
                                'type': 'string',
                                },
                             'datatype': {
                                'required': False,
                                'type': 'string',
                                'allowed': ['numerical','text']
                                }
                                }
                            }
        }


        self.schema_sequences = {
                                'sequence': {
                                'required': True,
                                'type': 'dict',
                                'schema': {
                                'name': {
                                'required': True,
                                'type': 'string',
                                },
                                'path': {
                                    'required': True,
                                    'type': 'string',
                                    },
                                 'preprocessing': {
                                    'required': True,
                                    'type': 'string',
                                    'allowed': ['l1','l2','percent','zero2one','none','std','binary']
                                    },
                                 'k_value': {
                                    'required': True,
                                    'type': 'number',
                                    'allowed': [1,2,3,4,5,6,7,8]
                                    }
                                }}}


    def create_zip_file(self, project_name):

        if self.generate_zip:
            self.rootLogger.info('The config has been correctly formatted and the zip generation is in progress')

            config_str = str(self.config)

            zipObj = ZipFile(F'{project_name}', 'w')

            # genotype tables
            self.rootLogger.info('Adding genotype tables to the zip file')
            for name, path in tqdm.tqdm(self.genotype_tables.items()):
                zipObj.write(path, F"{name}.csv")
                config_str = config_str.replace(path, F"{name}.csv")

            # sequences
            self.rootLogger.info('Adding sequence files to the zip file')
            for fasta_file in  tqdm.tqdm(self.fasta_files):
                zipObj.write(fasta_file, F"sequences/{fasta_file.split('/')[-1]}" , zipfile.ZIP_DEFLATED )
                config_str = config_str.replace('/'.join(fasta_file.split('/')[0:-1]), "sequences/")

            # phenotypes
            self.rootLogger.info('Adding the phenotypes to the zip file')
            zipObj.write(self.phenotype_path, 'phenotype.csv')
            config_str = config_str.replace(self.phenotype_path, 'phenotype.csv')

            # trees
            self.rootLogger.info('Adding the tree to the zip file')
            zipObj.write(self.tree_path, 'tree.csv')
            config_str = config_str.replace(self.tree_path, 'tree.csv')

            # config
            self.rootLogger.info('Adding the config to the zip file')
            config = eval(config_str)
            timestamp = datetime.datetime.now()
            unix_time = str(time.mktime(timestamp.timetuple()))
            md5_digest = str(hashlib.md5((unix_time).encode('utf-8')).hexdigest())


            with open(md5_digest, 'w') as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False,explicit_start=False,version=False,sort_keys=True)
            zipObj.write(md5_digest, 'config.yml')
            os.remove(md5_digest)

            zipObj.close()
        else:
            self.rootLogger.error('Please carefully review the errors and the config file and fix the config and rerun the validator!')

    @staticmethod
    def recursive_glob(treeroot, pattern):
        '''
        :param treeroot: the path to the directory
        :param pattern:  the pattern of files
        :return:
        '''
        results = []
        for base, dirs, files in os.walk(treeroot):
            good_files = fnmatch.filter(files, pattern)
            results.extend(os.path.join(base, f) for f in good_files)
        return results

    def check_prediction_block(self, field, value, error):
        try:
            for idx, prediction in enumerate(value):
                vtemp = Validator()
                res = vtemp.validate(prediction, self.schema_prediction)
                if vtemp.errors:
                    self.rootLogger.error( F"in prediction block-{idx+1} {str(vtemp.errors)}")
                    error(field, F"in prediction block-{idx+1} {str(vtemp.errors)}")
                else:
                    self.rootLogger.info('Schema passed, entered the detailed checking of the prediction block')
                    # Label mapping
                    for k,v in prediction['prediction']['label_mapping'].items():
                        if k not in self.phentype_unique_values:
                            self.rootLogger.error(F"{k} in the label_mapping does not exist in the phenotype values {self.phentype_unique_values}")
                            error(field, F"{k} in the label_mapping does not exist in the phenotype values {self.phentype_unique_values}")
                        if v not in [0,1]:
                            self.rootLogger.error(F"Only the binary classification is supported please map labels to 0 or 1")
                            error(field, F"Only the binary classification is supported please map labels to 0 or 1")

                    # features
                    for feature in prediction['prediction']['features']:
                        for feature_type in feature['list']:
                            if feature_type not in self.feature_types:
                                self.rootLogger.error(F"Utilitized feature {feature_type} in {feature['feature']} does not exist in {self.feature_types}")
                                error(field, F"Utilitized feature {feature_type} in {feature['feature']} does not exist in {self.feature_types}")
                        # exisiting validation
                        if 'validation_tuning' in feature and 'use_validation_tuning' in feature:
                            self.rootLogger.error(F"You need to either provide validation_tuning or use_validation_tuning for feature set of {feature['feature']} and not both ")
                            error(field, F"You need to either provide validation_tuning or use_validation_tuning for feature set of {feature['feature']} and not both ")

                        if 'validation_tuning' not in feature and 'use_validation_tuning' not in feature:
                            self.rootLogger.error(F"You need to at least provide one of the validation_tuning or use_validation_tuning for feature set of {feature['feature']} ")
                            error(field, F"You need to at least provide one of the validation_tuning or use_validation_tuning for feature set of {feature['feature']} ")

                        if 'validation_tuning' in feature:
                            self.existing_validations.append(feature['validation_tuning']['name'])
                            if 'train' not in feature['validation_tuning']:
                                self.rootLogger.error(F"You need to define the training scheme for the feature set of {feature['feature']} ")
                                error(field, F"You need to define the training scheme for the feature set of {feature['feature']} ")
                            else:
                                if 'method' not in feature['validation_tuning']['train'] or feature['validation_tuning']['train']['method'] not in ['treebased','random']:
                                    self.rootLogger.error(F"The method for the training scheme of the feature set {feature['feature']} is not valid, has to be in {['treebased','random']}")
                                    error(field, F"The method for the training scheme of the feature set {feature['feature']} is not valid, has to be in {['treebased','random']}")

                                if 'method' not in feature['validation_tuning']['test'] or feature['validation_tuning']['test']['method'] not in ['treebased','random']:
                                    self.rootLogger.error(F"The method for the test scheme of the feature set {feature['feature']} is not valid, has to be in {['treebased','random']}")
                                    error(field, F"The method for the test scheme of the feature set {feature['feature']} is not valid, has to be in {['treebased','random']}")

                                if type(feature['validation_tuning']['test']['ratio']) == float:
                                    if feature['validation_tuning']['test']['ratio'] < 0 or feature['validation_tuning']['test']['ratio'] >1:
                                        self.rootLogger.error(F"The test ratio has to be a real number between 0 and 1.")
                                        error(field, F"The test ratio has to be a real number between 0 and 1.")
                                else:
                                    self.rootLogger.error(F"The test ratio has to be a real number between 0 and 1.")
                                    error(field, F"The test ratio has to be a real number between 0 and 1.")

                                if type(feature['validation_tuning']['train']['folds'])==int:
                                    if  feature['validation_tuning']['train']['folds'] < 1 or feature['validation_tuning']['train']['folds'] >20:
                                        self.rootLogger.error(F"The train folds must be an integer between 2 and 20")
                                        error(field, F"The train folds must be an integer between 2 and 20")
                                else:
                                    self.rootLogger.error(F"The train folds must be an integer between 2 and 20")
                                    error(field, F"The train folds must be an integer between 2 and 20")

                        if 'use_validation_tuning' in feature:
                            if feature['use_validation_tuning'] not in self.existing_validations:
                                self.rootLogger.error(F" The validation {feature['use_validation_tuning']} is not defined!")
                                error(field, F" The validation {feature['use_validation_tuning']} is not defined!")
        except:
            self.rootLogger.error(F" Error in the prediction block")
            error(field, F" Error in the prediction block")
        return True

def save_list(filename, list_names, overwrite=True, logger=None):
    if not ValidateGenML.exists(filename) or overwrite:
        f = codecs.open(filename, 'w', 'utf-8')
        for x in list_names:
            f.write(x + '\n')
        f.close()
        if logger:
            logger.info(F"file created: {filename} ")
    elif logger:
        logger.info(F"file existed and remained unchanged: {filename}")

def load_list(filename):
    return [line.rstrip() for line in codecs.open(filename, 'r', 'utf-8').readlines()]


import warnings
import argparse
import os
import logging
import sys
def checkArgs(args):
    '''
        This function checks the input arguments and returns the errors (if exist) otherwise reads the parameters
    '''
    # keep all errors
    err = "";
    # Using the argument parser in case of -h or wrong usage the correct argument usage
    # will be prompted
    parser = argparse.ArgumentParser()

    # parse #################################################################################################
    parser.add_argument('--config', action='store', dest='genyml_path', default=False, type=str,
                        help='Provide the path to the GenYML file (the pipeline config)')

    parser.add_argument('--log', action='store', dest='log_path', default=False, type=str,
                        help='Provide the path to generate the log file')

    parser.add_argument('--zip', action='store', dest='zip_path', default=False, type=str,
                        help='Provide the path and file name to generate the zip file')

    parsedArgs = parser.parse_args()

    if (not os.access(parsedArgs.genyml_path, os.F_OK)):
        err = err + "\nError: Permission denied or could not find the file"
        return err

    val = ValidateGenML(parsedArgs.genyml_path, parsedArgs.log_path)
    val.create_zip_file(parsedArgs.zip_path)

    return False



if __name__ == '__main__':
    err = checkArgs(sys.argv)
    if err:
        print(err)
        exit()
