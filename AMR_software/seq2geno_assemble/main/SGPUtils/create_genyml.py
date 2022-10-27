#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import yaml
import os
import sys
from pprint import pprint
class Items(list):
    pass

def items_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def make_parser():
    arg_formatter = lambda prog: argparse.RawTextHelpFormatter(prog,
            max_help_position=4, width = 80)

    parser = argparse.ArgumentParser(
            formatter_class= arg_formatter,
            description='create yaml file from Seq2Geno output for Geno2Pheno')

    parser.add_argument('-v', action= 'version', 
        version='v.Beta')
    parser.add_argument('--seq2geno', dest= 'sg', 
            help= 'seq2geno project folder', required= True)
    parser.add_argument('--yaml', dest= 'yaml', 
            help= 'the yaml file to be generated', required= True)
    parser.add_argument('--proj', dest= 'proj', 
            help= 'project name', default= 'sgp')

    ## prediction block
    pred_args=  parser.add_argument_group('predict')
    pred_args.add_argument('--report', dest= 'report', default= ['f1_macro'], 
            nargs= '+',
            help= 'the metrics to include in the report', 
            choices=['accuracy',  'auc_score_macro',  'auc_score_micro',  
                     'f1_macro',  'f1_micro',  'f1_neg',  
                     'f1_pos',  'p_neg',  'p_pos',  
                     'precision_macro',  'precision_micro',  'r_neg',  
                     'r_pos',  'recall_macro',  'recall_micro'])
    pred_args.add_argument('--opt', dest= 'optimize', default= 'f1_macro', 
            nargs= 1, 
            help= 'target performance metric to optimize', 
            choices=['accuracy',  'auc_score_macro',  'auc_score_micro',  
                     'f1_macro',  'f1_micro',  'f1_neg',  
                     'f1_pos',  'p_neg',  'p_pos',  
                     'precision_macro',  'precision_micro',  'r_neg',  
                     'r_pos',  'recall_macro',  'recall_micro'])
    pred_args.add_argument('--fold_n', dest= 'fold_n', 
            help= 'number of folds during validation', 
            default= 10)
    pred_args.add_argument('--test_ratio', dest= 'test_ratio', 
            help= 'proportion of samples for testing', 
            default= 0.1)
    pred_args.add_argument('--part', dest= 'part', default= 'treebased',
            nargs= 1, 
            help= 'method to partition dataset', choices= ['treebased','random'])
    pred_args.add_argument('--models', dest= 'models', nargs= '*',  
            default= ['svm'],
            help= 'machine learning algorithms', 
            choices= ['svm', 'rf','lsvm',  'lr'])
    pred_args.add_argument('--k-mer', dest= 'kmer',type= int, 
            help= 'the k-mer size for prediction', 
            default= 6)
    pred_args.add_argument('--cls', dest= 'classes_f', 
            help= 'a two-column file to specify labels and prediction groups')
    pred_args.add_argument('--cpu', dest= 'cpu', type= int,default= 1,
            help= 'number of cpus for parallel computation')
    return(parser)

def parse_usr_opts():
    parser= make_parser()
    args = parser.parse_args()
    return(args)

def make_genyml(args):

    #' ensure the seq2geno file exists
    results_dir= os.path.join(args.sg, 'RESULTS')
    try:
        assert os.path.isdir(results_dir)
    except AssertionError:
        print('{} not found'.format(results_dir))
        exit()

    blocks= dict()
    ####
    ## metadata block
    ####
    blocks['metadata']=dict(
        project=args.proj,
        phylogenetic_tree= os.path.join(args.sg, 'RESULTS',
                                        'phylogeny',
                                        'tree.nwk'),
        phenotype_table= os.path.join(args.sg, 'RESULTS',
                                      'phenotype',
                                      'phenotypes.mat'),
        output_directory= 'results/', 
        number_of_cores=args.cpu
    )
    ####
    ## genotype tables
    ####
    poss_tabs= [
        dict(sequence= dict(
            name= '{}mer'.format(str(args.kmer)),  
            path= os.path.join(args.sg,'RESULTS','assemblies'), 
            preprocessing= 'l1', 
            k_value= args.kmer)),
        dict(table= dict(
            name= 'gpa',
            delimiter='\t', 
            path= os.path.join(args.sg,'RESULTS','bin_tables/gpa.mat_NONRDNT'), 
            preprocessing= 'binary', 
            datatype= 'numerical')),
        dict(table= dict(
            name= 'indel',
            delimiter='\t', 
            path= os.path.join(args.sg,'RESULTS','bin_tables/indel.mat_NONRDNT'), 
            preprocessing= 'binary', 
            datatype= 'numerical')),
        dict(table= dict(
            name= 'snp', 
            delimiter='\t', 
            path= os.path.join(args.sg,'RESULTS','bin_tables/nonsyn_SNPs_final.bin.mat_NONRDNT'), 
            preprocessing= 'binary', 
            datatype= 'numerical')),
        dict(table= dict(
            name= 'expr', 
            delimiter='\t', 
            path= os.path.join(args.sg, 'RESULTS', 'num_tables/expr.log.mat'), 
            preprocessing= 'none', 
            datatype= 'numerical'))]
    blocks['genotype_tables']=dict(tables= [])
    for t in poss_tabs:
        if 'table' in t:
            if os.path.isfile(t['table']['path']):
                blocks['genotype_tables']['tables'].append(t)
        elif 'sequence' in t:
            if os.path.isdir(t['sequence']['path']):
                blocks['genotype_tables']['tables'].append(t)

    ####
    ## prediction block
    ####
    def parse_classes(classes_f):
        classses_dict= {}
        if args.classes_f is None:
            classses_dict= {1: 1, 0: 0}
        else:
            with open(classes_f, 'r') as cls_fh:
                for l in cls_fh:
                    d=l.strip().split('\t')
                    classses_dict[str(d[0])]= d[1]
        return(classses_dict)

    available_feat_names= [t['table']['name'] if 'table' in t else t['sequence']['name'] 
                           for t in blocks['genotype_tables']['tables']]
    blocks['predictions']=[dict(
        prediction= dict(
            name= args.sg.strip('/').split('/')[-1],
            label_mapping= parse_classes(args.classes_f),
            optimized_for= str(args.optimize),
            reporting= Items(args.report),
            features= [
                dict(feature= "seq2geno_feats",
                     list= Items(available_feat_names),
                     validation_tuning= dict(
                         name= 'cv_tree', 
                         train= {'method':args.part,'folds': args.fold_n},
                         test={'method':args.part,'ratio': args.test_ratio}, 
                         inner_cv= 10))],
            classifiers= args.models
        ))]
    yaml_f= args.yaml
    with open(yaml_f, 'w') as outfile:
        yaml.representer.SafeRepresenter.add_representer(Items, items_representer)
        yaml.safe_dump(blocks, outfile, default_flow_style=False,
                       sort_keys=False)

if __name__== '__main__':
    args= parse_usr_opts()
    make_genyml(args)

