#!/usr/bin/python

import amr_utility.name_utility
import argparse
import amr_utility.load_data
import subprocess
import pandas as pd


def extract_info(path_sequence,s,f_all,f_prepare_meta,f_tree,cv,level,n_jobs,f_finished,f_ml,f_phylotree,f_kma,f_qsub):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    if f_all == False:
        data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(data)

    for species in df_species:
        print(species)
        _, _, dna_list, assemble_list, yml_file, run_file_name, wd = amr_utility.name_utility.s2g_GETname(level,
                                                                                                          species,
                                                                                                          '')

        aln=wd+ '/results/denovo/roary/core_gene_alignment_renamed.aln'
        tree=wd+ '/results/denovo/roary/nj_tree.newick'
        aln=amr_utility.file_utility.get_full_d(aln)
        tree = amr_utility.file_utility.get_full_d(tree)
        cmd='Rscript --vanilla ./cv_folders/phylo_tree.r -f %s -o %s' %(aln,tree)
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # print(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_finished', '--f_finished', dest='f_finished', action='store_true',
                        help='delete large unnecessary tempt files')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_tree', '--f_tree', dest='f_tree', action='store_true',
                        help='Kma cluster')  # c program
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='ML')  # c program
    parser.add_argument('-f_qsub', '--f_qsub', dest='f_qsub',
                        help='Prepare scriptd for qsub.', action='store_true', )
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.f_all,parsedArgs.f_prepare_meta,parsedArgs.f_tree,
                 parsedArgs.cv,parsedArgs.level,parsedArgs.n_jobs,parsedArgs.f_finished,parsedArgs.f_ml,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_qsub)
