import os
import argparse
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path





def extract_info(path_sequence,s,f_all,f_prepare_meta,f_tree,cv,level,n_jobs,f_finished):

    # if path_sequence=='/net/projects/BIFO/patric_genome':
    #     path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/s2g2p'#todo, may need a change
    # else:
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    # path_large_temp = os.path.join(fileDir, 'large_temp')
    # # print(path_large_temp)
    #
    # amr_utility.file_utility.make_dir(path_large_temp)

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
        amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/'+str(species.replace(" ", "_")))
        amr_utility.file_utility.make_dir('log/results/' + str(level) +'/'+ str(species.replace(" ", "_")))


    if f_prepare_meta:
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            for outer in cv:
                # prepare features for each training and testing sets, to prevent information leakage



                ALL=[]
                for anti in antibiotics:
                    name,_=amr_utility.name_utility.Pts_GETname(level, species, anti)
                    name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")

                    if Path(fileDir).parts[1] == 'vol':
                        # path_list=np.genfromtxt(path, dtype="str")
                        name_list['Addresses'] = '/vol/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                    else:
                        name_list['Addresses']= '/net/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                    name_list['ID'] = 'iso_' + name_list['genome_id'].astype(str)

                    name_list.rename(columns={'resistant_phenotype': anti}, inplace=True)

                    ALL.append(name_list)
                    # print(name_list)
                _,meta_txt = amr_utility.name_utility.Pts_GETname(level, species, '')
                amr_utility.file_utility.make_dir(meta_txt)
                #combine the list for all antis
                species_dna=ALL[0]

                for i in ALL[1:]:
                    species_dna = pd.merge(species_dna, i, how="outer", on=["ID",'Addresses'])# merge antibiotics within one species
                print(species_dna)
                species_dna_final=species_dna.loc[:,['ID','Addresses']+antibiotics]
                species_dna_final.to_csv(meta_txt+'/'+str(outer)+'_data.pheno', sep="\t", index=False,header=True)



    if f_tree == True:

        pass


    if f_finished:# delete large unnecessary tempt files(folder:spades)
        pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')

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
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.f_all, parsedArgs.f_prepare_meta,
                 parsedArgs.f_tree, parsedArgs.cv, parsedArgs.level, parsedArgs.n_jobs, parsedArgs.f_finished)
