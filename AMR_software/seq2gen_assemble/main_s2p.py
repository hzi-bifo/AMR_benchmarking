#!/usr/bin/python
import os
import numpy as np
from src.amr_utility import name_utility, file_utility, load_data
import argparse
import itertools
import pandas as pd
import subprocess
from pathlib import Path



def extract_info(path_sequence,temp_path,s,f_all,f_prepare_meta,f_tree,level,f_finished):


    fileDir = os.path.dirname(os.path.realpath('__file__'))
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    temp_path=temp_path+'log/temp/seg2geno/'
    file_utility.make_dir(temp_path)


    if f_prepare_meta:
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):
            amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/'+str(species.replace(" ", "_")))
            amr_utility.file_utility.make_dir('log/results/' + str(level) +'/'+ str(species.replace(" ", "_")))

            antibiotics, ID, Y =  load_data.extract_info(species, False, level)
            ALL=[]
            for anti in antibiotics:
                name,path,_,_,_,_,_=amr_utility.name_utility.s2g_GETname(level, species, anti)
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                if Path(fileDir).parts[1] == 'vol':
                    # path_list=np.genfromtxt(path, dtype="str")
                    name_list['path'] = '/vol/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                else:
                    name_list['path']= '/net/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                name_list['genome_id'] = 'iso_' + name_list['genome_id'].astype(str)

                pseudo=np.empty(len(name_list.index.to_list()),dtype='object')
                pseudo.fill(fileDir+'/example_sg_dataset/reads_subset/dna/CH2500.1.fastq.gz,'+fileDir+'/example_sg_dataset/reads_subset/dna/CH2500.2.fastq.gz')
                name_list['path_pseudo'] = pseudo
                ALL.append(name_list)
                # print(name_list)
            _, _, dna_list, assemble_list, yml_file, run_file_name,wd = amr_utility.name_utility.s2g_GETname(level, species, '')

            #combine the list for all antis
            species_dna=ALL[0]
            for i in ALL[1:]:
                species_dna = pd.merge(species_dna, i, how="outer", on=["genome_id",'path','path_pseudo'])# merge antibiotics within one species
            print(species_dna)
            species_dna_final=species_dna.loc[:,['genome_id','path']]
            species_dna_final.to_csv(assemble_list, sep="\t", index=False,header=False)
            species_pseudo = species_dna.loc[:, ['genome_id', 'path_pseudo']]
            species_pseudo.to_csv(dna_list, sep="\t", index=False, header=False)


            #prepare yml files based on a basic version at working directory

            # cmd_cp='cp seq2geno_inputs.yml %s' %wd
            # subprocess.run(cmd_cp, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            wd_results = wd + '/results'
            amr_utility.file_utility.make_dir(wd_results)

            fileDir = os.path.dirname(os.path.realpath('__file__'))
            wd_results = os.path.join(fileDir, wd_results)
            assemble_list=os.path.join(fileDir, assemble_list)
            dna_list=os.path.join(fileDir, dna_list)
            # #modify the yml file
            a_file = open("seq2geno_inputs.yml", "r")
            list_of_lines = a_file.readlines()
            list_of_lines[12] = "    100000\n"
            list_of_lines[14] = "    %s\n" % dna_list
            list_of_lines[26] = "    %s\n" % wd_results
            list_of_lines[28] = "    %s\n" % assemble_list


            a_file = open(yml_file, "w")
            a_file.writelines(list_of_lines)
            a_file.close()


            #prepare bash scripts
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                   str(species.replace(" ", "_")),
                                                                   20,'snakemake_env','all.q')
            cmd='seq2geno -f ./%s -l ./%s' % (yml_file,wd+'/'+str(species.replace(" ", "_"))+'log.txt')
            run_file.write(cmd)
            run_file.write("\n")


    if f_tree == True:

        #kma cluster: use the results in multi-species model.
        #phylo-tree: build a tree w.r.t. each species. use the roary results after the s2g is finished.
        #todo add names


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


    if f_finished:# delete large unnecessary tempt files(folder:spades)
        for species in df_species:
            _, _, dna_list, assemble_list, yml_file, run_file_name, wd = amr_utility.name_utility.s2g_GETname(level,
                                                                                                              species,
                                                                                                              '')
            # spades_cp = amr_utility.file_utility.get_full_d(wd)+'/results/denovo/spades'
            pan_cp = amr_utility.file_utility.get_full_d(wd) + '/results/denovo/roary/pan_genome_sequences'
            # as_cp = amr_utility.file_utility.get_full_d(wd) + '/results/RESULTS/assemblies/'
            cmd = 'rm -r %s' % (pan_cp)
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)




def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='The log file')
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
    parser.add_argument('-f_tree', '--f_tree', dest='f_tree', action='store_true',
                        help='Kma cluster')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence,parsedArgs.temp_path,parsedArgs.species,parsedArgs.f_all,parsedArgs.f_prepare_meta,parsedArgs.f_tree,parsedArgs.level, parsedArgs.f_finished)
