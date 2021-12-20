#!/usr/bin/python
import os
import numpy as np
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import itertools
import amr_utility.load_data
import pandas as pd
import subprocess
from pathlib import Path
import cv_folders.cluster_folders
import hyper_range
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
import pickle
from sklearn import svm,preprocessing


def extract_info(path_sequence,s,f_all,f_prepare_meta,f_tree,cv,level,n_jobs,f_finished,f_ml,f_phylotree,f_kma,f_qsub):

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
    if f_qsub:#prepare bash scripts for each species for ML
        for species, antibiotics in zip(df_species, antibiotics):
            amr_utility.file_utility.make_dir('log/qsub')
            run_file_name='log/qsub/'+str(species.replace(" ", "_"))+'_kmer.sh'
            amr_utility.file_utility.make_dir('log/qsub')
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_"))+'g2p',
                                                                    100, 'amr','uv2000.q')
            # run_file = amr_utility.file_utility.header_THREADS(run_file,
            #                                                    n_jobs)
            cmd = 'python main_s2p.py -f_ml --n_jobs %s -s \'%s\' -f_kma' % (100,species)
            run_file.write(cmd)
            run_file.write("\n")

            #------------------------------------------------------------
            run_file_name = 'log/qsub/' + str(species.replace(" ", "_")) + '_kmer2.sh'
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_")+'g2p'),
                                                                    20, 'amr', 'all.q')
            # run_file = amr_utility.file_utility.header_THREADS(run_file,
            #                                                     20)
            cmd = 'python main_s2p.py -f_ml --n_jobs %s -s \'%s\' -f_kma' % (20, species)
            run_file.write(cmd)
            run_file.write("\n")
            #------------------------------------------------------------
            run_file_name = 'log/qsub/' + str(species.replace(" ", "_")) + '_kmer3.sh'
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_")+'g2p'),
                                                                    20, 'amr', 'all.q')
            # run_file = amr_utility.file_utility.header_THREADS(run_file,
            #                                                     20)
            cmd = 'python main_s2p.py -f_ml --n_jobs %s -s \'%s\' -f_phylotree' % (20, species)
            run_file.write(cmd)
            run_file.write("\n")
            #3. random--------------------------------------
            run_file_name = 'log/qsub/' + str(species.replace(" ", "_")) + '_kmer4.sh'
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_")+'g2p'),
                                                                    20, 'amr', 'all.q')
            # run_file = amr_utility.file_utility.header_THREADS(run_file,
            #                                                     20)
            cmd = 'python main_s2p.py -f_ml --n_jobs %s -s \'%s\'' % (20, species)
            run_file.write(cmd)
            run_file.write("\n")





    if f_ml:

        for species, antibiotics in zip(df_species, antibiotics):
            # amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/' + str(species.replace(" ", "_")))
            amr_utility.file_utility.make_dir('log/results/' + str(level) + '/' + str(species.replace(" ", "_")))
            print(species)
            run_file = None

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            # antibiotics, ID, Y = antibiotics[1:], ID[1:], Y[1:]
            # antibiotics, ID, Y = antibiotics[8:], ID[8:], Y[8:]

            i_anti = 0
            for anti in antibiotics:


                id_all = ID[i_anti]  # sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
                y_all = Y[i_anti]
                i_anti+=1
                mer6_file = '/vol/projects/khu/amr/patric_Mar/log/feature/kmer/cano_' + str(
                    species.replace(" ", "_")) + '_6_mer.h5'

                s2g_file='./log/temp/loose/'+ str(
                            species.replace(" ", "_"))+'/results/RESULTS/bin_tables'
                data_feature1 = pd.read_csv(s2g_file+'/gpa.mat_NONRDNT', index_col=0,sep="\t")
                data_feature2 = pd.read_csv(s2g_file + '/indel.mat_NONRDNT',index_col=0,sep="\t")
                data_feature3 = pd.read_hdf(mer6_file)
                data_feature3 = data_feature3.T

                init_feature = np.zeros((len(id_all), 1), dtype='uint16')


                data_model_init = pd.DataFrame(init_feature, index=id_all, columns=['initializer'])
                X_all = pd.concat([data_model_init, data_feature3.reindex(data_model_init.index)], axis=1)
                #only scale the 6mer data
                scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X_all)
                X_all = pd.DataFrame(data=scaler.transform(X_all),
                               index=X_all.index,
                               columns=X_all.columns)
                X_all.index= 'iso_'+X_all.index
                id_all=['iso_'+ s  for s in id_all]
                X_all = pd.concat([X_all, data_feature1.reindex(X_all.index)], axis=1)
                X_all = pd.concat([X_all, data_feature2.reindex(X_all.index)], axis=1)#todo check
                X_all = X_all.drop(['initializer'], axis=1)


                id_all = np.array(id_all)
                y_all = np.array(y_all)
                for chosen_cl in ['svm','lr', 'rf','lsvm']:

                    hyper_space,cl=hyper_range.hyper_range(chosen_cl)
                    # 1. by each classifier.2. by outer loop. 3. by inner loop. 4. by each hyper-para


                    mcc_test = []  # MCC results for the test data
                    f1_test = []
                    score_report_test = []
                    aucs_test = []
                    hyperparameters_test = []
                    meta_original, meta_txt,save_name_score=amr_utility.name_utility.Pts_GETname(level, species, anti,chosen_cl)
                    for out_cv in range(cv):
                        print(anti,'. Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)
                        # 1. exrtact CV folders----------------------------------------------------------------
                        save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
                        Random_State = 42
                        p_clusters = amr_utility.name_utility.GETname_folder(species, anti, level)
                        if f_phylotree:  # phylo-tree based cv folders
                            folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, anti, p_names,
                                                                                                  False)
                        elif f_kma:  # kma cluster based cv folders
                            folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                                   p_clusters,
                                                                                                   'new')
                        else:#random
                            folders_index = cv_folders.cluster_folders.prepare_folders_random(cv, species, anti, p_names,
                                                                                                  False)


                        test_samples_index = folders_index[out_cv]# a list of index
                        # print(test_samples)
                        # print(id_all)
                        id_test = id_all[test_samples_index]#sample name list
                        y_test = y_all[test_samples_index]

                        train_val_train_index =folders_index[:out_cv] +folders_index[out_cv + 1:]
                        id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                        y_val_train = y_all[list(itertools.chain.from_iterable(train_val_train_index))]

                        X_val_train=X_all.loc[id_val_train,:]
                        X_test=X_all.loc[id_test,:]



                        pipe = Pipeline(steps=[('cl', cl)])

                        search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=n_jobs,
                                                  scoring='f1_macro',
                                                  cv=create_generator(train_val_train_index), refit=True)

                        search.fit(X_all, y_all)
                        hyperparameters_test_sub=search.best_estimator_
                        current_pipe=hyperparameters_test_sub
                        # -------------------------------------------------------
                        # retrain on train and val
                        current_pipe.fit(X_val_train, y_val_train)
                        y_test_pred = current_pipe.predict(X_test)
                        # scores
                        f1 = f1_score(y_test, y_test_pred, average='macro')
                        report = classification_report(y_test, y_test_pred, labels=[0, 1], output_dict=True)
                        mcc = matthews_corrcoef(y_test, y_test_pred)
                        fpr, tpr, _ = roc_curve(y_test, y_test_pred, pos_label=1)
                        # tprs.append(interp(mean_fpr, fpr, tpr))
                        # tprs[-1][0] = 0.0
                        roc_auc = auc(fpr, tpr)

                        f1_test.append(f1)
                        score_report_test.append(report)
                        aucs_test.append(roc_auc)
                        mcc_test.append(mcc)
                        hyperparameters_test.append(hyperparameters_test_sub)
                    score = [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]
                    with open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',
                              'wb') as f:  # overwrite mode
                        pickle.dump(score, f)



def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test

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
