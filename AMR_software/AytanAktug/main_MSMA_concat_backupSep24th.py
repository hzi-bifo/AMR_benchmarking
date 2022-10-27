import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility, file_utility,load_data
from AMR_software.AytanAktug.data_preparation import ResFinder_PointFinder_concat,merge_scaffolds,scored_representation_blast,ResFinder_analyser_blast,merge_resfinder_pointfinder,merge_input_output_files,merge_resfinder
import argparse,json
import pandas as pd
import numpy as np
from AMR_software.AytanAktug.nn import nn_MSMA_concat
from AMR_software.AytanAktug.nn import nn_MSMA_discrete
import subprocess
from src.cv_folds import cluster2folds

def extract_info(path_sequence,list_species,selected_anti,level,f_cluster_folds,f_all,f_pre_meta,f_run_res,f_res,threshold_point,
                 min_cov_point,f_merge_mution_gene,f_matching_io,f_nn,f_nn_score,f_nn_all,f_nn_all_io,cv,i_CV,
                   epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_kma,n_jobs,temp_path):

    merge_name = []
    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        data = data.loc[list_species, :]
        data = data.loc[:, (data.sum() > 1)]
    data = data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa


    if f_pre_meta==True:
        'First run main_discrete_merge.py' \
        'This scripts only move existing metadata from discrete part'
        # prepare the ID list and phenotype list for all species.
        # the same as discrete merge.
        _,_,_,_,_,_,_,_,_,_,_,_,_,path_feature,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level, '',merge_name,temp_path,'MSMA_concat')
        file_utility.make_dir(path_feature)

        for species in list_species:
            print(species,': \t moving metadata from discrete part...')

            _,path_ID_multi_each_main,_,path_metadata_multi_each_main,_,_,_,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')

            cmd1= 'cp %s %s' % (path_metadata_multi_each_main, path_feature)
            subprocess.run(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            cmd2 = 'cp %s %s' % (path_ID_multi_each_main, path_feature)
            subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        _,_,_,_,path_ID_multi,path_metadata_multi,_,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,'', '',merge_name,temp_path,'MSMA_discrete')

        cmd3 = 'cp %s %s' % (path_metadata_multi, path_feature)
        subprocess.run(cmd3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        cmd4 = 'cp %s %s' % (path_ID_multi, path_feature)
        subprocess.run(cmd4, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    if f_cluster_folds:#prepare CV folds for leave-one-out concatenated MSMA model, based on clusters built from discrete databases MSMA model.
        for n_species in list_species:

            _,path_ID_multi_eachS,_,_,_,_,path_cluster,path_cluster_results_eachS,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,n_species, '',merge_name,temp_path,'MSMA_discrete')
            # temp='/vol/projects/khu/amr/benchmarking2_kma/log/temp/loose/multi_species/Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj/'+str(n_species.replace(" ", "_"))+"_clustered_90.txt"
            _, _, folders_sampleName = cluster2folds.prepare_folders(cv, 42, path_ID_multi_eachS,path_cluster_results_eachS, 'new')
            # _, _, folders_sampleName = cluster2folds.prepare_folders(cv, 42, path_ID_multi_eachS,temp, 'new')
            folds_txt=name_utility.GETname_foldsMSMA_concatLOO(n_species,level,f_kma,f_phylotree)
            file_utility.make_dir(os.path.dirname(folds_txt))
            with open(folds_txt, 'w') as f:
                json.dump(folders_sampleName, f)


    # 2. run resfinder
    if f_run_res==True:
        _,_,_,_,path_ID_multi,_,_,_,_,_,_,_,path_res_concat,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,'', merge_name,temp_path,'MSMA_concat')

        ID_strain=np.genfromtxt(path_ID_multi, dtype="str")

        # 'merge_species' corresponds to the concatenated database under db_pointfinder.
        ResFinder_PointFinder_concat.determination('merge_species',path_sequence,path_res_concat,ID_strain,threshold_point,min_cov_point,n_jobs)


    # =================================
        # 3. Analysing PointFinder results
        # Analysing ResFinder results
        # =================================
    if f_res==True or f_merge_mution_gene == True :
        _,_,_,_,path_ID_multi,\
           _,_,_,_,path_point_repre_results,\
           path_res_repre_results,path_mutation_gene_results,path_res_result,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name,merge_name,temp_path,'MSMA_concat')
        print(path_mutation_gene_results)
        scored_representation_blast.extract_info(path_res_result, path_ID_multi, path_point_repre_results,
                                                                              True, True)  # SNP,no zip
        print('1')
        ResFinder_analyser_blast.extract_info(path_res_result, path_ID_multi, path_res_repre_results,
                                                                           True)  # GPA, no zip
        print('2')
        merge_resfinder_pointfinder.extract_info(path_point_repre_results, path_res_repre_results,path_mutation_gene_results)
        print('3')

    if f_matching_io == True:
        _,_,_,_,_,_,_,_,_,_,_,path_mutation_gene_results,_,path_feature,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name,merge_name, temp_path,'MSMA_concat')
        count=0
        for species_testing in list_species:
            print('species_testing',species_testing)
            list_species_training=list_species[:count] + list_species[count+1 :]
            count+=1
            # print(list_species_training)
            #
            merge_name_train=[]
            for n in list_species_training:
                merge_name_train.append(n[0] + n.split(' ')[1][0])
            merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
            merge_name_test = species_testing.replace(" ", "_")



            _,path_ID_multi_train,path_feature_train,path_metadata_multi_train,_,\
               _,_,_,_,path_point_repre_results_train,\
               path_res_repre_results_train ,path_mutation_gene_results_train,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name_train, merge_name,temp_path,'MSMA_concat')


            _,path_ID_multi_test,path_feature_test,path_metadata_multi_test,_,\
               _,_,_,_,path_point_repre_results_test,\
               path_res_repre_results_test ,path_mutation_gene_results_test,_,_,_,_,_=\
                        name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name_test, merge_name,temp_path,'MSMA_concat')



            ##########
            #1. train
            # ------
            train_meta=[]
            for s in list_species_training:
                sub=pd.read_csv(path_feature+'/'+str(s.replace(" ", "_"))+'_meta.txt', sep="\t", index_col=0, header=0,dtype={'id': object})
                train_meta.append(sub)
            df_path_meta_train=train_meta[0]
            for i in train_meta[1:]:
                df_path_meta_train = pd.concat([df_path_meta_train, i], ignore_index=True, sort=False)

            print('checking anti order \n',df_path_meta_train)
            df_path_meta_train['id'].to_csv(path_ID_multi_train, sep="\t", index=False, header=False)# Note!!! cluster folders will index the name acoording to this ID list

            df_path_meta_train=df_path_meta_train.loc[:, np.insert(np.array(All_antibiotics,dtype='object'), 0, 'id')]
            df_path_meta_train=df_path_meta_train.fillna(-1)
            print('checking anti order \n', df_path_meta_train)
            df_path_meta_train.to_csv(path_metadata_multi_train,sep="\t", index=False, header=False)
            # df_path_meta_train.loc[:,All_antibiotics].to_csv(path_y_train, sep="\t", index=False, header=False)
            # df_path_meta_train['id'].to_csv(path_name_train, sep="\t", index=False, header=False)

            #########
            #2. test
            # -----
            id_test=np.genfromtxt(path_ID_multi_test, dtype="str")
            df_path_meta_test=pd.DataFrame(index=np.arange(len(id_test)),columns=np.insert(np.array(All_antibiotics, dtype='object'), 0, 'id'))#initialize with the right order of anti.
            df_path_meta_test_all=pd.read_csv(path_feature+'/'+str(species_testing.replace(" ", "_"))+'_meta.txt', sep="\t", index_col=0, header=0,dtype={'id': object}) #
            #  add all the antibiotic completely for testing dataset. And delete the antibiotis that are not included in this set of species combination.
            print('check anti order test')
            print(df_path_meta_test_all)
            df_path_meta_test.loc[:,'id']=df_path_meta_test_all.loc[:,'id']
            for anti in All_antibiotics:
                if anti in df_path_meta_test_all.columns:
                    df_path_meta_test.loc[:, anti] = df_path_meta_test_all.loc[:, anti]

            df_path_meta_test = df_path_meta_test.fillna(-1)
            print('check anti order test')
            print(df_path_meta_test)
            df_path_meta_test.to_csv(path_metadata_multi_test,sep="\t", index=False, header=False)# multi_log + merge_name_train + '_metaresfinder'





            #get train from whole
            id_train=np.genfromtxt(path_ID_multi_train, dtype="str")
            feature=np.genfromtxt(path_mutation_gene_results, dtype="str")
            n_feature = feature.shape[1] - 1  # number of features
            # df_feature=pd.DataFrame(feature, index=None, columns=np.insert(np.array(np.arange(n_feature),dtype='object'), 0, 'id'))
            # df_feature=df_feature.set_index('id')
            df_feature = pd.DataFrame(feature[:,1:], index=feature[:,0],
                                      columns=np.array(np.arange(n_feature), dtype='object'))

            df_feature_train = df_feature.loc[id_train,:]
            df_feature_test = df_feature.loc[id_test, :]
            df_feature_train.to_csv(path_mutation_gene_results_train,sep="\t", index=True, header=False)
            print('df_feature_train',df_feature_train)
            df_feature_test.to_csv(path_mutation_gene_results_test,sep="\t", index=True, header=False)
            #need to check. checked.
            print('df_feature_test',df_feature_test)
            # preparing x y

            merge_input_output_files.extract_info(path_ID_multi_train,path_mutation_gene_results_train,
                                                                               path_metadata_multi_train, path_feature_train)

            merge_input_output_files.extract_info(path_ID_multi_test,path_mutation_gene_results_test,
                                                                               path_metadata_multi_test, path_feature_test)


    if f_nn == True:

        # =================================
        # 4.  model.
        # Each time set a species out for testing, with the rest species' samples for training.
        # =================================
        count = 0
        for species_testing in list_species:


            print('the species to test: ', species_testing)
            list_species_training=list_species[:count] + list_species[count+1 :]

            count += 1
            if count-1 ==i_CV:#all
                # do CV on list_species_training using folds from discrete multi-species models, select the best estimator for testing on the standing out species


                #prepare folds for CV. This only include species_testing.
                folders_sample_name=[]
                for i in range(cv):
                    folders_sample_name_i=[]
                    for n_species in list_species_training:
                        folds_txt=name_utility.GETname_foldsMSMA_concatLOO(n_species,level,f_kma,f_phylotree)
                        folders_sample_name_each = json.load(open(folds_txt, "rb"))
                        folders_sample_name_i=folders_sample_name_i +folders_sample_name_each[i]
                    folders_sample_name.append(folders_sample_name_i)

                merge_name_train=[]
                for n in list_species_training:
                    merge_name_train.append(n[0] + n.split(' ')[1][0])
                merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
                merge_name_test = species_testing.replace(" ", "_")

                # prepare metadata for training, testing  no need(use the single species meta data)

                _,_,_,_,_,_,_,_,_,_,_,_,_,_,path_x_train, path_y_train, path_name_train=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name_train, merge_name,temp_path,'MSMA_concat')
                _,_,_,_,_,_,_,_,_,_,_,_,_,_,path_x_test, path_y_test, path_name_test=\
                    name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name_test, merge_name,temp_path,'MSMA_concat')

                save_name_score,save_name_weights, save_name_loss = name_utility.GETname_AAscoreMSMA_concat('AytanAktug',merge_name,merge_name_test,learning, epochs,
                         f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_concatLOO')#if learning=0.0, and epoch = 0, it means hyper parameter selection mode.

                file_utility.make_dir(os.path.dirname(save_name_score))
                file_utility.make_dir(os.path.dirname(save_name_weights))
                file_utility.make_dir(os.path.dirname(save_name_loss))

                # sys.stdout = open(save_name_loss, 'w')
                score=nn_MSMA_concat.concat_eval(merge_name_train, 'MSMA', level, path_x_train, path_y_train, path_name_train,
                                        folders_sample_name,path_x_test, path_y_test,
                                                  cv, f_scaler, f_fixed_threshold, f_nn_base, f_phylotree,f_kma,
                                                  f_optimize_score, save_name_weights)





                #folders_sample_name
                score['samples'] =np.genfromtxt(path_name_test, dtype="str").tolist()

                with open(save_name_score+ '_TEST.json', 'w') as f:
                    json.dump(score, f)

    if f_nn_all==True:
        '''Do a normal CV for all species, for a comparison with multi-discrete model.'''

        _,_,_,path_metadata_multi_all,path_ID_all,\
               path_metadata_multi,_,_,_,_,\
               _ ,path_mutation_gene_results_all,_,path_feature_all,path_x_all, path_y_all, path_name_all=\
                        name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name, merge_name,temp_path,'MSMA_concat')


        if f_nn_all_io:# prepare feature matrix, y_data,x_data,path_name.
            # replace nan with -1 in path_metadata_all
            df_metadata_all = pd.read_csv(path_metadata_multi, sep="\t",
                                          index_col=0, header=0, dtype={'id': object})  # multi_log + 'pheno.txt'
            print(df_metadata_all)
            df_metadata_all = df_metadata_all.loc[:, np.insert(np.array(All_antibiotics, dtype='object'), 0, 'id')]
            df_metadata_all = df_metadata_all.fillna(-1)
            df_metadata_all.to_csv(path_metadata_multi_all, sep="\t", index=False, header=False)
            merge_input_output_files.extract_info(path_ID_all,path_mutation_gene_results_all,path_metadata_multi_all, path_feature_all+'/'+str(merge_name))


        else:
            print('Start training..')

            save_name_score,save_name_weights, save_name_loss = name_utility.GETname_AAscoreMSMA('AytanAktug',merge_name,learning, epochs,
                     f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_concat_mixedS')#if learning=0.0, and epoch = 0, it means hyper parameter selection mode.

            file_utility.make_dir(os.path.dirname(save_name_score))
            file_utility.make_dir(os.path.dirname(save_name_weights))
            file_utility.make_dir(os.path.dirname(save_name_loss))

            if f_nn_score: #retrain and then test on the hold-out test set# after above finished for each of 5 folds.
                # sys.stdout = open(save_name_loss+"_test", 'w')
                nn_MSMA_discrete.multi_test(merge_name,'MSMA', level, path_x_all, path_y_all, path_name_all,cv, f_scaler, f_fixed_threshold,
                               f_nn_base, f_phylotree,f_kma,  f_optimize_score,save_name_weights, save_name_score)

            else:
                sys.stdout = open(save_name_loss+ str(i_CV), 'w')
                nn_MSMA_discrete.multi_eval(merge_name,'MSMA', level, path_x_all, path_y_all, path_name_all,[i_CV], f_scaler,
                            f_nn_base,  f_phylotree,f_kma,  f_optimize_score, save_name_weights,save_name_score)

if __name__== '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-path_sequence', '--path_sequence', type=str,required=False,
                        help='Path of the directory with PATRIC sequences.')
    parser.add_argument('-f_pre_meta', '--f_pre_meta', dest='f_pre_meta', action='store_true',
                        help=' prepare metadata for multi-species model.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_cluster_folds', '--f_cluster_folds', dest='f_cluster_folds', action='store_true',
                        help=' Generate KMA folds.')
    parser.add_argument('-f_run_res', '--f_run_res', dest='f_run_res', action='store_true',
                        help='Running Point/ResFinder tools.')
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum (breadth-of) coverage of Pointfinder. ')
    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')
    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene',
                        action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')

    # para for CV
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument('-f_nn_score', '--f_nn_scorer', dest='f_nn_score', action='store_true',
                        help='Wrap NN model scores from outer loops.')
    parser.add_argument('-f_nn_all', '--f_nn_all', dest='f_nn_all', action='store_true',
                        help='Do a nested CV for all species, for comparison with multi-discrete model.')
    parser.add_argument('-f_nn_all_io', '--f_nn_all_io', dest='f_nn_all_io', action='store_true',
                        help='Prepare for a nested CV for all species(for comparison with multi-discrete model).')
    parser.add_argument("-cv", "--cv_number", default=6, type=int,
                        help='CV splits number')
    parser.add_argument("-i_CV", "--i_CV", type=int,
                        help=' the number of CV iteration to run.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument("-e", "--epochs", default=0, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.0, type=float,
                        help='learning rate')
    parser.add_argument('-f_scaler', '--f_scaler', dest='f_scaler', action='store_true',
                        help='normalize the data')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument('-anti', '--anti', default=[], type=str, nargs='+', help='one antibioticseach time to run: e.g.\'ciprofloxacin\' \
	            \'gentamicin\' \'ofloxacin\' \'tetracycline\' \'trimethoprim\' \'imipenem\' \
	            \'meropenem\' \'amikacin\'...')#to be coming in the future..
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel for Resfinder tool.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                help='Directory to store temporary files.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.anti, parsedArgs.level, parsedArgs.f_cluster_folds,
                 parsedArgs.f_all,parsedArgs.f_pre_meta,parsedArgs.f_run_res,
                 parsedArgs.f_res,parsedArgs.threshold_point,parsedArgs.min_cov_point,
                 parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io,parsedArgs.f_nn,parsedArgs.f_nn_score,
                 parsedArgs.f_nn_all,parsedArgs.f_nn_all_io,parsedArgs.cv_number, parsedArgs.i_CV, parsedArgs.epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,
                 parsedArgs.f_optimize_score,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.n_jobs,parsedArgs.temp_path)


# future work todo: if anti can be selected, then some saving names should be reconsidered.
