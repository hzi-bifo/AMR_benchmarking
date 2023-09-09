#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pandas as pd



# ======================
# Names for general use
# ======================
def GETname_main_meta(level):
    main_meta='./data/PATRIC/meta/'+str(level)+'_Species_antibiotic_FineQuality.csv'
    main_multi_meta='./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv'
    return main_meta,main_multi_meta

def GETname_meta(species,anti,level):
    ID='./data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    ## pheno='./data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
    ##     anti.translate(str.maketrans({'/': '_', ' ': '_'}))) +'_pheno.txt'
    return ID

def GETname_folds(species,anti,level,f_kma,f_phylotree):
    if f_kma:
        f_folds='_KMA_cv.json'
    elif f_phylotree==True:
        f_folds='_phylotree_cv.json'
    else:
        f_folds='_random_cv.json'

    folds_folder='./data/PATRIC/cv_folds/'+str(level)+'/single_S_A_folds/'+str(species.replace(" ", "_"))+'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+f_folds
    return folds_folder


def GETname_model(software, level,species, anti,cl,temp_path):
    '''usage: s2g, resfinder_folds'''
    meta = './data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
        anti.translate(str.maketrans({'/': '_', ' ': '_'}))) +'_pheno.txt'
    meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/'+   str(species.replace(" ", "_"))  + '/' + \
        str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/'+ str(species.replace(" ", "_"))  + '/' + \
    str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl)
    ## save_name_score='Results/' + str(software) + '/' + str(species.replace(" ", "_"))  + '/' + \
    ## str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl)

    return meta, meta_temp,save_name_score_temp

def GETname_model2(software, level,species, anti,cl,temp_path,f_kma,f_phylotree):
    '''usage: kover, Phenotypeseeker'''
    meta = './data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
        anti.translate(str.maketrans({'/': '_', ' ': '_'}))) +'_pheno.txt'
    if f_phylotree:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/phylotree/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
        # save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/phylotree/'+ str(species.replace(" ", "_"))  + '/' + \
        #     str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl)
        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/phylotree/'+ str(species.replace(" ", "_")) +'/anti_list'
    elif f_kma:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/kma/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
        # save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/kma/'+ str(species.replace(" ", "_"))  + '/' + \
        #     str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl)
        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/kma/'+ str(species.replace(" ", "_")) +'/anti_list'
    else:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/random/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/random/'+ str(species.replace(" ", "_")) +'/anti_list'

    save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/'+ str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl) #the same as GETname_model, which is important as used in result_analysis.py(pts)

    return anti_list,meta, meta_temp,save_name_score_temp

def GETname_model2_val(software, level,species, anti,cl,temp_path,f_kma,f_phylotree):
    '''usage: kover validation.'''
    meta = './data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
        anti.translate(str.maketrans({'/': '_', ' ': '_'}))) +'_pheno.txt'
    if f_phylotree:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output_val/phylotree/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
        # save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/phylotree/'+ str(species.replace(" ", "_"))  + '/' + \
        #     str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl)
        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output_val/phylotree/'+ str(species.replace(" ", "_")) +'/anti_list'
    elif f_kma:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output_val/kma/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
        # save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/kma/'+ str(species.replace(" ", "_"))  + '/' + \
        #     str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl)
        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output_val/kma/'+ str(species.replace(" ", "_")) +'/anti_list'
    else:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output_val/random/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output_val/random/'+ str(species.replace(" ", "_")) +'/anti_list'

    save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/'+ str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl) #the same as GETname_model, which is important as used in result_analysis.py

    return anti_list,meta, meta_temp,save_name_score_temp


def GETname_model3(software, level,species,anti,cl,temp_path):
    '''usage: kover, phenotypeseeker multi-species single-antibiotic model '''

    meta = './data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
        anti.translate(str.maketrans({'/': '_', ' ': '_'}))) +'_pheno.txt'
    meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/MS/'+   str(species.replace(" ", "_"))  + '/' + \
        str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/MS/'+ str(species.replace(" ", "_")) +'/anti_list'

    save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/MS/'+ str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl) #only for phenotypeseeker.
    return anti_list,meta, meta_temp,save_name_score_temp

def GETname_model4(software,base_software, level,species, anti,cl,temp_path,f_kma,f_phylotree):
    '''usage: ensemble base methods'''
    meta = './data/PATRIC/meta/'+str(level)+'_by_species/Data_' + str(species.replace(" ", "_")) + '_' + str(\
        anti.translate(str.maketrans({'/': '_', ' ': '_'}))) +'_pheno.txt'
    if f_phylotree:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/phylotree/'+ base_software +'/'+ str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/phylotree/'+ base_software +'/'+  str(species.replace(" ", "_")) +'/anti_list'
    elif f_kma:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/kma/'+ base_software +'/'+   str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/kma/'+ base_software +'/'+  str(species.replace(" ", "_")) +'/anti_list'
    else:
        meta_temp = str(temp_path)+'log/software/' +  str(software) +'/software_output/random/'+ base_software +'/'+    str(species.replace(" ", "_"))  + '/' + \
            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
        anti_list=str(temp_path)+'log/software/'+str(software) +'/software_output/random/'+ base_software +'/'+  str(species.replace(" ", "_")) +'/anti_list'

    # save_name_score_temp=str(temp_path)+'log/software/'+str(software) +'/analysis/'+ str(species.replace(" ", "_"))  + '/' + \
    #         str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_cl_'+str(cl) #the same as GETname_model, which is important as used in result_analysis.py

    return anti_list,meta, meta_temp




def GETname_result(software,species,fscore,f_kma,f_phylotree,chosen_cl,output_path):
    '''
    resfinder_folds , majority, AA results will not be stored by fscore (selection criteria).
    Only Kover, S2G2P,Phenotypeseeker will be stored by fscore (folder names) due to multiple classifiers.
    Aug 2023 update:  Kover, S2G2P,Phenotypeseeker  also don't need special " fscore " folder.
    '''

    save_name_score=output_path+'Results/software/'+str(software) +'/' + str(species.replace(" ", "_")) +'/' + str(species.replace(" ", "_"))+ \
                    '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree)+'_'+chosen_cl
    # if software in ['seq2geno','phenotypeseeker', 'kover'] and "clinical_" in fscore:
    #     save_name_final = output_path+'Results/software/'+str(software) +'/' + str(species.replace(" ", "_")) +'/' +fscore+ '/' +  str(species.replace(" ", "_"))+\
    #                 '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree)
    # else:
    save_name_final = output_path+'Results/software/'+str(software) +'/' + str(species.replace(" ", "_")) +'/' +  str(species.replace(" ", "_"))+\
                    '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree)

    return save_name_score,save_name_final

def GETname_result2(software,species,fscore,chosen_cl,output_path):
    '''
     multi-species LOSO kover, PhenotypeSeeker
    '''
    save_name_score=output_path+'Results/software/'+str(software) +'/MS/' + str(species.replace(" ", "_")) +\
                    '/' + str(species.replace(" ", "_"))+  '_'+chosen_cl + '_'+fscore
    return save_name_score





def GETname_AAresult(software,species,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,version,output_path):

    save_name_final = output_path+'Results/software/'+str(software) +'/'+ version+'/'+ str(species.replace(" ", "_")) + \
        '/'+  'lr_' + str(learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+\
                      '_fixT_'+str(f_fixed_threshold) +'_kma_' + str(f_kma) + '_tree_' + str(f_phylotree)
    return save_name_final

# ======================
# Names for data preprocess
# ======================
def GETname_quality(species,level):
    save_all_quality='./data/PATRIC/quality/' + str(species.replace(" ", "_"))+".csv"
    save_quality="./data/PATRIC/meta/fine_quality/GenomeFineQuality_" +str(level)+'_'+ str(species.replace(" ", "_")) + '.txt'

    return save_all_quality,save_quality
def load_metadata(SpeciesFile):#for metadata.py
    '''
    :param SpeciesFile: species list
    :return: Metadata for each strain, which belongs to the species in the parameter file
    '''
    data = pd.read_csv('./data/PATRIC/PATRIC_genomes_AMR.txt', dtype={'genome_id': object, 'genome_name': object}, sep="\t")
    data['genome_name'] = data['genome_name'].astype(str)
    data['species'] = data.genome_name.apply(lambda x: ' '.join(x.split(' ')[0:2]))
    data = data.loc[:, ("genome_id", 'species', 'antibiotic', 'resistant_phenotype')]
    df_species = pd.read_csv(SpeciesFile, dtype={'genome_id': object}, sep="\t", header=0)
    info_species = df_species['species'].tolist()  # 10 species that should be modelled!
    data = data.loc[data['species'].isin(info_species)]
    data = data.dropna()
    data = data.reset_index(drop=True)  # new index now
    return data, info_species

# ======================
# Names for Resfinder
# ======================

def GETname_ResfinderResults(species,version,output_path):
    '''version: resfinder_k, resfinder_b'''
    save_name_score = output_path+'Results/software/'+ version + '/'  + str(species.replace(" ", "_"))

    return save_name_score

# ======================
# Names for Seq2Geno
# ======================

def GETname_S2Gfeature(species,temp_path,k):

    if '.' in temp_path:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        temp_path=temp_path.replace(".",fileDir)

    log_feature=str(temp_path)+'log/software/seq2geno/software_output/' + str(species.replace(" ", "_"))+'/'
    dna_list=log_feature+'dna_list'
    assemble_list=log_feature+'assemble_list'
    yml_file=log_feature+'seq2geno_inputs.yml'
    run_file=log_feature+'run.sh'
    save_name_kmer=log_feature +  'cano_'+ str(species.replace(" ", "_")) +'_'+str(k) + '_mer.h5'


    #only for the sake of original pipeline checking procedure. Not used actually in our benchmarking pipeline.
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    phe_table= fileDir+'/AMR_software/seq2geno/examples/mem_phenotypes.mat'
    ref_fa= fileDir+'/AMR_software/seq2geno/examples/reference/Pseudomonas_aeruginosa_PA14.edit.fasta.sub'
    ref_gff=fileDir+ '/AMR_software/seq2geno/examples/reference/RefCln_UCBPP-PA14.edit.gff.sub.coor_shift'
    ref_gbk= fileDir+'/AMR_software/seq2geno/examples/reference/Pseudomonas_aeruginosa_PA14_ncRNA.edit.utf-8.gbk.sub'
    rna_reads= fileDir+'/AMR_software/seq2geno/examples/rna_list'
    pseudo=[phe_table,ref_fa,ref_gff,ref_gbk,rna_reads]



    return save_name_kmer,dna_list,assemble_list,yml_file,run_file,log_feature,pseudo




# ======================
# Names for Aytan-Aktug
# ======================
def GETname_AAfeatureSSSA(software,level,species, anti,temp_path):
    '''Single-species-antibiotic model'''
    save_name_ID=GETname_meta(species,anti,level)

    save_name_meta=save_name_ID+'_pheno.txt'
    path_feature = temp_path+'log/software/' + str(software)+'/software_output/SSSA/'+str(species.replace(" ", "_")) # all feature temp data(except large files)
    save_name_anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    anti_list = path_feature + '/anti_list'

    path_cluster = path_feature+'/cluster/temp/' + save_name_anti + '_all_strains_assembly.txt'
    path_cluster_results=path_feature+'/cluster/'+save_name_anti+'_clustered_90.txt'
    path_cluster_temp=path_feature+'/cluster/temp/clustered_90_'+save_name_anti

    path_metadata_pheno=path_feature+'/'+save_name_anti+'_pheno.txt' #without header
    path_point_repre_results=path_feature+'/'+save_name_anti+'_mutations.txt'
    path_res_repre_results = path_feature + '/' + save_name_anti + '_acquired_genes.txt'
    path_mutation_gene_results=path_feature + '/' + save_name_anti + '_res_point.txt'
    path_x_y = path_feature + '/' + save_name_anti
    path_x = path_x_y+'data_x.txt'
    path_y = path_x_y+'data_y.txt'
    path_name = path_x_y + 'data_names.txt'
    if species =='Neisseria gonorrhoeae':
        path_res_result=temp_path+ 'log/software/resfinder_b/software_output/'+ str(species.replace(" ", "_"))
    else:
        path_res_result=temp_path+ 'log/software/resfinder_k/software_output/'+ str(species.replace(" ", "_"))
    return save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,path_res_result,path_point_repre_results,\
           path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name

def GETname_AAscoreSSSA(software,species, antibiotics,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,f_kma,f_phylotree):
    ''' For saving CV scores. Single-species-antibiotic model'''

    save_name_score_temp=temp_path+'log/software/'+str(software) +'/analysis/SSSA/'+ str(species.replace(" ", "_"))  + '/' + \
    str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)
    save_name_weights=temp_path+'log/software/'+str(software) +'/analysis/SSSA/'+ str(species.replace(" ", "_"))  + '/weights/' + \
    str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    save_name_loss=temp_path+'log/software/'+str(software) +'/analysis/SSSA/'+ str(species.replace(" ", "_"))  + '/loss/' + \
    str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    #------------------------------------------------------------------------------------------------------------------------------------------
    save_name_score_temp=save_name_score_temp+'_all_score_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json'
    save_name_weights=save_name_weights+ '_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)
    save_name_loss=save_name_loss+'_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)+'_traininglog.txt'


    return save_name_score_temp,save_name_weights, save_name_loss

def GETname_AAfeatureSSMA(software,level,species, anti,temp_path):
    '''Single-species multi-antibiotics model'''
    save_name_ID=GETname_meta(species,anti,level)
    save_name_meta=save_name_ID+'_pheno.txt'
    path_feature = temp_path+'log/software/' + str(software)+'/software_output/SSMA/'+str(species.replace(" ", "_")) # all feature temp data(except large files)
    path_ID_multi = path_feature + '/ID'
    path_metadata_multi_temp=path_feature  + '/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_pheno.txt' #without header
    path_metadata_multi = path_feature  + '/meta.txt' #with header

    path_cluster = path_feature+'/cluster/temp/all_strains_assembly.txt'
    path_cluster_results=path_feature+'/cluster/clustered_90.txt'
    path_cluster_temp=path_feature+'/cluster/temp/clustered_90'

    path_point_repre_results=path_feature+'/mutations.txt'
    path_res_repre_results = path_feature + '/acquired_genes.txt'
    path_mutation_gene_results=path_feature + '/res_point.txt'
    if species =='Neisseria gonorrhoeae':
        path_res_result=temp_path+ 'log/software/resfinder_b/software_output/'+ str(species.replace(" ", "_"))
    else:
        path_res_result=temp_path+ 'log/software/resfinder_k/software_output/'+ str(species.replace(" ", "_"))
    path_x = path_feature + '/data_x.txt'
    path_y = path_feature + '/data_y.txt'
    path_name = path_feature + '/data_names.txt'

    return  save_name_meta,path_metadata_multi_temp,path_ID_multi,path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,\
            path_res_result,path_point_repre_results,path_res_repre_results,path_mutation_gene_results,path_x ,path_y,path_name

def GETname_foldsSSMA(species,level,f_kma,f_phylotree):
    if f_kma:
        f_folds='_KMA_cv.json'
    elif f_phylotree==True:
        f_folds='_phylotree_cv.json'
    else:
        f_folds='_random_cv.json'

    folds_folder='./data/PATRIC/cv_folds/'+str(level)+'/single_S_multi_A_folds/'+str(species.replace(" ", "_"))+ f_folds
    return folds_folder




def GETname_AAscoreSSMA(software,species,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,f_kma,f_phylotree):
    ''' For saving CV scores. Single-species-antibiotic model'''

    save_name_score_temp=temp_path+'log/software/'+str(software) +'/analysis/SSMA/'+ str(species.replace(" ", "_"))  + '/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    save_name_weights=temp_path+'log/software/'+str(software) +'/analysis/SSMA/'+ str(species.replace(" ", "_"))  + '/weights/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    save_name_loss=temp_path+'log/software/'+str(software) +'/analysis/SSMA/'+ str(species.replace(" ", "_"))  + '/loss/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)


    save_name_score_temp=save_name_score_temp+'_all_score_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree)
    save_name_weights=save_name_weights+ '_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)
    save_name_loss=save_name_loss+'_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)+'_traininglog.txt'


    return save_name_score_temp,save_name_weights, save_name_loss

def GETname_AAfeatureMSMA(software,level,species,anti,merge_name,temp_path,version):
    '''
    Usage:
    MSMA discrete model
    MSMA concat mixed-dspecies model
     '''
    save_name_ID=GETname_meta(species,anti,level)
    save_name_meta=save_name_ID+'_pheno.txt'

    path_feature_eachS = temp_path+'log/software/' + str(software)+'/software_output/'+str(version)+'/'+str(merge_name)+'/'+str(species.replace(" ", "_"))
    path_feature = temp_path+'log/software/' + str(software)+'/software_output/'+str(version)+'/'+str(merge_name)

    path_metadata_multi_each=path_feature_eachS+'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_meta.txt'
    path_metadata_multi_each_main=path_feature_eachS+'_meta.txt' #with header
    path_ID_multi_each= path_feature_eachS+'_id'

    path_point_repre_results_each=path_feature_eachS+'/mutations.txt'
    path_res_repre_results_each = path_feature_eachS + '/acquired_genes.txt'
    path_mutation_gene_results_each=path_feature_eachS + '/res_point.txt'

    if version=="MSMA_concat": # usage:  concat mixed-species model.
        path_res_result=temp_path+ 'log/software/'+ str(software)+'/software_output/'+str(version) +'/resfinder_outputs'
    else:
        if species =='Neisseria gonorrhoeae':
            path_res_result=temp_path+ 'log/software/resfinder_b/software_output/'+ str(species.replace(" ", "_"))
        else:
            path_res_result=temp_path+ 'log/software/resfinder_k/software_output/'+ str(species.replace(" ", "_"))

    path_x = path_feature + '/data_x.txt'
    path_y = path_feature + '/data_y.txt'
    path_name = path_feature + '/data_names.txt'


    # -------------------------------
    path_ID_multi = path_feature + '/ID'
    path_metadata_multi = path_feature  + '/meta.txt'

    path_cluster = path_feature_eachS+'/cluster/temp/all_strains_assembly.txt'
    path_cluster_results=path_feature_eachS+'/cluster/clustered_90.txt'
    path_cluster_temp=path_feature_eachS+'/cluster/temp/clustered_90'

    return save_name_meta,path_ID_multi_each,path_metadata_multi_each,path_metadata_multi_each_main,path_ID_multi,\
           path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_point_repre_results_each,\
           path_res_repre_results_each,path_mutation_gene_results_each,path_res_result,path_feature,path_x,path_y,path_name




def GETname_foldsMSMA(merge_name,level,f_kma,f_phylotree):
    '''
    Usage:
    MSMA discrete model
    MSMA concat mixed-dspecies model
    '''
    if f_kma:
        f_folds='_KMA_cv.json'
    elif f_phylotree==True:
        f_folds='_phylotree_cv.json'
    else:
        f_folds='_random_cv.json'

    folds_folder='./data/PATRIC/cv_folds/'+str(level)+'/multi_S_folds/'+str(merge_name)+f_folds
    return folds_folder
def GETname_foldsMSMA_concatLOO(species,level,f_kma,f_phylotree):
    if f_kma:
        f_folds='_KMA_cv.json'
    elif f_phylotree==True:
        f_folds='_phylotree_cv.json'
    else:
        f_folds='_random_cv.json'

    folds_folder='./data/PATRIC/cv_folds/'+str(level)+'/multi_S_LOO_folds/'+str(species.replace(" ", "_"))+ f_folds
    return folds_folder


def GETname_AAscoreMSMA(software,merge_name,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,f_kma,f_phylotree,version):
    ''' For saving CV scores. Usage:
    MSMA discrete model
    MSMA concat mixed-dspecies model

    '''

    save_name_score_temp=temp_path+'log/software/'+str(software) +'/analysis/'+ str(version) +'/'+str(merge_name) + '/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    save_name_weights=temp_path+'log/software/'+str(software) +'/analysis/'+  str(version) +'/'+str(merge_name) + '/weights/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    save_name_loss=temp_path+'log/software/'+str(software) +'/analysis/'+  str(version) +'/'+str(merge_name)  + '/loss/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)


    save_name_score_temp=save_name_score_temp+'_all_score_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree)
    save_name_weights=save_name_weights+ '_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)
    save_name_loss=save_name_loss+'_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)+'_traininglog'


    return save_name_score_temp,save_name_weights, save_name_loss

def GETname_AAscoreMSMA_concat(software,merge_name,merge_name_test,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,f_kma,f_phylotree,version):
    ''' For saving CV scores. Usage:
    MSMA concat leave-one-out model

    '''

    save_name_score_temp=temp_path+'log/software/'+str(software) +'/analysis/'+ str(version) +'/'+str(merge_name) + '/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)
    save_name_weights=temp_path+'log/software/'+str(software) +'/analysis/'+  str(version) +'/'+str(merge_name) + '/weights/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    save_name_loss=temp_path+'log/software/'+str(software) +'/analysis/'+  str(version) +'/'+str(merge_name)  + '/loss/lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)


    save_name_score_temp=save_name_score_temp+'_all_score_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree)+'_testOn_'+merge_name_test
    save_name_weights=save_name_weights+ '_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)+'_'+merge_name_test
    save_name_loss=save_name_loss+'_KMA_'+ str(f_kma) + '_Tree_' + str(f_phylotree)+'_testOn_'+merge_name_test+'_traininglog.txt'


    return save_name_score_temp,save_name_weights, save_name_loss

def GETname_AAfeatureMSMA_concat(software,level,merge_name_train,merge_name,temp_path,version):
    '''MSMA concat leave-one-out model feature part.
    here "train" can be replace by "test"
     '''
    # save_name_ID=GETname_meta(species,anti,level)
    # save_name_meta=save_name_ID+'_pheno.txt'
    save_name_meta='' #only used to keep the same number of return parameters of GETname_AAfeatureMSMA

    path_feature_train = temp_path+'log/software/' + str(software)+'/software_output/'+str(version)+'/'+str(merge_name)+'/'+ str(merge_name_train)
    path_feature = temp_path+'log/software/' + str(software)+'/software_output/'+str(version)+'/'+str(merge_name)


    # path_metadata_multi_each=path_feature_train+'_meta.txt' #with  header
    path_metadata_multi_train=path_feature_train+'_pheno.txt' #without  header #todo
    path_ID_multi_train= path_feature_train+'_id'
    path_x = path_feature_train + 'data_x.txt'
    path_y = path_feature_train + 'data_y.txt'
    path_name = path_feature_train + 'data_names.txt'
    path_point_repre_results =path_feature_train+'_mutations.txt' # * differ form discrete
    path_res_repre_results  = path_feature_train + '_acquired_genes.txt' # * differ form discrete
    path_mutation_gene_results =path_feature_train + '_res_point.txt' # * differ form discrete


    path_ID_multi = path_feature + '/ID'
    path_metadata_multi = path_feature  + '/meta.txt' #with header

    path_res_result=temp_path+ 'log/software/'+ str(software)+'/software_output/'+str(version) +'/resfinder_outputs/' # * differ form discrete
    # -------------------------------

    path_cluster = path_feature+'/cluster/temp/all_strains_assembly.txt' #no use
    path_cluster_results=path_feature+'/cluster/clustered_90.txt' #no use
    path_cluster_temp=path_feature+'/cluster/temp/clustered_90' #no use

    return save_name_meta,path_ID_multi_train,path_feature_train,path_metadata_multi_train,path_ID_multi,\
           path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_point_repre_results,\
           path_res_repre_results ,path_mutation_gene_results,path_res_result,path_feature,path_x,path_y,path_name











