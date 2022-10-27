#!/usr/bin/python
from src.amr_utility import file_utility,name_utility,load_data
from src.cv_folds import cluster2folds
import os,argparse
import json
import pandas as pd

def extract_info(level,s, cv,f_phylotree, f_kma,f_all,temp_path):

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if -f_phylotree:
        #no use. As we directly provided Phylogeny folds via Seq2Geno2Pheno
        print('we directly provided Phylogeny folds via Seq2Geno2Pheno')
        exit()
        s=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae',\
           'Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']
        data = data.loc[s, :]
    else:
        if f_all == False:
            data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(data)
    for species, antibiotics in zip(df_species, antibiotics):
        antibiotics, ID, Y = load_data.extract_info(species, False, level)
        i_anti = 0

        for anti in antibiotics:

            # 1. exrtact CV folders----------------------------------------------------------------
            p_names = name_utility.GETname_meta(species,anti,level)
            Random_State = 42

            if f_phylotree:  # phylo-tree based cv folders
                pass
                # folders_index = cluster2folds.prepare_folders_tree(cv, species, anti, p_names,
                #                                                                       False)
            elif f_kma:  # kma cluster based cv folders
                _,_,_,_,p_clusters,_,_,_,_,\
                        _,_,_,_,_,_ = name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)


                folders_index, _, _ = cluster2folds.prepare_folders(cv, Random_State, p_names,
                                                                                       p_clusters,
                                                                                       'new')
            else:#random
                pass
                # folders_index = cluster2folds.prepare_folders_random(cv, species, anti, p_names,
                #                                                                       False)

            id=ID[i_anti]
            i_anti+=1
            idname=[]
            for each_folds in folders_index:
                id_sub=[]
                for each_s in each_folds:
                    id_sub.append(id[each_s])
                idname.append(id_sub)


            folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
            file_utility.make_dir(os.path.dirname(folds_txt))
            with open(folds_txt, 'w') as f:
                json.dump(idname, f)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                    help='Directory to store temporary files.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.level,parsedArgs.species,parsedArgs.cv_number,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_all,parsedArgs.temp_path)
# if __name__ == '__main__':
#     # extract_info('loose', True, False )
#     extract_info('loose', False, True )
#     # extract_info('loose', False, False )

