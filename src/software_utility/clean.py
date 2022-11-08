#!/usr/bin/python
import argparse,os
import shutil
from src.amr_utility import name_utility,load_data
import pandas as pd

'Cleaning large intermediate files after running'

def clean(temp_path,software,level):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    df_species = data.index.tolist()
    # antibiotics = data['modelling antibiotics'].tolist()


    if software=='phenotypeseeker': # intermediate files for generating features.
        for species in df_species:
            antibiotics, _, _ =  load_data.extract_info(species, False, level)
            for anti in antibiotics:
                temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/random/'+ str(species.replace(" ", "_"))  + '/' + \
                        str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_temp'
                if os.path.isdir(temp_folder):
                    print('remove: '+temp_folder)
                    shutil.rmtree(temp_folder)

                temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/kma/'+ str(species.replace(" ", "_"))  + '/' + \
                        str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_temp'
                if os.path.isdir(temp_folder):
                    print('remove: '+temp_folder)
                    shutil.rmtree(temp_folder)

                temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/phylotree/'+ str(species.replace(" ", "_"))  + '/' + \
                        str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_temp'
                if os.path.isdir(temp_folder):
                    print('remove: '+temp_folder)
                    shutil.rmtree(temp_folder)


    elif software=='AytanAktug': #intermediate files for generating kma-based folds.
        for species in df_species:

            temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/SSSA/'+str(species.replace(" ", "_")) +'/cluster/temp'
            if os.path.isdir(temp_folder):
                print('remove: '+temp_folder)
                shutil.rmtree(temp_folder)

            temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/SSMA/'+str(species.replace(" ", "_")) +'/cluster/temp'
            if os.path.isdir(temp_folder):
                print('remove: '+temp_folder)
                shutil.rmtree(temp_folder)

            temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/MSMA_discrete/Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj/'+str(species.replace(" ", "_")) +'/cluster/temp'
            if os.path.isdir(temp_folder):
                print('remove: '+temp_folder)
                shutil.rmtree(temp_folder)




    elif  software=='seq2geno':
        for species in df_species:

            temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/'+str(species.replace(" ", "_")) +'/results/denovo/prokka'
            if os.path.isdir(temp_folder):
                print('remove: '+temp_folder)
                shutil.rmtree(temp_folder)

            temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/'+str(species.replace(" ", "_")) +'/results/denovo/spades'
            if os.path.isdir(temp_folder):
                print('remove: '+temp_folder)
                shutil.rmtree(temp_folder)
            temp_folder=str(temp_path)+'log/software/' +  str(software) +'/software_output/'+str(species.replace(" ", "_")) +'/results/denovo/extracted_proteins_nt'
            if os.path.isdir(temp_folder):
                print('remove: '+temp_folder)
                shutil.rmtree(temp_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                    help='Quality control: strict or loose')
    parser.add_argument('-s', '--software', type=str, required=False,
                        help='Cleaning large intermediate files after running. ')


    parsedArgs = parser.parse_args()
    clean(parsedArgs.temp_path,parsedArgs.software,parsedArgs.level)
