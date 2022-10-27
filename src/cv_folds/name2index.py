#!/usr/bin/python
import pandas as pd
import numpy as np

'''Turn sample name folds to index folds, based on the p-names provided'''

def Get_index(folders_sample,p_names):

    sample_list_open = open(p_names, "r")
    names_read = sample_list_open.readlines()
    sample_list = []
    for each in range(len(names_read)):
        sample_list.append(names_read[each].replace("\n", ""))

    # print(sample_list)
    folders_index=[]
    for each_cv in folders_sample:
        folders_index_sub=[]
        for sample_name in each_cv:
            folders_index_sub.append(sample_list.index(sample_name.replace("iso_","").replace("ISO_","")))
        folders_index.append(folders_index_sub)
    return folders_index

