import pandas as pd
import numpy as np
'''
This script is to create the mapping file for PATRIC ID to NCBI ID. 9th Feb 2024.
'''


data = pd.read_csv('./data/PATRIC/genome_metadata', dtype={'genome_id': object}, sep="\t")
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also#
#     print(data.loc[data['genome_id'] == "1773.19907"])
#     print("-------------------------------------------------")
#     print(data.loc[data['genome_id'] == "1310800.122"])
data = data.loc[:, ("genome_id", "biosample_accession","bioproject_accession","strain","assembly_accession","genbank_accessions","refseq_accessions","publication")]
id_list=data["genome_id"].to_list()

data2=np.genfromtxt('./data/PATRIC/meta/genome_list', dtype="str")
data=data.loc[data['genome_id'].isin(data2)]
print(data)
data.to_csv('./data/PATRIC/meta/PATRICtoNCBI', sep="\t",  index = False)

# print(data2)
# print(len(data))
### check if all genome we selected are in the mapping file.
### 289012.3 is genome is gone from PATRIC, but we also did not include it in our datasets.
# i=0
# for each in data2:
#     i+=1
#     if each not in data:
#         print(each)
#         print(i)
# # data = list(dict.fromkeys(data))
# # print(len(data))

### 289012.3 is in AMR list but now removed from PATRIC. We also (luckily) did not include it in our datasets.
# data = pd.read_csv('./data/PATRIC/PATRIC_genomes_AMR.txt', dtype={'genome_id': object}, sep="\t")
# print(data.loc[data['genome_id'] == "289012.3"])
