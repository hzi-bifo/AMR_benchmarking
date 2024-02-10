import pandas as pd
import numpy as np
'''
This script is to create the mapping file for PATRIC ID to NCBI ID. 9th Feb 2024.
'''


data = pd.read_csv('./data/PATRIC/genome_metadata', dtype={'genome_id': object}, sep="\t")
data = data.loc[:, ("genome_id", "biosample_accession","bioproject_accession","strain","assembly_accession","genbank_accessions","refseq_accessions","publication")]
id_list=data["genome_id"].to_list()

### genomes selected before QC (note: this list contains more genomes than our final datasets)
data2=np.genfromtxt('./data/PATRIC/meta/genome_list', dtype="str")
data=data.loc[data['genome_id'].isin(data2)]

### add SRA information
### run get_sra.sh
data3 = pd.read_csv('./data/PATRIC/meta/sra.txt', dtype={' genome.genome_id': object}, sep="\t")
data3.drop(data3.loc[data3['genome.genome_id']=='genome.genome_id'].index, inplace=True)
data3.rename(columns={"genome.genome_id": "genome_id", "genome.sra_accession": "sra_accession"}, inplace=True)
data3.reset_index(drop=True, inplace=True)
data3.drop_duplicates(inplace=True)




result = pd.merge(data, data3, how="left", on=["genome_id"])
second_column = result.pop('sra_accession')
result.insert(3, 'sra_accession', second_column)
result.to_csv('./data/PATRIC/meta/PATRICtoNCBI', sep="\t",  index = False)


### check if all genome we selected are in the mapping file.
# print(data2)
# print(len(data))
### 289012.3 is genome is gone from PATRIC, but we also did not include it in our datasets.
# i=0
# for each in data2:
#     i+=1
#     if each not in data:
#         print(each)
#         print(i)
# # data = list(dict.fromkeys(data))
# # print(len(data))

## check if all genome we selected are equipped with at least NCBI and genbank. 327 such cases.
result_check=result.loc[:, ("genome_id", "biosample_accession","bioproject_accession","sra_accession","genbank_accessions")]
result_check.set_index('genome_id', inplace=True)
selected_rows = result_check[result_check.isnull().all(axis=1)]
print(selected_rows)



### 289012.3 is in AMR list but now removed from PATRIC. We also (luckily) did not include it in our datasets.
# data = pd.read_csv('./data/PATRIC/PATRIC_genomes_AMR.txt', dtype={'genome_id': object}, sep="\t")
# print(data.loc[data['genome_id'] == "289012.3"])
