#!/usr/bin/python
import sys
import os
sys.path.append('../')
# sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def hzi_cpu_header(run_file,name,cpu_n):

    run_file.write("#$ -N %s" % name)
    run_file.write("\n")
    run_file.write('#$ -l arch=linux-x64')
    run_file.write("\n")
    run_file.write("#$ -pe multislot %s" %cpu_n)
    run_file.write("\n")
    run_file.write('#$ -b n')
    run_file.write("\n")
    run_file.write("#$ -o /vol/cluster-data/khu/sge_stdout_logs/")
    run_file.write("\n")
    run_file.write("#$ -e /vol/cluster-data/khu/sge_stdout_logs/")
    run_file.write("\n")
    run_file.write("#$ -q all.q")
    run_file.write("\n")
    run_file.write("#$ -cwd")
    run_file.write("\n")
    run_file.write("hostname -f")
    run_file.write("\n")
    run_file.write("echo $PATH")
    run_file.write("\n")
    run_file.write("export PATH=~/miniconda2/bin:$PATH \nexport PYTHONPATH=$PWD \nsource activate multi_bench_phylo \n")

    return run_file



def plot_kma_split_dif(split_original_all,split_new_k_all,level):
    # dif=list(map(operator.sub, split_original_all, split_new_k_all))
    # plt.plot(dif)
    ind = np.arange(len(split_original_all))
    width = 0.3
    plt.bar(ind, split_original_all, width, label='Original split method')
    plt.bar(ind + width, split_new_k_all, width, label='New split method')
    plt.legend(loc=0)
    plt.xlabel('Each species and antibiotic combinations')
    plt.title('Standard deviation of sample number in the CV folders')
    plt.savefig('cv_folders/' + str(level) + '/kma_split_dif.png')

    # X = np.arange(len(split_original_all))
    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.bar(X + 0.00,split_original_all, width=0.25,label='Original split method')
    # ax.bar(X + 0.25, split_new_k_all,  width=0.25,label='New split method')
    # ax.set_xlabel('Each species and antibiotic combinations')
    # ax.set_ylabel('')
    # # plt.xlabel('Each species and antibiotic combinations')
    # plt.legend(loc=0)
    # ax.set_title('Standard deviation of sample number in the CV folders')
    # plt.savefig('cv_folders/' + str(level) + '/kma_split_dif.png')

    # n=len(split_original_all)
    # df = pd.DataFrame(list(zip(split_original_all, split_new_k_all)),columns=['Original split method','New split method'] )
    # x=np.array(n)
    # y=split_original_all
    # z=split_new_k_all
    # df = pd.DataFrame(zip(x * n, ["Original split method"] * n + ["New split method"] * n , y + z ),
    #                   columns=["Original split method", "New split method"])
    # g=sns.barplot(x="Original split method", hue="New split method", data=df)
    # g.savefig('cv_folders/'+str(level)+'/kma_split_dif.png')

    # np.savetxt('cv_folders/'+str(level)+'/split_kma_original.csv',
    #            split_original_all,
    #            delimiter=", ",
    #            fmt='% s')
    # np.savetxt('cv_folders/' + str(level) + '/split_kma_khu.csv',
    #            split_new_k_all,
    #            delimiter=", ",
    #            fmt='% s')