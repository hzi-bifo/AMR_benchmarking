#!/usr/bin/python
import sys
import os
sys.path.append('../')
# sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import math

def get_full_d(wd_results):

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    wd_results = os.path.join(fileDir, wd_results)
    return wd_results
def get_directory(path):
    os.path.dirname(path)
    return path


def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

def get_absolute_pathname(p_names):
    # get full path from a relative path.
    fileDir=os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, p_names)
    filename = os.path.abspath(os.path.realpath(filename))
    return filename


def roundup(x):
    return int(math.ceil(x / 50.0)) * 50

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

def hzi_cpu_header3(run_file,name,cpu_n):

    run_file.write("#$ -N %s" % name)
    run_file.write("\n")
    run_file.write('#$ -l arch=linux-x64')
    run_file.write("\n")
    run_file.write("#$ -pe multislot %s" %cpu_n)
    run_file.write("\n")
    run_file.write('#$ -b n')
    run_file.write("\n")
    run_file.write('#$ -l vf=4G')
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
    run_file.write("export PATH=~/miniconda2/bin:$PATH \nexport PYTHONPATH=$PWD \nsource activate multi_bench_phylo \nexport PATH=~/bin:$PATH \nexport PYTHONPATH=$PWD")
    run_file.write("\n")

    return run_file

def hzi_cpu_header2(run_file,name,cpu_n):
    #large memory usage
    run_file.write("#$ -N %s" % name)
    run_file.write("\n")
    run_file.write('#$ -l arch=linux-x64')
    run_file.write("\n")
    run_file.write("#$ -pe multislot %s" %cpu_n)
    run_file.write("\n")
    run_file.write('#$ -l vf=200G')
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
def hzi_cpu_header4(run_file,name,cpu_n,env_name,node):

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
    run_file.write("#$ -q %s" %node)
    run_file.write("\n")
    run_file.write("#$ -cwd")
    run_file.write("\n")
    run_file.write("hostname -f")
    run_file.write("\n")
    run_file.write("echo $PATH")
    run_file.write("\n")
    run_file.write("export PATH=~/miniconda2/bin:$PATH \nexport PYTHONPATH=$PWD \nsource activate %s \n" %env_name)

    return run_file
def hzi_cpu_header5(run_file,name,env_name,node):

    run_file.write("#$ -N %s" % name)
    run_file.write("\n")
    run_file.write('#$ -l arch=linux-x64')
    run_file.write("\n")
    run_file.write("#$ -l t4_gpu=1")
    run_file.write("\n")
    run_file.write('#$ -b n')
    run_file.write("\n")
    run_file.write("#$ -o /vol/cluster-data/khu/sge_stdout_logs/")
    run_file.write("\n")
    run_file.write("#$ -e /vol/cluster-data/khu/sge_stdout_logs/")
    run_file.write("\n")
    run_file.write("#$ -q %s" %node)
    run_file.write("\n")
    run_file.write("#$ -cwd")
    run_file.write("\n")
    run_file.write("hostname -f")
    run_file.write("\n")
    run_file.write("echo $PATH")
    run_file.write("\n")
    run_file.write("export PATH=~/miniconda2/bin:$PATH \nexport PYTHONPATH=$PWD \nsource activate %s \n" %env_name)

    return run_file
def header_THREADS(run_file,n):
    run_file.write("\n")
    run_file.write("OMP_NUM_THREADS=%s \n" % n)
    run_file.write("OPENBLAS_NUM_THREADS=%s \n" % n)
    run_file.write("BLIS_NUM_THREADS=%s \n" % n)
    run_file.write("MKL_NUM_THREADS=%s \n" % n)
    run_file.write("VECLIB_MAXIMUM_THREADS=%s \n" % n)
    run_file.write("NUMEXPR_NUM_THREADS=%s \n" % n)
    run_file.write("\n")
    return run_file




def plot_kma_split_dif(split_original_all,split_new_k_all,level,combination):
    #plot the std of samples in each folder
    # dif=list(map(operator.sub, split_original_all, split_new_k_all))
    # plt.plot(dif)
    ind = np.arange(len(split_original_all))
    width = 0.3
    plt.bar(ind, split_original_all, width, label='Original split method')
    plt.bar(ind + width, split_new_k_all, width, label='New split method')
    plt.legend(loc=0)
    y_pos = np.arange(len(combination))
    # plt.xticks(y_pos, combination, color='orange', rotation=45, fontweight='bold', fontsize='17', horizontalalignment='right')
    # plt.xticks(y_pos, combination, rotation=45, fontsize='7',
    #            horizontalalignment='right')
    # plt.show()
    plt.xlabel('Each species and antibiotic combinations')
    plt.title('Standard deviation of sample number in the CV folders')
    plt.savefig('cv_folders/' + str(level) + '/kma_split_dif.png')


def plot_kma_split(split_original,split_new_k,level,list_species,merge_name):
    # plot the sample number in each folder, w.r.t. each species in a multi-species model.
    fig, axs = plt.subplots(1,2)
    # fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=0.8, top=None, wspace=0.3, hspace=None)
    fig.suptitle('Each species\' sample number in each CV folder')
    ind = np.arange(split_original.shape[1])
    width=0.1
    n=0
    for s in np.arange(split_original.shape[0]):
        axs[0].bar(ind+n*width,split_original[s], width, label="$\it{%s}$"%list_species[s].split()[0][0]+'. '+list_species[s].split()[1])

        axs[1].bar(ind+n*width,split_new_k[s], width, label="$\it{%s}$"%list_species[s].split()[0][0]+'. '+list_species[s].split()[1])
        n += 1
    max_v=max([np.max(split_original),np.max(split_new_k)])
    # axs[1].legend(loc=1,fontsize='small')
    axs[1].legend(bbox_to_anchor=(1.02, 1.02), fontsize='small',handlelength=0.5)
    # axs[0].legend(loc = 'upper center', bbox_to_anchor = (0., -0.05, 1., 0.), fontsize='small')
    axs[1].set_ylim(0,roundup(max_v))
    axs[0].set_ylim(0, roundup(max_v))
    axs[1].set_xlabel('folder in nested CV')
    axs[0].set_xlabel('folder in nested CV')
    axs[0].set_ylabel('sample number')
    # axs[1].set_ylabel('sample number')
    axs[0].set_title('Aytan-Aktug\'s split',loc ='right')
    axs[1].set_title('Modified split',loc ='right')
    plt.savefig('cv_folders/' + str(level) + '/'+ merge_name+'kma_split_multi.png')

# def plot_kma_split(split_original, split_new_k, level, list_species, merge_name):
#     # plot the sample number in each folder, w.r.t. each species in a multi-species model.
#     fig, axs = plt.subplots(1, 2)
#     # fig.tight_layout()
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
#     fig.suptitle('Each species\' sample number in each CV folder')
#     ind = np.arange(split_original.shape[1])
#     width = 0.1
#     n = 0
#     for s in np.arange(split_original.shape[0]):
#         axs[0].bar(ind + n * width, split_original[s], width, label=list_species[s])
#
#         axs[1].bar(ind + n * width, split_new_k[s], width, label=list_species[s])
#         n += 1
#     max_v = max([np.max(split_original), np.max(split_new_k)])
#     # axs[1].legend(loc=1,fontsize='small')
#     axs[1].legend(bbox_to_anchor=(1.02, 1.02), fontsize='small')
#     axs[1].set_ylim(0, roundup(max_v))
#     axs[0].set_ylim(0, roundup(max_v))
#     axs[1].set_xlabel('folder in nested CV')
#     axs[0].set_xlabel('folder in nested CV')
#     axs[0].set_ylabel('sample number')
#     # axs[1].set_ylabel('sample number')
#     axs[0].set_title('Aytan-Aktug\'s split', loc='right')
#     axs[1].set_title('Modified split', loc='right')
#     plt.savefig('cv_folders/' + str(level) + '/' + merge_name + 'kma_split_multi.png')



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
