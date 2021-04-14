import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def dis(df,x_name,output_name,xlabel):
    '''
    :param df: pandas damefram
    :param x_name: vectors or keys in data
    :param output_name: graph name
    '''
    sns.set_theme(style="darkgrid")
    ax=sns.displot(
        df, x=x_name,bins=30)
    # plt.legend()
    ax.set(xlabel=xlabel, ylabel='')
    ax.figure.savefig(output_name)