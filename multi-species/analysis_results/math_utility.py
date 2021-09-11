import numpy as np
import pandas as pd

def get_most_fre_hyper(hy_para_collection):
    '''
    :param hy_para_collection: cv(10) dimension. each reapresents one outer loop's hyper-parameters, which is selected via CV inner loop.
    :return: the most frequenct hyper-para setting.
    '''
    df = pd.DataFrame(hy_para_collection)
    df_count=df.groupby(df.columns.tolist(), as_index=False).size()
    df_count.rename(columns={'size': 'frequency'}, inplace=True)
    ind=df_count['frequency'].argmax()
    common=df_count.loc[ind]
    fre=df_count.loc[df_count.index[ind], 'frequency']
    # print(fre)
    return common,fre