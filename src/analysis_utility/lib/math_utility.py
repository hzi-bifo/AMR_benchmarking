import numpy as np
import pandas as pd

def get_most_fre_hyper(hy_para_collection,f_pipe):
    '''
    :param hy_para_collection: cv(10) dimension. each reapresents one outer loop's hyper-parameters, which is selected via CV inner loop.
    :return: the most frequenct hyper-para setting.
    '''
    if f_pipe:
        hy_para_collection=[i.steps[0][1].get_params() for i in hy_para_collection]
        # hy_para_collection_2=[i.steps[0][1].__class__.__name__ for i in hy_para_collection]

    df = pd.DataFrame(hy_para_collection)
    # print(df)
    df_count=df.groupby(df.columns.tolist(), as_index=False, dropna=False).size()
    # print(df_count)
    df_count.rename(columns={'size': 'frequency'}, inplace=True)

    ind=df_count['frequency'].argmax()
    common=df_count.loc[ind]
    fre=df_count.loc[df_count.index[ind], 'frequency']
    # print(fre)
    return common,fre
