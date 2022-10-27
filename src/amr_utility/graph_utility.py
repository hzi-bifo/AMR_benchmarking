import seaborn as sns
import pandas as pd
import numpy as np
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

def dataset_plot(summary_plot,anti,cv,Tscore,report_list,hue_list):
    '''
    building pandas fataframe for later plotting.
    final_score| 'antibiotic'|'classifier/selection Method'
    ceftazidime|0.85|...
    ciprofloxacin|
    :param Tscore: the score used for comparison
    :param report_list: classfication reports list w.r.t. all
    :param hue_list:
    :return:
    '''
    for i in np.arange(cv):
        report_list_subcv=[]
        for n in range(len(report_list)):
            # print(report_list[n])
            report_list_subcv.append(report_list[n][i])


        final_score_subcv=[]
        if Tscore == 'f1_macro':
            for n in range(len(report_list)):
                report_=report_list_subcv[n]
                report=pd.DataFrame(report_).transpose()
                if report.loc['1', 'support']!=0 and report.loc['0', 'support']!=0:
                    final_score_subcv.append(report.loc['macro avg', 'f1-score'])

        elif Tscore == 'f1_negative':
             for n in range(len(report_list)):
                report_=report_list_subcv[n]
                report=pd.DataFrame(report_).transpose()
                if report.loc['0', 'support']!=0:
                    final_score_subcv.append(report.loc['0', 'f1-score'])


        elif Tscore == 'f1_positive':
            for n in range(len(report_list)):
                report_=report_list_subcv[n]
                report=pd.DataFrame(report_).transpose()
                if report.loc['1', 'support']!=0:
                    final_score_subcv.append(report.loc['1', 'f1-score'])


        elif Tscore == 'accuracy':
            for n in range(len(report_list)):
                report_=report_list_subcv[n]
                report=pd.DataFrame(report_).transpose()
                # if report.loc['1', 'support']!=0 and report.loc['0', 'support']!=0:
                final_score_subcv.append(report.iat[2,2])


        # summary_plot_sub = pd.DataFrame(columns=[Tscore, 'antibiotic', 'selection Method'])
        columns_name=summary_plot.columns.tolist()
        summary_plot_sub = pd.DataFrame(columns=columns_name)

        for n in range(len(report_list)):
            if final_score_subcv!=[]:
                summary_plot_sub.loc['s'] = [final_score_subcv[n], anti, hue_list[n]]
                summary_plot = summary_plot.append(summary_plot_sub, sort=False)


    return summary_plot




def box_plot_multi(Tscore,summary_plot,save_name_score_final,title):
    ax = sns.boxplot(x="antibiotic", y=Tscore, hue="selection Method",
                                 data=summary_plot, dodge=True, width=0.4)
    fig = ax.get_figure()
    ax.set_title(title)
    plt.legend(loc='lower right')
    fig.tight_layout()
    fig.set_size_inches(10, 8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, horizontalalignment='right', fontsize=9)
    fig.savefig(save_name_score_final + '_' + Tscore + ".png")
    fig.clf()
