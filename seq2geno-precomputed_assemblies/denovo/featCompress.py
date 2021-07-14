# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

'''
Compress binary features by their patterns among the samples.
A representiative is selected for each pattern. The new feature name
is composed of the representative feature and the size of group.
'''

import pandas as pd


def check_input(df):
    # Valid values are '1', '0', 'NA'
    # check if the features contain non-determined value
    check_df = df.fillna('.')
    outcome = False
    if ((check_df != '1') & (check_df != '0') &
       (check_df != 'NA')).sum().sum() == 0:
        outcome = True
    if not outcome:
        raise ValueError("File unrecognizable: "
                         "only 'NA', '1', and '0' are acceptable")


f = snakemake.input[0]
group_out_f = snakemake.output['GROUPS']
comp_out_f = snakemake.output['NONRDNT']

# read table
df = pd.read_csv(f, sep='\t', header=0, index_col=0,
                 dtype='str', na_values=[''],
                 keep_default_na=False)
check_input(df)

# group the features
t_df = df.transpose()  # features in rows
grouped_t_df = t_df.groupby(t_df.columns.values.tolist())

# list the groups
# and convert the feature names
new_feat_dict = {}
group_dict = {}
for name, group in grouped_t_df:
    grouped_t_df.get_group(name)
    repre = str(group.index.tolist()[0])
    new_repre = '|'.join([repre, str(len(group))])
    feat = t_df.loc[repre, :]

    new_feat_dict[new_repre] = feat
    group_dict[new_repre] = group.index.tolist()

# print the compressed features
new_df = pd.DataFrame(data=new_feat_dict)  # features in columns
new_df.to_csv(comp_out_f, sep='\t', header=True,
              index=True, index_label='Isolates', encoding='utf-8')

# print the cluster information
with open(group_out_f, 'w') as group_out_fh:
    for new_repre in group_dict:
        group_out_fh.write('{}:\t{}\n'.format(
            new_repre, '\t'.join(group_dict[new_repre])))
