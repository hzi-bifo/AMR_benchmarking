#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Refactored and translated from the old scripts

import pandas as pd
import re
import argparse
def LoadFile(f):
    # The files might have different encoding methods
    # To void the problem, this is a generalized loader to create file handels
    encoding_set = ['utf-8', 'latin1', 'windows-1252']
    right_encoding = encoding_set.pop()
    fh = open(f, 'r', encoding=right_encoding)
    found = False
    while (not found) and (len(encoding_set) > 0):
        try:
            # test whether the file can be read
            fh.readlines()
            found = True
        except UnicodeDecodeError:
            # shift the decoding to the next
            right_encoding = encoding_set.pop()
        finally:
            # open the file with either the same or another decoding method
            fh.close()
            fh = open(f, 'r', encoding=right_encoding)

    if found:
        return(fh)
    else:
        raise UnicodeDecodeError(
            'The encodings of {} is not recognizable'.format(f))


def parse_gff(gff_f, tmp_gff_f):
    # replace the old command :
    # cat /net/metagenomics/data/from_moni/old.tzuhao/seq2geno/test/v3/data/reference01/RefCln_UCBPP-PA14.edit.gff | sed '/^>/,$d' | tail -n+2 | grep -v '#' |grep -v '^\s*$' > roary/tmp.gff
    gff_fh = LoadFile(gff_f)
    good_lines = []
    for line in gff_fh.readlines()[1:]:
        line = re.sub('^>', ',$d', line)
        if not re.match('^\s*$', line):
            good_lines.append(line)
    gff_fh.close()

    with open(tmp_gff_f, 'w') as tmp_gff_fh:
        tmp_gff_fh.writelines(good_lines)

    # replace the old command:
    # field_map_wrapper.edit < <(grep -v @ annotation.tab) -s <(cat roary/tmp.gff) -f 6 -m 4 -i
    good_gff_df = pd.DataFrame([line.strip().split('\t')
                                for line in good_lines if
                                re.match('#', line) is None])
    good_gff_df.columns = ['CHROM', 'source', 'feature',
                           'Start', 'Stop', 'score',
                           'strand', 'frame', 'attributes']
    good_gff_df['Start'] = good_gff_df['Start'].astype(str)
    good_gff_df['Stop'] = good_gff_df['Stop'].astype(str)

    # to mimic bash 'sort', which solve ties from the first charcter to the
    # end when both start and end are specified
    good_gff_df['secondary_key'] = good_gff_df.apply(
        lambda r: '\t'.join(r), axis=1)
    # for field_map_wrapper.edit:line53
    good_gff_df.sort_values(by=['Start', 'secondary_key'],
                            ascending=True, inplace=True)
    # for field_map.edit.py:line31
    good_gff_df.drop_duplicates(subset=['Start'],
                                inplace=True, keep='first')
    print(good_gff_df.shape)
    print(good_gff_df.head())
    print(good_gff_df.dtypes)
    return(good_gff_df)


def parse_tab(tab_f):
    tab_df = pd.read_csv(tab_f, sep='\t', comment='@', header=None)
    with open(tab_f, 'r') as tab_fh:
        cmt_lines = [line for line in tab_fh.readlines()
                     if re.match('@', line)]
        name_line = cmt_lines[-1].strip()
        tab_df.columns = name_line.split('\t')
    # for string comparison like field_map.edit.py
    tab_df['Start'] = tab_df['Start'].astype(str)
    tab_df['Stop'] = tab_df['Stop'].astype(str)
    # for field_map.edit.py:line31
    tab_df.drop_duplicates(subset=['Start'], inplace=True, keep='first')
    return(tab_df)


def map_two_data(tab_df, good_gff_df):
    # in the previous command, 'Start' was the selected columns of the two data
    # frames
    # Use inner, because of field_map_wrapper.edit:line53 and field_map.edit.py:line158
    annotation_mapped_df = pd.merge(tab_df, good_gff_df, how='inner',
                                    left_on='Start', right_on='Start')
    print('merged')
    print(annotation_mapped_df.shape)
    seleccted_columns = [annotation_mapped_df.iloc[:, 3],
                         annotation_mapped_df.iloc[:, 8],
                         annotation_mapped_df.iloc[:, 17]]
    annot_mapped_df = pd.DataFrame(seleccted_columns).T
    print(annot_mapped_df.head())
    return(annot_mapped_df)


def main(Args):
    gff_f = Args.gff_f
    tab_f = Args.tab_f
    tmp_gff_f = Args.tmp_gff
    annot_mapped_tab = Args.merged

    good_gff_df = parse_gff(gff_f, tmp_gff_f)
    tab_df = parse_tab(tab_f)
    annot_mapped_df = map_two_data(tab_df, good_gff_df)
    with open(annot_mapped_tab, 'w') as annot_mapped_tab_h:
        annot_mapped_df.sort_values(by=['Locus_Tag']).to_csv(
            annot_mapped_tab_h, na_rep='',
            sep='\t', header=None, index=False)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("-g", dest="gff_f", required=True,
                        help="input gff")
    Parser.add_argument("-t", dest="tab_f", required=True,
                        help="input annotation tab")
    Parser.add_argument("-out_gff", dest="tmp_gff", required=True)
    Parser.add_argument("-out_annot", dest="merged", required=True)
    Args = Parser.parse_args()
    main(Args)

