#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# 1. Compute the minimum and maximum sequences of each family
# 2. Sort the seqeunces from by strain to by coding region
from Bio import SeqIO
import os
import argparse


def run(f_l, out_d, stats_f):
    if not os.path.exists(out_d):
        os.makedirs(out_d)

    families = {}
    # parse the list of files per strain
    f_lh = open(f_l, 'r')
    for line in f_lh.readlines():
        strain, f = line.strip().split('\t')
        records = SeqIO.parse(f, 'fasta')
        for rec in records:
            family_name = str(rec.id)
            rec.id = strain
            # remove the unuseful information
            rec.description = ''
            rec.name = ''
            # sort the sequences
            if family_name in families:
                families[family_name].append(rec)
            else:
                families[family_name] = [rec]

    f_lh.close()

    # write sequences per family
    for f in families:
        SeqIO.write(families[f], os.path.join(out_d, f+'.fa'), 'fasta')

    # statistics for furthur selection
    stats_fh = open(stats_f, 'w')
    for f in families:
        lengths = [len(rec.seq) for rec in families[f]]
        max_l = int(max(lengths))
        min_l = int(min(lengths))
        stats_fh.write('{}\t{}\t{}\n'.format(f, min_l, max_l))
    stats_fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compute the minimum and
        maximum sequences of each family''')
    parser.add_argument('--l', dest='f_list',
                        required=True,
                        help='list of sequence files of each strain')
    parser.add_argument('--d', dest='g_d',
                        required=True,
                        help='folder of output fasta files')
    parser.add_argument('--o', dest='stats_f',
                        required=True,
                        help='output statistics of each family')
    args = parser.parse_args()
    run(args.f_list, args.g_d, args.stats_f)
