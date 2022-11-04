#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

from Bio import AlignIO
import argparse


def parse_and_conc(file_list, out_file):
    # read each alignment
    files = [f.strip() for f in open(file_list, 'r').readlines()]
    # load the first alignment
    aln = AlignIO.read(files[0], 'fasta')
    aln.sort()
    # read the others and concatenate them
    for f in files[1:]:
        partition = AlignIO.read(f, 'fasta')
        partition.sort()
        aln += partition

    with open(out_file, 'w') as out_fh:
        AlignIO.write(aln, out_fh, 'fasta')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate alignments')
    parser.add_argument('--l', dest='aln_list',
                        required=True,
                        help='list of alignments to be concatenated')
    parser.add_argument('--o', dest='out', required=True, help='output')

    args = parser.parse_args()
    parse_and_conc(args.aln_list, args.out)
