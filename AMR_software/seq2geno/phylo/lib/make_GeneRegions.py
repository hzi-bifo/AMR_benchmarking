#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# To create the consensus sequences, this script determines the coordinates
# for bcftools
import sys
import re
import argparse


def select(f, target_feature, comment_pattern):
    lines = [line.strip().split('\t') for line in open(f, 'r').readlines()
             if (not re.search(comment_pattern, line))
             and (len(line.split('\t')) == 9)]
    regions = ['{}:{}-{}'.format(line[0], line[3], line[4])
               for line in lines if line[2] == target_feature]

    if len(regions) == 0:
        sys.exit('No target lines found')
    else:
        print('\n'.join(regions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create gene regions list')
    parser.add_argument('--g', dest='gff',
                        required=True, help='gff file')
    parser.add_argument('--f', dest='feature',
                        required=True,
                        help='target features (column 3 of gff file)')
    parser.add_argument('--c', dest='comment',
                        required=False, default='^#',
                        help='comment pattern')
    args = parser.parse_args()

    f = args.gff
    target_feature = args.feature
    comment_pattern = args.comment
    select(f, target_feature, comment_pattern)
