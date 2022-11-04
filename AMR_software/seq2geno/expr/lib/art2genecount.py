#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Tanslated and refactored from Andreas Dötsch's script (2012)
import os
import re
import argparse


def main(args):
    art_file = args.a
    ref_type = args.t
    ref_file = args.r
    blankmode = args.b
    cds_only = args.c

    assert os.path.isfile(art_file)
    ART = open(art_file, 'r')
    assert os.path.isfile(ref_file)
    REF = open(ref_file, 'r')

    # initalize vectors ####
    # check for header in annotation
    line = REF.readline()

    if re.match('@', line):
        parts = line.split('|')
        genomesize = int(parts[2].strip())

    plus = [0] * (genomesize + 1)
    minus = [0] * (genomesize + 1)

    # ### parse ART ######
    for line in ART.readlines():
        if re.match('#', line) is not None:
            # skip header
            continue
        parts = line.strip().split()
        i = int(parts[0])
        plus[i] = int(parts[1])
        minus[i] = int(parts[2])
    ART.close()

    # #### parse annotation #####
    n_genes = 0
    genecount = dict()
    antisensecount = dict()
    for line in REF.readlines():
        genetype = ''
        genestart = 0
        geneend = 0
        genestrand = ''
        genetxt = ''
        geneID = ''
        # parse current gene
        if (ref_type == "gtf"):
            # use GTF format
            parts = line.split('\t')
            genetype = parts[2]
            genestart = int(parts[3])
            geneend = int(parts[4])
            genestrand = parts[6]
            genetxt = parts[8]
            parts = genetxt.split('"')
            geneID = parts[1]
        elif (ref_type == "tab"):
            # skip header
            if re.match('@', line):
                continue
            # use tab separated format from pseudomonas.com
            parts = line.split('\t')
            geneID = parts[3]
            if len(parts) >= 9:
                geneID = ','.join([parts[3], parts[8]])
            genetype = parts[4]
            genestart = int(parts[5])
            geneend = int(parts[6])
            genestrand = parts[7]
        else:
            raise TypeError('No format description found!')

        n_genes += 1

        # skip non-coding genes, if this option is set
        if cds_only and (genetype != 'CDS'):
            continue

    #   This check is done when assigning integer type to them
    #    #check annotation for errors
    #    if(!( (genestart =~ /^[0-9]+$/) & ($geneend =~ /^[0-9]+$/) )){
    #            print STDERR "!!! Annotation error for gene $geneID !!!\n";
    #    }

        # count reads
        # initiate the count
        if not (geneID in genecount):
            genecount[geneID] = 0
        if not (geneID in antisensecount):
            antisensecount[geneID] = 0

        for i in range(genestart, geneend + 1):
            if i > genomesize:
                raise IndexError('Out of range values for gene {}'.format(
                    geneID))

            if genestrand == "+":
                genecount[geneID] += plus[i]
                antisensecount[geneID] += minus[i]
            elif genestrand == "-":
                genecount[geneID] += minus[i]
                antisensecount[geneID] += plus[i]
            else:
                raise ValueError('''Unrecognized strand for gene
                                 {} ({}:{})'''.format(
                                     geneID, genestart, geneend))
    REF.close()

    geneIDs = list(genecount.keys())
    geneIDs.sort()
    for geneID in geneIDs:
        if blankmode:
            print("{}".format(genecount[geneID]))
        else:
            print('\t'.join([geneID,
                             str(genecount[geneID]),
                             str(antisensecount[geneID])]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='art2genecount.py',
        description='''Calculate raw counts per gene from art files''')

    parser.add_argument('-v', action='version',
                        version='v.Beta')
    parser.add_argument('--art', dest='a', required=True,
                        type=str, default='',
                        help='''art (artemis readable pileup) file''')
    parser.add_argument('-t', dest='t', type=str,
                        choices=['gtf', 'tab'],
                        help='''reference genome annotation type''')
    parser.add_argument('-r', dest='r', type=str,
                        required=True,
                        help='''reference genome''')
    parser.add_argument('-b', dest='b', action='store_true',
                        help='''Blank mode. Only readcounts are printed in the
                        output without gene IDs and anti-sense counts.''')
    parser.add_argument('-c', dest='c', action='store_true',
                        help='''only coding sequences''')

    args = parser.parse_args()
    main(args)
