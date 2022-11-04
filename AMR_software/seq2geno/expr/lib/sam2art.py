#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import re
from math import ceil, floor


def main(args):
    input_file = args.input_file  # SAM file
    strands = args.s  # stranded option
    sinister = args.l  # sinister option (forces -4 in paired end mode)
    dexter = args.d  # dexter option
    window = args.window  # windowsize
    paired = args.p  # paired end option
    minmapq = args.mapping_quality  # minimum mapping quality
    # independent output of mates (in paired end mode)
    pe_sepout = args.print_mates
    # Flat mode. Prints a flat file for fasta database access. Readcounts are
    # printed in 6-digit fields for subsequent positions througout the genome.
    flatmode = args.f
    usecontig = args.chromosome  # processed contig/chromosome

    if (not pe_sepout) and sinister and paired:
        pe_sepout = 1
        print('-l and -p were selected, forcing -4')

    if (not paired) and pe_sepout:
        pe_sepout = 0
        print('-4 was selected without -p, ignoring -4')

    if window > 1 and (not sinister):
        sinister = True 
        print('''You selected a window size > 1.
              Sinister option has been set automatically.''')

    no_overflow = 1
    highest_pos_found = 0

    maxpos = 8000000
    readcount = 0
    mappedcount = 0
    basecount = 0

    # inititate the arrays
    plus_left = [0] * (maxpos + 1)
    plus_right = [0] * (maxpos + 1)
    minus_left = [0] * (maxpos + 1)
    minus_right = [0] * (maxpos + 1)

    with open(input_file, 'r') as input_fh:
        for line in input_fh.readlines():
            # skip header lines
            if re.match('@', line):
                continue

            # parse read
            parts = line.strip().split('\t')
            flag = parts[1]
            contig = parts[2]
            pos = int(parts[3])
            mapq = float(parts[4])
            readlength = len(parts[9])

            readcount += 1
            # if several contigs/chromosomes are present,
            # skip if the wrong one was hit
            if usecontig != '' and usecontig != contig:
                continue

            # skip read if mapping quality is below threshold
            # (i.e. unmapped reads or non-unique hits)
            if mapq < minmapq:
                continue

            mappedcount += 1

            # count strand hits, paired end mode
            if paired:
                if sinister:
                    if window == 1:
                        if flag == '99':
                            # left (5') read of a plus strand fragment
                            plus_left[pos] += 1
                        if flag == '147':
                            # right (3') read of a plus strand fragment
                            plus_right[pos+readlength-1] += 1
                        if flag == '163':
                            # right (3') read of a minus strand fragment
                            minus_right[pos] += 1
                        if flag == '83':
                            # left (5') read of a minus strand fragment
                            minus_left[pos+readlength-1] += 1
                    else:
                        if flag == '99':
                            # left (5') read of a plus strand fragment
                            plus_left[ceil(pos/window)] += 1
                        if flag == '147':
                            # right (3') read of a plus strand fragment
                            plus_right[ceil((pos+readlength-1)/window)] += 1
                        if flag == '163':
                            # right (3') read of a minus strand fragment
                            minus_right[ceil(pos/window)] += 1
                        if flag == '83':
                            # left (5') read of a minus strand fragment
                            minus_left[ceil((pos+readlength-1)/window)] += 1
                elif dexter:
                    if window == 1:
                        if flag == '99':
                            # left (5') read of a plus strand fragment
                            plus_left[pos+readlength-1] += 1
                        if flag == '147':
                            # right (3') read of a plus strand fragment
                            plus_right[pos] += 1
                        if flag == '163':
                            # right (3') read of a minus strand fragment
                            minus_right[pos+readlength-1] += 1
                        if flag == '83':
                            # left (5') read of a minus strand fragment
                            minus_left[pos] += 1
                    else:
                        if flag == '99':
                            #  left (5') read of a plus strand fragment
                            plus_left[ceil((pos+readlength-1)/window)] += 1
                        if flag == '147':
                            # right (3') read of a plus strand fragment
                            plus_right[ceil(pos/window)] += 1
                        if flag == '163':
                            # right (3') read of a minus strand fragment
                            minus_right[ceil((pos+readlength-1)/window)] += 1
                        if flag == '83':
                            # left (5') read of a minus strand fragment
                            minus_left[ceil(pos/window)] += 1
                else:
                    if flag == '99':
                        # left (5') read of a plus strand fragment
                        for i in range(pos, pos + readlength):
                            plus_left[i] += 1
                        basecount += readlength
                    if flag == '147':
                        # right (3') read of a plus strand fragment
                        for i in range(pos, pos + readlength):
                            plus_right[i] += 1
                        basecount += readlength
                    if flag == '163':
                        # right (3') read of a minus strand fragment
                        for i in range(pos, pos + readlength):
                            minus_right[i] += 1
                        basecount += readlength
                    if flag == '83':
                        # left (5') read of a minus strand fragment
                        for i in range(pos, pos + readlength):
                            minus_left[i] += 1
                        basecount += readlength
            else:
                # count strand hits, single end mode
                if sinister:
                    if window == 1:
                        if flag == '0':  # plus strand
                            plus_left[pos] += 1
                        if flag == '16':  # minus strand
                            minus_left[pos + readlength - 1] += 1
                    else:
                        if flag == '0':
                            # plus strand
                            plus_left[ceil(pos / window)] += 1
                        if flag == '16':
                            # minus strand
                            minus_left[
                                ceil((pos + readlength - 1) / window)] += 1
                elif dexter:
                    if window == 1:
                        if flag == '0':  # plus strand
                            plus_left[pos + readlength - 1] += 1
                        if flag == '16':  # minus strand
                            minus_left[pos] += 1
                    else:
                        if flag == '0':  # plus strand
                            plus_left[ceil((pos + readlength - 1)/window)] += 1
                        if flag == '16':  # minus strand
                            minus_left[ceil(pos/window)] += 1
                else:
                    if flag == '0':  # plus strand
                        for i in range(readlength):
                            plus_left[i + pos] += 1
                    if flag == '16':  # minus strand
                        for i in range(readlength):
                            minus_left[i + pos] += 1

            if highest_pos_found < pos:
                highest_pos_found = pos

    mappedpercent = mappedcount/readcount * 100

    # print header
    if flatmode:
        print("      ", end='')
    elif pe_sepout:
        print(("# base forward_left reverse_left forward_right reverse_right "
              "# high quality (MQ >= {}): {}/{} ({}"
              "%)").format(str(minmapq), str(mappedcount), str(readcount),
                           str(mappedpercent)))
        print("# colour 255:0:0 0:255:0 255:0:255 0:255:255")
    elif strands == 1:
        print(("# base coverage  # high quality (MQ >= {}): "
              "{}/{} ({} %)").format(str(minmapq), str(mappedcount),
                                     str(readcount), str(mappedpercent)),
              end='')
        print("# colour 0:0:0")
    elif strands == 2:
        print(("# base forward reverse # high quality (MQ >= {}): "
              "{}/{} ({} %)").format(str(minmapq), str(mappedcount),
                                     str(readcount), str(mappedpercent)))
        print("# colour 255:0:0 0:255:0")

    # workaround for out-of-genome hits
    if no_overflow > 0:
        maxpos = highest_pos_found

    # print output
    for i in range(1, maxpos + 1):
        if flatmode:
            if i >= len(plus_left):
                plus_left.append(0)
            if i >= len(minus_left):
                minus_left.append(0)
            if i >= len(plus_right):
                plus_right.append(0)
            if i >= len(minus_right):
                minus_right.append(0)
            print("{:06d}".format(plus_left[i] + minus_left[i] + plus_right[i]
                  + minus_right[i]), end='')
        else:
            # skip empty positions
            if i >= len(plus_left):
                plus_left.append(0)
            if i >= len(minus_left):
                minus_left.append(0)
            if paired:
                if i >= len(plus_right):
                    plus_right.append(0)
                if i >= len(minus_right):
                    minus_right.append(0)
                if (plus_left[i] + minus_left[i] + plus_right[i]
                   + minus_right[i]) == 0:
                    continue
            elif (plus_left[i] + minus_left[i]) == 0:
                continue

            # determine position in windowed mode
            pos = floor(i * window - (window - 1) / 2)

            # print values
            if paired:
                if pe_sepout:
                    print('\t'.join([str(pos), str(plus_left[i]),
                                     str(minus_left[i]), str(plus_right[i]),
                                     str(minus_right[i])]))
                else:
                    tmp1 = plus_left[i] + plus_right[i]
                    tmp2 = minus_left[i] + minus_right[i]
                    print('\t'.join([str(pos), str(tmp1), str(tmp2)]))
            elif strands == 1:
                print('\t'.join([str(pos), str(plus_left[i] + minus_left[i])]))
            elif strands == 2:
                print('\t'.join([str(pos), str(plus_left[i]),
                                 str(minus_left[i])]))


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(
        prog='sam2art.py',
        description='''Convert SAM (.sam) to Artemis
        readable (.art) format''')

    parser.add_argument('-v', action='version',
                        version='v.Beta')
    parser.add_argument('--sam', dest='input_file', required=True,
                        type=str, default='',
                        help="""input sam file""")
    parser.add_argument('-s', dest='s', type=int,
                        choices=[1, 2], default=2,
                        help='''1 for unstranded or single stranded, 2 for double
                        stranded''')
    parser.add_argument('-l', dest='l',
                        action='store_true',
                        help="""'sinister' profile (5'-coverage, Filiatrault et
                        al.). When -p is also used, both 5' and 3'- ends are
                        simulateously detected for both plus and minus strand.
                        Default is unset (output of read depth).""")
    parser.add_argument('-d', dest='d', default=0, type=int,
                        help="""'dexter' profile. Reads only 3'-ends
                        of reads (not to be confused
                        with 3'-ends of fragments).""")
    parser.add_argument('-p', dest='p', action='store_true',
                        help="""paired end (mate pairs) mode.""")
    parser.add_argument('-4', dest='print_mates',
                        action='store_true',
                        help="""print values of mates independently (paired end
                        mode only.""")
    parser.add_argument('-mh', dest='map_holes', default=0, type=int,
                        help="""map 'holes' of coverage (coverage is less than
                        <threshold>).""")
    parser.add_argument('-mq', dest='mapping_quality', default=0, type=int,
                        help="""minimum mapping quality""")
    parser.add_argument('-c', dest='chromosome',
                        type=str, default='',
                        help="""name of contig/chromosome to analyse.
                        If the reference contains multiple contigs or
                        chromosomes, this option should be used to
                        analyse each contig independently.""")
    parser.add_argument('-f', dest='f', action='store_true',
                        help="""Flat mode. Prints a flat file for
                        fasta database access. Readcounts are
                        printed in 6-digit fields for
                        subsequent positions througout the genome.""")
    parser.add_argument('-w', dest='window', default=1, type=int,
                        help="""If set, read counts will be calculated
                        for windows of the specified size.
                        'Sinister' option will be set
                        automatically for w > 1.
                        Default is 1 (base-wise).""")

    args = parser.parse_args()
    main(args)
