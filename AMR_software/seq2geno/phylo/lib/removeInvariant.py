#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Remove invariant sites from the alignment
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import argparse

# read an alignment
parser = argparse.ArgumentParser()
parser.add_argument('--in', required=True,
                    dest='i', help='input alignment')
parser.add_argument('--out', required=True,
                    dest='o', help='output alignment')
parser.add_argument('--cn', dest='cutoff_num',
                    help='''least number of variant residue should be included
                    in a column''')
parser.add_argument('--s', dest='case_sensitive',
                    default=False, action='store_true',
                    help='''case-sensitive (ie. A and a should be viewed
                    differently)''')

args = parser.parse_args()

aln_f = args.i
out_f = args.o
cutoff = int(args.cutoff_num)
case = args.case_sensitive

aln = SeqIO.to_dict(SeqIO.parse(aln_f, 'fasta'))

# detect invariant sites
detect_result = []
for name in aln:
    for n in range(len(str(aln[name].seq))):
        char = (str(aln[name].seq[n]).upper()
                if (not case) else str(aln[name].seq[n]))
        if len(detect_result) < len(str(aln[name].seq)):
            detect_result.append({char: 1})
        elif not (char in detect_result[n]):
            detect_result[n][char] = 1
        else:
            detect_result[n][char] += 1

variant_counts = [len(aln) - compo[sorted(compo, key=lambda x:-compo[x])[0]]
                  for compo in detect_result]
variant_loc = [n for n in range(len(variant_counts))
               if int(variant_counts[n]) >= cutoff]

new_recs = []
for name in aln:
    new_seq = ''.join([str(aln[name].seq[n]) for n in variant_loc])
    new_rec = SeqRecord(Seq(new_seq, IUPAC.IUPACAmbiguousDNA),
                        id=name, name='', description='')
    new_recs.append(new_rec)

with open(out_f, 'w') as out_h:
    SeqIO.write(new_recs, out_h, 'fasta')
