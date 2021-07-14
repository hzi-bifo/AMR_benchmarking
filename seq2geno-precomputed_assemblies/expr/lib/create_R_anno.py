#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Only the column "gene_name" will be used in the subsequently steps...

from Bio import SeqIO
import re
import argparse
import sys
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

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description='create R annotation table')
parser.add_argument('-r', dest='ref_gbk', type=str,
                    help='the reference genbank file')
parser.add_argument('-o', dest='out_f', type=str,
                    help='output file')

args = parser.parse_args()
gbk_f = args.ref_gbk
out_f = args.out_f

# read the gbk
rec = ''
try:
    rec = SeqIO.read(LoadFile(gbk_f), 'gb')
    chr_len = str(len(rec.seq))
    acc = rec.id
except IOError as ioe:
    print(ioe)
    print('Unable to open {}'.format(gbk_f))
    sys.exit()

with open(out_f, 'w') as out_fh:
    columns = ['locus', 'ID', 'gene_name', 'type', 'Start', 'End']
    out_fh.write('\t'.join(columns)+'\n')

    target_fea = [fea for fea in rec.features
                  if ((fea.type == 'gene') or
                      (not (re.search('RNA', fea.type) is None)))]
    for fea in [fea for fea in rec.features
                if ((fea.type == 'gene') or
                    (not (re.search('RNA', fea.type) is None)))]:
        if not ('locus_tag' in fea.qualifiers):
            continue
        acc = fea.qualifiers['locus_tag'][0]
        gene_name = (fea.qualifiers['name'][0] if 'name' in fea.qualifiers else
                     '-')
#        locus= ','.join([acc, gene_name])
        locus = ','.join([acc, fea.qualifiers['name'][0]]) if \
                ('name' in fea.qualifiers) else acc
        # be careful about the coordinates
        d = [locus, acc, gene_name, str(fea.type),
             str(fea.location.start+1),
             str(fea.location.end)]
        out_fh.write('\t'.join(d)+'\n')
