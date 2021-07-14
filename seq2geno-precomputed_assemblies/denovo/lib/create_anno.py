#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

from Bio import SeqIO
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

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description='create annotation table')
parser.add_argument('-r', dest='ref_gbk', type=str,
                    help='the reference genbank file')
parser.add_argument('-n', dest='ref_name', type=str,
                    help='the reference strain name')
parser.add_argument('-o', dest='out_f', type=str,
                    help='output file')

args = parser.parse_args()
gbk_f = args.ref_gbk
ref_strain = args.ref_name
out_f = args.out_f

# read the gbk
rec = SeqIO.read(LoadFile(gbk_f), 'gb')
chr_len = str(len(rec.seq))
acc = rec.id
sequence_type = 'Chromosome'

with open(out_f, 'w') as out_fh:
    header_line = '@'+'|'.join([ref_strain, acc, chr_len])
    columns = ['@Strain', 'Refseq_Accession', 'Replicon', 'Locus_Tag',
               'Feature_Type', 'Start', 'Stop', 'Strand', 'Gene_Name',
               'Product_Name']
    out_fh.write(header_line+'\n')
    out_fh.write('\t'.join(columns)+'\n')

    target_feature = 'gene'
    for fea in [fea for fea in rec.features if fea.type == target_feature]:
        if not ('locus_tag' in fea.qualifiers):
            continue
        d = [ref_strain, acc, sequence_type, fea.qualifiers['locus_tag'][0],
             target_feature, str(fea.location.start+1), str(fea.location.end),
             '+' if fea.strand == 1 else '-',
             fea.qualifiers['name'][0] if 'name' in fea.qualifiers else '',
             re.sub('\s+', '_', fea.qualifiers['function'][0])
             if ('function' in fea.qualifiers) else '.']
        out_fh.write('\t'.join(d)+'\n')
