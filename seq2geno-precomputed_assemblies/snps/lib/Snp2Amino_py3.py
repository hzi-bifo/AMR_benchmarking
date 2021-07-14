#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Rewritten and restyled from the old script by spo12 2013
#     - removing global variables
#     - python3
#     - encapsulating repeated commands into functions
#     - improving readibility for maintenance

import math
import argparse
import sys
import Bio
from Bio import SeqIO
from Bio.Seq import Seq
import logging
from collections import defaultdict
import re
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


def write_aa_table(out_f, SnpDict, nonsyn, head):
    head = head.split("\t")
    head.insert(4, "ref aa")
    head.insert(5, "alt aa")

    logging.info("Writing output table.")
    with open(out_f, "w") as out:
        out.write("\t".join(head))
        if nonsyn == "all":
            for gene in sorted(SnpDict.keys()):
                for item in SnpDict[gene]:
                    out.write("\t".join(item[5][:4] + item[3:5] + item[5][4:])
                              + "\n")
        elif nonsyn == "non-syn":
            for gene in sorted(SnpDict.keys()):
                for item in SnpDict[gene]:
                    if ((item[3] != "none") and (item[4] != "none")
                       and (item[3] != item[4])):
                        out.write("\t".join(item[5][:4] + item[3:5]
                                            + item[4][6:]) + "\n")


def determine_pos(snp, gene, strand, GenDict):
    # Be careful about the positions because division in python3 and python2
    # differ. Python2 division for integers applies floor after the division.
    # For example, in python2:
    # 38/3 == 12
    # However in python3:
    # 38/3 == 12.6666666667
    ref = -1
    alt = -1
    snppos = -1
    start = -1
    if strand == -1:
        ref = str(Seq(snp[1]).complement())
        alt = str(Seq(snp[2]).complement())
        snppos = (snp[0] - GenDict[gene][0]) * -1
        start = 3 * math.floor(snppos / 3)
    else:
        ref = snp[1]
        alt = snp[2]
        snppos = snp[0] - GenDict[gene][0] - 1
        start = 3 * math.floor(snppos / 3)
    end = start + 3
    return(ref, alt, snppos, start, end)


def count_aminoacids(SnpDict, GenDict):
    logging.info("Exchanging amino acids.")
    for gene in SnpDict:
        if gene in GenDict:
            for snp in SnpDict[gene]:
                if len(snp[1]) > 1 or len(snp[2]) > 1:
                    snp.insert(3, "none")
                    snp.insert(4, "none")
                else:
                    strand = GenDict[gene][3]
                    logging.info("gene: %s, strand: %s" % (gene, strand))
                    logging.info("gene fpos: %d, gene lpos: %d"
                                 % (GenDict[gene][0], GenDict[gene][1]))
                    logging.info("SNP pos: %d, reference: %s, alternative: %s"
                                 % (snp[0], snp[1], snp[2]))
                    ref, alt, snppos, start, end = determine_pos(snp,
                                                                 gene,
                                                                 strand,
                                                                 GenDict)
                    gene_seq = GenDict[gene][2]
                    snp_seq = list(gene_seq)
                    snp_seq[snppos] = alt
                    snp_seq = Seq("".join(snp_seq))
                    logging.info(("rel pos: %d, start: %d, end: %d, gene_seq:"
                                 "%s, snp_seq: %s") % (snppos, start, end,
                                 gene_seq[start:end], snp_seq[start:end]))
                    if len(gene_seq[start:end]) % 3 == 0:
                        orig = str(gene_seq[start:end].translate())
                        muta = str(snp_seq[start:end].translate())
                        snp.insert(3, orig)
                        snp.insert(4, muta)
                        logging.info(
                            "orig: %s, muta: %s, orig aa: %s, muta aa %s" % (
                                gene_seq[start:end], snp_seq[start:end],
                                orig, muta))
                    else:
                        logging.warning(("Gene %s contains a partial codon. "
                                        "Make sure that's all right.") % gene)
                        snp.insert(3, "none")
                        snp.insert(4, "none")
        else:
            for snp in SnpDict[gene]:
                snp.insert(3, "none")
                snp.insert(4, "none")


def read_gb(gb_f, SnpDict):
    logging.info("Reading GBK file.")
    GenDict = {}
    gb_fh = LoadFile(gb_f)

    for seq_record in SeqIO.parse(gb_fh, "genbank"):
        sequence = seq_record.seq
        if isinstance(sequence, Bio.Seq.UnknownSeq):
            sys.exit(
                "There seems to be no sequence in your GenBank file!")
        for feature in seq_record.features:
            if (not feature.type == "misc_feature" and
               not feature.type == "unsure"):
                if "locus_tag" in list(feature.qualifiers.keys()):
                    locus = feature.qualifiers["locus_tag"][0]
                    if locus in list(SnpDict.keys()):
                        start = feature.location.start
                        end = feature.location.end
                        strand = feature.location.strand
                        if strand == 1:
                            gene_seq = seq_record.seq[start:end]
                        else:
                            gene_seq = (
                                seq_record.seq[start:end].reverse_complement())
                        GenDict[locus] = [int(start), int(end),
                                          gene_seq, strand]
    gb_fh.close()
    return(GenDict)


def read_snps_tab(table_f):
    Table = defaultdict(list)
    SnpDict = defaultdict(list)
    logging.info("Reading SNP table.")
    head = ''
    with open(table_f) as snp:
        for line in snp:
            if not line.startswith("gene"):
                line = line.rstrip("\n")
                info = line.split("\t")
                gene = info[0].split(",")[0]
                pos = int(info[1])
                ref = info[2]
                alt = info[3]
                Table[gene].append(info)
                SnpDict[gene].append([pos, ref, alt, info])
            else:
                head = line
    return(Table, SnpDict, head)


def main(Args):
    if "." in Args.OutFile:
        logfile = re.sub('[^\.]+$', 'log', Args.OutFile)
    else:
        logfile = "%s.log" % Args.OutFile

    logging.basicConfig(filename=logfile, level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("You have specified that you want %s SNPs." %
                 Args.NonSyn)

    Table, SnpDict, head = read_snps_tab(Args.Table)
    GenDict = read_gb(Args.GbkFile, SnpDict)
    count_aminoacids(SnpDict, GenDict)
    out_f = Args.OutFile
    nonsyn = Args.NonSyn
    write_aa_table(out_f, SnpDict, nonsyn, head)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument("-f", dest="Table", default="",
                        metavar="<mutation table>", required=True,
                        help="""path and name of the already created mutation
                        table""")
    Parser.add_argument("-g", dest="GbkFile",
                        metavar="<genbank>", required=True,
                        help="path and name of the corresponding genbank file")
    Parser.add_argument("-n", dest="NonSyn",
                        choices=['all', 'non-syn'], default="all",
                        help="""specify if you want all SNPs returned or only
                        non-synonymous ones""")
    Parser.add_argument("-o", dest="OutFile",
                        metavar="<filename>", required=True,
                        help="path and name of the output file")

    Args = Parser.parse_args()
    main(Args)
