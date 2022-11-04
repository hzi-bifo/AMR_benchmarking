#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

'''
Rewritten and restyled from the old script by spo12 2013
    - removing global variables
    - python3
    - encapsulating repeated commands into functions
    - improving readibility for maintenance
'''

import re
import os
import argparse
from collections import defaultdict


def parse_annot(anno_f, size):
    with open(anno_f) as anno:
        if not size:
            head = anno.readline()
            finfo = head.split("|")
            try:
                size = int(finfo[-1])
            except ValueError:
                print("It seems I can't find the genome "
                      "size in the annotation file!")
                print("Please enter it manually with "
                      "the argument -s <number>.")

            '''
            The coordinates in this script are pseudo 1-based, so that the
            1-based coordinates of genes can fit this python script.
            Therefore, it should be the following line instead of
            the genome size described in the annot file that helps to
            adjust the coordinates. Otherwise it looks confusing and hard to
            maintain it in the future.
            '''
            genome = [0]*(size+1)
            for line in anno:
                if not line.startswith("@"):
                    line = line.rstrip()
                    if line:
                        info = line.split("\t")
                        gene = info[3]
                        try:
                            name = info[8]
                        except IndexError:
                            name = ""
                        start = int(info[5])
                        end = int(info[6])
                        for i in range(start, end):
                            if not name:
                                genome[i] = gene
                            else:
                                genome[i] = gene + "," + name

            return(genome)


def reads(Args, mutations, prob, pos, ref, alt, name, qual, flat):
    if not Args.Region:
        region(Args, mutations, prob, pos, ref, alt, name, qual)
    else:
        flt_pos = int(pos)*6
        flat.seek(flt_pos, 0)
        string = flat.read(6)
        if string:
            region(Args, mutations, prob, pos, ref, alt, name, qual)


def region(Args, mutations, prob, pos, ref, alt, name, qual):
    if Args.Region:
        first = int(region.split("-")[0])
        last = int(region.split("-")[1])

    if not Args.Region:
        score(Args, mutations, prob, pos, ref, alt, name, qual)
    else:
        if int(pos) >= first and int(pos) <= last:
            score(Args, mutations, prob, pos, ref, alt, name, qual)


def score(Args, mutations, prob, pos, ref, alt, name, qual):
    if Args.Score:
        force_homozygous(Args, mutations, prob, pos, ref, alt, name, qual)
    else:
        if float(qual) >= Args.Score:
            force_homozygous(Args, mutations, prob, pos, ref, alt, name, qual)


def force_homozygous(Args, mutations, prob, pos, ref, alt, name, qual):
    if Args.force_homozygous:
        if prob.startswith("0/1") and len(ref.split(",")) == 1:
            probs = prob.split(":")[1].split(",")
            if int(probs[2]) < int(probs[0]):
                assign_key(mutations, pos, ref, alt, name, qual)
        else:
            assign_key(mutations, pos, ref, alt, name, qual)
    else:
        assign_key(mutations, pos, ref, alt, name, qual)


def assign_key(mutations, pos, ref, alt, name, qual):
    key = pos + "_" + ref + "_" + alt
    mutations[key].append((name, qual))


def determine_mutations(Args, dict_f):
    names = set()
    mutations = defaultdict(list)
    with open(dict_f) as infile:
        for line in infile:
            name = line.rstrip()
            names.add(name)
            # filename = name +".raw.vcf"
            filename = name + ".flt.vcf"
            flat = None
            if Args.Region:
                flatcount = name + ".flatcount"
                flat = open(flatcount)
            with open(filename) as vcf:
                for line in vcf:
                    if not line.startswith("#"):
                        line = line.rstrip()
                        info = line.split("\t")
                        pos = info[1]
                        ref = info[3]
                        alt = info[4]
                        qual = info[5]
                        prob = info[9]
                        reads(Args, mutations, prob, pos, ref, alt,
                              name, qual, flat)
            if Args.Region:
                flat.close()

    print(list(mutations.keys())[:5])
    return(mutations, names)


#  empty samples list
def confirm_nr(item, mut2sample, mutations, filename):
    with open(filename, 'r') as flat:
        for key in mutations:
            # muts = []
            if item not in mut2sample[key]:
                pos = int(key.split("_")[0])*6
                flat.seek(pos, 0)
                string = flat.read(6)
                if string:
                    number = int(string)
                    if number == 0:
                        mutations[key].append((item, "NR"))
                    else:
                        mutations[key].append((item, ""))


def add_intergenic(mutations, out_f, genome, names):
    with open(out_f, "w") as out:
        out.write("gene\tpos\tref\talt")
        for item in sorted(names):
            try:
                name = item.split("/")[-1]
            except IndexError:
                name = item
            out.write("\t" + name)
        out.write("\n")
        for key in sorted(mutations):
            info = key.split("_")
            pos = info[0]
            ref = info[1]
            alt = info[2]
            gene = genome[int(pos)]
            if gene == 0:
                out.write("intergenic\t" + pos + "\t" + ref + "\t" + alt)
            else:
                out.write(gene + "\t" + pos + "\t" + ref + "\t" + alt)
            for item in sorted(mutations[key]):
                if item[0] in names:
                    out.write("\t" + item[1])
            out.write("\n")


def main(Args):
    genome = parse_annot(Args.AnnoFile, Args.Size)
    mutations, names = determine_mutations(Args, Args.DictFile)
    print(list(mutations.keys())[:5])
    if Args.restrict_samples:
        with open(Args.restrict_samples) as f:
            names = set([s.strip() for s in f])
    if len([sample for sample in names if re.search('\w', sample)]) >= 1:
        mut2sample = {}
        for mut in mutations:
            mut2sample[mut] = set()
            for x in mutations[mut]:
                mut2sample[mut].add(x[0])
        for item in names:
            filename = item + ".flatcount"
            confirm_nr(item, mut2sample,  mutations, filename)
        add_intergenic(mutations, Args.OutFile, genome, names)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("-f", dest="DictFile", default="dictionary.txt",
                        metavar="<dictionary>",
                        help="""
                        path and name of the dictionary file with
                        all filenames (without extensions!)""")
    Parser.add_argument("-a", dest="AnnoFile", metavar="<annotation>",
                        required=True,
                        help="""path and name of the corresponding annotation
                        file""")
    Parser.add_argument("-g", dest="Size", metavar="<genome size>", type=int,
                        help="""(approximate) size of the reference
                        genome, better too big than too small""")
    Parser.add_argument("-c", dest="Reads",
                        metavar="<coverage/number of reads>", type=int,
                        default=1,
                        help="""minimum number of reads required at each SNP
                        position""")
    Parser.add_argument("-s", dest="Score", metavar="<SNP score>",
                        type=int, default=1,
                        help="minimum SNP score")
    Parser.add_argument("-r", dest="Region", metavar="<genomic region>",
                        default=False,
                        help="""genomic region of interest (instead of
                        whole genome), format: start-end""")
    Parser.add_argument("-o", dest="OutFile", default="all_mutations.tab",
                        metavar="<filename>",
                        help="path and name of the output file")
    Parser.add_argument("--force_homozygous", action="store_true",
                        help="""if set, convert heterozygous to
                        the next most likely SNP call""")
    Parser.add_argument("--restrict_samples",
                        help="restrict to these samples for coverage check")
    Args = Parser.parse_args()
    main(Args)
