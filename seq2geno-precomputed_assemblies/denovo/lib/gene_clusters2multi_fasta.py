#!/usr/bin/env python2
import os
import sys

def read_clustered_proteins(clustered_proteins):
    import re
    """read in clustered roary proteins and make a dictionary from that"""
    with open(clustered_proteins, 'r') as cp:
        family2protein = {} 
        protein2family = {}
        for l in cp:
            fields = l.strip().split("\t")
            family_id, field0 = [i.strip() for i in fields[0].split(":")]
            #family_id = family_id.replace("/", "_")
            family_id = re.sub('\W', '_', family_id)
            family2protein[family_id] = [field0] + fields[1:]
            for i in family2protein[family_id]:
                if i not in protein2family:
                    protein2family[i] = family_id 
    return family2protein, protein2family

def proteins2multi_fasta(family2protein, protein2family, fasta, out_dir):
    """read in all fasta files and distribute proteins into gene family specific multi fastas"""
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    family2seqs= {}
    for f in fasta:
        print f
        with open(f, 'r') as fasta_open:
            current_seq = []
            fasta_header = None
            for l in fasta_open:
                if l.startswith(">"):
                    if not len(current_seq) == 0:
                        #write current protein
                        if protein_id not in protein2family:
                            #sys.stderr.write("%s\n" % protein_id)
                            pass
                        else:
                            if not protein2family[protein_id] in family2seqs:
                                family2seqs[protein2family[protein_id]] = ["".join([fasta_header] + current_seq)]
                            else:
                                family2seqs[protein2family[protein_id]].append("".join([fasta_header] + current_seq))
                        current_seq = [] 
                    fasta_header = l
                    protein_id = l.split(" ")[0][1:]
                else:
                    current_seq.append(l)
            #last protein in fasta
            if protein_id not in protein2family:
                #sys.stderr.write("%s\n" % protein_id)
                continue
            if not protein2family[protein_id] in family2seqs:
                family2seqs[protein2family[protein_id]] = ["".join([fasta_header] + current_seq)]
            else:
                family2seqs[protein2family[protein_id]].append("".join([fasta_header] + current_seq))
    for f in family2seqs:
        with open("%s/%s.fasta" % (out_dir, f), 'w') as fo:
                fo.write("".join(family2seqs[f]))

def run(out_dir, clustered_proteins, fasta):
    family2protein, protein2family = read_clustered_proteins(clustered_proteins) 
    proteins2multi_fasta(family2protein, protein2family, fasta, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("extract amino acid or nucleotide acid sequences from the Roary and Prokka output")
    #parser.add_argument("fasta_type", choose = ["nucleotide", "amino_acid"], help='choose between nucleotide and amino acid input')
    parser.add_argument("out_dir", help="directory for output fast (will be created if it doesn't exist)")
    parser.add_argument("clustered_proteins", help='Roary output of clustered proteins')
    parser.add_argument("fasta", nargs = "*",  help='gene / aa or nt fasta')
    args = parser.parse_args()
    run(**vars(args))
