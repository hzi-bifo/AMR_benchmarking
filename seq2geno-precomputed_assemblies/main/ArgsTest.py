# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from Bio import SeqIO
from LoadFile import LoadFile


def test_dna_reads_list(f):
    print('Checking the DNA-seq list...')
    # the file should exist
    assert os.path.isfile(f), 'DNA-seq list "{}" not found'.format(f)
    with LoadFile(f) as fh:
        for line in fh.readlines():
            # the two columns
            d = line.strip().split('\t')
            assert len(d) == 2
            print(d[0])
            # the files should be paired
            r_pair_files = d[1].split(',')
            # the listed paths should exist
            assert os.path.isfile(r_pair_files[0])
            assert os.path.isfile(r_pair_files[1])

    return(0)


def test_rna_reads_list(f):
    print('Checking the RNA-seq list...')
    # the file should exist
    assert os.path.isfile(f)
    with LoadFile(f) as fh:
        for line in fh.readlines():
            # the two columns
            d = line.strip().split('\t')
            assert len(d) == 2
            print(d[0])
            # the files should be paired
            r_file = d[1]
            # the listed paths should exist
            assert os.path.isfile(r_file)
    return(0)


def test_reference_seq(f):
    print('Checking the reference sequences...')
    # the file should exist
    assert os.path.isfile(f), 'Reference genome file "{}" not found'.format(f)

    # ensure encoding method
    fh = LoadFile(f)

    # ensure a single sequence formatted in fasta
    seq_dict = SeqIO.to_dict(SeqIO.parse(fh, 'fasta'))
    assert len(seq_dict) == 1

    fh.close()

    return(0)


def test_functions(funcs_d):
    # choices
    choices_dict = {'denovo': ['Y', 'N'],
                    'snps': ['Y', 'N'],
                    'expr': ['Y', 'N'],
                    'phylo': ['Y', 'N'],
                    'de': ['Y', 'N'],
                    'ar': ['Y', 'N']}

    for k in funcs_d:
        # ensure understoodable function
        assert k in choices_dict
        # ensure the choice
        assert funcs_d[k] in choices_dict[k]

    return(0)
