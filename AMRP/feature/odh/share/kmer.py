# -*- coding: utf-8 -*-
# @Author: Den
# @Date:   2020-12-26 20:05:55
# @Last Modified by:   Den
# @Last Modified time: 2020-12-29 11:28:18

import gzip
import itertools
import numpy as np
from mimetypes import guess_type
from pathlib import Path
from functools import partial
from collections import OrderedDict
from share.seq_parser import seq_parser
from share.canonicam_kmer import canonical_kmer
from multiprocessing import Pool

# seqs = ['ACTGTTTATATCTATCGATTT']  # , 'AGTCCGCGATGCTATCGATCGATTTAAA']
# max_dist = 5


def get_seq_format(seq_file):
    fa_exts = [".fasta", ".fa", ".fna", ".fas"]
    fq_exts = [".fq", ".fastq"]
    encoding = guess_type(seq_file)[1]  # uses file extension
    if encoding is None:
        encoding = ""
    elif encoding == "gzip":
        encoding = "gz"
    else:
        raise ValueError('Unknown file encoding: "{}"'.format(encoding))
    seq_filename = Path(
        seq_file).stem if encoding == 'gz' else Path(seq_file).name
    seq_file_ext = Path(seq_filename).suffix
    if seq_file_ext not in (fa_exts + fq_exts):
        raise ValueError("""Unknown extension {}. Only fastq and fasta sequence formats are supported.
And the file must end with one of ".fasta", ".fa", ".fna", ".fas", ".fq", ".fastq"
and followed by ".gz" or ".gzip" if they are gzipped.""".format(seq_file_ext))
    seq_format = "fa" + encoding if seq_file_ext in fa_exts else "fq" + encoding
    return seq_format


def kmer(seq_file, k=3, canonical=True):
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith(
        "gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"

    if canonical:
        kmers, kmer_map = canonical_kmer(k)

        kmer_count = OrderedDict({kmer: 0 for kmer in kmers})
        with _open(seq_file) as fh:
            for record in seq_parser(fh, seq_type):
                seq = record[1]
                seq_len = len(seq)
                for pos in range(seq_len - k + 1):
                    try:
                        kmer_count[kmer_map[seq[pos:pos + k]]] += 1
                    except KeyError:
                        continue

    else:
        vocab = 'ACGT'
        kmers = (''.join(nucleotides)
                 for nucleotides in itertools.product(vocab, repeat=k))

        kmer_count = OrderedDict({kmer: 0 for kmer in kmers})

        with _open(seq_file) as fh:
            for record in seq_parser(fh, seq_type):
                seq = record[1]
                seq_len = len(seq)
                for pos in range(seq_len - k + 1):
                    try:
                        kmer_count[seq[pos:pos + k]] += 1
                    except KeyError:
                        continue

    return kmer_count


def kmer_array(seq_file, k=3, canonical=True):
    return np.array(list(kmer(seq_file, k=k, canonical=canonical).values())).reshape(-1)
