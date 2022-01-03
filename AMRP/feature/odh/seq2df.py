import click
import pandas as pd
from functools import partial
from multiprocessing import Pool

from share.odh import odh_array
from share.kmer import kmer_array


@click.command()
@click.argument('seq_file_list', type=click.Path(exists=True))
@click.option('-k', '--ksize', type=int, help='Kmer size', default=3)
@click.option('-m', '--min_distance', type=int, help='Mimimum distance', default=0)
@click.option('-d', '--distance', type=int, help='Maximum distance for ODH (must be set for ODH)', default=None)
@click.option('--canonical', is_flag=True, default=False)
@click.option('--header', is_flag=True, default=False)
@click.option('-t', '--threads', type=int, help='Number of threads to use', default=20)
@click.option('-o', '--out', type=str, help='Output file', default=None)
def seq_to_df(seq_file_list, ksize, min_distance, distance, canonical, header, threads, out):

    if out is None:
        out = seq_file_list.rsplit(
            '.', 1)[0] + '.k{}m{}d{}.hd5'.format(ksize, min_distance, distance)

    genome_names = []
    seq_files = []
    with open(seq_file_list, 'r') as fh:
        if header:
            _ = fh.readline()
        for line in fh:
            line = line.strip()
            genome_name, seq_file = line.split('\t')

            genome_names.append(genome_name.strip())
            seq_files.append(seq_file.strip())

    print('Loading sequences and generating features...')

    # Kmer count
    if distance is None:
        with Pool(threads) as p:
            features = p.map(partial(kmer_array, k=ksize,
                                     canonical=canonical), seq_files)
    # ODH count
    else:
        with Pool(threads) as p:
            features = p.map(partial(odh_array, k=ksize, min_dist=min_distance,
                                     max_dist=distance), seq_files)

    df = pd.DataFrame(features, index=genome_names)
    df.to_hdf(out, key='df', mode='w', complevel=9)


seq_to_df()
