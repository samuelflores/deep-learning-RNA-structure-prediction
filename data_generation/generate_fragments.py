import argparse
import os

import utils

from classes import Chain, Fragment
from typing import Type


def get_fragments(chains:list[Type[Chain]], fragment_length:int=8) -> list[Type[Fragment]]:
    fragment_extension = int((fragment_length-8)/2)
    fragments = []
    for chain in utils.progress_bar_for(chains):
        for i in range(len(chain)-fragment_length+1):
            clust_id = chain.clust_ids[i + fragment_extension]
            seq_nums, res_names, res_nums, ins_codes = tuple([j[i:i+fragment_length] for j in [chain.seq_nums, chain.res_names, chain.res_nums, chain.ins_codes]])
            fragments += [Fragment(chain.pdb_id, clust_id, chain.chain_id, seq_nums, res_names, res_nums, ins_codes)]
    return fragments


def main(args):

    # Make data folder
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    chains_annotated_filtered = utils.load(args.chains_data)
    
    print(f'Retrieving fragments of length {args.fragment_length}')
    fragments_raw = get_fragments(chains_annotated_filtered, args.fragment_length)
    utils.save(fragments_raw, f'fragments_{args.fragment_length}_raw', args.data_dir, 'pickle')
    
    print('Removing duplicate decoy fragments')
    tloop_fragments = [i for i in fragments_raw if i.clust_id != 0]
    decoy_fragments = [i for i in fragments_raw if i.clust_id == 0]
    decoy_fragments_filtered = utils.filter(decoy_fragments, ['res_names'])
    fragments_filtered = tloop_fragments + decoy_fragments_filtered
    utils.save(fragments_filtered, f'fragments_{args.fragment_length}_filtered', args.data_dir, 'pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chains_data', type=str, default='data/chains_annotated_filtered.pickle')
    parser.add_argument('-d', '--data_dir', type=str, default='data/fragments')
    parser.add_argument('-f', '--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)