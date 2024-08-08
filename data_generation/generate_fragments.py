import argparse
import os

import utils

from classes import Chain, Fragment
from typing import Type


# * This script will generate the following files (in .pickle format):
#     fragments_x_raw: All fragments of length x that can be generates from the input chains_annotated_filtered.pickle file
#     fragments_x_filtered: fragments_x_raw with duplicate decoy fragments removed


# * Arguments
#     '-c', '--chains_data': .pickle file containing the Chains from which the Fragments should be generated 
#     '-d', '--data_dir': Folder into which the generated files should be placed
#     '-p', '--prefix': Prefix for generated files
#     '-f', '--fragment_length': The desired length of the fragments
#     '-m', '--multi_clust_id': Whether to return a matrix of cluster IDs (indicating position) as opposed to a singular cluster ID per tetraloop (in the center of a sequence)
#! Most of the default values for these args were set up for the folder structure that I used so that I didn't need to re-input every argument every time while testing. If you need to change them then scroll down to the bottom of the script to the if __name__ == '__main__': section


# * Generates a list of Fragment objects along the length of every Chain in the input Chain
# Inputs:
#     chains: list of Chain objects from which to generate Fragments
#     fragment_length: Desired fragment size (in bp)
# Output: list of Fragment objects

def get_fragments(chains:list[Type[Chain]], fragment_length:int=8, multi:bool=True) -> list[Type[Fragment]]:
    fragment_extension = int((fragment_length-8)/2)
    fragments = []
    for chain in utils.progress_bar_for(chains):
        for i in range(len(chain)-fragment_length+1):
            if multi:  
                clust_id = chain.clust_ids[i:i+fragment_length]
                clust_id[-7:] = [0]*7 # Remove cluster IDs for cut-off tetraloops at the end of the sequence
            else:
                clust_id = chain.clust_ids[i + fragment_extension]
            seq_nums, res_names, res_nums, ins_codes = tuple([j[i:i+fragment_length] for j in [chain.seq_nums, chain.res_names, chain.res_nums, chain.ins_codes]])
            fragments += [Fragment(chain.pdb_id, clust_id, chain.chain_id, seq_nums, res_names, res_nums, ins_codes)]
    return fragments


def main(args):

    # Make data folder
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    # Load in Chains file
    chains_annotated_filtered = utils.load(args.chains_data)
    
    print(f'Retrieving fragments of length {args.fragment_length}')
    fragments_raw = get_fragments(chains_annotated_filtered, args.fragment_length, args.multi_clust_id)
    utils.save(fragments_raw, f'{args.prefix}_{args.fragment_length}_raw', args.data_dir, 'pickle')
    
    print('Removing duplicate decoy fragments')
    tloop_fragments = [i for i in fragments_raw if i.clust_id != 0]
    decoy_fragments = [i for i in fragments_raw if i.clust_id == 0]
    decoy_fragments_filtered = utils.filter(decoy_fragments, ['res_names'])
    fragments_filtered = tloop_fragments + decoy_fragments_filtered
    utils.save(fragments_filtered, f'{args.prefix}_{args.fragment_length}_filtered', args.data_dir, 'pickle')
    # utils.save(fragments_filtered, f'fragments_{args.fragment_length}_filtered', args.data_dir, 'csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chains_data', type=str, default='data/chains_annotated_filtered.pickle')
    parser.add_argument('-d', '--data_dir', type=str, default='data/fragments')
    parser.add_argument('-p', '--prefix', type=str, default='fragments')
    parser.add_argument('-f', '--fragment_length', type=int, default=8)
    parser.add_argument('-m', '--multi_clust_id', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)