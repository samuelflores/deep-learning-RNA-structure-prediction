import argparse
import ast
import json
import os
import random

import numpy as np
import numpy.typing as npt
import pandas as pd

from Bio import pairwise2 as pw
from Bio.PDB import PDBList, FastMMCIFParser
from pdbecif.mmcif_tools import MMCIF2Dict


# weird tloops: ['3a3a', '3adb', '3adc', '3hl2', '3rg5', '3w3s', '4rqf']

# DONE parse every non-duplicate .align.pdb ATOM file for the sequence and residue numbers
# DONE check that a) the chain is long enough to contain the residue number b) the first residues and number align c) the whole sequence matches
# DONE allow for residue number skipping in decoy generation. generate decoys only from the chain from which the tetraloop originates
# DONE (not really, i just don't touch the original data) update chain ID in bottaro's dataset

# 1. get tloop data from clusters folder
# 2. get all sequences of all chains
# 3. combine tloop data and sequences into one large matrix


def get_tloop_sequences(clusters_folder: str) -> pd.DataFrame:
    sequences = pd.DataFrame(columns=['pdb_id','resnames','resnums','cluster'])
    for folder in os.listdir(clusters_folder):
        cluster = int(folder[1:])
        for file in os.listdir(f'{clusters_folder}/{folder}'):
            pdb_id = file[:4].lower()
            resnames = ''
            resnums = []
            filepath = f'{clusters_folder}/{folder}/{file}'
            with open(filepath, 'r') as f:
                for line in f:
                    record_name = line[:6].strip()
                    atom_name = line[13:16].strip()
                    if record_name == 'ATOM' and atom_name == 'P':
                        resnames += line[17:20].strip()
                        resnums += [int(line[22:26].strip())]
            resnums = tuple(resnums)
            entry = pd.DataFrame({
                'pdb_id': pdb_id,
                'resnames': resnames,
                'resnums': [resnums],
                'cluster': cluster
            })
            sequences = pd.concat([sequences, entry], ignore_index=True)
    sequences = sequences.drop_duplicates()
    return sequences


# TODO make this faster somehow. it takes like 20+ min right now. switch this out to use PDBeCIF instead
def get_all_sequences(pdb_ids: list[str], structures_folder: str) -> pd.DataFrame:
    parser = FastMMCIFParser(QUIET=True)
    sequences = pd.DataFrame(columns=['pdb_id','chain_id','resnames','resnums','icodes'])
    for pdb_id in pdb_ids:
        structure = parser.get_structure(pdb_id, f'{structures_folder}/{pdb_id}.cif')
        for chain in structure.get_chains():
            resnames = ''
            resnums = []
            icodes = []
            for res in chain.get_residues():
                if len(res.resname) > 1: # TODO is this stringent enough
                    continue
                resnames += res.resname
                resnums += [res.id[1]] # https://biopython.org/docs/1.76/api/Bio.PDB.Entity.html
                icodes += [res.id[2]]
            if resnames:
                resnums = tuple(resnums)
                icodes = tuple(icodes)
                entry = pd.DataFrame({
                    'pdb_id': pdb_id,
                    'chain_id': chain.id,
                    'resnames': resnames,
                    'resnums': [resnums],
                    'icodes': [icodes]
                })
                sequences = pd.concat([sequences, entry], ignore_index=True)
        print('Retrieved sequence for ' + pdb_id)
    return sequences


def remove_redundancy(all_seqs: pd.DataFrame) -> pd.DataFrame:
    all_seqs = all_seqs.drop_duplicates(['pdb_id','resnames', 'resnums','icodes']) # * For some reason, many of the .cif files contain a lot of duplicate chains

    pdb_id_dfs = [group.sort_values('resnames', key=lambda x: x.str.len(), ascending=False) for _, group in all_seqs.groupby('pdb_id')]
    
    # collect all chains under single pdb id
    # sort chains in order of length
    # for chain in chains
    # compare all other chains to longest chain. if best alignment > 90% identity, delete said chain
    return all_seqs


def make_tloop_matrices(tloop_seqs: pd.DataFrame, all_seqs: pd.DataFrame, frag_len: int = 8) -> dict[str, npt.ArrayLike]:
    data = {}
    frag_extension = int((8-frag_len)/2)
    for seq in all_seqs.itertuples(index=False):
        clusters = np.zeros((len(seq.resnames),), dtype=int)
        matching_tloops = tloop_seqs[tloop_seqs.iloc[:, 0].str.fullmatch(seq.pdb_id, case=False)]
        for tloop in matching_tloops.itertuples(index=False):
            tloop_resnum = tloop.resnums[0]
            seq_idxs = [i for i, x in enumerate(seq.resnums) if x == tloop_resnum]
            if seq_idxs: # If the sequence is long enough
                for idx in seq_idxs:
                    seq_resnames = seq.resnames[idx:idx + 8]
                    seq_resnums = seq.resnums[idx:idx + 8]
                    if tloop.resnames == seq_resnames and tloop.resnums == seq_resnums:
                        clusters[idx-frag_extension] = tloop.cluster # Depending on the fragment length, mark the tetraloop further back
        # Convert cluster, sequence, residue number, and insertion code data into matrix
        key = f'{seq.pdb_id}_{seq.chain_id}'
        matrix = np.row_stack((list(seq.resnames), seq.resnums, seq.icodes, clusters))
        data[key] = matrix
    return data


def convert_matrices_to_df(all_matrices: dict[str, npt.ArrayLike]) -> None:
    df = pd.DataFrame(columns=['pdb_id','chain_id','residues'])
    df['pdb_id'] = [i.split('_')[0] for i in all_matrices.keys()]
    df['chain_id'] = [i.split('_')[1] for i in all_matrices.keys()]
    res_lists = []
    for array in all_matrices.values():
        res_lists += [[','.join(i) for i in array.tolist()]]
    df['residues'] = res_lists
    df = df.explode('residues')
    return df


def get_all_fragments(matrices: dict[str, npt.ArrayLike], frag_len: int = 8) -> dict[str, npt.ArrayLike]:
    all_fragments = {}
    for chain in chains.itertuples(index=False):
        print(f'Fragmenting chain {chain.pdb_id}_{chain.chain_id}')
        fragment_seqs = [chain.sequence[i:i+fragment_length] for i in range(len(chain.sequence) - fragment_length + 1)]
        fragment_resnums = [chain.residue_numbers[i:i+fragment_length] for i in range(len(chain.residue_numbers) - fragment_length + 1)]
        entry = pd.DataFrame({'pdb_id': chain.pdb_id, 'chain_id': chain.chain_id, 'sequence': fragment_seqs, 'residue_numbers': fragment_resnums})
        all_fragments = pd.concat([all_fragments, entry], ignore_index=True)
    return all_fragments


# TODO make tensorflow dataset


# def get_tloop_chains(tloop_fragments: pd.DataFrame, all_chains: pd.DataFrame) -> pd.DataFrame:
#     tloop_chains = pd.DataFrame(columns=['pdb_id','chain_id','sequence', 'residue_numbers'])

#     for tloop in tloop_fragments.itertuples(index=False):
#         tloop_resnum = tloop.residue_numbers[0]
#         chains = all_chains[all_chains.iloc[:, 0].str.fullmatch(tloop.pdb_id, case=False)]
#         for chain in chains.itertuples(index=False):
#             try:
#                 chain_idx = chain.residue_numbers.index(tloop_resnum)
#             except ValueError:
#                 continue
#             chain_seq = chain.sequence[chain_idx:chain_idx+8]
#             chain_resnums = chain.residue_numbers[chain_idx:chain_idx+8]
#             if tloop.sequence == chain_seq and tloop.residue_numbers == chain_resnums:
#                 entry = pd.DataFrame({'pdb_id': tloop.pdb_id, 'chain_id': chain.chain_id, 'sequence': chain.sequence, 'residue_numbers': [chain.residue_numbers]})
#                 tloop_chains = pd.concat([tloop_chains, entry], ignore_index=True)
    
#     tloop_chains = tloop_chains.drop_duplicates(['pdb_id','sequence','residue_numbers'])
#     return tloop_chains


# def drop_tloops(all_fragments: pd.DataFrame, tloop_fragments:pd.DataFrame) -> pd.DataFrame:
#     merged_fragments = all_fragments.merge(tloop_fragments, how='left', on=['pdb_id','sequence','residue_numbers'], indicator=True)
#     decoy_fragments = merged_fragments[merged_fragments['_merge'] == 'left_only'].drop(columns=['_merge','cluster'])
#     return decoy_fragments


# def encode_sequences(sequences: list[str], residue_map: dict[str, int] = {'A':0,'U':1,'C':2,'G':3}) -> list[str]:
#     encoded_sequences = []
#     for seq in sequences:
#         seq_array = np.array([residue_map[i] for i in seq])
#         encoded_array = np.zeros((seq_array.size, seq_array.max()+1), dtype=int)
#         encoded_array[np.arange(seq_array.size), seq_array] = 1
#         encoded_sequences += [encoded_array]
#     return encoded_sequences


# # 1:3 split of encoded matrices (all, including decoys and tloops) into test and training data
# # two modes: unannotated (labels not saved) and annotated (labels saved)
# # TODO i'm guessing each datapoint in the matrices list should have a corresponding item in the labels list. e.g. decoys are labeled 0, and tloop are labeled with whatever cluster they originate from
# # TODO finish type hints. how to do it for numpy datatypes?
# def split_dataset(matrices: list, labels: list[str], ratio: float = 1/3) -> tuple:
#     # Randomize item order of both arrays in unison
#     combined_lists = list(zip(matrices, labels))
#     shuffled_combined = random.sample(combined_lists, len(combined_lists))
#     matrices_shuffled, labels_shuffled = zip(*shuffled_combined)

#     split_index = int((len(matrices) + 1) * ratio)

#     test_matrices = matrices_shuffled[:split_index]
#     train_matrices = matrices_shuffled[split_index:]
#     test_labels = labels_shuffled[:split_index]
#     train_labels = labels_shuffled[split_index:]

#     return (test_matrices, test_labels), (train_matrices, train_labels)


def main(args):

    # # load existing data
    # tloop_sequences = pd.read_csv('tloop_sequences.csv', sep='\t')
    # tloop_sequences['resnums'] = tloop_sequences['resnums'].apply(ast.literal_eval)

    # pdb_ids = list(set(tloop_sequences['pdb_id'].to_list()))

    # all_sequences = pd.read_csv('all_sequences.csv', sep='\t')
    # all_sequences['resnums'] = all_sequences['resnums'].apply(ast.literal_eval)
    # all_sequences['icodes'] = all_sequences['icodes'].apply(ast.literal_eval)

    # all_matrices = np.load('all_matrices_8.npz')

    ###################################

    # # tloop sequences
    # tloop_sequences = get_tloop_sequences(args.clusters_folder)
    # tloop_sequences.to_csv('tloop_sequences.csv', sep='\t', index=False)

    # # unique PDB ids
    # pdb_ids = list(set(tloop_sequences['pdb_id'].to_list()))

    # # download mmcif files
    # if args.download_structures:
    #     PDBList().download_pdb_files(pdb_ids, pdir=args.structures_folder, obsolete=True, overwrite=True)
    
    # # all sequences
    # all_sequences = get_all_sequences(pdb_ids, args.structures_folder)
    all_sequences = get_all_sequences(['4yb0'], args.structures_folder)
    # all_sequences = remove_redundancy(all_sequences[:10])
    # all_sequences.to_csv('all_sequences.csv', sep='\t', index=False)
    
    # # all_matrices
    # all_matrices = make_tloop_matrices(tloop_sequences, all_sequences, args.fragment_length)
    # np.savez(f'all_matrices_{args.fragment_length}.npz', **all_matrices)
    # convert_matrices_to_df(all_matrices).to_csv(f'all_matrices_{args.fragment_length}.csv', sep='\t', index=False)


    # # encode sequences
    # sequences = tloop_fragments['sequence'].to_list()
    # encoded_sequences = encode_sequences(sequences)
    # labels = ['l']*len(encoded_sequences)
    # (test_matrices, test_labels), (training_matrices, training_labels) = split_dataset(encoded_sequences, labels)
    
    # # export and load .npz files
    # np.savez('matrices_1.npz', *test_matrices)
    # with np.load('matrices_1.npz') as data:
    #     for i in data:
    #         print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clusters_folder', type=str)
    parser.add_argument('structures_folder', type=str)
    parser.add_argument('--fragment_length', type=int, default=8)
    parser.add_argument('--download_structures', type=bool, default=False)
    args = parser.parse_args()
    main(args)