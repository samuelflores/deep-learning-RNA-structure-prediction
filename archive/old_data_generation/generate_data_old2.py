import argparse
import os

import numpy as np
import numpy.typing as npt
import utils_old

from Bio import pairwise2 as pw2


# Sequence array format:
# 0: [sequence numbers]
# 1: [chain ids]
# 2: [cluster]
# 3: [residue names]
# 4: [residue numbers]
# 5: [insertion codes]


def get_tloop_sequences(clust_dir: str) -> dict[str, list[npt.ArrayLike]]:
    seqs = {}
    for folder in os.listdir(clust_dir):
        for file in os.listdir(f'{clust_dir}/{folder}'):
            pdb_id = file[:4].lower()
            if pdb_id not in seqs.keys():
                seqs[pdb_id] = []
            filepath = f'{clust_dir}/{folder}/{file}'
            arr = np.row_stack(utils_old.parse_pdb(filepath))
            if not any(np.array_equal(arr, i) for i in seqs[pdb_id]):
                seqs[pdb_id] += [arr]
    return seqs


def get_full_sequences(pdb_ids: list[str], struct_dir: str) -> dict[str, npt.ArrayLike]:
    seqs = {}
    for pdb_id in pdb_ids:
        arr = np.row_stack(utils_old.parse_cif(f'{struct_dir}/{pdb_id}.cif'))
        seqs[pdb_id] = arr
    return(seqs)


def remove_duplicates(full_seqs: dict[str, npt.ArrayLike], percentage_id_limit: float = 0.9) -> dict[str, npt.ArrayLike]:
    for pdb_id, arr in full_seqs.items():

        # Separate full PDB array into subarrays by chain ID
        chains = []
        chain_ids = np.unique(arr[1])
        for id in chain_ids:
            chain_arr = arr[:, np.where(arr[1] == id)[0]]
            if not any(np.array_equal(chain_arr, i[1]) for i in chains): # Filter out identical chains
                chains += [(id, chain_arr)]
        
        # Sort chains by length
        chains.sort(key=lambda x: len(x[1][3]), reverse=True)
        
        # Remove chains with high enough percentage identity
        i = 0
        while i < len(chains):
            j = i + 1
            while j < len(chains):
                seq1 = ''.join(chains[i][1][3])
                seq2 = ''.join(chains[j][1][3])
                alignments = pw2.align.globalxx(seq1, seq2)
                if alignments:
                    seq1_aligned, seq2_aligned, score, start, end = alignments[0]
                    alignment_len = end - start
                    num_matches = sum(a == b for a, b in zip(seq1_aligned, seq2_aligned))
                    percentage_id = (num_matches / alignment_len)
                    if percentage_id > percentage_id_limit:
                        chains.pop(j)
                        continue # Keep pointer position
                j += 1
            i += 1
        
        chains_concat = np.concatenate([i[1] for i in chains], axis=1)
        full_seqs[pdb_id] = chains_concat
    
    return full_seqs


def align_tloops_to_full(tloop_seqs: dict[str, list[npt.ArrayLike]], full_seqs: dict[str, npt.ArrayLike]) -> dict[str, npt.ArrayLike]:
    seqs = full_seqs.copy()
    for pdb_id, full_arr in seqs.items():
        full_resnames, full_resnums = full_arr[3], full_arr[4]
        for tloop_arr in tloop_seqs[pdb_id]:
            tloop_resnames, tloop_resnums = tloop_arr[3], tloop_arr[4]
            possible_idxs = np.where(full_resnums == tloop_resnums[0])[0]
            for idx in possible_idxs:
                idx_resnames = full_resnames[idx:idx+8]
                idx_resnums = full_resnums[idx:idx+8]
                if (
                    len(idx_resnames) == 8 and len(idx_resnums) == 8 and
                    np.all(idx_resnames == tloop_resnames) and np.all(idx_resnums == tloop_resnums)
                    ):
                    full_arr[2, idx] = tloop_arr[2, 0]
    return seqs


# TODO is this correct? even if the tloop is off-center, should the whole fragment still be counted as a tloop?
def get_fragments(all_seqs, frag_len: int = 8) -> dict[str, list[npt.ArrayLike]]:
    frag_extension = int((frag_len-8)/2)
    seqs = {}
    for pdb_id, arr in all_seqs.items():
        frags = []
        for i in range(len(arr[0])-frag_len+1):
            frag = arr[:,i:i+frag_len].copy() # Assignment by value, not reference
            if len(np.unique(frag[1])) > 1 or len(frag[0]) < frag_len: # Remove chain-crossing fragments
                continue
            frag[2,:] = frag[2, frag_extension] # Replace all cluster numbers with cluster ID, looking at offset
            frags += [frag]
        seqs[pdb_id] = frags
    return seqs


def main(args):
    tloop_seqs = get_tloop_sequences(args.clusters_dir)
    np.savez_compressed('tloop_seqs.npz', **tloop_seqs)
    utils_old.arrdict_to_df(tloop_seqs).to_csv('tloop_seqs.csv', sep='\t', index=False)
    print('Tetraloop sequences retrieved')

    pdb_ids = tloop_seqs.keys()

    full_seqs = get_full_sequences(pdb_ids, args.structures_dir)
    full_seqs = remove_duplicates(full_seqs)
    np.savez_compressed('full_seqs.npz', **full_seqs)
    utils_old.arrdict_to_df(full_seqs).to_csv('full_seqs.csv', sep='\t', index=False)
    print('Full sequences retrieved')

    all_seqs = align_tloops_to_full(tloop_seqs, full_seqs)
    np.savez_compressed('all_seqs.npz', **all_seqs)
    utils_old.arrdict_to_df(all_seqs).to_csv('all_seqs.csv', sep='\t', index=False)
    print('All sequences retrieved')

    all_frags = get_fragments(all_seqs, args.fragment_length)
    np.savez_compressed(f'all_frags_{args.fragment_length}.npz', **all_frags)
    print('All fragments retrieved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-fl', '--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)