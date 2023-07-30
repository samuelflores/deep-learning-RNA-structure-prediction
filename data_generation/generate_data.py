import argparse
import os

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import utils


# Sequence array format:
# 0: [chain ids]
# 1: [cluster]
# 2: [residue names]
# 3: [residue numbers]
# 4: [insertion codes]


def get_tloop_sequences(clust_dir: str) -> dict[str, list[npt.ArrayLike]]:
    seqs = {}
    for folder in os.listdir(clust_dir):
        for file in os.listdir(f'{clust_dir}/{folder}'):
            pdb_id = file[:4].lower()
            if pdb_id not in seqs.keys():
                seqs[pdb_id] = []
            filepath = f'{clust_dir}/{folder}/{file}'
            arr = np.row_stack(utils.parse_pdb(filepath))
            if not any(np.array_equal(arr, i) for i in seqs[pdb_id]):
                seqs[pdb_id] += [arr]
    return seqs


def get_full_sequences(pdb_ids: list[str], struct_dir: str) -> dict[str, npt.ArrayLike]:
    seqs = {}
    for pdb_id in pdb_ids:
        arr = np.row_stack(utils.parse_cif(f'{struct_dir}/{pdb_id}.cif'))
        seqs[pdb_id] = arr
    return(seqs)


def remove_redundancy(full_seqs: dict[str, npt.ArrayLike]) -> dict[str, npt.ArrayLike]:
    # split array into subarrays by chain name
    # TODO this
    pass


def align_tloops_to_full(tloop_seqs, full_seqs) -> dict[str, npt.ArrayLike]:
    seqs = full_seqs.copy()
    for pdb_id, full_arr in seqs.items():
        full_resnames, full_resnums = full_arr[2], full_arr[3]
        for tloop_arr in tloop_seqs[pdb_id]:
            tloop_resnames, tloop_resnums = tloop_arr[2], tloop_arr[3]
            possible_idxs = np.where(full_resnums == tloop_resnums[0])[0]
            for idx in possible_idxs:
                idx_resnames = full_resnames[idx:idx+8]
                idx_resnums = full_resnums[idx:idx+8]
                if (
                    len(idx_resnames) == 8 and len(idx_resnums) == 8 and
                    np.all(idx_resnames == tloop_resnames) and np.all(idx_resnums == tloop_resnums)
                    ):
                    full_arr[1, idx] = tloop_arr[1, 0]
    return seqs


# TODO is this correct? even if the tloop is off-center, should the whole fragment still be counted as a tloop?
def get_fragments(all_seqs, frag_len: int = 8) -> dict[str, list[npt.ArrayLike]]:
    frag_extension = int((frag_len-8)/2)
    seqs = {}
    for pdb_id, arr in all_seqs.items():
        frags = []
        for i in range(len(arr[0])-frag_len+1):
            frag = arr[:,i:i+frag_len].copy() # Assignment by value, not reference
            if len(np.unique(frag[0])) > 1 or len(frag[0]) < frag_len: # Remove chain-crossing fragments
                continue
            frag[1,:] = frag[1, frag_extension] # Replace all cluster numbers with cluster ID, looking at offset
            frags += [frag]
        seqs[pdb_id] = frags
    return seqs


def main(args):
    # tloop_seqs = get_tloop_sequences(args.clusters_dir)
    # np.savez_compressed('tloop_seqs.npz', **tloop_seqs)
    # utils.arrdict_to_df(tloop_seqs).to_csv('tloop_seqs.csv', sep='\t', index=False)
    # print('Tetraloop sequences retrieved')

    # pdb_ids = tloop_seqs.keys()

    # full_seqs = get_full_sequences(pdb_ids, args.structures_dir)
    # # full_seqs = remove_redundancy(full_seqs)
    # np.savez_compressed('full_seqs.npz', **full_seqs)
    # utils.arrdict_to_df(full_seqs).to_csv('full_seqs.csv', sep='\t', index=False)
    # print('Full sequences retrieved')

    # all_seqs = align_tloops_to_full(tloop_seqs, full_seqs)
    # np.savez_compressed('all_seqs.npz', **all_seqs)
    # utils.arrdict_to_df(all_seqs).to_csv('all_seqs.csv', sep='\t', index=False)
    # print('All sequences retrieved')

    # all_frags = get_fragments(all_seqs, args.fragment_length)
    # np.savez_compressed('all_frags.npz', **all_frags)
    # print('All fragments retrieved')

    all_frags = np.load('all_frags.npz')
    make_tf_dataset(all_frags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-f', '--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)