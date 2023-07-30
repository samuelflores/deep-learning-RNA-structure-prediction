import argparse
import os

import numpy as np
import numpy.typing as npt
import pandas as pd


# [chain ids]
# [cluster]
# [residue names]
# [residue numbers]
# [insertion codes]


def parse_pdb(filepath: str):
    res_names, res_nums = [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.split()[0] == 'MODEL':
                break
        for line in file:
            if line.split()[0] == 'TER':
                break
            rec_name, res_name, res_num, element = line[:6].strip(), line[17:20].strip(), line[22:26].strip(), line[13:16].strip()
            if rec_name == 'ATOM' and element == 'P': # All bases start with P
                res_names += [res_name]
                res_nums += [res_num]
    clust_nums = [int(filepath.split('/')[-2][1:])]*8
    chain_ids, ins_codes = ['']*8, ['']*8
    return chain_ids, clust_nums, res_names, res_nums, ins_codes


def parse_cif(filepath: str):
    chain_ids, res_names, res_nums, ins_codes = [], [], [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() == '_pdbx_poly_seq_scheme.hetero':
                break
        for line in file:
            if line.strip() == '#':
                break
            data = line.split()
            chain_id, res_name, res_num, ins_code = data[9], data[7], data[5], data[10]
            if len(res_name) > 1 or res_name == '?':
                continue
            chain_ids += [chain_id]
            res_names += [res_name]
            res_nums += [res_num]
            ins_codes += [ins_code]
    clust_nums = [0]*len(res_names)
    return chain_ids, clust_nums, res_names, res_nums, ins_codes


def get_tloop_sequences(clust_dir: str) -> dict[str, list[npt.ArrayLike]]:
    seqs = {}
    for folder in os.listdir(clust_dir):
        for file in os.listdir(f'{clust_dir}/{folder}'):
            pdb_id = file[:4].lower()
            if pdb_id not in seqs.keys():
                seqs[pdb_id] = []
            filepath = f'{clust_dir}/{folder}/{file}'
            arr = np.row_stack(parse_pdb(filepath))
            if not any(np.array_equal(arr, i) for i in seqs[pdb_id]):
                seqs[pdb_id] += [arr]
    return seqs


def get_full_sequences(pdb_ids: list[str], struct_dir: str) -> dict[str, npt.ArrayLike]:
    seqs = {}
    for pdb_id in pdb_ids:
        arr = np.row_stack(parse_cif(f'{struct_dir}/{pdb_id}.cif'))
        seqs[pdb_id] = arr
    return(seqs)


def remove_redundancy(full_seqs: dict[str, npt.ArrayLike]) -> dict[str, npt.ArrayLike]:
    # split array into subarrays by chain name
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


def arrdict_to_df(arrdict: dict) -> pd.DataFrame:
    df = pd.DataFrame(columns=['pdb_id','index','category','values'])
    categories = ['chain_ids', 'clust_nums', 'res_names', 'res_nums', 'ins_codes']
    for pdb_id, value in arrdict.items():
        if type(value) is list:
            for idx, arr in enumerate(value):
                arr_stings = [','.join(i) for i in arr.tolist()]
                entry = pd.DataFrame({'pdb_id': pdb_id, 'index':int(idx), 'category':[categories], 'values':[arr_stings]})
                entry = entry.explode(['category', 'values'])
                df = pd.concat([df, entry], ignore_index=True)
        else:
            arr_stings = [','.join(i) for i in value.tolist()]
            entry = pd.DataFrame({'pdb_id': pdb_id, 'category':[categories], 'values':[arr_stings]})        
            entry = entry.explode(['category', 'values'])
            df = pd.concat([df, entry], ignore_index=True)
    df = df.dropna(axis=1)
    return df


def main(args):
    tloop_seqs = get_tloop_sequences(args.clusters_dir)
    np.savez_compressed('tloop_seqs.npz', **tloop_seqs)
    arrdict_to_df(tloop_seqs).to_csv('tloop_seqs.csv', sep='\t', index=False)
    print('Tetraloop sequences retrieved')

    pdb_ids = tloop_seqs.keys()

    full_seqs = get_full_sequences(pdb_ids, args.structures_dir)
    # full_seqs = remove_redundancy(full_seqs)
    np.savez_compressed('full_seqs.npz', **full_seqs)
    arrdict_to_df(full_seqs).to_csv('full_seqs.csv', sep='\t', index=False)
    print('Full sequences retrieved')

    all_seqs = align_tloops_to_full(tloop_seqs, full_seqs)
    np.savez_compressed('all_seqs.npz', **all_seqs)
    arrdict_to_df(all_seqs).to_csv('all_seqs.csv', sep='\t', index=False)
    print('All sequences retrieved')

    all_frags = get_fragments(all_seqs, args.fragment_length)
    np.savez_compressed('all_frags.npz', **all_frags)
    arrdict_to_df(all_frags).to_csv('all_frags.csv', sep='\t', index=False)
    print('All fragments retrieved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-f', '--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)