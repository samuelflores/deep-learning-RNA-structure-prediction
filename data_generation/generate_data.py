import argparse
import os

import numpy as np
import numpy.typing as npt
import pandas as pd


def parse_pdb(filepath: str):
    res_names, res_seqs = [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.split()[0] == 'MODEL':
                break
        for line in file:
            if line.split()[0] == 'TER':
                break
            rec_name, res_name, res_seq, element = line[:6].strip(), line[17:20].strip(), int(line[22:26].strip()), line[13:16].strip()
            if rec_name == 'ATOM' and element == 'P':
                res_names += [res_name]
                res_seqs += [res_seq]
    return res_names, res_seqs


def parse_cif(filepath: str):
    pdb_strand_ids, pdb_mon_ids, pdb_seq_nums, pdb_ins_codes = [], [], [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() == '_pdbx_poly_seq_scheme.hetero':
                break
        for line in file:
            if line.strip() == '#':
                break
            data = line.split()
            pdb_strand_id, pdb_mon_id, pdb_seq_num, pdb_ins_code = data[9], data[7], data[5], data[10]
            if len(pdb_mon_id) > 1 or pdb_mon_id == '?':
                continue
            pdb_strand_ids += [pdb_strand_id]
            pdb_mon_ids += [pdb_mon_id]
            pdb_seq_nums += [pdb_seq_num]
            pdb_ins_codes += [pdb_ins_code]
    return pdb_strand_ids, pdb_mon_ids, pdb_seq_nums, pdb_ins_codes


def get_tloop_sequences(clusters_folder: str) -> dict[str, list[tuple[int, npt.ArrayLike]]]:
    sequences = {}
    for folder in os.listdir(clusters_folder):
        cluster = int(folder[1:])
        for file in os.listdir(f'{clusters_folder}/{folder}'):
            pdb_id = file[:4].lower()
            if pdb_id not in sequences.keys():
                sequences[pdb_id] = []
            filepath = f'{clusters_folder}/{folder}/{file}'
            array = np.row_stack(parse_pdb(filepath))
            if not any(np.array_equal(array, i[1]) for i in sequences[pdb_id]):
                sequences[pdb_id] += [(cluster, array)]
    return sequences


def get_full_sequences(pdb_ids: list[str], structures_folder: str) -> dict[str, npt.ArrayLike]:
    sequences = {}
    for pdb_id in pdb_ids:
        array = np.row_stack(parse_cif(f'{structures_folder}/{pdb_id}.cif'))
        sequences[pdb_id] = array
    return(sequences)


def align_tloops(tloop_seqs, full_seqs) -> dict[str, npt.ArrayLike]:
    sequences = {}
    for pdb_id, full_arr in full_seqs.items():
        full_resnames, full_resnums = full_arr[1], full_arr[2]
        clusters = np.zeros((len(full_resnames),), dtype=int)
        for cluster, tloop_arr in tloop_seqs[pdb_id]:
            tloop_resnames, tloop_resnums = tloop_arr[0], tloop_arr[1]
            possible_idxs = np.where(full_resnums == tloop_resnums[0])[0]
            for idx in possible_idxs:
                idx_resnames = full_resnames[idx:idx+8]
                idx_resnums = full_resnums[idx:idx+8]
                if (
                    len(idx_resnames) == 8 and len(idx_resnums) == 8 and
                    np.all(idx_resnames == tloop_resnames) and np.all(idx_resnums == tloop_resnums)
                    ):
                    clusters[idx] = cluster
        sequences[pdb_id] = np.vstack([full_arr, clusters])
    return sequences


# frag_len: int = 8
# frag_extension = int((8-frag_len)/2)
# clusters[idx-frag_extension] = cluster


# TODO make this table better
def array_to_df(array):
    df = pd.DataFrame(columns=['pdb_id','array'])
    df['pdb_id'] = array.keys()
    arr_list = []
    for arr in array.values():
        arr_list += [[','.join(i) for i in arr.tolist()]]
    df['array'] = arr_list
    df = df.explode('array')
    return df


def main(args):
    tloop_sequences = get_tloop_sequences(args.clusters_folder)
    pdb_ids = tloop_sequences.keys()
    full_sequences = get_full_sequences(pdb_ids, args.structures_folder)
    all_sequences = align_tloops(tloop_sequences, full_sequences, args.fragment_length)
    np.savez('all_sequences.npz', **all_sequences)
    array_to_df(all_sequences).to_csv('all_sequences_test.csv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clusters_folder', type=str)
    parser.add_argument('structures_folder', type=str)
    parser.add_argument('--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)