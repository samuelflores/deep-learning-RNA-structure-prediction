import argparse
import os

import numpy as np
import utils

from Bio.Align import PairwiseAligner
from sequence_classes import Tetraloop, PDB, Fragment
from typing import Type


def get_tloop_sequences(clust_dir: str) -> list[Type[Tetraloop]]:
    seqs = []
    for folder in utils.progressBar(os.listdir(clust_dir), prefix = 'Progress:', suffix = 'Complete', length = 50):
    # for folder in os.listdir(clust_dir):
        clust_id = int(folder[1:])
        for file in os.listdir(f'{clust_dir}/{folder}'):
            pdb_id = file[:4].lower()
            filepath = f'{clust_dir}/{folder}/{file}'
            seq_nums, res_names, res_nums = utils.parse_pdb(filepath)
            seqs += [Tetraloop(pdb_id, clust_id, seq_nums, res_names, res_nums)]
    return list(set(seqs))


def get_pdb_sequences(pdb_ids: list[str], struct_dir: str) -> list[Type[PDB]]:
    seqs = []
    for pdb_id in utils.progressBar(pdb_ids, prefix = 'Progress:', suffix = 'Complete', length = 50):
    # for pdb_id in pdb_ids:
        filepath = f'{struct_dir}/{pdb_id}.cif'
        seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes = utils.parse_cif(filepath)
        seqs += [PDB(pdb_id, seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes)]
    return seqs


def remove_chain_redundancy(pdb_seqs: list[Type[PDB]], max_percent_id: float = 0.9) -> list[Type[PDB]]:
    
    aligner = PairwiseAligner()
    for pdb_seq in utils.progressBar(pdb_seqs, prefix = 'Progress:', suffix = 'Complete', length = 50):
    # for pdb_seq in pdb_seqs:
        chains = {chain_id: pdb_seq.res_seq[indices[0]:indices[1]+1] for chain_id, indices in pdb_seq.chain_indices.items()}
        
        # remove identical chains
        chains = {v: k for k, v in chains.items()}
        chains = {v: k for k, v in chains.items()}
        
        # sort chains by length
        chains = dict(sorted(chains.items(), key=lambda x:len(x[1]), reverse=True))

        # Remove chains with high enough percentage identity
        i = 0
        while i < len(chains):
            j = i + 1
            while j < len(chains):
                chain_ids, chain_seqs = list(chains.keys()), list(chains.values())
                seq1, seq2 = chain_seqs[i], chain_seqs[j]
                alignment = aligner.align(seq1, seq2)[0]
                subseq_idxs = alignment.aligned[0]
                if subseq_idxs:
                    alignment_length = subseq_idxs[-1][1] - subseq_idxs[0][0]
                    identical_positions = sum([subseq[1] - subseq[0] for subseq in subseq_idxs])
                    percent_id = identical_positions / alignment_length
                    if percent_id > max_percent_id:
                        chains.pop(chain_ids[j])
                        continue # Keep pointer position
                j += 1
            i += 1
        
        deleted_chains = set(pdb_seq.chain_indices.keys()) - chains.keys()
        for i in deleted_chains:
            pdb_seq.remove_chain(i)
    
    return pdb_seqs


def align_tloops_to_pdb(tloop_seqs: list[Type[Tetraloop]], pdb_seqs: list[Type[PDB]]) -> list[Type[PDB]]:
    for pdb_seq in utils.progressBar(pdb_seqs, prefix = 'Progress:', suffix = 'Complete', length = 50):
    # for pdb_seq in pdb_seqs:
        pdb_tloops = [i for i in tloop_seqs if i.pdb_id == pdb_seq.pdb_id]
        for tloop_seq in pdb_tloops:
            possible_idxs = [idx for idx, res_num in enumerate(pdb_seq.res_nums) if res_num == tloop_seq.res_nums[0]]
            for idx in possible_idxs:
                idx_res_names, idx_res_nums = pdb_seq.res_names[idx:idx+8], pdb_seq.res_nums[idx:idx+8]
                if (
                    len(idx_res_names) == 8 and len(idx_res_nums) == 8 and
                    np.all(idx_res_names == tloop_seq.res_names) and np.all(idx_res_nums == tloop_seq.res_nums)
                    ):
                    pdb_seq.clust_ids[idx] = tloop_seq.clust_id
    return pdb_seqs


def get_fragments(all_seqs: list[Type[PDB]], fragment_length: int = 8) -> list[Type[Fragment]]:
    fragment_extension = int((fragment_length-8)/2)
    fragments = []
    for seq in utils.progressBar(all_seqs, prefix = 'Progress:', suffix = 'Complete', length = 50):
    # for seq in all_seqs:
        for i in range(len(seq)-fragment_length+1):
            unique_chain_ids = list(set(seq.chain_ids[i:i+fragment_length]))
            if len(unique_chain_ids) > 1:
                continue
            pdb_id = seq.pdb_id
            clust_id = seq.clust_ids[i + fragment_extension]
            chain_id = unique_chain_ids[0]
            seq_nums, res_names, res_nums, ins_codes = tuple([j[i:i+fragment_length] for j in [seq.seq_nums, seq.res_names, seq.res_nums, seq.ins_codes]])
            fragments += [Fragment(pdb_id, clust_id, chain_id, seq_nums, res_names, res_nums, ins_codes)]
    return list(set(fragments))


def main(args):

    # # Load existing data
    # tloop_seqs = utils.load(f'{args.data_dir}/tloop_seqs.pickle')
    # pdb_seqs = utils.load(f'{args.data_dir}/pdb_seqs.pickle')
    # all_seqs = utils.load(f'{args.data_dir}/all_seqs.pickle')
    # all_fragments = utils.load(f'{args.data_dir}/all_fragments_{args.fragment_length}.pickle')

    # Make data folder
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    print('Retrieving tetraloop sequences')
    tloop_seqs = get_tloop_sequences(args.clusters_dir)
    print('Pickling tetraloop sequences')
    utils.save(f'{args.data_dir}/tloop_seqs.pickle', tloop_seqs)
    print('Saving tetraloop sequences as CSV')
    utils.seq_list_to_df(tloop_seqs).to_csv(f'{args.data_dir}/tloop_seqs.csv', sep='\t', index=False)
    print('Tetraloop sequences retrieved\n')
    
    pdb_ids = set([i.pdb_id for i in tloop_seqs])
    
    print('Retrieving PDB sequences')
    pdb_seqs = get_pdb_sequences(pdb_ids, args.structures_dir)
    print('Removing redundant chains')
    pdb_seqs = remove_chain_redundancy(pdb_seqs)
    print('Pickling PDB sequences')
    utils.save(f'{args.data_dir}/pdb_seqs.pickle', pdb_seqs)
    print('Saving PDB sequences as CSV')
    utils.seq_list_to_df(pdb_seqs).to_csv(f'{args.data_dir}/pdb_seqs.csv', sep='\t', index=False)
    print('PDB sequences retrieved\n')

    print('Aligning tetraloops to all PDB sequences')
    all_seqs = align_tloops_to_pdb(tloop_seqs, pdb_seqs)
    print('Pickling all sequences')
    utils.save(f'{args.data_dir}/all_seqs.pickle', all_seqs)
    print('Saving all sequences as CSV')
    utils.seq_list_to_df(all_seqs).to_csv(f'{args.data_dir}/all_seqs.csv', sep='\t', index=False)
    print('All sequences retrieved\n')

    print(f'Retrieving all fragments of length {args.fragment_length}')
    all_fragments = get_fragments(all_seqs, args.fragment_length)
    print('Pickling all fragments')
    utils.save(f'{args.data_dir}/all_fragments_{args.fragment_length}.pickle', all_fragments)
    # print('Saving all fragments as CSV')
    # utils.seq_list_to_df(all_fragments).to_csv(f'{args.data_dir}/all_fragments_{args.fragment_length}.csv', sep='\t', index=False)
    print('All fragments retrieved\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-f', '--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)