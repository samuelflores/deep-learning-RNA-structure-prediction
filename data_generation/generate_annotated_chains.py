import argparse
import os
from tempfile import NamedTemporaryFile

import utils

from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.PDB import PDBList
from classes import Tetraloop, Chain, PDBAlignment
from typing import Type


def get_tloops(clust_dir:str) -> list[Type[Tetraloop]]:
    tloops = []
    for folder in utils.progress_bar_for(os.listdir(clust_dir)):
        clust_id = int(folder[1:])
        for file in os.listdir(f'{clust_dir}/{folder}'):
            pdb_id = file[:4].lower()
            filepath = f'{clust_dir}/{folder}/{file}'
            seq_nums, res_names, res_nums = utils.parse_pdb(filepath)
            tloops += [Tetraloop(pdb_id, clust_id, seq_nums, res_names, res_nums)]
    return tloops


def get_chains(pdb_ids:list[str], struct_dir:str) -> list[Type[Chain]]:
    chains = []
    for pdb_id in utils.progress_bar_for(pdb_ids):
        filepath = f'{struct_dir}/{pdb_id}.cif'
        seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes = utils.parse_cif(filepath)
        for chain_id in set(chain_ids):
            start_idx, stop_idx = chain_ids.index(chain_id), utils.list_rindex(chain_ids, chain_id)
            c_seq_nums, c_clust_ids, c_res_names, c_res_nums, c_ins_codes = tuple([i[start_idx:stop_idx+1] for i in [seq_nums, clust_ids, res_names, res_nums, ins_codes]])
            if c_seq_nums: # If chain isn't empty
                chains += [Chain(pdb_id, chain_id, c_seq_nums, c_clust_ids, c_res_names, c_res_nums, c_ins_codes)]
    return chains


def annotate_chains_tloops(tloops:list[Type[Tetraloop]], chains:list[Type[Chain]]) -> list[Type[Chain]]:
    for chain in utils.progress_bar_for(chains):
        pdb_tloops = [i for i in tloops if i.pdb_id == chain.pdb_id]
        for tloop in pdb_tloops:
            chain.align_tetraloop(tloop)
    return chains


def remove_similar_chains(chains:list[Type[Chain]], max_pident:float=95) -> list[Type[Chain]]:
    chain_lens = {i.seq_id:len(i) for i in chains}
    chain_clust_ids = {i.seq_id:set(i.clust_ids) for i in chains}
    with NamedTemporaryFile(mode='w') as fasta:
        fasta.write('\n'.join([f'>{i.seq_id}\n{i.res_seq}'for i in chains]))
        fasta.seek(0)
        out = NcbiblastnCommandline(query=fasta.name, subject=fasta.name, outfmt=6)()[0]
    out_array = [i.split('\t') for i in out.split('\n') if any(i)]
    pdb_alignments = [PDBAlignment(i[0],i[1],float(i[2]),int(i[6]),int(i[7]),int(i[8]),int(i[9])) for i in out_array]
    del_chains = []
    for alignment in pdb_alignments:
        short_seq, long_seq = tuple(sorted([alignment.qseqid, alignment.sseqid], key=lambda x:chain_lens[x]))
        if (
            alignment.qseqid != alignment.sseqid and
            alignment.pident > max_pident # and CHECK THAT ALL THE TLOOPS IN THE SHORTER SEQUENCE ARE FOUND IN THE LONGER SEQUENCE
            # chain_clust_ids[short_seq] in chain_clust_ids[long_seq]
        ):
            del_chains += [short_seq]
    chains = [i for i in chains if i.seq_id not in set(del_chains)]
    return chains


def main(args):

    # Make data folder
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    print('Retrieving raw tetraloops')
    tloops_raw = get_tloops(args.clusters_dir)
    utils.save(tloops_raw, 'tloops_raw', args.data_dir, 'pickle')
    utils.save(tloops_raw, 'tloops_raw', args.data_dir, 'csv')

    pdb_ids = list(set([i.pdb_id for i in tloops_raw]))
    if args.download_pdbs:
        PDBList().download_pdb_files(pdb_ids, obsolete=True, pdir=args.structures_dir, overwrite=True)
    
    print('Retrieving raw chains')
    chains_raw = get_chains(pdb_ids, args.structures_dir)
    
    print('Removing duplicate tetraloops')
    tloops_filtered = utils.filter(tloops_raw, ['pdb_id','res_names','res_nums'])

    print('Annotating raw chains with tetraloop positions')
    chains_annotated_raw = annotate_chains_tloops(tloops_filtered, chains_raw)
    utils.save(chains_annotated_raw, 'chains_annotated_raw', args.data_dir, 'pickle')
    utils.save(chains_annotated_raw, 'chains_annotated_raw', args.data_dir, 'csv')
    
    print('Removing duplicate and similar annotated chains')
    chains_annotated_filtered = utils.filter(chains_annotated_raw, ['clust_ids','res_names']) # Remove identical chains
    chains_annotated_filtered = remove_similar_chains(chains_annotated_filtered) # Remove similar chains (alignment above a certain percent identity)
    utils.save(chains_annotated_filtered, 'chains_annotated_filtered', args.data_dir, 'pickle')
    utils.save(chains_annotated_filtered, 'chains_annotated_filtered', args.data_dir, 'csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-p', '--download_pdbs', type=bool, default=False)
    args = parser.parse_args()
    main(args)