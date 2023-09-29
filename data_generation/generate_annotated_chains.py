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


def get_chains(pdb_ids:dict[str,Type[Chain]], struct_dir:str) -> list[Type[Chain]]:
    chains = {}
    for pdb_id in utils.progress_bar_for(pdb_ids):
        filepath = f'{struct_dir}/{pdb_id}.cif'
        seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes = utils.parse_cif(filepath)
        for chain_id in set(chain_ids):
            start_idx, stop_idx = chain_ids.index(chain_id), utils.list_rindex(chain_ids, chain_id)
            c_seq_nums, c_clust_ids, c_res_names, c_res_nums, c_ins_codes = tuple([i[start_idx:stop_idx+1] for i in [seq_nums, clust_ids, res_names, res_nums, ins_codes]])
            if c_seq_nums: # If chain isn't empty
                new_chain = Chain(pdb_id, chain_id, c_seq_nums, c_clust_ids, c_res_names, c_res_nums, c_ins_codes)
                chains[new_chain.seq_id] = new_chain
    return chains


def annotate_chains_tloops(tloops:list[Type[Tetraloop]], chains:dict[str, Type[Chain]]) -> dict[str, Type[Chain]]:
    for chain in utils.progress_bar_for(chains.values()):
        pdb_tloops = [i for i in tloops if i.pdb_id == chain.pdb_id]
        for tloop in pdb_tloops:
            chain.align_tetraloop(tloop)
            chains[chain.seq_id] = chain
    return chains


def remove_similar_chains(chains:dict[str, Type[Chain]], max_pident:float=95) -> dict[str, Type[Chain]]:
    with NamedTemporaryFile(mode='w') as fasta:
        fasta.write('\n'.join([f'>{i.seq_id}\n{i.res_seq}'for i in chains.values()]))
        fasta.seek(0)
        out = NcbiblastnCommandline(query=fasta.name, subject=fasta.name, outfmt=6)()[0]
    out_array = [i.split('\t') for i in out.split('\n') if any(i)]
    pdb_alignments = [PDBAlignment(i[0],i[1],float(i[2]),int(i[6]),int(i[7]),int(i[8]),int(i[9])) for i in out_array]
    del_chains = []
    for alignment in pdb_alignments:
        short_seq = tuple(sorted([len(chains[alignment.qseqid]), len(chains[alignment.sseqid])]))[0]
        tloops_1 = sorted([i.res_seq for i in chains[alignment.qseqid].tloops])# if i.res_nums[0] >= alignment.qstart-1 and i.res_nums[-1] <= alignment.qend-1])
        tloops_2 = sorted([i.res_seq for i in chains[alignment.sseqid].tloops])# if i.res_nums[0] >= alignment.sstart-1 and i.res_nums[-1] <= alignment.send-1])
        if (
            alignment.qseqid != alignment.sseqid and
            alignment.pident > max_pident and
            tloops_1 == tloops_2
        ):
            del_chains += [short_seq]
    del_chains = set(del_chains)
    chains = {seq_id:chain for seq_id, chain in chains.items() if seq_id not in del_chains}
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
    utils.save(tloops_filtered, 'tloops_filtered', args.data_dir, 'pickle')
    utils.save(tloops_filtered, 'tloops_filtered', args.data_dir, 'csv')

    print('Annotating raw chains with tetraloop positions')
    chains_annotated_raw = annotate_chains_tloops(tloops_filtered, chains_raw)
    utils.save(list(chains_annotated_raw.values()), 'chains_annotated_raw', args.data_dir, 'pickle')
    utils.save(list(chains_annotated_raw.values()), 'chains_annotated_raw', args.data_dir, 'csv')
    
    print('Removing duplicate and similar annotated chains')
    chains_annotated_filtered = {i.seq_id:i for i in utils.filter(list(chains_annotated_raw.values()), ['clust_ids','res_names'])} # Remove identical chains
    chains_annotated_filtered = remove_similar_chains(chains_annotated_filtered) # Remove similar chains (alignment above a certain percent identity)
    utils.save(list(chains_annotated_filtered.values()), 'chains_annotated_filtered', args.data_dir, 'pickle')
    utils.save(list(chains_annotated_filtered.values()), 'chains_annotated_filtered', args.data_dir, 'csv')

    # TODO COUNT ABUNDANCE
    # TODO KEEP UNIQUE TETRALOOPS. even when adding the tloop check, there are still some lost. why?
    # TODO run noAnno with current data
    # TODO CROSSREF CHIHFANS REPORT
    # TODO DATA ANALYSIS, COMPARE TO SAM'S DATA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-p', '--download_pdbs', type=bool, default=False)
    args = parser.parse_args()
    main(args)