import argparse
import os
from tempfile import NamedTemporaryFile

import utils

from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.PDB import PDBList
from classes import Tetraloop, Chain, PDBAlignment
from typing import Type 

# NOTE: If you're using VSCode to view this file, I recommend installing the Better Comments extension so the comments formats highlight properly
# NOTE: To run this script in terminal, first activate the appropriate conda env (in my case, 'tensorflow'), navigate to the folder where this script is located, and type 'python generate_annotated_chains.py'
# ! The classes.py and utils.py scripts are dependencies and must be located in the same folder


# * This script will generate the following files (in both .csv and .pickle formats):
#     tloops_raw: All tetraloops parsed from the Bottaro dataset
#     tloops_filtered: tloops_raw with structural duplicates removed (most files in the clusters folder have multiple versions, _00000, _00001, _00005, etc.). 
#     chains_annotated_raw: All chains from every single PDB ID found in the Bottaro dataset, annotated with possible tetraloop positions. 
#     chains_annotated_filtered: chains_annotated_raw with duplicate and similar chains removed (to account for highly conserved rRNA that dominates the dataset). This file is used by the downstream generate_fragments.py script.


# * Arguments
#     '-c', '--clusters_dir': path to clusters folder
#     '-s', '--structures_dir': path to mmCIF structures folder
#     '-d', '--data_dir': folder into which the generated files should be placed
#     '-p', '--download_pdbs': re-download all mmCIF files y/n
#! Most of the default values for these args were set up for the folder structure that I used so that I didn't need to re-input every argument every time while testing. If you need to change them then scroll down to the bottom of the script to the if __name__ == '__main__': section


# NOTE: Since this runs on Python 3.9, typing (i.e. when you specify what data type your function parameters should be, e.g. str) doesn't actually seem to work. For now they're just there for clarity.


# * Generates a Tetraloop object for every .aling.pdb file in the Bottaro clusters folder
# Inputs:
#     clust_dir: path to parent clusters directory
# Output: list of Tetraloop objects

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


# * Generates a dict of Chain objects from a given list of PDB IDs
# Inputs:
#     pdb_ids: list of PDB IDs
#     struct_dir: path to directory containing all downloaded mmCIF files.
# Output: dict of Chain objects with their seq_ids as keys

def get_chains(pdb_ids:list[str], struct_dir:str) -> dict[str, Type[Chain]]:
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


# * Return an annotated version of the input Chains dict where all possible aligned Tetraloop positions are marked
# Inputs:
#     tloops: list of Tetraloop objects
#     chains: dict of Chain objects
# Output: dict of annotated Chain objects

def annotate_chains_tloops(tloops:list[Type[Tetraloop]], chains:dict[str, Type[Chain]]) -> dict[str, Type[Chain]]:
    for chain in utils.progress_bar_for(chains.values()):
        pdb_tloops = [i for i in tloops if i.pdb_id == chain.pdb_id]
        for tloop in pdb_tloops:
            chain.align_tetraloop(tloop)
            chains[chain.seq_id] = chain
    return chains


# * Perform a BLAST alignment between all Chains in the input dict. Between every pairwise Chain comparison, remove the shorter Chain IF its alignment percent identity to the longer Chain is above the defined maximum AND both chains contains the same list of Tetraloops in the alignment area.
# Inputs:
#     chains: dict of Chain objects
#     max_pident: Maximum percentage identity needed for two sequences to count as similar
# Output: dict of Chain objects
# ! This step removes some of the unique tetraloops in the dataset, and I'm not entirely sure why. I've troubleshot it as far as I understand, but a couple tetraloops are just irrevocably lost. Keep this in mind when analyzing the filtered chains data.

def remove_similar_chains(chains:dict[str, Type[Chain]], max_pident:float=95) -> dict[str, Type[Chain]]:

    # Make temporary FASTA file for BLAST alignment and store the resulting pairwise comparisons as a list of PDBAlignment objects
    with NamedTemporaryFile(mode='w') as fasta:
        fasta.write('\n'.join([f'>{i.seq_id}\n{i.res_seq}'for i in chains.values()]))
        fasta.seek(0)
        out = NcbiblastnCommandline(query=fasta.name, subject=fasta.name, outfmt=6)()[0]
    out_array = [i.split('\t') for i in out.split('\n') if any(i)]
    pdb_alignments = [PDBAlignment(i[0],i[1],float(i[2]),int(i[6]),int(i[7]),int(i[8]),int(i[9])) for i in out_array]

    del_chains = [] # Chains to be deleted
    for alignment in pdb_alignments:
        short_seq = tuple(sorted([len(chains[alignment.qseqid]), len(chains[alignment.sseqid])]))[0]
        tloops_1 = sorted([i.res_seq for i in chains[alignment.qseqid].tloops]) #// if i.res_nums[0] >= alignment.qstart-1 and i.res_nums[-1] <= alignment.qend-1])
        tloops_2 = sorted([i.res_seq for i in chains[alignment.sseqid].tloops]) #// if i.res_nums[0] >= alignment.sstart-1 and i.res_nums[-1] <= alignment.send-1])
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

    # Make folders
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.structures_dir):
        os.makedirs(args.structures_dir)
    
    print('Retrieving raw tetraloops')
    tloops_raw = get_tloops(args.clusters_dir)
    utils.save(tloops_raw, 'tloops_raw', args.data_dir, 'pickle')
    utils.save(tloops_raw, 'tloops_raw', args.data_dir, 'csv')
    
    pdb_ids = list(set([i.pdb_id for i in tloops_raw]))

    # Download all required mmCIF files
    # NOTE: This step takes a long time, so if the structures folder already exists then don't create a new one or re-download.
    if args.download_pdbs:
        PDBList().download_pdb_files(pdb_ids, obsolete=True, pdir=args.structures_dir, overwrite=True)
    
    print('Removing duplicate tetraloops')
    tloops_filtered = utils.filter(tloops_raw, ['pdb_id','res_names','res_nums'])
    utils.save(tloops_filtered, 'tloops_filtered', args.data_dir, 'pickle')
    utils.save(tloops_filtered, 'tloops_filtered', args.data_dir, 'csv')

    print('Retrieving raw chains')
    chains_raw = get_chains(pdb_ids, args.structures_dir)

    print('Annotating raw chains with tetraloop positions')
    chains_annotated_raw = annotate_chains_tloops(tloops_filtered, chains_raw)
    utils.save(list(chains_annotated_raw.values()), 'chains_annotated_raw', args.data_dir, 'pickle')
    utils.save(list(chains_annotated_raw.values()), 'chains_annotated_raw', args.data_dir, 'csv')
    
    print('Removing duplicate and similar annotated chains')
    chains_annotated_filtered = {i.seq_id:i for i in utils.filter(list(chains_annotated_raw.values()), ['clust_ids','res_names'])} # Remove identical chains
    chains_annotated_filtered = remove_similar_chains(chains_annotated_filtered) # Remove similar chains (alignment above a certain percent identity)
    utils.save(list(chains_annotated_filtered.values()), 'chains_annotated_filtered', args.data_dir, 'pickle')
    utils.save(list(chains_annotated_filtered.values()), 'chains_annotated_filtered', args.data_dir, 'csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-p', '--download_pdbs', type=bool, default=False)
    args = parser.parse_args()
    main(args)