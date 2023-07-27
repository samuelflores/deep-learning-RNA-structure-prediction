import argparse
import ast
import json
import os

import pandas as pd

from Bio.PDB import PDBList, FastMMCIFParser


# weird tloops: ['3a3a', '3adb', '3adc', '3hl2', '3rg5', '3w3s', '4rqf']

# TODO correct bottaro's original chain IDs (if they're wrong) 

# DONE parse every non-duplicate .align.pdb ATOM file for the sequence and residue numbers
# DONE check that a) the chain is long enough to contain the residue number b) the first residues and number align c) the whole sequence matches
# DONE allow for residue number skipping in decoy generation. generate decoys only from the chain from which the tetraloop originates
# DONE (not really, i just don't touch the original data) update chain ID in bottaro's dataset

# Extract PDB data from cluster folder filenames
def get_tloop_data(clusters_folder):
    data = pd.DataFrame(columns=['pdb_id','sequence','residue_numbers','cluster'])
    for folder in os.listdir(clusters_folder):
        cluster = int(folder[1:])
        for file in os.listdir(f'{clusters_folder}/{folder}'):

            pdb_id = file[:4].lower()
            filepath = f'{clusters_folder}/{folder}/{file}'

            sequence = ''
            residue_numbers = []
            with open(filepath, 'r') as f:
                for line in f:
                    record_name = line[:6].strip()
                    atom_name = line[13:16].strip()
                    if record_name == 'ATOM' and atom_name == 'P':
                        sequence += line[17:20].strip()
                        residue_numbers += [int(line[22:26].strip())]
            residue_numbers = tuple(residue_numbers)
            
            entry = pd.DataFrame({'pdb_id': pdb_id, 'sequence': sequence, 'residue_numbers': [residue_numbers], 'cluster': cluster})
            data = pd.concat([data, entry], ignore_index=True)
    
    data = data.drop_duplicates()
    return data


# * For some reason, many of the .cif files contain a lot of duplicate chains
def get_full_sequences(pdb_ids, structures_folder):
    parser = FastMMCIFParser(QUIET=True)
    sequences = pd.DataFrame(columns=['pdb_id','chain','sequence','residue_numbers'])
    
    for pdb_id in pdb_ids:
        print('Retrieving sequences for ' + pdb_id)
        structure = parser.get_structure(pdb_id, f'{structures_folder}/{pdb_id}.cif')

        for chain in structure.get_chains():
            sequence = ''
            residue_numbers = []

            for res in chain.get_residues():
                if len(res.resname) > 1:
                    continue
                sequence += res.resname
                residue_numbers += [res.id[1]] # https://biopython.org/docs/1.76/api/Bio.PDB.Entity.html
            
            if sequence:
                residue_numbers = tuple(residue_numbers)
                entry = pd.DataFrame({'pdb_id': pdb_id, 'chain': chain.id, 'sequence': sequence, 'residue_numbers': [residue_numbers]})
                sequences = pd.concat([sequences, entry], ignore_index=True)
    
    sequences = sequences.drop_duplicates()
    return sequences


def get_tloop_chains(tloops, full_sequences):
    tloop_chains = pd.DataFrame(columns=['pdb_id','chain','sequence', 'residue_numbers'])

    for tloop in tloops.itertuples(index=False):
        tloop_resnum = tloop.residue_numbers[0]
        chains = full_sequences[full_sequences.iloc[:, 0].str.fullmatch(tloop.pdb_id, case=False)]
        for chain in chains.itertuples(index=False):
            try:
                chain_idx = chain.residue_numbers.index(tloop_resnum)
            except ValueError:
                continue
            chain_seq = chain.sequence[chain_idx:chain_idx+8]
            chain_resnums = chain.residue_numbers[chain_idx:chain_idx+8]
            if tloop.sequence == chain_seq and tloop.residue_numbers == chain_resnums:
                entry = pd.DataFrame({'pdb_id': tloop.pdb_id, 'chain': chain.chain, 'sequence': chain.sequence, 'residue_numbers': [chain.residue_numbers]})
                tloop_chains = pd.concat([tloop_chains, entry], ignore_index=True)
    
    tloop_chains = tloop_chains.drop_duplicates(['pdb_id','sequence','residue_numbers'])
    return tloop_chains


def get_all_fragments(tloop_chains, fragment_length):
    all_fragments = pd.DataFrame(columns=['pdb_id','chain','sequence', 'residue_numbers'])
    for chain in tloop_chains.itertuples(index=False):
        print(f'Fragmenting chain {chain.pdb_id}_{chain.chain}')
        fragment_seqs = [chain.sequence[i:i+fragment_length] for i in range(len(chain.sequence) - fragment_length + 1)]
        fragment_resnums = [chain.residue_numbers[i:i+fragment_length] for i in range(len(chain.residue_numbers) - fragment_length + 1)]
        entry = pd.DataFrame({'pdb_id': chain.pdb_id, 'chain': chain.chain, 'sequence': fragment_seqs, 'residue_numbers': fragment_resnums})
        all_fragments = pd.concat([all_fragments, entry], ignore_index=True)
    return all_fragments


def drop_tloops(all_fragments, tloop_data):
    merged_fragments = all_fragments.merge(tloop_data, how='left', on=['pdb_id','sequence','residue_numbers'], indicator=True)
    decoy_fragments = merged_fragments[merged_fragments['_merge'] == 'left_only'].drop(columns=['_merge','cluster'])
    return decoy_fragments


def main(args):

    # # load existing data
    tloop_data = pd.read_csv('tloop_data.csv', sep='\t')
    tloop_data['residue_numbers'] = tloop_data['residue_numbers'].apply(ast.literal_eval)

    # full_sequences = pd.read_csv('full_sequences.csv', sep='\t')
    # full_sequences['residue_numbers'] = full_sequences['residue_numbers'].apply(ast.literal_eval)

    # tloop_chains = pd.read_csv('tloop_chains.csv', sep='\t')
    # tloop_chains['residue_numbers'] = tloop_chains['residue_numbers'].apply(ast.literal_eval)

    all_fragments = pd.read_csv('all_fragments.csv', sep='\t')
    all_fragments['residue_numbers'] = all_fragments['residue_numbers'].apply(ast.literal_eval)

    # decoy_fragments = pd.read_csv('decoy_fragments.csv', sep='\t')
    # decoy_fragments['residue_numbers'] = decoy_fragments['residue_numbers'].apply(ast.literal_eval)

    ###################################

    # # tloop data
    # tloop_data = get_tloop_data(args.clusters_folder)
    # tloop_data.to_csv('tloop_data.csv', sep='\t', index=False)

    # pdb_ids = set(tloop_data['pdb_id'].to_list())

    # # download mmcif files
    # PDBList().download_pdb_files(pdb_ids, pdir=args.structures_folder, obsolete=True)

    # # full sequences
    # full_sequences = get_full_sequences(pdb_ids, args.structures_folder)
    # full_sequences.to_csv('full_sequences.csv', sep='\t', index=False)

    # # tloop chains
    # tloop_chains = get_tloop_chains(tloop_data, full_sequences)
    # tloop_chains.to_csv('tloop_chains.csv', sep='\t', index=False)

    # # fragments
    # all_fragments = get_all_fragments(tloop_chains, args.fragment_length)
    # all_fragments.to_csv('all_fragments.csv', sep='\t', index=False)

    # decoys
    # decoy_fragments = drop_tloops(all_fragments, tloop_data)
    # decoy_fragments.to_csv('decoy_fragments.csv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clusters_folder', type=str)
    parser.add_argument('structures_folder', type=str)
    parser.add_argument('--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)