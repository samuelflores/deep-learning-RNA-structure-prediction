import argparse
import json
import os

import requests
import pandas as pd

from Bio.PDB import MMCIFParser

# {
#     pdbid: {
#         sequence: ''
#         fragments = [Fragment()]
#     }
# }

class Fragment:
    def __init__(self, pdb_id, start_pos, end_pos, sequence, cluster):
        self.pdb_id = pdb_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.sequence = sequence
        self.cluster = cluster # if the cluster is 0, it's a decoy


# Extract PDB metadata from cluster folder filenames
def get_pdb_metadata(clusters_path):
    metadata = pd.DataFrame(columns=['pdb_id','base','position','cluster'])
    for folder in os.listdir(clusters_path):
        cluster = int(folder[1:])
        for file in os.listdir(f'{clusters_path}/{folder}'):
            filename = file.split('_')
            pdb_id, base, position = filename[0][:4].lower(), filename[1][0], filename[1][1:]
            entry = pd.DataFrame({'pdb_id': [pdb_id], 'base': [base], 'position': [position], 'cluster': [cluster]})
            metadata = pd.concat([metadata, entry], ignore_index=True)
    metadata = metadata.drop_duplicates()
    return metadata


# https://biopython.org/docs/1.76/api/Bio.PDB.Entity.html
def get_sequences(pdb_ids):
    sequences = {}
    for id in pdb_ids:
        sequences[id] = []
        molecules = requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{id}').json()[id.lower()]
        for m in molecules:
            if m['molecule_type'] == 'polyribonucleotide':
                sequences[id].append(m['sequence'])
    return sequences


def get_all_fragments(sequences, fragment_length):
    fragments = {}
    for id, chains in sequences.items():
        longest_sequence = max(chains, key=len)
        fragments[id] = [sequence[i:i+8] for i in range(len(sequence) - fragment_length + 1)]
    return fragments


def main(args):
    # metadata = get_pdb_metadata(args.clusters_path)
    # metadata.to_csv('pdb_metadata.csv', sep='\t')
    # pdb_ids = set(metadata['pdb_id'].to_list())
    # sequences = get_sequences(pdb_ids)
    # json.dump(sequences, open('sequences.json','w'), indent=6)
    # metadata = pd.read_csv('pdb_metadata.csv')
    # sequences = json.load(open('sequences.json', 'r'))

    # for id, chains in sequences.items():
    #     longest_sequence = max(chains, key=len)
    #     fragments = get_all_fragments(longest_sequence, args.fragment_length)
    #     print(fragments)
    #     break

    structure = MMCIFParser(QUIET=True).get_structure('1fjg', '../../../all_pdb_files/1fjg.cif')
    sequence = {}
    for chain in structure.get_chains():
        for res in chain.get_residues():
            res_id = int(res.id[1])
            sequence[res_id] = res.resname
        break
    print(sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('clusters_path', type=str)
    parser.add_argument('--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)