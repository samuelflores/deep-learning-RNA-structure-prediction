import os
import argparse
import requests
import pandas as pd
import numpy as np
from Bio.PDB import PDBList, MMCIFParser, Selection
from Bio import SeqIO

# TODO rewrite this script into doing everything, including the 44_classes script and make_dataset(?)

# generate_44classes_dataset.py: take path to all folders -> for each folder, generate one-hot encoded 8nt-length sequences for each PDB entry and save into group of matrices based on cluster

# 1. get unique PDB IDs and positions from a cluster folder. store these in a np df (per cluster)
# 2. download the original PDB files
# 3. use unique PDB IDs and positions to retrieve the sequences of the all 8nt sequences (per cluster). if the position is annotated, note that down somehow. NOTE: make it so this can be expanded to n nucleotides
# A2. construct a list of one hot matrices
# A3. export matrix list into an npy file
# export CSV

# 1. from PDB files in clusters, get all unique PDB IDs (and starting positions)
# 2. download the original PDB files with whole sequences
# 3. depending on the mode (decoys or annotated), make dataframe of all 8nt sequences and output the result as a CSV file (?)
# 4. one hot encode the sequences to output appropriate training data


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


# Save all nucleotide chains in PDB files as fasta files (to cache them)
def download_sequences(structure_ids, pdb_path, fasta_path):
    existing_files = os.listdir(fasta_path)

    for id in structure_ids:
        # if f'{id}.fasta' in existing_files:
        #     continue
        structure = MMCIFParser(QUIET=True).get_structure(id, f'{pdb_path}{id}.cif')
        sequences = {}
        for chain in structure.get_chains():
            sequences[chain.id] = {}
            for res in chain.get_residues():
                res_id = int(res.id[1])
                if res.resname in ['A','U','C','G']:
                    sequences[chain.id][res_id] = res.resname
        sequences = {k: v for k, v in sequences.items() if v}

        # check whether the sequences in a chain are discontinuous. if yes, break them up
        

        # save fasta file
        fasta.dump(sequences, open(f'{fasta_path}{id}.fasta', 'w'), indent = 6)
        print(f'Sequence {id} saved to {fasta_path}{id}.fasta')

    # store the positional information in the fasta file


def get_all_oligos(seq, oligo_len):
    # df = pd.DataFrame(columns=['pdb_id','base','position'])
    # for chain in list
    # turn the chain into strings, with cutoffs at points where the residue positions are discontinuous
    # 
    for i in range(len(seq) - oligo_len + 1):
        oligos += [seq[i:i+oligo_len]]
    return oligos

    # get oligos of length from pdb id
    # return a dataframe with pdb id, sequence, and starting position


# * remove all decoy tetraloops from a dataframe
# def annotated_only():
#     pass


# * remove all annotated/confirmed tetraloops from a dataframe
# def decoy_only():
#     pass


# * one hot encode a list of oligos, returning a matrix
def one_hot():
    pass


def export_fasta():
    pass


# ? what was the point of this again
def export_matrix():
    pass


# * export a dataframe of tetraloops as a CSV
def export_csv():
    pass


def main(args):
    metadata = get_pdb_metadata(args.clusters_path)
    pdb_ids = set(metadata['pdb_id'].to_list())
    # PDBList().download_pdb_files(pdb_ids, pdir=args.pdb_path)
    download_sequences(pdb_ids, args.pdb_path, args.fasta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clusters_path', type=str)
    parser.add_argument('--pdb_path', type=str, default='all_pdb_files')
    parser.add_argument('--fasta_path', type=str, default='all_fasta_files')
    parser.add_argument('--oligo_length', type=int, default=8)
    args = parser.parse_args()
    main(args)


# # Create list of unique pdb ids. Number of unique pdb id checked.
# def get_unique_pdb(all_clusters_folderpath):
#     # p = PDB.PDBParser(PERMISSIVE=1)
#     clusters_list = os.listdir(all_clusters_folderpath)  # File name of each pdb entry of tloop.
#     unique_pdb_list = []
#     for cluster in clusters_list:
#         for pdb_file in os.listdir(all_clusters_folderpath + '/' + cluster):
#             pdb_id = pdb_file.split('_')[0]
#             if pdb_id not in unique_pdb_list:
#                 unique_pdb_list.append(pdb_id)
#     return unique_pdb_list


# # Download the original pdb files with whole sequences.
# def download_pdb_files(pdb_list):
#     for pdb in pdb_list:
#         if len(pdb) == 4:
#             pdbl = PDB.PDBList()
#             pdbl.retrieve_pdb_file(pdb, file_format='pdb', pdir='all_pdb_files')
#         else:
#             url = 'https://files.rcsb.org/pub/pdb/compatible/pdb_bundle/' + \
#                   pdb[1:3] + '/' + pdb[0:4] + '/' + pdb[0:15] + '.tar.gz'
#             r = requests.get(url)
#             with open('C:/Users/raecb/PycharmProjects/RNA_TETRALOOPS/all_pdb_files/' + pdb[0:15] + '.tar.gz',
#                       'wb') as f:
#                 f.write(r.content)


# # Create list of unique pdb-positions (starting nucleotide and index).
# def unique_pdb_pos(allfolderpath):
#     cluster_list = os.listdir(allfolderpath)
#     unique_pdb_position_list = []
#     for cluster in cluster_list:
#         filename_list = os.listdir(allfolderpath + '/' + cluster)  # File name of each pdb entry of tloop.

#         for filename in filename_list:
#             pdb_position = filename[0:-16]
#             if pdb_position not in unique_pdb_position_list:
#                 unique_pdb_position_list.append(pdb_position)
#             else:
#                 pass

#     return unique_pdb_position_list


# # Make dataframe of all 8nt sequences including both tloop-annotated and non-tloop ones.
# def write_seq_to_file(path):
#     seq8_list = []
#     num_list = []
#     pdbid_list = []
#     for pdb_filename in os.listdir(path):
#         pdb_id = pdb_filename[0:-4]
#         p = PDB.PDBParser(PERMISSIVE=1)
#         structure = p.get_structure(pdb_id, path + pdb_filename)
#         seq = ''
#         num = []

#         for res in PDB.Selection.unfold_entities(structure, target_level='R'):
#             if res.get_resname() in ['A', 'U', 'C', 'G']:
#                 seq += res.get_resname()
#                 num.append(res.get_id()[1])
#             else:
#                 pass
#         for i in range(len(seq) - 7):
#             seq_8 = seq[i:(i + 8)]
#             index_seq_8 = seq[i] + str(num[i])
#             pdb_id = pdb_id
#             seq8_list.append(seq_8)
#             num_list.append(index_seq_8)
#             pdbid_list.append(pdb_id)

#     d = {'pdbid': pdbid_list, 'seq8_index': num_list, 'seq8': seq8_list}
#     df = pd.DataFrame(data=d)
#     # print(df)
#     df.to_csv('./seq8_df.csv', index=False)

#     return df


# # Remove the sequences with tloop annotations.
# def dropANNO(df_path):
#     df = pd.read_csv(df_path)
#     for i in range(len(df)):
#         if str(df.loc[i, 'pdbid'] + '_' + df.loc[i, 'seq8_index']) in unique_pdb_position_list:
#             # print(str(i) + str(df.loc[i, 'pdbid'] + '_' + df.loc[i, 'seq8_index']))
#             df = df.drop(i)
#     df.to_csv('./noAnno_seq8_df.csv', index=False)
#     return df


# # if __name__ == '__main__':
# #     download_pdb_files(get_unique_pdb('./clusters_folder'))
# #     unique_pdb_position_list = unique_pdb_pos('clusters_folder')

# #     if not os.path.isdir('./all_pdb_files'):
# #         os.makedirs('./all_pdb_files')
    
# #     pdbs_path = './all_pdb_files/'
# #     write_seq_to_file(pdbs_path)
# #     dropANNO('seq8_df.csv')
