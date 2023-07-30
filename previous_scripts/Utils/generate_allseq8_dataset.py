import os
from Bio import PDB
import requests
import pandas as pd


# Create list of unique pdb ids. Number of unique pdb id checked.
def get_unique_pdb(all_clusters_folderpath):
    p = PDB.PDBParser(PERMISSIVE=1)
    clusters_list = os.listdir(all_clusters_folderpath)  # File name of each pdb entry of tloop.
    unique_pdb_list = []
    for cluster in clusters_list:
        for pdb_file in os.listdir(all_clusters_folderpath + '/' + cluster):
            pdb_id = pdb_file.split('_')[0]
            if pdb_id not in unique_pdb_list:
                unique_pdb_list.append(pdb_id)
            else:
                pass
    # print(len(unique_pdb_list))
    return unique_pdb_list


# Download the original pdb files with whole sequences.
def download_pdb_files(pdb_list):
    for pdb in pdb_list:
        if len(pdb) == 4:
            pdbl = PDB.PDBList()
            pdbl.retrieve_pdb_file(pdb, file_format='pdb', pdir='all_pdb_files')
        else:
            url = 'https://files.rcsb.org/pub/pdb/compatible/pdb_bundle/' + \
                  pdb[1:3] + '/' + pdb[0:4] + '/' + pdb[0:15] + '.tar.gz'
            r = requests.get(url)
            with open('C:/Users/raecb/PycharmProjects/RNA_TETRALOOPS/all_pdb_files/' + pdb[0:15] + '.tar.gz',
                      'wb') as f:
                f.write(r.content)


# Create list of unique pdb-positions (starting nucleotide and index).
def unique_pdb_pos(allfolderpath):
    cluster_list = os.listdir(allfolderpath)
    unique_pdb_position_list = []
    for cluster in cluster_list:
        filename_list = os.listdir(allfolderpath + '/' + cluster)  # File name of each pdb entry of tloop.

        for filename in filename_list:
            pdb_position = filename[0:-16]
            if pdb_position not in unique_pdb_position_list:
                unique_pdb_position_list.append(pdb_position)
            else:
                pass

    return unique_pdb_position_list


# Make dataframe of all 8nt sequences including both tloop-annotated and non-tloop ones.
def write_seq_to_file(path):
    seq8_list = []
    num_list = []
    pdbid_list = []
    for pdb_filename in os.listdir(path):
        pdb_id = pdb_filename[0:-4]
        p = PDB.PDBParser(PERMISSIVE=1)
        structure = p.get_structure(pdb_id, path + pdb_filename)
        seq = ''
        num = []

        for res in PDB.Selection.unfold_entities(structure, target_level='R'):
            if res.get_resname() in ['A', 'U', 'C', 'G']:
                seq += res.get_resname()
                num.append(res.get_id()[1])
            else:
                pass
        for i in range(len(seq) - 7):
            seq_8 = seq[i:(i + 8)]
            index_seq_8 = seq[i] + str(num[i])
            pdb_id = pdb_id
            seq8_list.append(seq_8)
            num_list.append(index_seq_8)
            pdbid_list.append(pdb_id)

    d = {'pdbid': pdbid_list, 'seq8_index': num_list, 'seq8': seq8_list}
    df = pd.DataFrame(data=d)
    # print(df)
    df.to_csv('./seq8_df.csv', index=False)

    return df


# Remove the sequences with tloop annotations.
def dropANNO(df_path):
    df = pd.read_csv(df_path)
    for i in range(len(df)):
        if str(df.loc[i, 'pdbid'] + '_' + df.loc[i, 'seq8_index']) in unique_pdb_position_list:
            # print(str(i) + str(df.loc[i, 'pdbid'] + '_' + df.loc[i, 'seq8_index']))
            df = df.drop(i)
    df.to_csv('./noAnno_seq8_df.csv', index=False)
    return df


if __name__ == "__main__":
    download_pdb_files(get_unique_pdb('./clusters_folder'))
    unique_pdb_position_list = unique_pdb_pos('clusters_folder')

    if not os.path.isdir("./all_pdb_files"):
        os.makedirs("./all_pdb_files")

    pdbs_path = './all_pdb_files/'
    write_seq_to_file(pdbs_path)
    dropANNO('seq8_df.csv')
