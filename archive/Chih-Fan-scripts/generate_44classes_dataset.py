import os
import sys
from Bio import PDB
import numpy as np
import glob


# To run this script, execute this in terminal: python generate_44classes_dataset.py "path_to_clusters_folder"

# First part: generate one-hot encoded sequences of 8nt for each entry and save into group of matrices based on cluster.
def get_unique_pdb_position_seq8(cluster_folderpath):
    p = PDB.PDBParser(PERMISSIVE=1)
    filename_list = os.listdir(cluster_folderpath)  # File name of each pdb entry of tloop.
    unique_pdb_position_list = []
    for filename in filename_list:
        pdb_position = filename[0:-16]
        # print(pdb)
        if pdb_position not in unique_pdb_position_list:
            unique_pdb_position_list.append(pdb_position)

        else:
            pass

    seq8_list = []
    count_uniqueindex = []
    for unique_pdb_position in unique_pdb_position_list:
        for file in os.listdir(cluster_folderpath):
            if file.startswith(unique_pdb_position) and file[0:-16] not in count_uniqueindex:
                # print(unique_pdb_position)
                count_uniqueindex.append(unique_pdb_position)
                structure = p.get_structure(unique_pdb_position, cluster_folderpath + file)
                seq8 = ''
                for res in PDB.Selection.unfold_entities(structure, target_level='R'):
                    seq8 += res.get_resname()

                seq8_list.append(seq8)

    # print(len(seq8_list))
    # print(len(count_uniqueindex))
    return seq8_list

def make_seqmatrix_one_hot(seq_list):
    matrices = []
    for seq8 in seq_list:
        seqmatrix_one_hot = np.zeros([8, 4])
        for i in range(8):
            if seq8[i] == 'A':
                seqmatrix_one_hot[i, 0] = 1
            elif seq8[i] == 'U':
                seqmatrix_one_hot[i, 1] = 1
            elif seq8[i] == 'C':
                seqmatrix_one_hot[i, 2] = 1
            elif seq8[i] == 'G':
                seqmatrix_one_hot[i, 3] = 1
            else:
                print(seq8[i])
        matrices.append(np.array(seqmatrix_one_hot))
    return matrices

# Take path to the folder with cluster files. List the cluster names.
clusters_folders_path_list = os.listdir(sys.argv[1])
print(clusters_folders_path_list)

for folder in clusters_folders_path_list:  # folder = cluster name
    # print(folder)
    # print(folder[2:])
    matrices = make_seqmatrix_one_hot(get_unique_pdb_position_seq8(sys.argv[1] + '/' + folder + '/'))
    np.save(folder + '_one_hot_matrices', matrices)


# Second part: save each entry into test and train dataset.

# Make train and test set.
test_matrices = []
test_labels = []
train_matrices = []
train_labels = []


def append_matrices_and_labels_to_list(matricesfilename):
    a = np.load(matricesfilename, allow_pickle=True)

    for i in range(len(a)):
        if i%3 == 0:
            test_matrices.append(a[i])
            test_labels.append(int(matricesfilename[1:3]))
        else:
            train_matrices.append(a[i])
            train_labels.append(int(matricesfilename[1:3]))
    # print(len(test_matrices), len(test_labels), len(train_matrices), len(train_labels))
    return test_matrices, test_labels, train_matrices, train_labels


matricesfile_list = glob.glob('*_one_hot_matrices.npy')
for matricesfile in matricesfile_list:
    append_matrices_and_labels_to_list(matricesfile)

# np.savez('test_array.npz', np.array(test_matrices))
# np.savez('train_array.npz', np.array(train_matrices))
#
# np.save('test_labels.npy', test_labels)
# np.save('train_labels.npy', train_labels)
