import numpy as np
import pandas as pd
import random
import glob


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

noAnno_df = pd.read_csv('noAnno_seq8_df.csv')
noAnno_seq8_list = noAnno_df['seq8'].values.tolist()


def make_noAnno_seqmatrix_one_hot(seq_list):
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

    random.shuffle(matrices)
    for i in range(len(matrices)):
        if i % 3 == 0:
            test_matrices.append(matrices[i])
            test_labels.append(0)

        else:
            train_matrices.append(matrices[i])
            train_labels.append(0)

    return test_matrices, test_labels, train_matrices, train_labels

# print(len(test_matrices), len(test_labels))
# print(len(train_matrices), len(train_labels))

make_noAnno_seqmatrix_one_hot(noAnno_seq8_list)

# print(len(test_matrices), len(test_labels))
# print(len(train_matrices), len(train_labels))

np.savez('noAnno_test_array.npz', np.array(test_matrices))
np.savez('noAnno_train_array.npz', np.array(train_matrices))

np.save('noAnno_test_labels.npy', test_labels)
np.save('noAnno_train_labels.npy', train_labels)


# Original code for noAnno
#
# noAnno_df = pd.read_csv('noAnno_df.csv')
# noAnno_seq8_list = noAnno_df['seq8'].values.tolist()
#
# test_matrices = []
# test_labels = []
# train_matrices = []
# train_labels = []
#
#
# def make_noAnno_seqmatrix_one_hot(seq_list):
#     matrices = []
#     for seq8 in seq_list:
#         seqmatrix_one_hot = np.zeros([8, 4])
#         for i in range(len(seq8)):
#             if seq8[i] == 'A':
#                 seqmatrix_one_hot[i, 0] = 1
#             elif seq8[i] == 'U':
#                 seqmatrix_one_hot[i, 1] = 1
#             elif seq8[i] == 'C':
#                 seqmatrix_one_hot[i, 2] = 1
#             elif seq8[i] == 'G':
#                 seqmatrix_one_hot[i, 3] = 1
#             else:
#                 print(seq8[i])
#         matrices.append(np.array(seqmatrix_one_hot))
#
#     random.shuffle(matrices)
#     for i in range(len(matrices)):
#         if i % 3 == 0:
#             test_matrices.append(matrices[i])
#             test_labels.append(0)
#
#         else:
#             train_matrices.append(matrices[i])
#             train_labels.append(0)
#
#     return test_matrices, test_labels, train_matrices, train_labels
#
#
# make_noAnno_seqmatrix_one_hot(noAnno_seq8_list)
# print(train_matrices)
# print(test_matrices)
# print(train_labels)
# print(test_labels)
#
# # Second part: save each entry into test and train dataset.
#
# np.savez('noAnno_test_array.npz', np.array(test_matrices))
# np.savez('noAnno_train_array.npz', np.array(train_matrices))
#
# np.save('noAnno_test_labels.npy', test_labels)
# np.save('noAnno_train_labels.npy', train_labels)