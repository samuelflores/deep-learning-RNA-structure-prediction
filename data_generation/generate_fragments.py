import argparse
import os

import utils

from Bio import AlignIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Align import PairwiseAligner
from Bio.PDB.PDBList import PDBList
from sequence_classes import Sequence, Tetraloop, Chain, Fragment
from typing import Type


def get_tloops(clust_dir: str) -> list[Type[Tetraloop]]:
    tloops = []
    for folder in utils.progress_bar_for(os.listdir(clust_dir)):
        clust_id = int(folder[1:])
        for file in os.listdir(f'{clust_dir}/{folder}'):
            pdb_id = file[:4].lower()
            filepath = f'{clust_dir}/{folder}/{file}'
            seq_nums, res_names, res_nums = utils.parse_pdb(filepath)
            tloops += [Tetraloop(pdb_id, clust_id, seq_nums, res_names, res_nums)]
    return tloops


# TODO there's something wrong with this, fix it
def get_chains(pdb_ids: list[str], struct_dir: str) -> list[Type[Chain]]:
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


def annotate_chains_tloops(tloops: list[Type[Tetraloop]], chains: list[Type[Chain]]) -> list[Type[Chain]]:
    for chain in utils.progress_bar_for(chains):
        pdb_tloops = [i for i in tloops if i.pdb_id == chain.pdb_id]
        for tloop in pdb_tloops:
            chain.align_tetraloop(tloop)
    return chains


#! This step is incredibly slow because of the all-against-all pairwise comparisons
# TODO think of other ways to reduce the chain amount?
#? maybe performing a MSA with all the chains and then doing a  would be faster?
# TODO perform MSA on all chains, then use a pairwise while loop (like below) to compare all alignments with each other. look into the biopython clustalomega command line tool + MultipleSeqAlignment

def remove_similar_chains(chains: list[Type[Chain]], max_percent_id: float = 0.9) -> list[Type[Chain]]:

#     ## MSA INCOMPLETE

#     chains.sort(reverse=True)

#     # 1. Make FASTA file from chains
#     utils.save(chains, f'{filename}_unaligned', folder, 'fasta')

#     # 2. multiple sequence align fasta file using clustalo command line tool
#     print(f'Saving {filename}_aligned.fasta')
#     in_file = f'{folder}/{filename}_unaligned.fasta'
#     out_file = f'{folder}/{filename}_aligned.fasta'
#     clustalomega_cline = ClustalOmegaCommandline(infile=in_file, outfile=out_file, verbose=True, force=True, percentid=True)
#     clustalomega_cline()

#     # 3. parse aligned fasta file into muliple sequence alignment
#     aligns = AlignIO.read(f'{folder}/{filename}_aligned.fasta', 'fasta')

#     def get_alignment_idxs(seq1: str, seq2: str):
#         return [seq1[i] == seq2[i] for i in range(len(seq1))]
    
#     def get_percent_identity(seq_idxs:list[bool]):
#         alignment_length = utils.list_rindex(seq_idxs, True) - seq_idxs.index(True)
#         identical_positions = sum(seq_idxs)
#         percent_id = identical_positions / alignment_length
#         return percent_id
    
#     return chains

#     # PAIRWISE

    aligner = PairwiseAligner()
    def get_alignment_idxs(seq1: str, seq2: str):
        alignment = aligner.align(seq1, seq2)[0]
        seq1_aligned, seq2_aligned = alignment.aligned
        seq1_idxs = [num for idxs in seq1_aligned for num in list(range(idxs[0], idxs[1]))]
        seq2_idxs = [num for idxs in seq2_aligned for num in list(range(idxs[0], idxs[1]))]
        return seq1_idxs, seq2_idxs
    
    def get_percent_identity(seq_idxs:list[int]):
        alignment_length = seq_idxs[-1] - seq_idxs[0]
        identical_positions = len(seq_idxs)
        percent_id = identical_positions / alignment_length
        return percent_id
    
    # Check whether all clusters in the alignment are the same
    def check_clusts(seq1_clust_ids, seq2_clust_ids, seq1_idxs:list[int], seq2_idxs:list[int]):
        return all([seq1_clust_ids[seq1_idxs[i]] == seq2_clust_ids[seq2_idxs[i]] for i in range(len(seq1_idxs))])
    
    def is_similar(seq1:Type[Chain], seq2:Type[Chain]):
        seq1_idxs, seq2_idxs = get_alignment_idxs(seq1.res_seq, seq2.res_seq)
        return (
            check_clusts(seq1.clust_ids, seq2.clust_ids, seq1_idxs, seq2_idxs) and 
            get_percent_identity(seq1_idxs) > max_percent_id
        )

    i = 0
    while i < len(chains):
        utils.progress_bar_while(0)
        j = i + 1
        while j < len(chains):
            if is_similar(chains[i], chains[j]):
                del chains[j]
                continue # Keep pointer position
            j += 1
        i += 1
        utils.progress_bar_while(i/len(chains))
    
    return chains


def get_fragments(chains: list[Type[Chain]], fragment_length: int = 8) -> list[Type[Fragment]]:
    fragment_extension = int((fragment_length-8)/2)
    fragments = []
    for chain in utils.progress_bar_for(chains):
        for i in range(len(chain)-fragment_length+1):
            clust_id = chain.clust_ids[i + fragment_extension]
            seq_nums, res_names, res_nums, ins_codes = tuple([j[i:i+fragment_length] for j in [chain.seq_nums, chain.res_names, chain.res_nums, chain.ins_codes]])
            fragments += [Fragment(chain.pdb_id, clust_id, chain.chain_id, seq_nums, res_names, res_nums, ins_codes)]
    return fragments


def main(args):
    
    # # Load existing data
    # chains_annotated_filtered = utils.load(f'{args.data_dir}/chains_annotated_filtered.pickle')
    
    # Make data folder
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    print('Retrieving raw tetraloops')
    tloops_raw = get_tloops(args.clusters_dir)
    utils.save(tloops_raw, 'tloops_raw', args.data_dir, 'pickle')
    utils.save(tloops_raw, 'tloops_raw', args.data_dir, 'csv')

    print('Filtering raw tetraloops for duplicates')
    tloops_filtered = utils.filter(tloops_raw, ['pdb_id','res_names','res_nums'])
    utils.save(tloops_filtered, 'tloops_filtered', args.data_dir, 'pickle')
    utils.save(tloops_filtered, 'tloops_filtered', args.data_dir, 'csv')

    pdb_ids = list(set([i.pdb_id for i in tloops_filtered]))
    # PDBList().download_pdb_files(pdb_ids, obsolete=True, pdir=args.structures_dir)
    
    print('Retrieving raw chains')
    chains_raw = get_chains(pdb_ids, args.structures_dir)
    utils.save(chains_raw, 'chains_raw', args.data_dir, 'pickle')
    utils.save(chains_raw, 'chains_raw', args.data_dir, 'csv')
    print(len(pdb_ids), len(set([i.pdb_id for i in chains_raw])))
    
    print('Annotating chains with raw tetraloop positions')
    chains_annotated_raw = annotate_chains_tloops(tloops_filtered, chains_raw)
    utils.save(chains_annotated_raw, 'chains_annotated_raw', args.data_dir, 'pickle')
    utils.save(chains_annotated_raw, 'chains_annotated_raw', args.data_dir, 'csv')
    
    print('Filtering annotated chains')
    chains_annotated_filtered = utils.filter(chains_annotated_raw, ['clust_ids','res_names']) # Remove identical chains
    chains_annotated_filtered = remove_similar_chains(chains_annotated_filtered) # Remove similar chains (alignment above a certain percent identity)
    utils.save(chains_annotated_filtered, 'chains_annotated_filtered', args.data_dir, 'pickle')
    utils.save(chains_annotated_filtered, 'chains_annotated_filtered', args.data_dir, 'csv')
    
    print(f'Retrieving fragments of length {args.fragment_length}')
    fragments_raw = get_fragments(chains_annotated_filtered, args.fragment_length)
    utils.save(fragments_raw, f'fragments_{args.fragment_length}_raw', args.data_dir, 'pickle')
    
    print('Filtering fragments')
    tloop_fragments = [i for i in fragments_raw if i.clust_id != 0]
    decoy_fragments = [i for i in fragments_raw if i.clust_id == 0]
    decoy_fragments_filtered = utils.filter(decoy_fragments, ['res_names'])
    fragments_filtered = tloop_fragments + decoy_fragments_filtered
    utils.save(fragments_filtered, f'fragments_{args.fragment_length}_filtered', args.data_dir, 'pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clusters_dir', type=str, default='../../../all_clusters')
    parser.add_argument('-s', '--structures_dir', type=str, default='../../../all_structures')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-f', '--fragment_length', type=int, default=8)
    args = parser.parse_args()
    main(args)