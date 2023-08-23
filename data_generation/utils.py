import pandas as pd
import pickle

from classes import Sequence, Tetraloop, Chain, Fragment
from typing import Type


def parse_pdb(filepath: str):
    res_names, res_nums = [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.split()[0] == 'MODEL':
                break
        for line in file:
            if line.split()[0] == 'TER':
                break
            rec_name, res_name, res_num, element = line[:6].strip(), line[17:20].strip(), line[22:26].strip(), line[13:16].strip()
            if rec_name == 'ATOM' and element == 'P': # All bases start with P
                res_names += [res_name]
                res_nums += [int(res_num)]
    seq_nums = list(range(len(res_names)))
    return seq_nums, res_names, res_nums


def parse_cif(filepath: str):
    chain_ids, res_names, res_nums, ins_codes = [], [], [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() == '_pdbx_poly_seq_scheme.hetero':
                break
        for line in file:
            if line.strip() == '#':
                break
            data = line.split()
            chain_id, res_name, res_num, ins_code = data[9], data[7], data[5], data[10]
            if len(res_name) > 1 or res_name == '?': # TODO is this stringent enough?
                continue
            chain_ids += [chain_id]
            res_names += [res_name]
            res_nums += [int(res_num)]
            ins_codes += [ins_code]
    seq_nums = list(range(len(res_names)))
    clust_ids = [0]*len(res_names)
    return seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes


def seq_list_to_df(seq_list: list[Type[Sequence]]) -> pd.DataFrame:
    categories = {
        'Tetraloop':['seq_nums', 'res_names', 'res_nums'],
        'Chain':['seq_nums', 'clust_ids', 'res_names', 'res_nums', 'ins_codes'],
        'Fragment':['seq_nums', 'res_names', 'res_nums', 'ins_codes']
        }
    df = pd.DataFrame()
    for index, seq in progress_bar_for(list(enumerate(seq_list))):
        if type(seq) ==  Tetraloop:
            values = seq.seq_nums, seq.res_names, seq.res_nums
            values = [','.join(map(str, i)) for i in values]
            entry = pd.DataFrame({'pdb_id':seq.pdb_id, 'clust_id':seq.clust_id, 'index':index, 'category':[categories['Tetraloop']], 'values':[values]})
        elif type(seq) ==  Chain:
            values = seq.seq_nums, seq.clust_ids, seq.res_names, seq.res_nums, seq.ins_codes
            values = [','.join(map(str, i)) for i in values]
            entry = pd.DataFrame({'pdb_id':seq.pdb_id, 'chain_id':seq.chain_id, 'category':[categories['Chain']], 'values':[values]})
        elif type(seq) == Fragment:
            values = seq.seq_nums, seq.res_names, seq.res_nums, seq.ins_codes
            values = [','.join(map(str, i)) for i in values]
            entry = pd.DataFrame({'pdb_id':seq.pdb_id, 'clust_id':seq.clust_id, 'chain_id':seq.chain_id ,'index':index, 'category':[categories['Fragment']], 'values':[values]})
        df = pd.concat([df, entry], ignore_index=True)
    df = df.sort_values('pdb_id')
    df = df.explode(['category', 'values'])
    return df


def filter(seqs:list[Type[Sequence]], args:list[str]):
    for i in seqs:
        i.id = tuple([str(getattr(i, a)) for a in args])
    return list(set(seqs))


def is_sublist(sublist, main_list):
    sublist_length = len(sublist)
    for i in range(len(main_list) - sublist_length + 1):
        if main_list[i:i+sublist_length] == sublist:
            return True
    return False


def save(data: list[Type[Sequence]], filename: str, folder: str, format: str) -> None:
    print(f'Saving {filename}.{format}')
    filepath = f'{folder}/{filename}.{format}'
    if format == 'csv':
        seq_list_to_df(data).to_csv(filepath, sep='\t', index=False)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'fasta':
        with open(filepath, 'w') as f:
            ids = [f'{i.pdb_id}_{i.chain_id}' for i in data]
            seqs = [i.res_seq for i in data]
            for i in range(len(data)):
                f.write(f'>{ids[i]}\n{seqs[i]}\n')


def load(filepath, format: str = 'pickle'):
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def list_rindex(alist: list, value):
    return len(alist) - alist[-1::-1].index(value) -1


# Terminal progress bar from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress_bar_for(iterable, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


# progressBar modified to work for a while-loop
def progress_bar_while(progress:float, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (progress))
    filledLength = int(length * progress)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if progress == 1.0:
        print()