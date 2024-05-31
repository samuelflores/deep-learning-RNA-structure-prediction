import pandas as pd


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
                res_nums += [res_num]
    seq_nums = [range(len(res_names))]
    clust_ids = [int(filepath.split('/')[-2][1:])]*8
    chain_ids, ins_codes = ['']*8, ['']*8
    return seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes


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
            res_nums += [res_num]
            ins_codes += [ins_code]
    seq_nums = [range(len(res_names))]
    clust_ids = [0]*len(res_names)
    return seq_nums, chain_ids, clust_ids, res_names, res_nums, ins_codes


# TODO make this more efficient
def arrdict_to_df(arrdict: dict) -> pd.DataFrame:
    df = pd.DataFrame(columns=['pdb_id','index','category','values'])
    categories = ['seq_nums', 'chain_ids', 'clust_ids', 'res_names', 'res_nums', 'ins_codes']
    for pdb_id, value in arrdict.items():
        if type(value) is list:
            for idx, arr in enumerate(value):
                arr_stings = [','.join(i) for i in arr.tolist()]
                entry = pd.DataFrame({'pdb_id': pdb_id, 'index':int(idx), 'category':[categories], 'values':[arr_stings]})
                entry = entry.explode(['category', 'values'])
                df = pd.concat([df, entry], ignore_index=True)
        else:
            arr_stings = [','.join(i) for i in value.tolist()]
            entry = pd.DataFrame({'pdb_id': pdb_id, 'category':[categories], 'values':[arr_stings]})        
            entry = entry.explode(['category', 'values'])
            df = pd.concat([df, entry], ignore_index=True)
    df = df.dropna(axis=1)
    return df