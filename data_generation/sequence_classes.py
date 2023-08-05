import utils


class Sequence:
    def __init__(self, pdb_id: str, seq_nums: list[int], res_names: list[str], res_nums: list[int]) -> None:
        self.pdb_id = pdb_id
        self.seq_nums = seq_nums
        self.res_names = res_names
        self.res_nums = res_nums
    
    def __len__(self):
        return len(self.seq_nums)
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Sequence):
            return False
        return self.id == __value.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    @property
    def res_seq(self):
        return ''.join(self.res_names)
    
    #* To remove redundancy, residue names and cluster ID are the only identifiers of uniqueness (PDB ID doesn't matter), as defined by the @id property. Account for tloop abundance later (include it as var in the Tetraloop object?)
    @property
    def id(self) -> tuple:
        return tuple([str(i) for i in [self.clust_id, self.res_names]])
    

class Tetraloop(Sequence):
    def __init__(self, pdb_id: str, clust_id: int, seq_nums: list[int], res_names: list[str], res_nums: list[int]) -> None:
        super().__init__(pdb_id, seq_nums, res_names, res_nums)
        self.clust_id = clust_id
    
    def __str__(self) -> str:
        return f'{self.pdb_id}_{self.clust_id}_{self.res_names[0]}{self.res_nums[0]}'
    
    # @property
    # def id(self) -> tuple:
    #     return tuple([str(i) for i in [self.pdb_id, self.clust_id, self.seq_nums, self.res_names, self.res_nums]])


class PDB(Sequence):
    def __init__(self, pdb_id: str, seq_nums: list[int], chain_ids: list[str], clust_ids: list[int], res_names: list[str], res_nums: list[int], ins_codes: list[str]) -> None:
        super().__init__(pdb_id, seq_nums, res_names, res_nums)
        self.chain_ids = chain_ids
        self.clust_ids = clust_ids
        self.ins_codes = ins_codes
    
    def __str__(self) -> str:
        return f'{self.pdb_id}\n{self.chain_ids}\n{self.clust_ids}\n{self.res_names}\n{self.res_nums}\n{self.ins_codes}\n'
    
    @property
    def id(self) -> tuple:
        return tuple([str(i) for i in [self.pdb_id, self.seq_nums, self.chain_ids, self.clust_ids, self.res_names, self.res_nums, self.ins_codes]])
    
    # Unique chains and their start + stop indices
    @property
    def chain_indices(self) -> list[tuple[str, int, int]]:
        return {chain_id: (self.chain_ids.index(chain_id), utils.list_rindex(self.chain_ids, chain_id)) for chain_id in set(self.chain_ids)}
    
    def remove_chain(self, chain_id: str) -> None:
        start_idx, stop_idx = self.chain_indices[chain_id]
        self.seq_nums, self.chain_ids, self.clust_ids, self.res_names, self.res_nums, self.ins_codes = tuple([i[:start_idx] + i[stop_idx+1:] for i in [self.seq_nums, self.chain_ids, self.clust_ids, self.res_names, self.res_nums, self.ins_codes]])


class Fragment(Sequence):
    def __init__(self, pdb_id: str, clust_id: int, chain_id: str, seq_nums: list[int], res_names: list[str], res_nums: list[int], ins_codes: list[str]) -> None:
        super().__init__(pdb_id, seq_nums, res_names, res_nums)
        self.clust_id = clust_id
        self.chain_id = chain_id
        self.ins_codes = ins_codes

    # @property
    # def id(self) -> tuple:
    #     return tuple([str(i) for i in [self.pdb_id, self.chain_id, self.clust_id, self.seq_nums, self.res_names, self.res_nums, self.ins_codes]])