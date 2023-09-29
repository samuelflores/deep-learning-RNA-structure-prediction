class Sequence:
    def __init__(self, pdb_id: str, seq_nums: list[int], res_names: list[str], res_nums: list[int]) -> None:
        self.pdb_id = pdb_id
        self.seq_nums = seq_nums
        self.res_names = res_names
        self.res_nums = res_nums
        self.id = '' # String used for hashing object
    
    def __len__(self):
        return len(self.seq_nums)
    
    def __lt__(self, obj) -> bool:
        if not isinstance(obj, Sequence):
            return False
        return (len(self)) < (len(obj)) #? Is this the right way to do this? Does it conflict with __eq__
    
    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, Sequence):
            return False
        return self.id == obj.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    @property
    def res_seq(self):
        return ''.join(self.res_names)


class Tetraloop(Sequence):
    def __init__(self, pdb_id: str, clust_id: int, seq_nums: list[int], res_names: list[str], res_nums: list[int]) -> None:
        super().__init__(pdb_id, seq_nums, res_names, res_nums)
        self.clust_id = clust_id


class Chain(Sequence):
    def __init__(self, pdb_id: str, chain_id: str, seq_nums: list[int], clust_ids: list[int], res_names: list[str], res_nums: list[int], ins_codes: list[str]) -> None:
        super().__init__(pdb_id, seq_nums, res_names, res_nums)
        self.chain_id = chain_id
        self.seq_id = f'{self.pdb_id}_{self.chain_id}'
        self.clust_ids = clust_ids
        self.ins_codes = ins_codes
        self.tloops = []
    
    def align_tetraloop(self, tloop: Tetraloop) -> None:
        possible_idxs = [idx for idx, res_num in enumerate(self.res_nums) if res_num == tloop.res_nums[0]]
        for idx in possible_idxs:
            idx_res_names, idx_res_nums = self.res_names[idx:idx+8], self.res_nums[idx:idx+8]
            if (
                len(idx_res_names) == 8 and idx_res_names == tloop.res_names and
                len(idx_res_nums) == 8 and idx_res_nums == tloop.res_nums
                ):
                self.clust_ids[idx] = tloop.clust_id
                self.tloops += [tloop] # Add tloop to list, including the alignment position


class Fragment(Sequence):
    def __init__(self, pdb_id: str, clust_id: int, chain_id: str, seq_nums: list[int], res_names: list[str], res_nums: list[int], ins_codes: list[str]) -> None:
        super().__init__(pdb_id, seq_nums, res_names, res_nums)
        self.clust_id = clust_id
        self.chain_id = chain_id
        self.seq_id = f'{self.pdb_id}_{self.chain_id}'
        self.ins_codes = ins_codes


class PDBAlignment:
    def __init__(self, qseqid, sseqid, pident, qstart, qend, sstart, send) -> None:
        self.qseqid = qseqid
        self.sseqid = sseqid
        self.pident = pident
        self.qstart = qstart
        self.qend = qend
        self.sstart = sstart
        self.send = send