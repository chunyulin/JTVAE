import nvtx
import torch
from torch.utils.data import Dataset, DataLoader
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from fast_jtnn.jtmpn import JTMPN
import pickle
import os, random

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from prefetch_generator import background

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

def split_list_into_n(data_list, n):
    if n <= 0:  raise ValueError("n_parts must be greater than 0")
    k = len(data_list) // n
    r = len(data_list) % n
    return [
        data_list[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)]
        for i in range(n)
    ]

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, mult_gpus=False, assm=True, replicate=None):
        NP = dist.get_world_size()
        self.data_folder = data_folder
        inpfiles = [fn for fn in os.listdir(data_folder)]

        #random.shuffle(self.data_files)   ### comment this out to clearly see that each GPU processes the whole inputs
        rank = dist.get_rank()
        self.data_files = split_list_into_n(inpfiles, NP)[rank]
        #print(self.data_files)

        print(f"Rank {rank} processes files: ", self.data_files)

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        self.mult_gpus = mult_gpus

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    @background()
    def __iter__(self):   ### for batch in loader:
        for fn in self.data_files:

            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)   ### a list of MolTreeObject

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            print(f"G:{int(os.environ['LOCAL_RANK'])}  Processing {fn} having {len(data)} data ...")

            ##print("LDATA(100): ", len(data))  ##CY# First file has 57 
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]

            ## why we waste some training data here??!!
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ##print("Getting ", idx, "in", self.data[idx])
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder  ### redudent !!???
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)

    ### Deprecated syntax: batch_idx = torch.LongTensor(batch_idx)
    batch_idx = torch.tensor(batch_idx, dtype=torch.long)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            ##print("--", node.wid,  node.smiles)
            tot += 1
