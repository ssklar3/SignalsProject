import sys
import os
import subprocess
import traceback
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data.dataset import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader

from toil.job import Job

from molmimic.common.voxels import ProteinVoxelizer
from molmimic.common.featurizer import ProteinFeaturizer

def get_structure(pdb_file, ligand=False):
    """Read structure and features (or calculate if not present)"""
    
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    features_path = os.path.dirname(pdb_file)
    
    if not os.path.isfile(os.path.join(features_path, "{}_atom.h5".format(pdb_name))):
        #Calculate features
        protein = ProteinFeaturizer(
            pdb_file, 
            pdb_name, 
            Job(), 
            os.path.dirname(pdb_file), 
            force_feature_calculation=True)
        
        #Only calculate accesible surface area
        [protein.get_accessible_surface_area_residue(protein._remove_altloc(a)) for a in protein.get_atoms()]
        
        protein.write_features()
        
    protein = ProteinVoxelizer(
        pdb_file, 
        pdb_name, 
        features_path=features_path,
        ligand=ligand,
        volume=65)

    return protein

class BM5Dataset(Dataset):
    def __init__(self, path, read_strutures=True, batch_size=4, num_workers=1, distributed=False):
        self.path = path
        self.read_strutures = read_strutures
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "HADDOCK-ready")):
            #Path is correct
            pass
        elif os.path.isdir(os.path.dirname(path)) and not os.path.isdir(path):
            #Download to this direcotry
            subprocess.call(["git", "checkout", "https://github.com/haddocking/BM5-clean.git", path])
        else:
            raise RuntimeError("Must include proper path")
        
        self.files_path = os.path.join(path, "HADDOCK-ready")
            
        _, pdbs, _ = next(os.walk(self.files_path))
        self.pdbs = list(sorted([pdb for pdb in pdbs if pdb not in ["data", "scripts", "ana_scripts"]]))
        

    def _get_data_loader(self):
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(self)

        return DataLoader(self, batch_size=self.batch_size, sampler=sampler, \
            shuffle=True, num_workers=self.num_workers, drop_last=True)
    
    def __len__(self):
        return len(self.pdbs)
    
    def __getitem__(self, index):
        pdb = self.pdbs[index]
        receptor_file = os.path.join(self.files_path, pdb, "{}_r_u.pdb".format(pdb))
        ligand_file = os.path.join(self.files_path, pdb, "{}_l_u.pdb".format(pdb))
        
        if not self.read_strutures:
            return receptor_file, ligand_file
        
        receptor = get_structure(receptor_file)
        ligand = get_structure(ligand_file, ligand=True)
        
        return receptor, ligand
            