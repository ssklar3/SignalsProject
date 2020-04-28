import sys
import os
import traceback
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data.dataset import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader

import numpy as np
from scipy.stats import special_ortho_group

def collate(data):
    rvs, volumes = zip(*data)
    rvs = np.stack(rvs, axis=0)
    volumes = torch.stack(volumes, axis=0)
    return rvs, volumes

def voxelize(protein, ligand=False, plot=False, size=264):
    """Rp=−1  on a surface layer and Rp=1 on the core of the receptor,
        Lp=1 on the entire ligand, and Rp=Lp=0 everywhere else. It is clear that
        this scoring function, which is essentially the one used by
        Katchalski-Katzir et al. (5), reaches its minimum on a conformation in
        which the ligand maximally overlaps with the surface layer of the receptor,
        thus providing optimal shape complementarity. https://doi.org/10.1073/pnas.1603929113"""
    indices, data, _, _, _ = protein.map_atoms_to_voxel_space(
        autoencoder=True,
        only_surface=False,
        simple_fft=True)
    volume = np.zeros((protein.volume,protein.volume,protein.volume,2))
    x, y, z = zip(*indices.astype(int).tolist())
    volume[x,y,z] = data
    del indices, data, _
    
    #volume = volume.reshape(size,size,size)
    
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(volume!=0, facecolors='g' if ligand else 'b', edgecolor='g' if ligand else 'b')
        plt.show()
    
    volume = torch.from_numpy(volume)

    return volume

class DataAugmenter(Dataset):
    def __init__(self, ligand, num_rotations=100, batch_size=4, num_workers=1, distributed=False):
        self.ligand = ligand
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.num_rotations = num_rotations
        
        #Set to standard orientation
        self.ligand.orient_to_pai()
    
    def _get_data_loader(self):
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(self)

        return DataLoader(self, batch_size=self.batch_size, sampler=sampler, \
            shuffle=True, num_workers=self.num_workers, \
            collate_fn=collate, drop_last=True)
    
    def __len__(self):
        return self.num_rotations
    
    def __getitem__(self, index):
        rot_mat = special_ortho_group.rvs(3)
        
        #Rotate ligand
        next(self.ligand.rotate(rot_mat))
        
        #Map rotated ligand to voxels
        ligand_volume = voxelize(self.ligand, ligand=True)
        
        #Reset to standard orientation
        #self.ligand.orient_to_pai()
        next(self.ligand.rotate(np.linalg.inv(rot_mat)))
        
        return rot_mat, ligand_volume