import os
import multiprocessing
from collections import OrderedDict

import numpy as np
import torch
import pytorch_lightning as pl

from toil.job import Job

from DataAugmenter import DataAugmenter, voxelize
from molmimic.util.pdb import tidy
#from molmimic.parsers.CNS import Minimize

def roll_n(X, axis, n):
    """https://github.com/tomrunia/PyTorchSteerablePyramid/blob/0b6514d81f669b52767689a5780c88087ea2c191/steerable/math_utils.py"""
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift(x):
    real, imag = torch.unbind(x, -1)
    print(real.size(), imag.size())
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def normalize(v, eps=1e-6):
    norm=np.linalg.norm(v, ord=1)
    return v/norm if norm!=0 else v/eps

class DockPair(pl.LightningModule):

    def __init__(self, receptor, ligand, hparams, rfft=True):
        # init superclass
        super(DockPair, self).__init__()
        
        self.hparams = hparams
        
        self.rotation = None
        self.translation = None
        self.energy = None
        
        self.receptor = receptor
        self.ligand = ligand
        
        self.receptor_size = self.receptor.get_max_length(buffer=25)
        self.ligand_size = self.ligand.get_max_length(buffer=25)
        
        if self.ligand_size > self.receptor_size:
            #Resize receptor
            self.receptor.resize_volume(self.ligand_size)
            self.ligand.resize_volume(self.ligand_size)
            
            #Swap receptor and ligand so receptor is always largest
            ligand_ = self.ligand
            ligand_size_ = self.ligand_size
            self.ligand = self.receptor
            self.receptor = ligand_
            self.ligand_side = self.receptor_size
            self.receptor_size = ligand_size_
        else:
            #Resize ligand
            self.receptor.resize_volume(self.receptor_size)
            self.ligand.resize_volume(self.receptor_size)
            
        #Voxelize receptor
        self.receptor_volume = voxelize(self.receptor)
        
        self.rfft = rfft

        #Create Fourier image of receptor
        if rfft and self.receptor_volume.size()[-1] == 2:
            self.receptor_volume = self.receptor_volume[:, : , :, 0].resize(*self.receptor_volume.size()[:3])
            self.receptor_fft = torch.rfft(self.receptor_volume, 3).cpu()
        else:
            self.receptor_fft = torch.fft(self.receptor_volume, 3).cpu()
        
        #Take the complex conjugate of receptor
        self.receptor_fft = self.receptor_fft.conj()
        #self.receptor_fft[:,1] *= -1
    
    def get_complex(self):
        assert self.energy is not None
        
        def write(r, l, h, f):
            with open(f, "w") as complex_pdb:
                print(h, file=complex_pdb)
                for line in r:
                    if not line.startswith("END"):
                        print(line.rstrip(), file=complex_pdb)
                for line in l:
                    print(line.rstrip(), file=complex_pdb)
        
        #Rotate ligand to best orientation
        next(self.ligand.rotate(self.rotation))

        complex_file = "{}_predicted_complex.pdb".format(os.path.basename(self.receptor.path).split("_")[0])
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, "", complex_file+".notrans.pdb")

        #self.ligand.shift_coords(self.receptor
        self.ligand.shift_coords(self.translation, from_origin=False)

        header = "REMARK Best Energy={}; Best Translation={}; \n".format(self.energy, self.translation)
        header += "REMARK Best Rotation Matrix:\n"
        for line in str(self.rotation).splitlines():
            header += "REMARK     "+line.rstrip()+"\n"
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, header, complex_file)
        
        self.ligand.shift_coords_to_volume_center()
        
        t2 = []
        for i in self.translation:
            if i>self.ligand.volume/2:
                t2.append(self.ligand.volume-i)
            else:
                t2.append(i)
        self.ligand.shift_coords(t2, from_origin=False)
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, header, complex_file+".shift.pdb")
        
        self.ligand.shift_coords_to_volume_center()
        
        t2 = []
        for i in self.translation:
            t2.append(self.ligand.volume-i)
        self.ligand.shift_coords(t2, from_origin=False)
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, header, complex_file+".shift2.pdb")
        
        self.ligand.shift_coords_to_volume_center()
        
        t2 = []
        for i in self.translation:
            t2.append(self.ligand.volume/2-i)
        self.ligand.shift_coords(t2, from_origin=False)
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, header, complex_file+".shift3.pdb")
        
        self.ligand.shift_coords_to_volume_center()
        
        t2 = np.array(self.translation)-np.array([self.ligand.volume/2]*3)
        self.ligand.shift_coords(t2, from_origin=False)
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, header, complex_file+".shift4.pdb")
        
        self.ligand.shift_coords_to_volume_center()
        
        self.ligand.shift_coords(self.translation, from_origin=True)
        
        receptor_pdb = self.receptor.save_pdb(file_like=True)
        ligand_pdb = self.ligand.save_pdb(file_like=True)
        
        write(receptor_pdb, ligand_pdb, header, complex_file+".shift4.pdb")
        
        
        #Minimize using CNS
        #complex_file, cns_results = Minimize(complex_file, work_dir=os.dirname(self.receptor.path), job=Job())

        print(list(self.receptor.structure.get_chains()), list(self.ligand.structure.get_chains()))

    def forward(self, rvs, ligand_volumes, shift=False):
        #Create Fourier image of ligand
        if self.rfft and ligand_volumes.size()[-1] == 2:
            ligand_volumes = ligand_volumes[:, :, : , :, 0].resize(*ligand_volumes.size()[:-1])
            ligand_fft = torch.rfft(ligand_volumes, 3)
        else:
            ligand_fft = torch.fft(ligand_volumes, 3)
            
        #Create Fourier image of rotated ligand
        #ligand_fft = torch.fft(ligand_volumes, 3)
        
        #Calculate energy using the convolution thm and correlation
        if self.rfft:
            energy = torch.irfft(self.receptor_fft.to(ligand_fft.device)*ligand_fft, 3)
        else:
            energy = torch.ifft(self.receptor_fft.to(ligand_fft.device)*ligand_fft, 3)
        
            if shift:
                energy = batch_ifftshift(energy)

            energy = energy[:, :, :, 0]
        
        print(energy)
        print(energy.size())

        #energy = energy.reshape(energy.size()[:-1])
        
        #Get index with lowest energy, 
        maxval_z, ind_z = torch.max(energy, dim=3, keepdim=False)
        maxval_y, ind_y = torch.max(maxval_z, dim=2)
        maxval_x, ind_x = torch.max(maxval_y, dim=1)
        maxval_batch, ind_batch = torch.max(maxval_x, dim=0)
        
        #print(ind_x)
        
        batch = ind_batch.item()
        x = ind_x[batch].item()
        y = ind_y[batch, x].item()
        z = ind_z[batch, x, y].item()
        
        translation = (x, y, z)
        rotation = rvs[batch]
        low_energy = energy[batch, x, y, z]
        
        return translation, rotation, low_energy

    def training_step(self, batch, batch_num):
        rvs, volumes = batch
        translation, rotation, energy = self.forward(rvs, volumes)
        
        #return translation, rotation, energy
        
        if self.energy is None or energy>self.energy:
            self.rotation = rotation
            self.translation = translation
            self.energy = energy
    
        tqdm_dict = {'loss': energy.item(), 'energy': self.energy, "translation":translation}
        output = OrderedDict({
            'loss': energy,
            'progress_bar': tqdm_dict,
            'log': {'loss': energy.item(), 'energy': self.energy}
        })
        return output
    
    def test_step(self, batch, batch_num):
        return self.training_step(batch, batch_num)
    
    def test_step_end(self, batch):
        print(batch)
        return batch
    
    def backward(self, closure_loss, optimizer, opt_idx):
        return
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i,
      second_order_closure=None):
        return

#     def training_end(self, batch):
#         print(batch)
#         print(batch[0][2])
#         translation, rotation, energy = min(batch, key=lambda b: b[2]) 
        
#         if self.energy is None or energy<self.energy:
#             self.rotation = rotation
#             self.translation = translation
#             self.energy = energy

#         tqdm_dict = {'energy': self.energy, "translation":self.translation}
#         output = OrderedDict({
#             'loss': energy,
#             'progress_bar': tqdm_dict,
#             'log': tqdm_dict
#         })

#         return output
    

    def train_dataloader(self):
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))-1
        return DataAugmenter(
            self.ligand,
            num_rotations=self.hparams.num_rotations,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            distributed=self.hparams.distributed_backend in ["ddp", "ddp2"]
            )._get_data_loader()
    
    def test_dataloader(self):
        return self.train_dataloader()
    
    def configure_optimizers(self):
        return [None]