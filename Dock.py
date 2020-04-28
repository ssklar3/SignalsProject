import os, sys
sys.path.append("/project/ppi_workspace/toil-env36/lib/python3.6/site-packages/")
os.environ["PYTHONPATH"] = "/project/ppi_workspace/toil-env36/lib/python3.6/site-packages/"

import argparse
import itertools as it
import subprocess
from datetime import datetime

import wandb

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger

from BM5Dataset import BM5Dataset
from DataAugmenter import DataAugmenter

from DockPair import DockPair

def main(hparams):
    # each trial has a separate version number which can be accessed from the cluster
    date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    version = date #if cluster is None else cluster.hpc_exp_number

    job_name = "{}-{}".format(hparams.prefix, date)

    tensorboard_logger = TensorBoardLogger("tb_logs", version=version, name=job_name)

    if torch.cuda.is_available() and (hparams.num_gpus == -1 or hparams.num_gpus > 0):
        if hparams.gpus is None or hparams.gpus == "":
            hparams.num_gpus = min(4, hparams.num_gpus)
            hparams.gpus = ",".join(map(str, range(hparams.num_gpus)))
        else:
            hparams.num_gpus = hparams.gpus.count(",")+1
        
        if hparams.distributed_backend is None:
            hparams.distributed_backend = "dp" #if hparams.num_gpus>1 else "ddp"
        if hparams.distributed_backend == "dp": #["dp", "ddp", "ddp2"]:
            hparams.batch_size *= hparams.num_gpus
    else:
        hparams.num_gpus = 0 #None
        hparams.gpus = ""
        hparams.distributed_backend = ""
        gpus = None
        distributed_backend = None
    
    if hparams.bm5_path is None:
        hparams.bm5_path = os.path.join(os.getcwd(), "BM5-fft")
    elif os.path.isdir(hparams.bm5_path):
        if os.path.isdir(os.path.join(hparams.bm5_path, "HADDOCK-ready")):
            pass
        else:
            hparams.bm5_path = os.path.join(hparams.bm5_path, "BM5-fft")
            
    

    bm5 = BM5Dataset(hparams.bm5_path, batch_size=1, num_workers=1, distributed=False)
    
    for i, (receptor, ligand) in enumerate(bm5):
        if i<1: continue
            
        dock_pair = DockPair(receptor, ligand, hparams)
        dock_trainer = Trainer(
            num_nodes=1, #hparams.num_nodes,
            #gpus=1, #xshparams.gpus,
            distributed_backend=None, #hparams.distributed_backend,
            logger=tensorboard_logger,
            early_stop_callback=False,
            num_sanity_val_steps=0,
            min_epochs=1, 
            max_epochs=1
        )
        dock_trainer.fit(dock_pair)
        dock_pair.get_complex()
        break

if __name__ ==  '__main__':
    # subclass of argparse
    parser = argparse.ArgumentParser()
    #parser = DockPair.add_model_specific_args(parser, os.getcwd())

    parser.add_argument('--bm5_path', default=None)
    parser.add_argument('--prefix', default="FFTDock")
    #parser.add_argument('--slurm_log_path', default=None, type=str, help='where slurm will save scripts to')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--gpus')
    parser.add_argument('--num_nodes', default=1, type=int, choices=list(range(1,5)))
    parser.add_argument('--distributed_backend', default=None, choices=["dp", "ddp", "ddp2"])
    parser.add_argument('--num_rotations', default=100, type=int)
    parser.add_argument('--batch_size', default=3, type=int)

    # compile (because it's argparse underneath)
    hparams = parser.parse_args()
    
    print(hparams)
    
    main(hparams)
