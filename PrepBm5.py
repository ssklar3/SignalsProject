#! /project/ppi_workspace/toil-env36/bin/python
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A muragroup
#SBATCH -t 10:00:00
#SBATCH -J bm5-clean
#SBATCH --mem=120000
#SBATCH --cpus-per-task=4
#SBATCH --array=1-463

import os, sys
sys.path.append("/project/ppi_workspace/toil-env36/lib/python3.6/site-packages")
sys.path.append("/project/ppi_workspace/notbooks/fft-ppi/SignalsProject")
from shutil import copyfileobj
from molmimic.generate_data.prepare_protein import extract_domain, prepare_domain
from toil.job import Job
from molmimic.common.voxels import ProteinVoxelizer
from molmimic.common.featurizer import ProteinFeaturizer
from molmimic.util.pdb import get_first_chain
from BM5Dataset import BM5Dataset

def prep_structure(pdb_file):
    pdb_fname = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_fname)[0]
    pdb_id = pdb_name.split("_")[0]
    chain = get_first_chain(pdb_file)
    
    features_path = os.path.join(os.getcwd(), "BM5-fft", pdb_id)
    if not os.path.isdir(features_path):
        os.makedirs(features_path)
    
    cleaned_file, _ = extract_domain(pdb_file, pdb_name, "BM5-fft", chain=chain, work_dir=features_path)
    prepared_file, _ = prepare_domain(cleaned_file, chain, pdb_name, work_dir=features_path, job=Job())
            
    protein = ProteinFeaturizer(
        prepared_file, 
        pdb_name, 
        Job(), 
        features_path, 
        force_feature_calculation=True,
        reset_chain=True
    )

    try:
        protein.calculate_flat_features()
    except:
        #Only calculate accesible surface area
        [protein.get_accessible_surface_area_residue(protein._remove_altloc(a)) for a in protein.get_atoms()]

    protein.write_features()
    
if __name__ == "__main__":
    bm5 = BM5Dataset(os.path.join(os.getcwd(), "BM5-clean"), batch_size=1, num_workers=1, read_strutures=False, distributed=False)
    stop = False
    i = 0
    for proteins in bm5:
        for p in proteins:
            if i==int(os.environ["SLURM_ARRAY_TASK_ID"])-1:
                prep_structure(p)
                stop = True
                break
            i += 1
        if stop:
            break