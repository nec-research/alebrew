import os

import numpy as np

from alebrew.data.data import AtomicStructures
from alebrew.datagen.references import SingleCalculatorReference
from alebrew.interfaces.ase import TorchMDCalculator
from alebrew.task_execution import CustomPaths, Task, Learner

if __name__ == '__main__':
    custom_paths = CustomPaths(folder='.')    # set custom paths
    
    init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_init.extxyz')

    calc = TorchMDCalculator(input_file=os.path.join(str(custom_paths.data_path), 'input.pdb'),
                            parm_file=os.path.join(str(custom_paths.data_path), 'parm7.prmtop'),
                            r_cutoff=9.0)

    reference = SingleCalculatorReference(calc=calc, n_threads=1, force_threshold=20.0)
                                            
    test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_test.extxyz')

    task = Task(task_name='ala2-300K-ffs', structures=init_structures, use_precomp_labels=False,
                init_structures=init_structures, reference=reference, r_cutoff=5.0,
                atomic_types=['H', 'C', 'N', 'O'], timestep=0.5, temperatures=[300.0],
                externalstresses=[0.0], mask=None, test_structures=test_structures)
    
    unbiased_learner = Learner(method_name='random_8_0.02-nvt_1.5_20000_200_8-posterior-max_det_8_512-0', 
                            pre_sampling_method='random', max_pre_sampling_step=8, amplitude_shift=0.02,
                            sampling_method='nvt', friction=0.02, atomic_unc_threshold=1.5, eval_sampling_step=200, 
                            max_sampling_step=20000, n_samplers=8, n_threads=8, write_traj_details=True, 
                            write_traj_properties=['energy', 'forces'], unc_method='posterior', posterior_sigma=0.01, 
                            compute_atomic_unc=True, atomic_unc_residual='forces_rmse', conformal_alpha=0.05, 
                            use_grad_features=True, n_random_projections=512, sel_method='max_det', 
                            max_sel_batch_size=8, calibr_batch_size=32, eval_batch_size=32, max_data_size=512, 
                            valid_fraction=0.1, data_seed=0, sampling_seed=0, model_seeds=[0], custom_paths=custom_paths)

    unbiased_learner.run_on_task(task)
