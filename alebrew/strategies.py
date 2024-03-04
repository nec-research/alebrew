"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     strategies.py 
  Authors:  Viktor Zaverkin (viktor.zaverkin@neclab.eu) 
            David Holzmüller (david.holzmuller@inria.fr)
            Henrik Christiansen (henrik.christiansen@neclab.eu)
            Federico Errica (federico.errica@neclab.eu)
            Francesco Alesiani (francesco.alesiani@neclab.eu)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)
            Johannes Kästner (kaestner@theochem.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
import os
import time
import copy
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Tuple

import numpy as np
import torch

from alebrew.datagen.samplers import ParallelEnsembleSampler, get_sampler
from alebrew.datagen.selection import BatchSizeScaling, get_sel_method
from alebrew.data.data import AtomicStructures, AtomicTypeConverter
from alebrew.interfaces.ase import ASEWrapper
from alebrew.model.calculators import get_torch_calculator
from alebrew.model.forward import ForwardAtomisticNetwork, build_model, load_models_from_folder
from alebrew.training.callbacks import FileLoggingCallback
from alebrew.training.loss_fns import config_to_loss
from alebrew.training.trainer import Trainer, eval_metrics
from alebrew.utils.config import update_config
from alebrew.utils.misc import save_object, get_available_devices
from alebrew.utils.torch_geometric import DataLoader


class TrainingStrategy:
    """Strategy for training interatomic potentials.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration file with parameters listed in 'utils/config.py'. 
                                                     The default parameters of 'utils/config.py' will be updated by those 
                                                     provided in 'config'. Defaults to None.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Update config containing all parameters (including the model, training, fine-tuning, and evaluation)
        # We store all parameters in one config for simplicity. In the log and best folders the last training
        # config is stored and used when loading a model for inference.
        self.config = update_config(config.copy())

    def run(self,
            train_structures: AtomicStructures,
            valid_structures: AtomicStructures,
            folder: Union[str, Path],
            model_seeds: List[int] = [],
            pretrained_models: Optional[List[ForwardAtomisticNetwork]] = None) -> List[ForwardAtomisticNetwork]:
        """Runs training using provided training and validation structures.

        Args:
            train_structures (AtomicStructures): Training structures.
            valid_structures (AtomicStructures): Validation structures.
            folder (Union[str, Path]): Folder where trained models are stored.
            model_seeds (List[int], optional): List of random seeds to initialize models. 
                                               Defaults to [].
            pretrained_models (Optional[List[ForwardAtomisticNetwork]], optional): List of pre-trained models. These models can 
                                                                                   be used for a fine-tuning task. Defaults to None.

        Returns:
            List[ForwardAtomisticNetwork]: List of trained models.
        """
        # define atomic type converter
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        # convert atomic numbers to type names
        train_structures = train_structures.to_type_names(atomic_type_converter, check=True)
        valid_structures = valid_structures.to_type_names(atomic_type_converter, check=True)
        # store the number of training and validation structures in config
        self.config['n_train'] = len(train_structures)
        self.config['n_valid'] = len(valid_structures)
        # build atomic data sets
        train_ds = train_structures.to_data(r_cutoff=self.config['r_cutoff'])
        valid_ds = valid_structures.to_data(r_cutoff=self.config['r_cutoff'])
        # build models
        models = []
        if model_seeds:
            # update model seeds if provided (can be used to re-run a calculation with different seeds)
            self.config['model_seeds'] = model_seeds
        for model_seed in self.config['model_seeds']:
            models.append(build_model(train_structures, model_seed=model_seed,
                                      n_species=atomic_type_converter.get_n_type_names(), **self.config))
        if pretrained_models is not None:
            # load modules/parameters from pretrained models
            assert len(models) <= len(pretrained_models)
            models = [self.from_pretrained_model(model, pretrained_model)
                      for model, pretrained_model in zip(models, pretrained_models)]
        # define losses from config
        train_loss = config_to_loss(self.config['train_loss'])
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        early_stopping_loss = config_to_loss(self.config['early_stopping_loss'])
        # define callbacks to track training
        callbacks = [FileLoggingCallback()]
        # define learning rates
        base_lr = self.config['base_lr']
        # for embeddings
        lrs = [base_lr * self.config['emb_lr_factor']]
        # for the fully connected part
        if isinstance(self.config['weight_lr_factors'], float):
            lrs.extend([base_lr * self.config['weight_lr_factors']] * (2 * (len(self.config['hidden_sizes']) + 1)))
        if isinstance(self.config['weight_lr_factors'], list):
            if len(self.config['weight_lr_factors']) != (len(self.config['hidden_sizes']) + 1):
                raise ValueError(f'Provided {len(self.config["weight_lr_factors"])} learning rates and'
                                 f' {len(self.config["hidden_sizes"]) + 1} layers, which must be equal!')
            for lr_factor in self.config['weight_lr_factors']:
                lrs.extend([base_lr * lr_factor] * 2)
        if 'atomic_scale_shift' in self.config['output_tfms']:
            # for atomic scale and shift
            lrs.extend([base_lr * self.config['scale_lr_factor']] + [base_lr * self.config['shift_lr_factor']])
        # define model training
        trainer = Trainer(models, lrs=lrs, lr_sched=lambda t: 1. - t, model_path=folder, callbacks=callbacks,
                          max_epoch=self.config['max_epoch'], save_epoch=self.config['save_epoch'],
                          train_batch_size=min(self.config['train_batch_size'], len(train_structures)),
                          valid_batch_size=min(self.config['eval_batch_size'], len(valid_structures)),
                          train_loss=train_loss, eval_losses=eval_losses, early_stopping_loss=early_stopping_loss,
                          with_sam=self.config['with_sam'], rho_sam=self.config['rho_sam'], adaptive_sam=self.config['adaptive_sam'],
                          device=self.config['device'])
        # train the model
        trainer.fit(train_ds=train_ds, valid_ds=valid_ds)
        # return best models and move them to device
        models = load_models_from_folder(folder, len(models), key='best')
        models = [model.to(self.config['device']) for model in models]
        return models

    def from_pretrained_model(self, 
                              model: ForwardAtomisticNetwork, 
                              pretrained_model: ForwardAtomisticNetwork) -> ForwardAtomisticNetwork:
        """Loads parameters from pre-trained models.

        Args:
            model (ForwardAtomisticNetwork): Current model.
            pretrained_model (ForwardAtomisticNetwork): Pre-trained model.

        Returns:
            ForwardAtomisticNetwork: Current model initialized with parameters from the pre-trained one.
        """
        for module_name in self.config['pretrained_modules']:
            if hasattr(model, module_name) and hasattr(pretrained_model, module_name):
                getattr(model, module_name).load_state_dict(getattr(pretrained_model, module_name).state_dict())
            else:
                raise RuntimeError(f'{hasattr(model, module_name)=} while {hasattr(pretrained_model, module_name)=}. '
                                   f'Make sure that both models have modules you wish to initialize!')
        return model


class EvaluationStrategy:
    """Strategy for evaluating the performance of interatomic potentials.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration file with parameters listed in 'utils/config.py'. 
                                                     The default parameters of 'utils/config.py' will be updated by those 
                                                     provided in 'config'. Defaults to None.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Update config containing all parameters (including the model, training, fine-tuning, and evaluation)
        # We store all parameters in one config for simplicity. In the log and best folders the last training
        # config is stored and used when loading a model for inference.
        self.config = update_config(config.copy())

    def run(self,
            models: List[ForwardAtomisticNetwork],
            test_structures: AtomicStructures,
            folder: Union[str, Path]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluates models using the provided test data set.

        Args:
            models (List[ForwardAtomisticNetwork]): List of models.
            test_structures (AtomicStructures): Test structures.
            folder (Union[str, Path]): Folder where the evaluation results are stored.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: Results dictionary, comprising ensemble/average and individual error metrics.
        """
        folder = Path(folder)
        # apply property and ensemble calculators to models
        ens = get_torch_calculator(models).to(self.config['device'])
        # define atomic type converter
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        # convert atomic numbers to type names
        test_structures = test_structures.to_type_names(atomic_type_converter, check=True)
        # build atomic data sets
        test_ds = test_structures.to_data(r_cutoff=self.config['r_cutoff'])
        # define losses from config
        eval_losses = {l['type']: config_to_loss(l) for l in self.config['eval_losses']}
        eval_output_variables = list(set(sum([l.get_output_variables() for l in eval_losses.values()], [])))
        # evaluate model on the test data
        use_gpu = self.config['device'].startswith('cuda')
        test_dl = DataLoader(test_ds, batch_size=self.config['eval_batch_size'], shuffle=False, drop_last=False,
                             pin_memory=use_gpu, pin_memory_device=self.config['device'] if use_gpu else '')
        # evaluate metrics on test data and store results as a .json file
        test_metrics = eval_metrics(ens=ens, dl=test_dl, eval_loss_fns=eval_losses,
                                    eval_output_variables=eval_output_variables, device=self.config['device'])
        save_object(folder / f'test_results.json', test_metrics, use_json=True)
        return test_metrics['average'], test_metrics['individual']

    def measure_inference_time(self,
                               models: List[ForwardAtomisticNetwork],
                               test_structures: AtomicStructures,
                               folder: Union[str, Path],
                               batch_size: int = 100,
                               n_reps: int = 100) -> Dict[str, Any]:
        """Provide inference time for the defined batch size, i.e., atomic system size.

        Args:
            models (List[ForwardAtomisticNetwork]): List of models.
            test_structures (AtomicStructures): Test structures
            folder (Union[str, Path]): Folder where the results of the inference time measurement are stored.
            batch_size (int, optional): Evaluation batch size. Defaults to 100.
            n_reps (int, optional): Number of repetitions. Defaults to 100.

        Returns:
            Dict[str, Any]: Results dictionary.
        """
        folder = Path(folder)
        ens = get_torch_calculator(models).to(self.config['device'])
        atomic_type_converter = AtomicTypeConverter.from_type_list(self.config['atomic_types'])
        test_structures = test_structures.to_type_names(atomic_type_converter, check=True)
        test_ds = test_structures.to_data(r_cutoff=self.config['r_cutoff'])
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        batch = next(iter(test_dl)).to(self.config['device'])
        # need to re-iterate before time measurement
        for i in range(10):
            ens(batch, forces=True, features=False)
        # start with the time measurement
        start_time = time.time()
        for i in range(n_reps):
            ens(batch, forces=True, features=False)
        if self.config['device'].startswith('cuda'):
            torch.cuda.synchronize()
        end_time = time.time()
        to_save = {'total_time': end_time - start_time,
                   'time_per_repetition': (end_time - start_time) / n_reps,
                   'time_per_structure': (end_time - start_time) / n_reps / batch_size,
                   'time_per_atom': (end_time - start_time) / n_reps / batch.n_atoms.sum().item()}
        save_object(folder / f'timing_results.json', to_save, use_json=True)
        return to_save


class SamplingStrategy:
    """Strategy for sampling/exploring configurational spaces with atomistic simulations. We interface our 
    data generation workflow with the atomic simulation environment (ASE).
    
    Args:
        sampling_method (str): Sampling method: 'random', 'nvt', 'nh-nvt', 'npt', 'berendsen-npt', 
                               or 'adversarial'.
        sampling_configs (List[Dict[str, Any]]): Configurations file with parameters specific to the 
                                                 sampling method such as temperatures, external stresses (pressures),
                                                 integration time step, learning rate (for adversarial attacks), etc.
        unc_method (Optional[str], optional): Uncertainty quantification method: None, 'random', 'posterior',
                                              'distance', or 'ensemble'. Defaults to None.
        bias_method (Optional[str], optional): The method employed to introduce a bias to the potential energy surface 
                                               and, thus, bias the respective atomistic simulation: None or 'linear_unc'. 
                                               Defaults to None.
        pre_sampling_method (Optional[str], optional): Pre-sampling method used to generate the initial training 
                                                       data set: None or 'random'. Defaults to None.
        max_pre_sampling_step (int, optional): Maximal number of pre-sampling steps. Defaults to 8.
        pre_sampling_mask (Optional[np.ndarray], optional): If np.eye(3) is used, then the cell is scaled only 
                                                            along x-, y-, and z- axis. Defaults to np.eye(3).
        max_sampling_step (int, optional): Maximal number of sampling steps, i.e., iterations of the underlying 
                                           atomistic simulations. Defaults to 1000.
        eval_sampling_step (int, optional): Frequency of storing atomic structures explored/sampled during 
                                            an atomistic simulation. Defaults to 10.
        n_samplers (int, optional): Number of samplers. Defaults to 8.
        n_threads (int, optional): Number of parallel threads, i.e., number of samplers run in parallel. Defaults to 1.
    """
    def __init__(self,
                 sampling_method: str,
                 sampling_configs: List[Dict[str, Any]],
                 unc_method: Optional[str] = None,
                 bias_method: Optional[str] = None,
                 pre_sampling_method: Optional[str] = None,
                 max_pre_sampling_step: int = 8,
                 pre_sampling_mask: Optional[np.ndarray] = np.eye(3),
                 max_sampling_step: int = 1000,
                 eval_sampling_step: int = 10,
                 n_samplers: int = 8,
                 n_threads: int = 1,
                 **config: Any):
        self.sampler = get_sampler(sampling_method=sampling_method)
        self.sampling_configs = sampling_configs
        self.unc_method = unc_method
        self.bias_method = bias_method

        self.max_sampling_step = max_sampling_step
        self.eval_sampling_step = eval_sampling_step
        self.n_samplers = n_samplers
        self.n_threads = n_threads

        if pre_sampling_method is None:
            self.pre_sampler = None
        else:
            self.pre_sampler = get_sampler(sampling_method=pre_sampling_method)
        self.max_pre_sampling_step = max_pre_sampling_step
        self.pre_sampling_mask = pre_sampling_mask

        self.config = config.copy()

    def pre_run(self,
                structures: AtomicStructures,
                folder: Union[str, Path],
                seed: int = 1234) -> AtomicStructures:
        """Generates an initial training data set by randomly displacing atoms 
        and applying random strain deformations to the periodic cell.

        Args:
            structures (AtomicStructures): Atomic structures used to initiate the pre-sampling step, 
                                           also a single initial structure is possible.
            folder (Union[str, Path]): Folder where generated atomic structures are stored.
            seed (int, optional): Random seed to set up pre-sampling. Defaults to 1234.

        Returns:
            AtomicStructures: Randomly distorted atomic structures or an empty list if pre-sampling is not needed.
        """
        folder = Path(folder)
        if self.pre_sampler is None:
            split = structures.random_split({'init': self.max_pre_sampling_step}, seed=seed)
            return split['init']
        else:
            random_structures = AtomicStructures([])
            max_pre_sampling_step = self.max_pre_sampling_step // len(structures)
            for i, structure in enumerate(structures):
                sampler = self.pre_sampler(structure.to_atoms(), seed=seed, max_step=max_pre_sampling_step - 1,
                                           mask=self.pre_sampling_mask, **self.config)
                new_random_structures = sampler.run(folder=folder / f'{i}')
                random_structures = random_structures + new_random_structures
            return structures + random_structures

    def run(self,
            models: List[ForwardAtomisticNetwork],
            train_structures: AtomicStructures,
            valid_structures: AtomicStructures,
            init_structures: AtomicStructures,
            folder: Union[str, Path],
            seed: int = 1234) -> AtomicStructures:
        """Runs exploration using the provided interatomic potential (or an ensemble of them).

        Args:
            models (List[ForwardAtomisticNetwork]): List of models.
            train_structures (AtomicStructures): Training structures used for uncertainty calculation.
            valid_structures (AtomicStructures): Validation structures used for uncertainty calibration.
            init_structures (AtomicStructures): Selected structures from which next atomistic simulations
                                                are started.
            folder (Union[str, Path]): Folder where the results of sampling are stored.
            seed (int, optional): Random seed to set up sampling. Defaults to 1234.

        Returns:
            AtomicStructures: A list of explored/sampled atomic structures.
        """
        # define atomic type converter
        atomic_type_converter = AtomicTypeConverter.from_type_list(models[0].config['atomic_types'])
        # convert atomic numbers to type names
        train_structures = train_structures.to_type_names(atomic_type_converter, check=True)
        valid_structures = valid_structures.to_type_names(atomic_type_converter, check=True)
        # build atomic data sets
        train_ds = train_structures.to_data(r_cutoff=models[0].config['r_cutoff'])
        valid_ds = valid_structures.to_data(r_cutoff=models[0].config['r_cutoff'])
        # get list of available devices, create a cycle object for devices and sampling configs
        devices_cycle = cycle(get_available_devices())
        sampling_configs_cycle = cycle(self.sampling_configs)
        # define an ensemble of samplers
        n_samplers = min(self.n_samplers, len(init_structures))
        samplers = []
        for i_sampler, structure in enumerate(init_structures[:n_samplers]):
            # get next available device
            next_device = next(devices_cycle)
            sampling_config = next(sampling_configs_cycle)
            # deepcopy models, build torch calculator and move everything to next device
            next_models = [copy.deepcopy(model).to(next_device) for model in models]
            next_calc = get_torch_calculator(next_models, unc_method=self.unc_method, bias_method=self.bias_method,
                                             train_ds=train_ds, valid_ds=valid_ds, **self.config).to(next_device)
            # wrap torch calculator to ase calculator and provide the next available device to move the model to it
            wrapped_calc = ASEWrapper(next_calc, models[0].config['r_cutoff'], models[0].config['atomic_types'],
                                      device=next_device, neighbors=models[0].config['neighbors'],
                                      unc_method=self.unc_method, bias_method=self.bias_method, **self.config)
            samplers.append(self.sampler(structure.to_atoms(), wrapped_calc, seed=seed+i_sampler,
                                         max_step=self.max_sampling_step, eval_step=self.eval_sampling_step,
                                         **sampling_config, **self.config))
        # sample new configurations
        n_threads = min(self.n_threads, n_samplers)
        parallel_sampler = ParallelEnsembleSampler(samplers, n_threads=n_threads)
        new_structures = parallel_sampler.run(folder=folder)
        return new_structures


class SelectionStrategy:
    """Strategy for selecting a batch of configurations from candidate pools generated, e.g., by running 
    atomistic simulations. The available selection strategies select structures based on their uncertainty 
    values and enforce diversity (and representativeness), depending on the chosen selection method.

    Args:
        sel_method (str): Selection method: 'random', 'max_diag', 'max_det', 'max_dist', or 'lcmd'.
        unc_method (str): Uncertainty method: 'random', 'posterior', 'distance', 'ensemble'.
        eval_batch_size (int, optional): Evaluation batch size. Adjust it for controlling the RAM usage. 
                                         Defaults to 100.
        retain_neighbors (bool, optional): If True, store neighbor lists during the calculation. 
                                           Choosing False reduces the RAM overload. Defaults to False.
    """
    def __init__(self,
                 sel_method: str,
                 unc_method: str,
                 eval_batch_size: int = 100,
                 retain_neighbors: bool = False,
                 **config: Any):
        self.sel_method = get_sel_method(sel_method=sel_method)
        self.unc_method = unc_method

        # compute desired selected batch size
        self.sel_batch_size_scaling = BatchSizeScaling(**config)

        # parameters for evaluating features on all sampled structures (may overload RAM because of neighbor
        # list storing)
        self.eval_batch_size = eval_batch_size
        self.retain_neighbors = retain_neighbors

        # other parameters specific for sampling, uncertainty, and bias methods
        self.config = config.copy()

    def run(self,
            models: List[ForwardAtomisticNetwork],
            pool_structures: AtomicStructures,
            train_structures: AtomicStructures,
            valid_structures: AtomicStructures,
            folder: Union[str, Path],
            min_sel_data_size: Optional[int] = None,
            max_sel_data_size: Optional[int] = None) -> Tuple[AtomicStructures, AtomicStructures]:
        """Selects candidates for the provided data pool.

        Args:
            models (List[ForwardAtomisticNetwork]): List of models.
            pool_structures (AtomicStructures): Pool structures.
            train_structures (AtomicStructures): Training structures.
            valid_structures (AtomicStructures): Validation structures.
            folder (Union[str, Path]): Folder where selected structures are stored.
            min_sel_data_size (Optional[int], optional): Minimal number of selected structures. Defaults to None.
            max_sel_data_size (Optional[int], optional): Maximal number of selected structures. Defaults to None.

        Returns:
            Tuple[AtomicStructures, AtomicStructures]: Selected and remaining atomic structures.
        """
        folder = Path(folder)
        if os.path.exists(folder / 'selected_idxs.npy'):
            # check if selection step has already been performed
            selected_idxs = np.load(folder / 'selected_idxs.npy')
            selected_structures, remaining_structures = pool_structures.split_by_indices(selected_idxs)
        else:
            # define atomic type converter
            atomic_type_converter = AtomicTypeConverter.from_type_list(models[0].config['atomic_types'])
            # convert atomic numbers to type names
            train_structures = train_structures.to_type_names(atomic_type_converter, check=True)
            valid_structures = valid_structures.to_type_names(atomic_type_converter, check=True)
            pool_structures = pool_structures.to_type_names(atomic_type_converter, check=True)
            # build atomic data sets for calibration and initialize unc_calc
            train_ds = train_structures.to_data(r_cutoff=models[0].config['r_cutoff'])
            # build torch calculator
            if 'compute_atomic_unc' in self.config:
                self.config['compute_atomic_unc'] = False
            calc = get_torch_calculator(models, unc_method=self.unc_method, train_ds=train_ds,
                                        **self.config).to(models[0].config['device'])
            # scale batch size according to the provided rule
            sel_batch_size = self.sel_batch_size_scaling(len(pool_structures), len(train_structures))
            # update the selection batch size by the minimal and maximal amount of selected data
            if min_sel_data_size is not None:
                sel_batch_size = max(sel_batch_size, min_sel_data_size)
            if max_sel_data_size is not None:
                sel_batch_size = min(sel_batch_size, max_sel_data_size)
            # combined train and valid data sets to make sure that no data similar to valid is selected
            selected_idxs = self.sel_method.select_from_structures(calc=calc, pool_structures=pool_structures,
                                                                   train_structures=train_structures + valid_structures,
                                                                   sel_batch_size=sel_batch_size,
                                                                   r_cutoff=models[0].config['r_cutoff'],
                                                                   eval_batch_size=self.eval_batch_size,
                                                                   retain_neighbors=self.retain_neighbors)
            # convert selected structures back to atomic numbers
            selected_idxs = selected_idxs.detach().cpu().numpy()
            selected_structures, remaining_structures = pool_structures.split_by_indices(selected_idxs)
            selected_structures = selected_structures.to_atomic_numbers(atomic_type_converter)
            remaining_structures = remaining_structures.to_atomic_numbers(atomic_type_converter)
            # store selected structures and indices
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(folder / 'selected_idxs.npy', selected_idxs)
            selected_structures.save_extxyz(folder / 'selected_structures.extxyz')
        return selected_structures, remaining_structures
