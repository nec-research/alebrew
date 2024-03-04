"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     trainer.py 
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
import shutil
import time
from pathlib import Path
from typing import Dict, Callable, Union, List, Optional, Any

import torch
import numpy as np

from alebrew.data.data import AtomicData
from alebrew.model.forward import ForwardAtomisticNetwork
from alebrew.model.calculators import EnsembleCalculator, get_torch_calculator
from alebrew.training.callbacks import TrainingCallback
from alebrew.training.loss_fns import LossFunction, TotalLossTracker
from alebrew.training.sam import SAM
from alebrew.utils.torch_geometric import Data
from alebrew.utils.torch_geometric.dataloader import DataLoader
from alebrew.utils.misc import load_object, save_object, get_default_device


def eval_metrics(ens: EnsembleCalculator,
                 dl: DataLoader,
                 eval_loss_fns: Dict[str, LossFunction],
                 eval_output_variables: List[str],
                 device: str = 'cuda:0',
                 early_stopping_loss_fn: Optional[LossFunction] = None) -> Dict[str, Any]:
    """Evaluates error metrics on the provided data.

    Args:
        ens (EnsembleCalculator): Ensemble calculator for atomistic models, see `calculators.py`.
        dl (DataLoader): Atomic data loader.
        eval_loss_fns (Dict[str, LossFunction]): Loss functions defined for evaluating model's 
                                                 performance.
        eval_output_variables (List[str]): Output variables: energy, forces, etc.
        device (str, optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to 'cuda:0'.
        early_stopping_loss_fn (Optional[LossFunction], optional): Optional early stopping loss 
                                                                   (used, e.g., during training). 
                                                                   Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary with values for evaluation metrics provided by loss functions.
    """
    metrics = {}

    avg_loss_trackers = {name: TotalLossTracker(loss_fn, requires_grad=False)
                         for name, loss_fn in eval_loss_fns.items()}

    indiv_loss_trackers = [{name: TotalLossTracker(loss_fn, requires_grad=False)
                            for name, loss_fn in eval_loss_fns.items()}
                           for _ in range(len(ens.calcs))]

    if early_stopping_loss_fn is not None:
        early_stopping_loss_trackers = [TotalLossTracker(early_stopping_loss_fn, requires_grad=False)
                                        for _ in range(len(ens.calcs))]
    else:
        early_stopping_loss_trackers = None

    n_structures_total = 0
    n_atoms_total = 0

    for batch_idx, batch in enumerate(dl):
        n_structures_total += len(batch.n_atoms)
        n_atoms_total += batch.n_atoms.sum().item()

        results = ens(batch.to(device), forces='forces' in eval_output_variables,
                      stress='stress' in eval_output_variables,
                      virials='virials' in eval_output_variables,
                      features=False, create_graph=True)

        for result, loss_trackers in zip(results['individual_results'], indiv_loss_trackers):
            for loss_tracker in loss_trackers.values():
                loss_tracker.append_batch(result, batch)

        if early_stopping_loss_fn is not None:
            for result, loss_tracker in zip(results['individual_results'], early_stopping_loss_trackers):
                loss_tracker.append_batch(result, batch)

        for loss_tracker in avg_loss_trackers.values():
            loss_tracker.append_batch(results, batch)

    metrics['individual'] = [{name: loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item()
                              for name, loss_tracker in loss_trackers.items()} for loss_trackers in indiv_loss_trackers]

    metrics['average'] = {name: loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item()
                          for name, loss_tracker in avg_loss_trackers.items()}

    if early_stopping_loss_fn is not None:
        metrics['early_stopping'] = [loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item()
                                     for loss_tracker in early_stopping_loss_trackers]

    return metrics


class Trainer:
    """Trains an atomistic model using the provided training data set. It also uses early stopping to prevent overfitting.

    Args:
        models (List[ForwardAtomisticNetwork]): List of atomistic models.
        lrs (List[float]): List of learning rates.
        lr_sched (Callable[[float], float]): Learning rate schedule.
        model_path (str): Path to the model.
        train_loss (LossFunction): Train loss function.
        eval_losses (Dict[str, LossFunction]): Evaluation loss function.
        early_stopping_loss (LossFunction): Early stopping loss function.
        device (Optional[str], optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to None.
        max_epoch (int, optional): Maximal training epoch. Defaults to 1000.
        save_epoch (int, optional): Frequency for storing models for restarting. Defaults to 100.
        validate_epoch (int, optional): Frequency for evaluating models on validation data set and storing 
                                        best models, if requested.  Defaults to 1.
        train_batch_size (int, optional): Training mini-batch size. Defaults to 32.
        valid_batch_size (int, optional): Validation mini-batch size. Defaults to 100.
        callbacks (Optional[List[TrainingCallback]], optional): Callbacks to track training process. 
                                                                Defaults to None.
        opt_class (optional): Optimizer class. Defaults to torch.optim.Adam.
        with_sam (bool, optional): If True, SAM is used for adversarial training. Defaults to False.
        rho_sam (float, optional): Radius for perturbing the parameter space within SAM. Defaults to 0.05.
        adaptive_sam (bool, optional): If True, adaptive SAM is used (e.g., for the Adam optimizer). 
                                       Defaults to False.
    """
    def __init__(self,
                 models: List[ForwardAtomisticNetwork],
                 lrs: List[float],
                 lr_sched: Callable[[float], float],
                 model_path: str,
                 train_loss: LossFunction,
                 eval_losses: Dict[str, LossFunction],
                 early_stopping_loss: LossFunction,
                 device: Optional[str] = None,
                 max_epoch: int = 1000,
                 save_epoch: int = 100,
                 validate_epoch: int = 1,
                 train_batch_size: int = 32,
                 valid_batch_size: int = 100,
                 callbacks: Optional[List[TrainingCallback]] = None,
                 opt_class=torch.optim.Adam,
                 with_sam: bool = False,
                 rho_sam: float = 0.05,
                 adaptive_sam: bool = False):
        self.models = models
        self.n_models = len(models)
        self.device = device or get_default_device()
        self.ens = get_torch_calculator(self.models).to(self.device)
        self.train_loss = train_loss
        self.eval_loss_fns = eval_losses
        self.early_stopping_loss_fn = early_stopping_loss
        self.train_output_variables = self.train_loss.get_output_variables()
        self.eval_output_variables = list(set(sum([l.get_output_variables() for l in self.eval_loss_fns.values()], [])))
        self.early_stopping_output_variables = self.early_stopping_loss_fn.get_output_variables()

        # create a separate parameter group for each parameter
        self.with_sam = with_sam
        if self.with_sam:
            self.optimizers = [SAM([{'params': [p]} for p in model.parameters()], opt_class, rho=rho_sam,
                                   adaptive=adaptive_sam) for model in models]
        else:
            self.optimizers = [opt_class([{'params': [p]} for p in model.parameters()]) for model in models]

        self.lrs = lrs
        self.lr_sched = lr_sched
        self.callbacks = callbacks
        self.model_path = model_path
        self.max_epoch = max_epoch
        self.save_epoch = save_epoch
        self.validate_epoch = validate_epoch
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.epoch = 0
        self.best_es_metrics = [np.Inf] * self.n_models
        self.best_epochs = [0] * self.n_models
        self.best_mean_es_metric = np.Inf
        self.best_avg_metrics = None

        # create best and log directories to save/restore training progress
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.log_dirs = [os.path.join(self.model_path, str(i), 'logs') for i in range(self.n_models)]
        self.best_dirs = [os.path.join(self.model_path, str(i), 'best') for i in range(self.n_models)]
        for dir in self.log_dirs + self.best_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def save(self,
             paths: List[Union[Path, str]]):
        """Saves an ensemble of models.

        Args:
            paths (List[Union[Path, str]]): Path to the ensemble.
        """
        # one file path per model
        for i, path in enumerate(paths):
            self.save_indiv(path, i)

    def save_indiv(self,
                   path: Union[Path, str],
                   model_idx: int):
        """Saves an individual model.

        Args:
            path (Union[Path, str]): Path to the model.
            model_idx (int): Model index.
        """
        to_save = {'opt': self.optimizers[model_idx].state_dict(),
                   'best_es_metric': self.best_es_metrics[model_idx], 'best_epoch': self.best_epochs[model_idx],
                   'epoch': self.epoch, 'best_mean_es_metric': self.best_mean_es_metric,
                   'best_avg_metrics': self.best_avg_metrics}

        old_folders = list(Path(path).iterdir())

        new_folder = Path(path) / f'ckpt_{self.epoch}'
        os.makedirs(new_folder)

        self.models[model_idx].save(new_folder)
        save_object(new_folder / f'training_state.pkl', to_save)
        # delete older checkpoints after the new one has been saved
        for folder in old_folders:
            if any([p.is_dir() for p in folder.iterdir()]):
                # folder contains another folder, this shouldn't occur, we don't want to delete anything important
                raise RuntimeError(f'Model saving folder {folder} contains another folder, will not be deleted')
            else:
                shutil.rmtree(folder)

    def try_load_indiv(self,
                       path: Union[Path, str],
                       model_idx: int):
        """Loads an individual model.

        Args:
            path (Union[Path, str]): Path to the model.
            model_idx (int): Model index.
        """
        # if no checkpoint exists, just don't load
        folders = list(Path(path).iterdir())
        if len(folders) == 0:
            return  # no checkpoint exists
        if len(folders) >= 2:
            folders = [f for f in folders if f.name.startswith('ckpt_')]
            file_epoch_numbers = [int(f.name[5:]) for f in folders]
            newest_file_idx = np.argmax(np.asarray(file_epoch_numbers))
            folder = folders[newest_file_idx]
        else:
            folder = folders[0]

        self.models[model_idx].load_params(folder / 'params.pkl')

        state_dict = load_object(folder / 'training_state.pkl')
        self.optimizers[model_idx].load_state_dict(state_dict['opt'])
        self.best_es_metrics[model_idx] = state_dict['best_es_metric']
        self.best_epochs[model_idx] = state_dict['best_epoch']

        # the last three do not depend on the model index, they should be the same for all model
        # (we just overwrite them for each loaded model)
        # this is the case for the model saved in log_dirs
        self.epoch = state_dict['epoch']
        self.best_mean_es_metric = state_dict['best_mean_es_metric']
        self.best_avg_metrics = state_dict['best_avg_metrics']

    def try_load(self,
                 paths: List[Union[Path, str]]):
        """Loads an ensemble of models.

        Args:
            paths (List[Union[Path, str]]): Path to the ensemble.
        """
        # if no checkpoint exists, just don't load
        for i, path in enumerate(paths):
            self.try_load_indiv(path, i)

    def _train_step(self,
                    batch: Data,
                    train_avg_loss_trackers: Dict[str, TotalLossTracker]):
        """Performs a training step using the provided batch.

        Args:
            batch (Data): Atomic data graph.
            train_avg_loss_trackers (Dict[str, TotalLossTracker]): Dictionary of loss trackers using during training; 
                                                                   see `loss_fns.py`.
        """
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        results = self.ens(batch, forces='forces' in self.train_output_variables,
                           stress='stress' in self.train_output_variables,
                           virials='virials' in self.train_output_variables,
                           features=False, create_graph=True)

        # compute sum of train losses for model
        loss_values = []
        for result in results['individual_results']:
            tracker = TotalLossTracker(self.train_loss, requires_grad=True)
            tracker.append_batch(result, batch)
            loss_values.append(tracker.compute_final_result(n_atoms_total=batch.n_atoms.sum(),
                                                            n_structures_total=batch.n_atoms.shape[0]))
        total_loss = sum(loss_values)
        total_loss.backward()

        # optimizer update
        if self.with_sam:
            # perform the first step of the SAM optimizer (compute loss regularization)
            for opt in self.optimizers:
                opt.first_step(zero_grad=True)
            results = self.ens(batch, forces='forces' in self.train_output_variables,
                               stress='stress' in self.train_output_variables,
                               virials='virials' in self.train_output_variables,
                               features=False, create_graph=True)
            loss_values = []
            for result in results['individual_results']:
                tracker = TotalLossTracker(self.train_loss, requires_grad=True)
                tracker.append_batch(result, batch)
                loss_values.append(tracker.compute_final_result(n_atoms_total=batch.n_atoms.sum(),
                                                                n_structures_total=batch.n_atoms.shape[0]))
            total_loss = sum(loss_values)
            total_loss.backward()
            # perform the second step of the SAM optimizer (step using the regularized loss)
            for opt in self.optimizers:
                opt.second_step(zero_grad=True)
        else:
            for opt in self.optimizers:
                opt.step()

        with torch.no_grad():
            for loss_tracker in train_avg_loss_trackers.values():
                loss_tracker.append_batch(results, batch)

    def fit(self,
            train_ds: List[AtomicData],
            valid_ds: List[AtomicData]):
        """Trains atomistic models to provided training structures. Validation data is used for early stopping.

        Args:
            train_ds (List[AtomicData]): Training data.
            valid_ds (List[AtomicData]): Validation data.
        """
        # todo: put model in train() mode in the beginning and in eval() mode (or the mode they had before) at the end?
        # reset in case this fit() is called multiple times and try_load() doesn't find a checkpoint
        self.epoch = 0
        self.best_es_metrics = [np.Inf] * self.n_models
        self.best_epochs = [0] * self.n_models
        self.best_mean_es_metric = np.Inf
        self.best_avg_metrics = None

        self.try_load(self.log_dirs)
        # start timing
        start_session = time.time()

        # generate data queues for efficient training
        use_gpu = self.device.startswith('cuda')
        train_dl = DataLoader(train_ds, batch_size=self.train_batch_size, shuffle=True, drop_last=True,
                              pin_memory=use_gpu, pin_memory_device=self.device if use_gpu else '')
        valid_dl = DataLoader(valid_ds, batch_size=self.valid_batch_size, shuffle=False, drop_last=False,
                              pin_memory=use_gpu, pin_memory_device=self.device if use_gpu else '')

        steps_per_epoch = len(train_dl)
        max_step = self.max_epoch * steps_per_epoch

        for callback in self.callbacks:
            callback.before_fit(self)

        while self.epoch < self.max_epoch:
            start_epoch = time.time()
            self.epoch += 1

            train_avg_loss_trackers = {name: TotalLossTracker(loss_fn, requires_grad=False)
                                       for name, loss_fn in self.eval_loss_fns.items()}

            n_structures_total = 0
            n_atoms_total = 0

            for batch_idx, batch in enumerate(train_dl):
                step = steps_per_epoch * (self.epoch - 1) + batch_idx
                n_structures_total += len(batch.n_atoms)
                n_atoms_total += batch.n_atoms.sum().item()

                for opt in self.optimizers:
                    for group, lr in zip(opt.param_groups, self.lrs):
                        group['lr'] = lr * self.lr_sched(step / max_step)
                self._train_step(batch.to(self.device), train_avg_loss_trackers)

            train_avg_metrics = {name: loss_tracker.compute_final_result(n_structures_total, n_atoms_total).item()
                                 for name, loss_tracker in train_avg_loss_trackers.items()}

            if self.epoch % self.save_epoch == 0:
                # save progress for restoring
                self.save(self.log_dirs)

            if self.epoch % self.validate_epoch == 0 or self.epoch == self.max_epoch:
                # check performance on validation step
                valid_metrics = eval_metrics(ens=self.ens, dl=valid_dl, eval_loss_fns=self.eval_loss_fns,
                                             eval_output_variables=self.eval_output_variables,
                                             early_stopping_loss_fn=self.early_stopping_loss_fn,
                                             device=self.device)

                # update best_avg_metric based on mean early stopping score
                mean_es_metric = np.mean(valid_metrics['early_stopping'])
                if mean_es_metric < self.best_mean_es_metric:
                    self.best_mean_es_metric = mean_es_metric
                    self.best_avg_metrics = valid_metrics['average']

                # save individual model if individual early stopping metrics improved
                for i in range(self.n_models):
                    if valid_metrics['early_stopping'][i] < self.best_es_metrics[i]:
                        self.best_es_metrics[i] = valid_metrics['early_stopping'][i]
                        self.best_epochs[i] = self.epoch
                        self.save_indiv(self.best_dirs[i], i)

                end_epoch = time.time()

                for callback in self.callbacks:
                    callback.after_epoch(self, train_avg_metrics, valid_metrics['average'], end_epoch - start_epoch)

        end_session = time.time()

        for callback in self.callbacks:
            callback.after_fit(self, end_session - start_session)
