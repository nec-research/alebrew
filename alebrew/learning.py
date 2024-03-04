"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     learning.py 
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
import math
import os
import shutil
import time
from itertools import cycle
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any

import numpy as np
import torch
from ase.io import read

from alebrew.data.data import AtomicStructures
from alebrew.strategies import TrainingStrategy, SamplingStrategy, SelectionStrategy, EvaluationStrategy
from alebrew.model.forward import ForwardAtomisticNetwork
from alebrew.datagen.references import BaseReference
from alebrew.utils.misc import save_object


class BaseLearning:
    """Trains machine-learned interatomic potentials."""
    def run(self,
            *args: Any,
            **kwargs: Any) -> Tuple[AtomicStructures, List[ForwardAtomisticNetwork]]:
        """Runs training."""
        raise NotImplementedError()


class OnlineLearning(BaseLearning):
    """Trains machine-learned interatomic potentials using online active learning.

    Args:
        training_strategy (TrainingStrategy): Training strategy; see `strategies.py`.
        sampling_strategy (SamplingStrategy): Sampling strategy; see `strategies.py`.
        selection_strategy (SelectionStrategy): Selection strategy; see `strategies.py`.
        reference (BaseReference): Method for computing reference energy, atomic force, and stress values.
        evaluation_strategy (Optional[EvaluationStrategy], optional): Evaluation strategy; see `strategies.py. 
                                                                      Defaults to None.
        max_learning_step (int, optional): Maximal number of active learning steps. Defaults to 1024.
        max_data_size (int, optional): Maximal size of the acquired data. Defaults to 1024.
        valid_fraction (float, optional): Faction of data used for early stopping. Defaults to 0.2.
    """
    def __init__(self,
                 training_strategy: TrainingStrategy,
                 sampling_strategy: SamplingStrategy,
                 selection_strategy: SelectionStrategy,
                 reference: BaseReference,
                 evaluation_strategy: Optional[EvaluationStrategy] = None,
                 max_learning_step: int = 1024,
                 max_data_size: int = 1024,
                 valid_fraction: float = 0.2,
                 **kwargs: Any):
        super().__init__()
        self.training_strategy = training_strategy
        self.sampling_strategy = sampling_strategy
        self.selection_strategy = selection_strategy
        self.reference = reference
        self.evaluation_strategy = evaluation_strategy
        self.max_learning_step = max_learning_step
        self.max_data_size = max_data_size
        self.valid_fraction = valid_fraction

    def run(self,
            structures: AtomicStructures,
            folder: Union[str, Path],
            data_seed: int = 1234,
            sampling_seed: int = 1234,
            model_seeds: List[int] = [1234],
            init_structures: Optional[AtomicStructures] = None,
            test_structures: Optional[AtomicStructures] = None,
            use_precomp_labels: bool = False,
            clean_reference_folder: bool = True,
            **kwargs: Any) -> Tuple[AtomicStructures, List[ForwardAtomisticNetwork]]:
        """Runs training using online active learning.

        Args:
            structures (AtomicStructures): Atomic structures used for generating the initial training data set.
                                           If `init_structures=None` they are also used to run atomistic simulations 
                                           in the first active learning step.
            folder (Union[str, Path]): Folder in which results of online active learning are stored.
            data_seed (int, optional): Random seed for splitting data sets. Defaults to 1234.
            sampling_seed (int, optional): Random seed for setting up sampling methods. Defaults to 1234.
            model_seeds (List[int], optional): Random seed for initializing atomistic models. Defaults to [1234].
            init_structures (Optional[AtomicStructures], optional): Optional atomic structures to initialize learning.
                                                                    They can be different from 'structures'. Defaults to None.
            test_structures (Optional[AtomicStructures], optional): Optional test structures to evaluate models during active learning. 
                                                                    Defaults to None.
            use_precomp_labels (bool, optional): If True, pre-computed labels for `init_structures` are used. Defaults to False.
            clean_reference_folder (bool, optional): If True, reference folders are cleaned after calculations. Defaults to True.

        Returns:
            Tuple[AtomicStructures, List[ForwardAtomisticNetwork]]: Generated data set and the final model.
        """
        # define folder in which all results are stored
        folder = Path(folder)

        # initialize the loop by sampling random structures or by using the provided structures
        labeled_structures = AtomicStructures([])
        selected_structures = self.sampling_strategy.pre_run(structures, folder / 'pre_sampling',
                                                             seed=sampling_seed)

        # define init structures to be those initially provided, exclude randomly distorted ones as they may have high
        # energy or internal pressure
        if init_structures is None:
            init_structures = structures
        init_structures_cycle = cycle(init_structures.structures)
        init_structures = AtomicStructures([next(init_structures_cycle)
                                            for _ in range(self.sampling_strategy.n_samplers)])

        # begin with the main loop
        for i_step in range(self.max_learning_step):

            rng = np.random.default_rng(i_step)
            step_seed = rng.integers(low=0, high=9999999).item()

            step_folder = folder / f'step_{i_step}'
            to_save = {}

            # check if the current iteration has been finished and results have been stored
            if os.path.exists(step_folder / 'step_results.json'):
                # load labeled structures
                traj = read(step_folder / 'reference' / 'completed_structures.extxyz', ':')
                new_labeled_structures = AtomicStructures.from_traj(traj)
                labeled_structures = labeled_structures + new_labeled_structures
                if i_step > 0:
                    # update init structure with the newly selected ones (with successfully computed labels)
                    init_structures = new_labeled_structures
                # load selected structures (required for the next, not finished step)
                traj = read(step_folder / 'selection' / 'selected_structures.extxyz', ':')
                selected_structures = AtomicStructures.from_traj(traj)
                continue
            elif os.path.exists(step_folder):
                # for correct times have to recompute everything
                shutil.rmtree(step_folder)

            # compute reference labels for newly sampled/selected structures
            # release memory in case a NNReference has been used, also release GPU memory after each calculation
            start_reference = time.time()
            if use_precomp_labels and i_step == 0:
                # use precomputed labels for the first interation to reduce computational cost
                # may be necessary if larger data sets are used to initialize the learning loop
                new_labeled_structures = selected_structures
                if not os.path.exists(step_folder / 'reference'):
                    os.makedirs(step_folder / 'reference')
                new_labeled_structures.save_extxyz(file_path=step_folder / 'reference' / 'completed_structures.extxyz')
            else:
                new_labeled_structures, _ = self.reference.label_structures(structures=selected_structures,
                                                                            folder=step_folder / 'reference')
            if clean_reference_folder:
                self.reference.clean_reference_folder(folder=step_folder / 'reference')
            end_reference = time.time()
            to_save['reference_time'] = end_reference - start_reference
            torch.cuda.empty_cache()

            # update labeled structures and split them into train and valid
            labeled_structures = new_labeled_structures + labeled_structures
            if i_step > 0:
                # update init structure with the newly selected ones (with successfully computed labels)
                init_structures = new_labeled_structures
            n_valid = max(math.floor(self.valid_fraction * len(labeled_structures)), 1)
            n_train = len(labeled_structures) - n_valid
            split = labeled_structures.random_split({'train': n_train, 'valid': n_valid},
                                                    seed=data_seed + step_seed)
            # store the number of training and validation structures
            to_save['n_train'] = n_train
            to_save['n_valid'] = n_valid

            # train models with the defined strategy, model specific parameters should be defined in config
            start_training = time.time()
            step_model_seeds = [model_seed + step_seed for model_seed in model_seeds]
            models = self.training_strategy.run(train_structures=split['train'], valid_structures=split['valid'],
                                                folder=step_folder / 'models', model_seeds=step_model_seeds)
            end_training = time.time()
            to_save['training_time'] = end_training - start_training
            torch.cuda.empty_cache()

            # evaluate on test data if provided and evaluate object is not None
            if self.evaluation_strategy is not None and test_structures is not None:
                avg_metrics, indiv_metrics = self.evaluation_strategy.run(models=models,
                                                                          test_structures=test_structures,
                                                                          folder=step_folder / 'models')
                to_save['avg_metrics'] = avg_metrics
                to_save['indiv_metrics'] = indiv_metrics
                torch.cuda.empty_cache()

            # check if stopping criterion has been fulfilled
            if len(labeled_structures) >= self.max_data_size:
                save_object(step_folder / f'step_results.json', to_save, use_json=True)
                return labeled_structures, models

            # sample new configurations, calculator and wrapper specific parameters should be
            start_sampling = time.time()
            new_unlabeled_structures = self.sampling_strategy.run(
                models=models,
                train_structures=split['train'],
                valid_structures=split['valid'],
                init_structures=init_structures,
                folder=step_folder / 'sampling',
                seed=sampling_seed + step_seed
            )
            end_sampling = time.time()
            to_save['sampling_time'] = end_sampling - start_sampling
            torch.cuda.empty_cache()
            # store the number of sampled structures and the total number of sampling steps run by the sampler
            to_save['n_sampled'] = len(new_unlabeled_structures)

            # select configurations
            min_sel_data_size = min(min(self.sampling_strategy.n_samplers, len(labeled_structures)),
                                    len(new_unlabeled_structures))
            max_sel_data_size = self.max_data_size - len(labeled_structures)
            start_selection = time.time()
            selected_structures, _ = self.selection_strategy.run(
                models=models,
                pool_structures=new_unlabeled_structures,
                train_structures=split['train'],
                valid_structures=split['valid'],
                folder=step_folder / 'selection',
                min_sel_data_size=min_sel_data_size,
                max_sel_data_size=max_sel_data_size
            )
            end_selection = time.time()
            to_save['selection_time'] = end_selection - start_selection
            torch.cuda.empty_cache()
            # store number of selected structures
            to_save['n_selected'] = len(selected_structures)

            save_object(step_folder / f'step_results.json', to_save, use_json=True)

        return labeled_structures, models

   
class BatchLearning(BaseLearning):
    """Trains machine-learned interatomic potentials using batch active learning.

    Args:
        training_strategy (TrainingStrategy): Training strategy; see `strategies.py`.
        selection_strategy (SelectionStrategy): Selection strategy; see `strategies.py`.
        reference (Optional[BaseReference], optional): Method for computing reference energy, atomic force, and stress values. 
                                                       Defaults to None.
        evaluation_strategy (Optional[EvaluationStrategy], optional): Evaluation strategy; see `strategies.py`. 
                                                                      Defaults to None.
        max_learning_step (int, optional): Maximal number of active learning steps. Defaults to 1024.
        max_data_size (int, optional): Maximal size of the acquired data. Defaults to 1024.
        n_train (int, optional): Initial training data set size. Defaults to 32.
        n_valid (int, optional): Validation data set size. Defaults to 512.
        n_pool (int, optional): Pool dat set size. Defaults to 4096.
    """
    def __init__(self,
                 training_strategy: TrainingStrategy,
                 selection_strategy: SelectionStrategy,
                 reference: Optional[BaseReference] = None,
                 evaluation_strategy: Optional[EvaluationStrategy] = None,
                 max_learning_step: int = 1024,
                 max_train_size: int = 256,
                 n_train: int = 32,
                 n_valid: int = 512,
                 n_pool: int = 4096,
                 **kwargs: Any):
        super().__init__()
        self.training_strategy = training_strategy
        self.selection_strategy = selection_strategy
        self.reference = reference
        self.evaluation_strategy = evaluation_strategy
        self.max_learning_step = max_learning_step
        self.max_train_size = max_train_size
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_pool = n_pool

    def run(self,
            structures: AtomicStructures,
            folder: Union[str, Path],
            data_seed: int = 1234,
            model_seeds: List[int] = [1234],
            test_structures: Optional[AtomicStructures] = None,
            **kwargs: Any) -> Tuple[AtomicStructures, List[ForwardAtomisticNetwork]]:
        """Runs training using batch active learning.

        Args:
            structures (AtomicStructures): Pre-sampled atomic structures. They can be labeled or not.
            folder (Union[str, Path]): Folder in which results of active learning are stored.
            data_seed (int, optional): Random seed to split the data set. Defaults to 1234.
            model_seeds (List[int], optional): Random seed for initializing models. Defaults to [1234].
            test_structures (Optional[AtomicStructures], optional): Optional test structures to evaluate models 
                                                                    during active learning. Defaults to None.

        Returns:
            Tuple[AtomicStructures, List[ForwardAtomisticNetwork]]: Acquired data set and the final model.
        """
        # define folder in which all results are stored
        folder = Path(folder)

        # initialize the loop by splitting structures into training, validation, and pool subsets
        split = structures.random_split({'train': self.n_train, 'valid': self.n_valid, 'pool': self.n_pool},
                                        seed=data_seed)

        train_structures = AtomicStructures([])
        selected_structures = split['train']

        valid_structures = split['valid']
        if self.reference is not None:
            valid_structures, _ = self.reference.label_structures(valid_structures, folder / 'reference_valid')

        pool_structures = split['pool']

        if 'test' in split and test_structures is None:
            test_structures = split['test']
        if test_structures is not None and self.reference is not None:
            test_structures, _ = self.reference.label_structures(test_structures, folder / 'reference_test')

        # begin with the main loop
        for i_step in range(self.max_learning_step):

            rng = np.random.default_rng(i_step)
            step_seed = rng.integers(low=0, high=9999999).item()

            step_folder = folder / f'step_{i_step}'
            to_save = {}

            # check if the current iteration has been finished and results have been stored
            if os.path.exists(step_folder / 'step_results.json'):
                if self.reference is not None:
                    traj = read(step_folder / 'reference' / 'completed_structures.extxyz', ':')
                    selected_structures = AtomicStructures.from_traj(traj)
                train_structures = train_structures + selected_structures
                selected_idxs = np.load(folder / 'selection' / 'selected_idxs.npy')
                selected_structures, pool_structures = pool_structures.split_by_indices(selected_idxs)
                continue
            elif os.path.exists(step_folder):
                # for correct times have to recompute everything
                shutil.rmtree(step_folder)

            if self.reference is not None:
                # compute reference labels if reference object is not None
                # release memory in case a NNReference has been used, also release GPU memory after each calculation
                start_reference = time.time()
                selected_structures, _ = self.reference.label_structures(structures=selected_structures,
                                                                         folder=step_folder / 'reference')
                end_reference = time.time()
                to_save['reference_time'] = end_reference - start_reference
                torch.cuda.empty_cache()

            # update train structures
            train_structures = train_structures + selected_structures

            # train models with the defined strategy, model specific parameters should be defined in config
            start_training = time.time()
            step_model_seeds = [model_seed + step_seed for model_seed in model_seeds]
            models = self.training_strategy.run(train_structures=train_structures, valid_structures=valid_structures,
                                                folder=step_folder / 'models', model_seeds=step_model_seeds)
            end_training = time.time()
            to_save['training_time'] = end_training - start_training
            torch.cuda.empty_cache()

            # evaluate on test data if provided and evaluate object is not None
            if self.evaluation_strategy is not None and test_structures is not None:
                avg_metrics, indiv_metrics = self.evaluation_strategy.run(models=models,
                                                                          test_structures=test_structures,
                                                                          folder=step_folder / 'models')
                to_save['avg_metrics'] = avg_metrics
                to_save['indiv_metrics'] = indiv_metrics
                torch.cuda.empty_cache()

            # check if stopping criterion has been fulfilled
            if len(train_structures) >= self.max_train_size:
                save_object(step_folder / f'step_results.json', to_save, use_json=True)
                return train_structures, models

            # select configurations
            start_selection = time.time()
            selected_structures, pool_structures = self.selection_strategy.run(models=models,
                                                                               pool_structures=pool_structures,
                                                                               train_structures=train_structures,
                                                                               valid_structures=valid_structures,
                                                                               folder=step_folder / 'selection',
                                                                               max_sel_data_size=self.max_train_size)
            end_selection = time.time()
            to_save['selection_time'] = end_selection - start_selection
            torch.cuda.empty_cache()
            # store number of selected structures
            to_save['n_selected'] = len(selected_structures)

            save_object(step_folder / f'step_results.json', to_save, use_json=True)

        return train_structures, models
