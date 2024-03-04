"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     calculators.py 
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
import copy
from typing import Dict, Union, Any, List, Optional, Tuple

import numpy as np
import torch
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
from bmdal_reg.bmdal.features import Features
from bmdal_reg.bmdal.selection import MaxDistSelectionMethod

from alebrew.data.data import AtomicData, AtomicStructures
from alebrew.model.features import FeatureComputation
from alebrew.model.forward import ForwardAtomisticNetwork
from alebrew.model.unc_scaling import (LogRegressionScaling, GaussianMaximumLikelihoodScaling,
                                       ConformalPredictionScaling)
from alebrew.utils.math import segment_sum
from alebrew.utils.torch_geometric import Data, DataLoader
from alebrew.utils.misc import recursive_detach, recursive_cat, get_batch_intervals
from alebrew.utils.process_pool import ProcessPoolMapper


class TorchCalculator:
    """Computes atomic properties, e.g., (total) energy, atomic forces, stress, 
    or gradient/last layer features/uncertainties.
    """
    def __call__(self,
                 graph: Data,
                 **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Performs calculation on the provided (batch) graph data.

        Args:
            graph (Data): Atomic data graph.

        Returns: 
            Dict[str, Union[torch.Tensor, Any]]: Results dictionary.
        """
        raise NotImplementedError()

    def eval_on_ds(self,
                   ds: List[AtomicData],
                   keys: Optional[List[str]] = None,
                   eval_batch_size: int = 100,
                   **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Computes requested atomic properties for atomic data.

        Args:
            ds (List[AtomicData]): Data set provided as a list of `AtomicData`.
            keys (Optional[List[str]], optional): Property keys (e.g., energy, forces). 
                                                  Defaults to None.
            eval_batch_size (int): Batch size used during the evaluation. Defaults to 100.

        Returns: 
            Dict[str, Union[torch.Tensor, Any]]: Results dictionary.
        """
        dl = DataLoader(ds, shuffle=False, drop_last=False, batch_size=eval_batch_size)
        batch_results = []
        for batch in dl:
            results = self(batch.to(self.get_device()), **kwargs)
            batch_results.append({key: recursive_detach(value)
                                  for key, value in results.items() if keys is None or key in keys})
        return recursive_cat(batch_results, dim=0)

    def eval_on_structures(self,
                           structures: AtomicStructures,
                           r_cutoff: float,
                           keys: Optional[List[str]] = None,
                           eval_batch_size: int = 100,
                           retain_neighbors: bool = True,
                           **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Computes requested atomic properties for atomic structures. 
        Reduces memory requirements as it does not require retaining neighbors.

        Args:
            structures (AtomicStructures): Atomic structures.
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            keys (Optional[List[str]], optional): Property keys (e.g., energy, forces). 
                                                  Defaults to None.
            eval_batch_size: Batch size used during the evaluation. Defaults to 100.
            retain_neighbors: If False, neighbors are not be stored. Defaults to True.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Results dictionary.
        """
        # adjust batch size if only a few structures have been provided
        eval_batch_size = min(eval_batch_size, len(structures))
        # allows to not store the neighbors permanently in case this overloads the RAM
        batch_results = []
        for start, stop in get_batch_intervals(len(structures), batch_size=eval_batch_size):
            if retain_neighbors:
                batch_structures = structures[start:stop]
            else:
                # copy AtomicStructure objects such that the neighbors are only stored in the copied objects,
                # which will be deleted at the end of the loop iteration
                batch_structures = AtomicStructures([copy.copy(s) for s in structures.structures[start:stop]])
            batch = batch_structures.to_data(r_cutoff=r_cutoff)
            batch_results.append(self.eval_on_ds(batch, keys, eval_batch_size, **kwargs))
        return recursive_cat(batch_results, dim=0)

    def get_device(self) -> str:
        """Provides the device on which calculations are performed.
        
        Returns: 
            str: Device on which calculations are performed.
        """
        raise NotImplementedError()

    def to(self, device: str) -> 'TorchCalculator':
        """Moves the calculator to the provided device.
        
        Args:
            device: Device to which calculator has to be moved.

        Returns: 
            TorchCalculator: The `TorchCalculator` object.
        """
        raise NotImplementedError()


def prepare_gradients(graph: Data,
                      forces: bool = False,
                      stress: bool = False,
                      virials: bool = False,
                      **kwargs: Any) -> Tuple[Data, List[str]]:
    """Prepares gradient calculation by setting `requires_grad=True` for the selected atomic features. 

    Args:
        graph (Data): Atomic data graph.
        forces (bool): If True, gradients with respect to positions/coordinates are calculated. 
                       Defaults to False.
        stress (bool): If True, gradients with respect to strain deformations are calculated. 
                       Defaults to False.
        virials (bool): If True, gradients with respect to strain deformations are calculated. 
                        Defaults to False.
    
        Returns:
            Tuple[Data, List[str]]: Updated graph and list of properties which require gradients.
    """
    require_gradients = []
    if forces:
        require_gradients.append('positions')
        if not graph.positions.requires_grad:
            # request gradients wrt. positions/coordinates
            graph.positions.requires_grad = True
    if stress or virials:
        require_gradients.append('strain')
        if not graph.strain.requires_grad:
            # define displacements corresponding to:
            # Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
            # similar implementations are provided by NequIP (https://github.com/mir-group/nequip)
            # and SchNetPack (https://github.com/atomistic-machine-learning/schnetpack)
            graph.strain.requires_grad = True
            # symmetrize to account for possible numerical issues
            symmetric_strain = 0.5 * (graph.strain + graph.strain.transpose(-1, -2))
            # update cell
            graph.cell = graph.cell + torch.matmul(graph.cell, symmetric_strain)
            # update positions
            symmetric_strain_i = symmetric_strain.index_select(0, graph.batch)
            graph.positions = graph.positions + torch.matmul(graph.positions.unsqueeze(-2),
                                                             symmetric_strain_i).squeeze(-2)
            # update the shifts
            symmetric_strain_ij = symmetric_strain_i.index_select(0, graph.edge_index[0, :])
            graph.shifts = graph.shifts + torch.matmul(graph.shifts.unsqueeze(-2), symmetric_strain_ij).squeeze(-2)
    return graph, require_gradients


class StructurePropertyCalculator(TorchCalculator):
    """Calculates total energy, atomic forces, stress tensors, and gradient/last-layer features from atomic energies.

    Args:
        model (ForwardAtomisticNetwork): Forward atomistic neural network object (provides atomic energies).
        use_grad_features (bool): If True, gradient features are computed. Defaults to False.
        n_random_projections (int): The number of random projections. Defaults to -1.
    """
    def __init__(self,
                 model: ForwardAtomisticNetwork,
                 use_grad_features: bool = False,
                 n_random_projections: int = -1,
                 **config: Any):
        self.model = model
        # init feature computation
        self.feature_computation = FeatureComputation(self.model, use_grad_features, n_random_projections)
        self.feature_computation.to(self.model.get_device())

    def __call__(self,
                 graph: Data,
                 forces: bool = False,
                 stress: bool = False,
                 virials: bool = False,
                 features: bool = False,
                 store_atomic_features: bool = False,
                 create_graph: bool = False,
                 **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Performs calculations for the atomic data graph.

        Args:
            graph (Data): Atomic data graph.
            forces (bool): If True, atomic forces are computed. Defaults to False.
            stress (bool): If True, stress tensor is computed. Defaults to False.
            virials (bool): If True, virials = - stress * volume are computed. Defaults to False.
            features (bool): If True, last-layer or sketched gradient features are computed. 
                             The resulting features are normalized by the number of atoms 
                             such that total uncertainty is independent of it. 
                             Defaults to False.
            store_atomic_features (bool): If True, atom-based last-layer or sketched gradient features 
                                          are stored in addition to features. Defaults to False.
            create_graph (bool): If True, computational graph is created allowing the computation of 
                                 backward pass for multiple times. Defaults to False.

        Returns: 
            Dict[str, torch.Tensor]: Results dict.
        """
        results = {}
        # prepare graph and the list containing graph attributes requiring gradients
        graph, require_gradients = prepare_gradients(graph=graph, forces=forces, stress=stress, virials=virials)
        if features:
            # register forward hooks
            self.feature_computation.before_forward()
        # compute atomic energy
        atomic_energies = self.model(graph)
        results['atomic_energies'] = atomic_energies
        # sum up atomic contributions for a structure
        total_energies = segment_sum(atomic_energies, idx_i=graph.batch, dim_size=graph.n_atoms.shape[0])
        # write total energy to results
        results['energy'] = total_energies
        if features:
            # compute (gradient) features is requested
            atomic_features = self.feature_computation.pop_features(atomic_energies)
            if store_atomic_features:
                results['atomic_features'] = atomic_features
            # divide features by the number of atoms to make the resulting uncertainties independent of it, 
            # i.e. providing uncertainty per atom
            results['features'] = segment_sum(atomic_features, idx_i=graph.batch, 
                                              dim_size=graph.n_atoms.shape[0]) / graph.n_atoms[:, None]
        if require_gradients:
            # compute gradients wrt. positions, strain, etc.
            grads = torch.autograd.grad([atomic_energies], [getattr(graph, key) for key in require_gradients],
                                        torch.ones_like(atomic_energies), create_graph=create_graph)
        if forces:
            # compute forces as negative of the gradient wrt. positions
            results['forces'] = torch.neg(grads[0])
        if virials:
            # compute virials as negative of the gradient wrt. strain (note that other conventions are possible,
            # but here we use virials = -1 * stress * volume)
            if grads[-1] is not None:
                results['virials'] = torch.neg(grads[-1])
            else:
                results['virials'] = torch.zeros_like(graph.cell)
        if stress:
            # compute stress as -1 * virials / volume
            volume = torch.einsum('bi, bi -> b', graph.cell[:, 0, :],
                                  torch.cross(graph.cell[:, 1, :], graph.cell[:, 2, :], dim=1))
            if grads[-1] is not None:
                results['stress'] = grads[-1] / volume[:, None, None]
            else:
                results['stress'] = torch.zeros_like(graph.cell) / volume[:, None, None]
        return results

    def get_device(self) -> str:
        return self.model.get_device()

    def to(self, device: str) -> TorchCalculator:
        self.feature_computation.to(device)
        self.model.to(device)
        return self


class EnsembleCalculator(TorchCalculator):
    """Computes atomic properties using an ensemble of `StructurePropertyCalculator`.

    Args:
        calcs (List[TorchCalculator]): List of `StructurePropertyCalculator`.
        n_threads (int): The number of parallel threads.
    """

    def __init__(self,
                 calcs: List[TorchCalculator],
                 n_threads: int = -1,
                 **config: Any):
        self.calcs = calcs
        self.n_threads = n_threads

    def __call__(self,
                 graph: Data,
                 forces: bool = False,
                 stress: bool = False,
                 virials: bool = False,
                 features: bool = False,
                 store_atomic_features: bool = False,
                 create_graph: bool = False,
                 **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Performs calculations using an ensemble of `StructurePropertyCalculator` for the provided atomic data graph.

        Args:
            graph (Data): Atomic data graph.
            forces (bool): If True, atomic forces are computed. Defaults to False.
            stress (bool): If True, stress tensor is computed. Defaults to False.
            virials (bool): If True, virials = - stress * volume are computed. Defaults to False.
            features (bool): If True, last-layer or sketched gradient features are computed. 
                             The resulting features are normalized by the number of atoms such 
                             that total uncertainty is independent of it. Defaults to False.
            store_atomic_features (bool): If True, atom-based last-layer or sketched gradient features 
                                          are stored in addition to features. Defaults to False.
            create_graph (bool): If True, computational graph is created allowing the computation of 
                                 backward pass for multiple times. Defaults to False.

        Returns: 
            Dict[str, Union[torch.Tensor, Any]]: Results dict (containing ensemble and individual results).
        """
        # prepare graph for gradient calculations
        graph, _ = prepare_gradients(graph=graph, forces=forces, stress=stress, virials=virials)
        # create thread pool and run ensemble
        pool = ProcessPoolMapper(n_threads=len(self.calcs) if self.n_threads <= 0 else self.n_threads,
                                 use_multiprocessing=False)
        args_tuples = [(graph, forces, stress, virials, features, store_atomic_features, create_graph)] * len(self.calcs)
        results = pool.map(self.calcs, args_tuples)
        ens_results = {}
        for key in results[0]:
            if key in ['features', 'atomic_features']:
                # cat features instead of computing an average
                ens_results[key] = torch.cat([r[key] for r in results], dim=1)
            else:
                ens_results[key] = sum([r[key] for r in results]) / len(results)
        # store results for individual model/interfaces
        ens_results['individual_results'] = results
        return ens_results

    def get_device(self) -> str:
        return self.calcs[0].get_device()

    def to(self, device: str) -> TorchCalculator:
        for calc in self.calcs:
            calc.to(device)
        return self


class UncertaintyAndPropertyCalculator(TorchCalculator):
    """Calculates total and atom-based uncertainties.

    Args:
        calc (TorchCalculator): Ensemble calculator.
        compute_energy_unc (bool): If True, total uncertainty is computed. Defaults to True.
        compute_atomic_unc (bool): If True, atom-based uncertainty is computed. Defaults to False.
        unc_scaling_method (str): Define uncertainty calibration method: 'conformal', 'quantile_reg',
                                  'gaussian', and 'log_reg'. Defaults to 'conformal'.
        energy_unc_residual (str): Define total uncertainty residual for calibration: 'energy_ae' and 'forces_rmse'. 
                                   'energy_ae' re-scales uncertainty normalized by the number of atoms. Thus, it is 
                                   divided by the number of atoms, too. Defaults to 'energy_ae'.
        atomic_unc_residual (str): Define atom-based uncertainty residual for calibration: 'energy_ae' and 'forces_rmse'. 
                                   Defaults to 'forces_rmse'.
        train_ds (Optional[List[Data]]): Training data set. Defaults to None.
        valid_ds (Optional[List[Data]]): Validation data set. Defaults to None.
        adversarial_temperature (float): Given validation and training data sets, an approximation to the partition function 
                                         at given adversarial temperature is computed. The approximated partition function is 
                                         used for adversarial attacks. Defaults to 1.0.
    """
    def __init__(self,
                 calc: TorchCalculator,
                 compute_energy_unc: bool = True,
                 compute_atomic_unc: bool = False,
                 unc_scaling_method: str = 'conformal',
                 energy_unc_residual: str = 'energy_ae',
                 atomic_unc_residual: str = 'forces_rmse',
                 train_ds: Optional[List[Data]] = None,
                 valid_ds: Optional[List[Data]] = None,
                 adversarial_temperature: float = 1.0,
                 **config):
        self.calc = calc
        self.features_keys = []
        if compute_energy_unc:
            self.features_keys.append('features')
        if compute_atomic_unc:
            self.features_keys.append('atomic_features')

        # define uncertainty scaling method
        if unc_scaling_method == 'conformal':
            self.unc_scaling = ConformalPredictionScaling(**config)
        elif unc_scaling_method == 'gaussian':
            self.unc_scaling = GaussianMaximumLikelihoodScaling(**config)
        elif unc_scaling_method == 'log_reg':
            self.unc_scaling = LogRegressionScaling(**config)
        else:
            raise NotImplementedError(f'{unc_scaling_method=} is not implemented yet.')

        # define residual keys for total energy and atom-based uncertainties
        assert energy_unc_residual in ['energy_ae', 'forces_rmse']
        assert atomic_unc_residual in ['energy_ae', 'forces_rmse', 'forces_norm']
        self.energy_unc_residual = energy_unc_residual
        self.atomic_unc_residual = atomic_unc_residual

        self.scale_params = {'uncertainty': 1.0,
                             'atomic_uncertainties': 1.0}
        
        self.prepared_with_train = False
        
        if train_ds is not None and valid_ds is not None:
            energy = torch.as_tensor([d.energy for d in train_ds + valid_ds])
            self.partition_function = torch.exp(-energy / adversarial_temperature).sum()
        else:
            self.partition_function = None


    def __call__(self,
                 graph: Data,
                 **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Performs calculations for the provided atomic data graph.

        Args:
            graph: Atomic data graph.

        Returns: 
            Dict[str, Union[torch.Tensor, Any]]: Results dict (incl. uncertainties).
        """
        results = self.calc(graph, **kwargs)
        if self.prepared_with_train:
            unc_results = self.get_model_uncertainties(results, graph)
            for key, val in unc_results.items():
                results[key] = val
        return results
    
    def get_model_uncertainties(self,
                                results: Dict[str, Union[torch.Tensor, Any]],
                                graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """Computes uncertainties from results.

        Args:
            results (Dict[str, Union[torch.Tensor, Any]]): Results dict containing necessary keys (e.g., 'features' or 'forces').
            graph (Optional[Data], optional): Optional atomic data graph (required for ensemble uncertainties).

        Returns: 
            Dict[str, torch.Tensor]: Dictionary containing uncertainties.
        """
        raise NotImplementedError()

    def get_tfm_features_on_structures(self,
                                       structures: AtomicStructures,
                                       r_cutoff: float,
                                       eval_batch_size: int = 100,
                                       retain_neighbors: bool = True) -> Dict[str, Features]:
        """Computes transformed features from structures. Allows to save memory by not retaining neighbors.

        Args:
            structures (AtomicStructures): The `AtomicStructures` object.
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            eval_batch_size (int): Batch size used during evaluation. Defaults to 100.
            retain_neighbors: If True, neighbors are stored during the calculation. Defaults to True.

        Returns: 
            Dict[str, Features]: Dictionary containing transformed features.
        """
        raise NotImplementedError()

    def get_tfm_features_on_ds(self,
                               ds: List[Data],
                               eval_batch_size: int = 100) -> Dict[str, Features]:
        """Computes transformed features from atomic data set. May require more memory as neighbors are 
        retained during the calculation.

        Args:
            ds (List[Data]): Data set containing the list of `AtomicData` objects.
            eval_batch_size (int): Batch size used during evaluation. Defaults to 100.

        Returns: 
            Dict[str, Features]: Dictionary containing transformed features.
        """
        raise NotImplementedError()

    def get_tfm_features_on_batch(self,
                                  results: Dict[str, Any],
                                  graph: Optional[Data] = None) -> Dict[str, Features]:
        """Computes transformed features on batch results.

        Args:
            results (Dict): Results dict containing necessary keys (e.g., 'features' or 'forces').
            graph (Optional[Data], optional): Optional atomic data graph (required for ensemble uncertainties). 
                                              Defaults to None.

        Returns: 
            Dict[str, Features]: Dictionary containing transformed features.
        """
        raise NotImplementedError()

    def prepare_with_train(self,
                           train_ds: List[Data],
                           eval_batch_size: int = 100):
        """Prepares uncertainty calculation using training data (method specific, i.e., for distance-based uncertainties 
        training features are stores, but for posterior-based uncertainty the posterior transformation is prepared (kernel is computed)).

        Args:
            train_ds (List[Data]): Training data set containing the list of `AtomicData` objects.
            eval_batch_size (int): Evaluation batch size. Defaults to 100.
        """
        raise NotImplementedError()

    def scale_uncertainty(self,
                          valid_ds: List[Data],
                          eval_batch_size: int = 100,
                          **kwargs: Any):
        """Calibrates total and atom-based uncertainties using the validation data set.

        Args:
            valid_ds (List[Data]): Validation data set.
            eval_batch_size (int): Evaluation batch size. Defaults to 100.
        """
        # compute scale parameters from validation data
        valid_dl = DataLoader(valid_ds, shuffle=False, drop_last=False, batch_size=eval_batch_size)
        batch_unc = []
        batch_residual = []
        for batch in valid_dl:
            # predict all properties and compute uncertainties and residuals
            results = self(batch.to(self.get_device()), **kwargs)
            unc, residual = self.compute_uncertainty_residual_pairs(results, batch)
            batch_unc.append(unc)
            batch_residual.append(residual)
        batch_unc = recursive_cat(batch_unc, dim=0)
        batch_residual = recursive_cat(batch_residual, dim=0)
        for key in batch_unc.keys():
            res = batch_residual[key]
            unc = batch_unc[key]
            self.scale_params[key] = self.unc_scaling(res, unc)

    def compute_uncertainty_residual_pairs(self,
                                           results: Dict[Any, torch.Tensor],
                                           graph: Data) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Computes residuals and uncertainties for the provided batch. They are used to calibrate uncertainties.

        Args:
            results (Dict[Any, torch.Tensor]): Results dictionary.
            graph (Data): Atomic data graph.

        Returns: 
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Dictionary containing batch uncertainty and residuals.
        """
        # detach results such that they will never require gradient
        results = recursive_detach(results)
        # compute uncertainties and residuals
        batch_unc = {}
        batch_residual = {}
        if 'features' in self.features_keys:
            if self.energy_unc_residual == 'energy_ae':
                reference = graph.energy
                reference = reference.to(results['energy'].device)
                # divide the total energy residual by the number of atoms to re-scale total uncertainty per atom
                residual = (reference - results['energy']).abs() / graph.n_atoms
            if self.energy_unc_residual == 'forces_rmse':
                reference = graph.forces
                reference = reference.to(results['forces'].device)
                residual = (reference - results['forces']).square().mean(-1)
                residual = segment_sum(residual, idx_i=graph.batch, dim_size=graph.n_atoms.shape[0]) / graph.n_atoms
                residual = residual.sqrt()
            # compute scale factor from residuals and uncertainty
            unc = results['uncertainty']
            assert residual.shape[0] == unc.shape[0]
            batch_unc['uncertainty'] = unc
            batch_residual['uncertainty'] = residual
        if 'atomic_features' in self.features_keys:
            if self.atomic_unc_residual == 'energy_ae':
                reference = graph.energy
                reference = reference.to(results['energy'].device)
                residual = (reference - results['energy']).abs()
                # sum atomic uncertainties to have the same shape as residual
                unc = segment_sum(results['atomic_uncertainties'], idx_i=graph.batch, dim_size=graph.n_atoms.shape[0])
            if self.atomic_unc_residual == 'forces_rmse':
                reference = graph.forces
                reference = reference.to(results['forces'].device)
                # use atomic uncertainties and reduce only the spatial dimension for forces
                residual = (reference - results['forces']).square().mean(-1).sqrt()
                unc = results['atomic_uncertainties']
            if self.atomic_unc_residual == 'forces_norm':
                reference = graph.forces
                reference = reference.to(results['forces'].device)
                # use atomic uncertainties and reduce only the spatial dimension for forces
                residual = (reference - results['forces']).square().sum(-1).sqrt()
                unc = results['atomic_uncertainties']
            assert residual.shape[0] == unc.shape[0]
            batch_unc['atomic_uncertainties'] = unc
            batch_residual['atomic_uncertainties'] = residual
        return batch_unc, batch_residual

    def get_device(self) -> str:
        return self.calc.get_device()

    def to(self, device: str) -> 'UncertaintyAndPropertyCalculator':
        raise NotImplementedError()


class RandomUncertaintyAndPropertyCalculator(UncertaintyAndPropertyCalculator):
    """Provides random features to use combined with the random selection.

    Args:
        calc (TorchCalculator): Ensemble calculator.
        random_seed (int): Random seed for generating random features.
    """

    def __init__(self,
                 calc: TorchCalculator,
                 random_seed: int = 1234,
                 **config: Any):
        super().__init__(calc=calc, **config)
        self.generator = torch.Generator('cpu')
        self.generator.manual_seed(random_seed)

    def get_model_uncertainties(self,
                                results: Dict[str, Union[torch.Tensor, Any]],
                                graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        raise RuntimeError(f'No valid uncertainties are provided by the random method.')

    def get_tfm_features_on_structures(self,
                                       structures: AtomicStructures,
                                       r_cutoff: float,
                                       eval_batch_size: int = 100,
                                       retain_neighbors: bool = True) -> Dict[str, Features]:
        # compute features from AtomicStructures (requires less RAM if retain_neighbors is set to False)
        results = self.eval_on_structures(structures, r_cutoff=r_cutoff, keys=['energy'],
                                          eval_batch_size=eval_batch_size, retain_neighbors=retain_neighbors)
        return self.get_tfm_features_on_batch(results)

    def get_tfm_features_on_ds(self,
                               ds: List[Data],
                               eval_batch_size: int = 100) -> Dict[str, Features]:
        # compute features from AtomicData (may require large RAM as all neighbors are stored)
        results = self.eval_on_ds(ds, keys=['energy'], eval_batch_size=eval_batch_size)
        return self.get_tfm_features_on_batch(results)

    def get_tfm_features_on_batch(self,
                                  results: Dict[str, Union[torch.Tensor, Any]],
                                  graph: Optional[Data] = None) -> Dict[str, Features]:
        features = torch.rand(*results['energy'].shape, generator=self.generator).unsqueeze(-1)
        features = Features(IdentityFeatureMap(n_features=features.shape[1]), TensorFeatureData(features))
        return {'tfm_features': features}

    def prepare_with_train(self,
                           train_ds: List[Data],
                           eval_batch_size: int = 100):
        raise RuntimeError(f'Preparation with training data is not required.')

    def to(self, device: str) -> UncertaintyAndPropertyCalculator:
        self.calc.to(device)
        return self


class PosteriorUncertaintyAndPropertyCalculator(UncertaintyAndPropertyCalculator):
    """Calculates total and atom-based uncertainties using posterior transformation of last-layer or sketched gradient features.

    Args:
        calc (TorchCalculator): Ensemble calculator.
        train_ds (List[Data]): Training data set, used to prepare posterior transformation.
        valid_ds (Optional[List[Data]], optional): Validation data set. If not provided, uncertainties are not calibrated. 
                                                   Defaults to None
        calibr_batch_size (int): Batch size used for calibration. Defaults to 32.
        posterior_sigma (float): Regularizes the inverse of the kernel. Defaults to 1e-2.
        detach_atomic_features (bool): If True, atomic features are detached from the computational graph.
    """
    def __init__(self,
                 calc: TorchCalculator,
                 train_ds: List[Data],
                 valid_ds: Optional[List[Data]] = None,
                 calibr_batch_size: int = 32,
                 posterior_sigma: float = 1e-2,
                 detach_atomic_features: bool = True,
                 **config: Any):
        super().__init__(calc=calc, train_ds=train_ds, valid_ds=valid_ds, **config)
        self.posterior_sigma = posterior_sigma
        self.detach_atomic_features = detach_atomic_features

        self.scale_tfm = {}
        self.post_tfm = {}

        # prepare uncertainties with training data set
        self.prepare_with_train(train_ds, eval_batch_size=calibr_batch_size)

        # compute uncertainty scale parameters
        if valid_ds is not None:
            forces = 'forces' in [self.energy_unc_residual.split('_')[0], self.atomic_unc_residual.split('_')[0]]
            self.scale_uncertainty(valid_ds=valid_ds, features=True,
                                   store_atomic_features='atomic_features' in self.features_keys,
                                   forces=forces, eval_batch_size=calibr_batch_size)

        # empty cache after uncertainty calibration
        torch.cuda.empty_cache()
    
    def get_model_uncertainties(self,
                                results: Dict[str, Union[torch.Tensor, Any]],
                                graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        unc_results = {}
        # get transformed features
        features = self.get_tfm_features_on_batch(results, graph)
        # compute uncertainties = sqrt(variances)
        if 'tfm_features' in features:
            unc = features['tfm_features'].get_kernel_matrix_diag().sqrt()
            unc_results['uncertainty'] = self.scale_params['uncertainty'] * unc
        if 'tfm_atomic_features' in features:
            unc = features['tfm_atomic_features'].get_kernel_matrix_diag().sqrt()
            unc_results['atomic_uncertainties'] = self.scale_params['atomic_uncertainties'] * unc
        return unc_results

    def get_tfm_features_on_structures(self,
                                       structures: AtomicStructures,
                                       r_cutoff: float,
                                       eval_batch_size: int = 100,
                                       retain_neighbors: bool = True) -> Dict[str, Features]:
        # compute features from AtomicStructures (requires less RAM if retain_neighbors is set to False)
        results = self.eval_on_structures(structures, r_cutoff=r_cutoff, keys=self.features_keys,
                                          eval_batch_size=eval_batch_size, retain_neighbors=retain_neighbors,
                                          features=True, store_atomic_features='atomic_features' in self.features_keys)
        return self.get_tfm_features_on_batch(results)

    def get_tfm_features_on_ds(self,
                               ds: List[Data],
                               eval_batch_size: int = 100) -> Dict[str, Features]:
        # compute features from AtomicData (may require large RAM as all neighbors are stored)
        results = self.eval_on_ds(ds, keys=self.features_keys, features=True, eval_batch_size=eval_batch_size,
                                  store_atomic_features='atomic_features' in self.features_keys)
        return self.get_tfm_features_on_batch(results)

    def get_tfm_features_on_batch(self,
                                  results: Dict[str, Union[torch.Tensor, Any]],
                                  graph: Optional[Data] = None) -> Dict[str, Features]:
        tfm_features = {}
        for features_key in self.features_keys:
            features = results[features_key]
            # detach atomic features to prevent memory leak
            if features_key == 'atomic_features' and self.detach_atomic_features:
                features = features.detach()
            features = Features(IdentityFeatureMap(n_features=features.shape[1]), TensorFeatureData(features))
            post_tfm, scale_tfm = self.post_tfm[features_key], self.scale_tfm[features_key]
            tfm_features['tfm_' + features_key] = post_tfm(scale_tfm(features))
        return tfm_features

    def prepare_with_train(self,
                           train_ds: List[Data],
                           eval_batch_size: int = 100):
        results = self.eval_on_ds(train_ds, keys=self.features_keys, features=True, eval_batch_size=eval_batch_size,
                                  store_atomic_features='atomic_features' in self.features_keys)
        for features_key in self.features_keys:
            features = results[features_key]
            # define feature map
            features = Features(IdentityFeatureMap(n_features=features.shape[1]), TensorFeatureData(features))
            # compute scale transformation
            scale_tfm = features.scale_tfm(factor=None)
            # re-scale train features
            features = scale_tfm(features)
            # compute posterior transformation
            post_tfm = features.posterior_tfm(sigma=self.posterior_sigma, allow_kernel_space_posterior=False)
            # store transforms
            self.scale_tfm[features_key] = scale_tfm
            self.post_tfm[features_key] = post_tfm
        # make sure that now uncertainties can be calculated
        self.prepared_with_train = True

    def to(self, device: str) -> UncertaintyAndPropertyCalculator:
        self.calc.to(device)
        return self


class DistanceUncertaintyAndPropertyCalculator(UncertaintyAndPropertyCalculator):
    """Calculates total and atom-based uncertainties using distances (similarity) between last-layer or sketched gradient features.

    Args:
        calc (TorchCalculator): Ensemble calculator.
        train_ds (List[Data]): Training data set. Uncertainties are evaluated as distance to features for atomic structures 
                               in the training data set.
        valid_ds (Optional[List[Data]], optional): Validation data set. If not provided, uncertainties are not calibrated. 
                                                   Defaults to None.
        sq_distance_clamp (float): Clamp distance for numerical stability. Defaults to 1e-8.
        calibr_batch_size (int): Batch size used for calibration. Defaults to 32.
        max_calibr_train_points (int): Maximal number of training points, N_train and N_train x N_atoms, used for total 
                                       and atom-based uncertainty calculations, respectively. Necessary to reduce 
                                       the computational cost and memory usage for atom-based uncertainty. 
                                       Defaults to 262144.
        max_feature_batch_size (int): Maximal batch size used for calculating distances, used to reduce the memory requirements 
                                      when using atom-based uncertainties. Defaults to 65536.
        detach_atomic_features (bool): If True, atomic features are detached from the computational graph.
    """
    def __init__(self,
                 calc: TorchCalculator,
                 train_ds: List[Data],
                 valid_ds: Optional[List[Data]] = None,
                 sq_distance_clamp: float = 1e-8,
                 calibr_batch_size: int = 32,
                 max_calibr_train_points: int = 262144,
                 max_feature_batch_size: int = 65536,
                 detach_atomic_features: bool = True,
                 **config: Any):
        super().__init__(calc=calc, train_ds=train_ds, valid_ds=valid_ds, **config)
        self.sq_distance_clamp = sq_distance_clamp
        self.max_calibr_train_points = max_calibr_train_points
        self.max_feature_batch_size = max_feature_batch_size
        self.detach_atomic_features = detach_atomic_features

        self.train_features = {}

        # prepare uncertainties
        self.prepare_with_train(train_ds, eval_batch_size=calibr_batch_size)

        # compute uncertainty scale parameters
        if valid_ds is not None:
            forces = 'forces' in [self.energy_unc_residual.split('_')[0], self.atomic_unc_residual.split('_')[0]]
            self.scale_uncertainty(valid_ds=valid_ds, features=True,
                                   store_atomic_features='atomic_features' in self.features_keys,
                                   forces=forces, eval_batch_size=calibr_batch_size)

        # empty cache after uncertainty calibration
        torch.cuda.empty_cache()

    def get_model_uncertainties(self,
                                results: Dict[str, Union[torch.Tensor, Any]],
                                graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        unc_results = {}
        # get transformed features
        features = self.get_tfm_features_on_batch(results, graph)
        # clamp distances to prevent small negative numbers due to numerical issues
        # uncertainties = sqrt(variances)
        if 'tfm_features' in features:
            tfm_features = features['tfm_features']
            variance_batch = []
            batch_size = min(len(self.train_features['features']), self.max_feature_batch_size)
            for start, stop in get_batch_intervals(len(self.train_features['features']), batch_size=batch_size):
                train_features_batch = self.train_features['features'][start:stop].simplify()
                variance_batch.append(torch.min(
                    tfm_features.get_sq_dists(train_features_batch).clamp(min=self.sq_distance_clamp), -1).values)
            variance_batch = torch.stack(variance_batch)
            variance_batch = torch.min(variance_batch, 0).values
            unc = variance_batch.sqrt()
            unc_results['uncertainty'] = self.scale_params['uncertainty'] * unc
        if 'tfm_atomic_features' in features:
            tfm_features = features['tfm_atomic_features']
            variance_batch = []
            batch_size = min(len(self.train_features['atomic_features']), self.max_feature_batch_size)
            for start, stop in get_batch_intervals(len(self.train_features['atomic_features']), batch_size=batch_size):
                train_features_batch = self.train_features['atomic_features'][start:stop].simplify()
                variance_batch.append(torch.min(
                    tfm_features.get_sq_dists(train_features_batch).clamp(min=self.sq_distance_clamp), -1).values)
            variance_batch = torch.stack(variance_batch)
            variance_batch = torch.min(variance_batch, 0).values
            unc = variance_batch.sqrt()
            unc_results['atomic_uncertainties'] = self.scale_params['atomic_uncertainties'] * unc
        return unc_results

    def get_tfm_features_on_structures(self,
                                       structures: AtomicStructures,
                                       r_cutoff: float,
                                       eval_batch_size: int = 100,
                                       retain_neighbors: bool = True) -> Dict[str, Features]:
        # compute features from AtomicStructures (requires less RAM if retain_neighbors is set to False)
        results = self.eval_on_structures(structures, r_cutoff=r_cutoff, keys=self.features_keys,
                                          eval_batch_size=eval_batch_size, retain_neighbors=retain_neighbors,
                                          features=True, store_atomic_features='atomic_features' in self.features_keys)
        return self.get_tfm_features_on_batch(results)

    def get_tfm_features_on_ds(self,
                               ds: List[Data],
                               eval_batch_size: int = 100) -> Dict[str, Features]:
        # compute features from AtomicData (may require large RAM as all neighbors are stored)
        results = self.eval_on_ds(ds, keys=self.features_keys, features=True, eval_batch_size=eval_batch_size,
                                  store_atomic_features='atomic_features' in self.features_keys)
        return self.get_tfm_features_on_batch(results)

    def get_tfm_features_on_batch(self,
                                  results: Dict[str, Union[torch.Tensor, Any]],
                                  graph: Optional[Data] = None) -> Dict[str, Features]:
        tfm_features = {}
        for features_key in self.features_keys:
            features = results[features_key]
            # detach atomic features to prevent memory leak
            if features_key == 'atomic_features' and self.detach_atomic_features:
                features = features.detach()
            tfm_features['tfm_' + features_key] = Features(IdentityFeatureMap(n_features=features.shape[1]),
                                                           TensorFeatureData(features))
        return tfm_features

    def prepare_with_train(self,
                           train_ds: List[Data],
                           eval_batch_size: int = 100):
        results = self.eval_on_ds(train_ds, keys=self.features_keys, features=True, eval_batch_size=eval_batch_size,
                                  store_atomic_features='atomic_features' in self.features_keys)
        for features_key in self.features_keys:
            features = results[features_key]
            # define feature map
            features = Features(IdentityFeatureMap(n_features=features.shape[1]), TensorFeatureData(features))
            # run max_dist algorithm to reduce number of training points, may be important when running
            # uncertainty-biased sampling with atomic uncertainties
            if len(features) > self.max_calibr_train_points:
                sel_method = MaxDistSelectionMethod(train_features=None, pool_features=features,
                                                    sel_with_train=False)
                idxs = sel_method.select(self.max_calibr_train_points)
                features = features[idxs].simplify()
            # store train features
            self.train_features[features_key] = features
        # make sure that now uncertainties can be calculated
        self.prepared_with_train = True

    def to(self, device: str) -> UncertaintyAndPropertyCalculator:
        self.calc.to(device)
        for features_key in self.features_keys:
            train_features = self.train_features[features_key].get_feature_matrix().to(device)
            self.train_features[features_key] = Features(IdentityFeatureMap(n_features=train_features.shape[1]),
                                                         TensorFeatureData(train_features))
        return self


class EnsembleUncertaintyAndPropertyCalculator(UncertaintyAndPropertyCalculator):
    """Calculates total and atom-based uncertainties using the ensemble variance.

    Args:
        calc (TorchCalculator): Ensemble calculator.
        train_ds (Optional[List[Data]], optional): Training data set. If not provided, 
                                                   approx. partition function is not computed. 
                                                   Defaults to None.
        valid_ds (Optional[List[Data]], optional): Validation data set. If not provided, 
                                                   uncertainties are not calibrated. 
                                                   Defaults to None
        variance_clamp (float): Clip ensemble variance by a small number for numerical stability. 
                                Defaults to 1e-8.
        energy_unc_key (str): Property key, for which disagreement the total uncertainty is computed: 
                              'energy', 'atomic_energies', and 'forces'. 'energy'-case uses energy 
                              features per atom. Defaults to 'energy'.
        atomic_unc_key (str): Property key, for which disagreement the atom-based uncertainty is computed: 
                              'atomic_energies' and 'forces'. Defaults to 'forces'.
        calibr_batch_size (int): Batch size used for calibration. Defaults to 32.
        detach_atomic_features (bool): If True, atomic features are detached from the computational graph.
    """
    def __init__(self,
                 calc: TorchCalculator,
                 train_ds: Optional[List[Data]] = None,
                 valid_ds: Optional[List[Data]] = None,
                 variance_clamp: float = 1e-8,
                 energy_unc_key: bool = 'energy',
                 atomic_unc_key: bool = 'forces',
                 calibr_batch_size: int = 32,
                 detach_atomic_features: bool = True,
                 **config: Any):
        super().__init__(calc=calc, train_ds=train_ds, valid_ds=valid_ds, **config)
        assert energy_unc_key in ['energy', 'atomic_energies', 'forces']
        assert atomic_unc_key in ['atomic_energies', 'forces']
        self.variance_clamp = variance_clamp
        self.energy_unc_key = energy_unc_key
        self.atomic_unc_key = atomic_unc_key
        self.detach_atomic_features = detach_atomic_features

        # no preparations with train are needed for ensemble uncertainty
        self.prepared_with_train = True

        # compute uncertainty scale parameters
        if valid_ds is not None:
            forces = 'forces' in [self.energy_unc_residual.split('_')[0], self.atomic_unc_residual.split('_')[0],
                                  self.energy_unc_key, self.atomic_unc_key]
            self.scale_uncertainty(valid_ds=valid_ds, forces=forces, eval_batch_size=calibr_batch_size)

        # empty cache after uncertainty calibration
        torch.cuda.empty_cache()

    def get_model_uncertainties(self,
                                results: Dict[str, Union[torch.Tensor, Any]],
                                graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        unc_results = {}
        # get transformed features
        features = self.get_tfm_features_on_batch(results, graph)
        # compute uncertainties = sqrt(variances)
        if 'tfm_features' in features:
            variance = features['tfm_features'].get_kernel_matrix_diag()
            # clamping ensemble variance by a small number for numerical stability
            unc = variance.clamp(min=self.variance_clamp).sqrt()
            unc_results['uncertainty'] = self.scale_params['uncertainty'] * unc
        if 'tfm_atomic_features' in features:
            variance = features['tfm_atomic_features'].get_kernel_matrix_diag()
            # clamping ensemble variance by a small number for numerical stability
            unc = variance.clamp(min=self.variance_clamp).sqrt()
            unc_results['atomic_uncertainties'] = self.scale_params['atomic_uncertainties'] * unc
        return unc_results

    def get_tfm_features_on_structures(self,
                                       structures: AtomicStructures,
                                       r_cutoff: float,
                                       eval_batch_size: int = 100,
                                       retain_neighbors: bool = True) -> Dict[str, Features]:
        # compute features from AtomicStructures (requires less RAM if retain_neighbors is set to False)
        forces = 'forces' in [self.energy_unc_residual.split('_')[0], self.atomic_unc_residual.split('_')[0],
                              self.energy_unc_key, self.atomic_unc_key]
        eval_batch_size = min(eval_batch_size, len(structures))
        batch_tfm_features = []
        for start, stop in get_batch_intervals(len(structures), batch_size=eval_batch_size):
            if retain_neighbors:
                batch_structures = structures[start:stop]
            else:
                batch_structures = AtomicStructures([copy.copy(s) for s in structures.structures[start:stop]])
            batch = batch_structures.to_data(r_cutoff=r_cutoff)
            graph = next(iter(DataLoader(batch, batch_size=len(batch), shuffle=False, drop_last=False)))
            results = self.eval_on_ds(batch, keys=['individual_results'], forces=forces,
                                        eval_batch_size=eval_batch_size)
            batch_tfm_features.append(
                {key: val.get_feature_matrix()
                    for key, val in self.get_tfm_features_on_batch(results, graph.to(self.get_device())).items()})
        tfm_features = {key: Features(IdentityFeatureMap(n_features=val.shape[1]), TensorFeatureData(val))
                        for key, val in recursive_cat(batch_tfm_features, dim=0).items()}
        return tfm_features

    def get_tfm_features_on_ds(self,
                               ds: List[Data],
                               eval_batch_size: int = 100) -> Dict[str, Features]:
        # compute features from AtomicData (may require large RAM as all neighbors are stored)
        forces = 'forces' in [self.energy_unc_residual.split('_')[0], self.atomic_unc_residual.split('_')[0],
                              self.energy_unc_key, self.atomic_unc_key]
        dl = DataLoader(ds, shuffle=False, drop_last=False, batch_size=eval_batch_size)
        batch_tfm_features = []
        for batch in dl:
            results = self(batch.to(self.get_device()), forces=forces)
            batch_tfm_features.append(
                {key: val.get_feature_matrix()
                    for key, val in self.get_tfm_features_on_batch(results, batch.to(self.get_device())).items()})
        tfm_features = {key: Features(IdentityFeatureMap(n_features=val.shape[1]), TensorFeatureData(val))
                        for key, val in recursive_cat(batch_tfm_features, dim=0).items()}
        return tfm_features

    def get_tfm_features_on_batch(self,
                                  results: Dict[str, Union[torch.Tensor, Any]],
                                  graph: Optional[Data] = None) -> Dict[str, Features]:
        # define all property keys which have to be computed
        keys = []
        if 'features' in self.features_keys:
            keys = keys + [self.energy_unc_key]
        if 'atomic_features' in self.features_keys:
            if self.atomic_unc_key not in keys:
                keys = keys + [self.atomic_unc_key]

        predictions = {}
        for key in keys:
            predictions[key] = torch.stack([r[key] for r in results['individual_results']], dim=-1)

        tfm_features = {}
        if 'features' in self.features_keys:
            assert graph is not None
            if self.energy_unc_key == 'energy':
                prediction = predictions[self.energy_unc_key]
                features = (prediction - prediction.mean(-1, keepdim=True)) / np.sqrt(prediction.shape[-1]) / graph.n_atoms[:, None]
            if self.energy_unc_key == 'atomic_energies':
                prediction = predictions[self.energy_unc_key]
                features = ((prediction - prediction.mean(dim=-1, keepdim=True)) ** 2).mean(-1)
                features = segment_sum(features, idx_i=graph.batch, dim_size=graph.n_atoms.shape[0]) / graph.n_atoms
                features = features.sqrt().unsqueeze(-1)
            if self.energy_unc_key == 'forces':
                prediction = predictions[self.energy_unc_key]
                features = ((prediction - prediction.mean(dim=-1, keepdim=True)) ** 2).mean(-1)
                features = features.mean(-1)
                features = segment_sum(features, idx_i=graph.batch, dim_size=graph.n_atoms.shape[0]) / graph.n_atoms
                features = features.sqrt().unsqueeze(-1)
            tfm_features['tfm_features'] = Features(IdentityFeatureMap(n_features=features.shape[1]),
                                                    TensorFeatureData(features))
        if 'atomic_features' in self.features_keys:
            if self.atomic_unc_key == 'atomic_energies':
                prediction = predictions[self.atomic_unc_key]
                features = ((prediction - prediction.mean(dim=-1, keepdim=True)) ** 2).mean(-1)
                features = features.sqrt().unsqueeze(-1)
            elif self.atomic_unc_key == 'forces':
                prediction = predictions[self.atomic_unc_key]
                features = ((prediction - prediction.mean(dim=-1, keepdim=True)) ** 2).mean(-1)
                features = features.mean(-1)
                features = features.sqrt().unsqueeze(-1)
            # detach atomic features to prevent memory issues
            if self.detach_atomic_features:
                features = features.detach()
            tfm_features['tfm_atomic_features'] = Features(IdentityFeatureMap(n_features=features.shape[1]),
                                                           TensorFeatureData(features))
        return tfm_features

    def prepare_with_train(self,
                           train_ds: List[Data],
                           batch_size: int = 100):
        raise RuntimeError(f'Preparation with training data is not required.')

    def to(self, device: str) -> UncertaintyAndPropertyCalculator:
        self.calc.to(device)
        return self


class AdaptiveBiasingStrength:
    """Re-scales uncertainty gradients relative to actual gradients

    Args:
        forces_burn_in_period (int, optional): Burn-in period for atomic forces. 
                                               Defaults to 100.
        stress_burn_in_period (int, optional): Burn-in period for stress tensor. 
                                               Defaults to 100.
        eps (float, optional): Small value used for numerical stability when computing ration 
                               between uncertainty and actual gradients. Defaults to 1e-8.
    """
    def __init__(self,
                 forces_burn_in_period: int = 100,
                 stress_burn_in_period: int = 100,
                 eps: float = 1e-8,
                 **config: Any):
        self.forces_burn_in_period = forces_burn_in_period
        self.stress_burn_in_period = stress_burn_in_period
        self.eps = eps

        self.true_forces_norm_sq = torch.zeros(forces_burn_in_period)
        self.true_stress_norm_sq = torch.zeros(stress_burn_in_period)
        self.bias_forces_norm_sq = torch.zeros(forces_burn_in_period)
        self.bias_stress_norm_sq = torch.zeros(stress_burn_in_period)

        self.forces_rescaling_factor = 0.0
        self.stress_rescaling_factor = 0.0

        self.forces_step = 0
        self.stress_step = 0

    def __call__(self,
                 results: Dict[str, Any],
                 **kwargs: Any) -> Dict[str, Any]:
        """Calculates re-scaling factors.

        Args:
            results (Dict[str, Any]): Results dictionary, containing uncertainty and actual gradients.

        Returns:
            Dict[str, Any]: Results dictionary, containing rescaling factors for uncertainty gradients.
        """
        # detach results to not expect gradients
        bias_results = {}
        for key, value in results.items():
            if key in ['forces', 'stress']:
                bias_results['true_' + key] = recursive_detach(value)
            else:
                bias_results[key] = recursive_detach(value)

        # compute exponential moving averages for true and bias forces norm and the scale factor from them
        if 'true_forces' and 'bias_forces' in bias_results:
            # update bias and true forces squared norm
            self.bias_forces_norm_sq = torch.cat((self.bias_forces_norm_sq[1:],
                                                  bias_results['bias_forces'].norm().square().unsqueeze(0)))
            self.true_forces_norm_sq = torch.cat((self.true_forces_norm_sq[1:],
                                                  bias_results['true_forces'].norm().square().unsqueeze(0)))
            if self.forces_step > self.forces_burn_in_period:
                # update re-scaling factor
                self.forces_rescaling_factor = self.true_forces_norm_sq.sum().sqrt() / (
                        self.bias_forces_norm_sq.sum().sqrt() + self.eps)
            self.forces_step += 1

        # compute exponential moving averages for true and bias stress norm and the scale factor from them
        if 'true_stress' and 'bias_stress' in bias_results:
            # update bias and true stress squared norm in the window
            self.bias_stress_norm_sq = torch.cat((self.bias_stress_norm_sq[1:],
                                                  bias_results['bias_stress'].norm().square().unsqueeze(0)))
            self.true_stress_norm_sq = torch.cat((self.true_stress_norm_sq[1:],
                                                  bias_results['true_stress'].norm().square().unsqueeze(0)))
            if self.stress_step > self.stress_burn_in_period:
                # update re-scaling factor
                self.stress_rescaling_factor = self.true_stress_norm_sq.sum().sqrt() / (
                        self.bias_stress_norm_sq.sum().sqrt() + self.eps)
            self.stress_step += 1

        # update rescaling factor or return zero if not computed
        bias_results['forces_rescaling_factor'] = self.forces_rescaling_factor
        bias_results['stress_rescaling_factor'] = self.stress_rescaling_factor
        return bias_results

    def to(self, device: str) -> 'AdaptiveBiasingStrength':
        """Moves calculations to the provided device.

        Args:
            device: Device to which calculations are moved.

        Returns: 
            AdaptiveBiasingStrength: The `AdaptiveBiasingStrength` object.
        """
        self.true_forces_norm_sq = self.true_forces_norm_sq.to(device)
        self.true_stress_norm_sq = self.true_stress_norm_sq.to(device)
        self.bias_forces_norm_sq = self.bias_forces_norm_sq.to(device)
        self.bias_stress_norm_sq = self.bias_stress_norm_sq.to(device)
        return self


class LinearUncertaintyBiasedStructurePropertyCalculator(TorchCalculator):
    """Computes the linear uncertainty bias. It uses automatic differentiation to add bias atomic forces and stress to the true ones.

    Args:
        calc (TorchCalculator): Calculator with an uncertainty method.
        forces_biasing_strength (float, optional): Force biasing strength. Defaults to 0.05.
        stress_biasing_strength (float, optional): Stress biasing strength. Defaults to 0.05.
        bias_with_atomic_uncertainties (bool, optional): If True, the sum of atom-based uncertainties is used for biasing. 
                                                         Defaults to False.
        atom_based_rescaling (Optional[np.ndarray], optional): Array containing atom-based re-scaling factor. Used to damp biasing of 
                                                               fast oscillating modes. Defaults to None.
    """
    def __init__(self,
                 calc: TorchCalculator,
                 forces_biasing_strength: float = 0.05,
                 stress_biasing_strength: float = 0.05,
                 bias_with_atomic_uncertainties: bool = False,
                 atom_based_rescaling: Optional[np.ndarray] = None,
                 **config: Any):
        self.calc = calc
        self.forces_biasing_strength = forces_biasing_strength
        self.stress_biasing_strength = stress_biasing_strength
        self.bias_with_atomic_uncertainties = bias_with_atomic_uncertainties
        self.atom_based_rescaling = atom_based_rescaling
        if self.atom_based_rescaling is not None:
            self.atom_based_rescaling = torch.as_tensor(self.atom_based_rescaling)
        self.adaptive_biasing_strength = AdaptiveBiasingStrength(**config)

    def __call__(self,
                 graph: Data,
                 **kwargs: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """Calculates biased atomic forces and stress.

        Args:
            graph (Data): Atomic data graph.

        Returns: 
            Dict[str, Union[torch.Tensor, Any]]: Dictionary with all properties for the provided atomic data graph.
        """
        # prepare graph and the list containing graph attributes requiring gradients
        graph, require_gradients = prepare_gradients(graph=graph, **kwargs)
        # compute results on the graph
        results = self.calc(graph, **kwargs)
        # use the total energy uncertainty or the sum of atomic energy uncertainties
        if self.bias_with_atomic_uncertainties:
            assert 'atomic_uncertainties' in results
            bias_energy = results['atomic_uncertainties'].sum()
        else:
            assert 'uncertainty' in results
            bias_energy = results['uncertainty']
        results['bias_energy'] = bias_energy
        # compute gradients for selected graph attributes (positions, strain, etc.)
        if require_gradients:
            grads = torch.autograd.grad([bias_energy], [getattr(graph, key) for key in require_gradients],
                                        torch.ones_like(bias_energy))
        # store bias forces and stress
        if 'positions' in require_gradients:
            results['bias_forces'] = grads[0].neg()
        if 'strain' in require_gradients:
            volume = torch.einsum('bi, bi -> b', graph.cell[:, 0, :],
                                  torch.cross(graph.cell[:, 1, :], graph.cell[:, 2, :], dim=1))
            if grads[-1] is not None:
                results['bias_stress'] = grads[-1] / volume[:, None, None]
            else:
                results['bias_stress'] = torch.zeros_like(graph.cell) / volume[:, None, None]
        # compute adaptive bias forces and stress
        bias_results = self.adaptive_biasing_strength(results)
        if 'true_forces' and 'bias_forces' in bias_results:
            # update forces biasing strength
            forces_biasing_strength_rescaled = self.forces_biasing_strength * bias_results['forces_rescaling_factor']
            if self.atom_based_rescaling is not None:
                forces_biasing_strength_rescaled = forces_biasing_strength_rescaled * self.atom_based_rescaling[:, None]
            bias_results['forces'] = bias_results['true_forces'] - forces_biasing_strength_rescaled * \
                                     bias_results['bias_forces']
        if 'true_stress' and 'bias_stress' in bias_results:
            # update stress biasing strength
            stress_biasing_strength_rescaled = self.stress_biasing_strength * bias_results['stress_rescaling_factor']
            bias_results['stress'] = bias_results['true_stress'] - stress_biasing_strength_rescaled * \
                                     bias_results['bias_stress']
        return bias_results
    
    def get_device(self) -> str:
        return self.calc.get_device()

    def to(self, device: str) -> TorchCalculator:
        self.calc.to(device)
        self.adaptive_biasing_strength.to(device)
        if self.atom_based_rescaling is not None:
            self.atom_based_rescaling = self.atom_based_rescaling.to(device)
        return self


def get_torch_calculator(models: List[ForwardAtomisticNetwork],
                         unc_method: Optional[str] = None,
                         bias_method: Optional[str] = None,
                         **kwargs: Any):
    """Provides a calculator from a list of models. It includes standard energy, atomic forces, and stress
    predictions as well as more advanced uncertainty and bias calculations if corresponding methods are defined.

    Args:
        models (List[ForwardAtomisticNetwork]): List of atomistic forward models.
        unc_method (str): Method for uncertainty calculation (total and atom-based). Defaults to None.
        bias_method (str): Method for biasing energies, atomic forces, and stresses (total and atom-based). 
                           Defaults to None.

    Returns: 
        Property calculator.
    """
    # attach structure and property as well as ensemble calculator to the list of models
    calc = EnsembleCalculator([StructurePropertyCalculator(model, **kwargs) for model in models], **kwargs)
    # define uncertainty calculator from uncertainty method (if provided)
    if unc_method is not None:
        if unc_method == 'random':
            calc = RandomUncertaintyAndPropertyCalculator(calc, **kwargs)
        elif unc_method == 'posterior':
            calc = PosteriorUncertaintyAndPropertyCalculator(calc, **kwargs)
        elif unc_method == 'distance':
            calc = DistanceUncertaintyAndPropertyCalculator(calc, **kwargs)
        elif unc_method == 'ensemble':
            calc = EnsembleUncertaintyAndPropertyCalculator(calc, **kwargs)
        else:
            raise NotImplementedError(f'{unc_method=} is not implemented yet.')
    if bias_method is not None:
        if bias_method == 'linear_unc':
            calc = LinearUncertaintyBiasedStructurePropertyCalculator(calc, **kwargs)
        else:
            raise NotImplementedError(f'{bias_method=} is not implemented yet.')
    return calc
