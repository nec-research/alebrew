"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     samplers.py 
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

import copy
from pathlib import Path
from typing import Optional, Union, List, Tuple, Callable, Any

import torch
import torch.nn as nn

import ase
import ase.io
import ase.md.md
import numpy as np
from ase import units
from ase.calculators.general import Calculator
from ase.io import read
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from alebrew.data.data import AtomicStructures
from alebrew.utils.misc import save_object, load_object, PickleStreamWriter
from alebrew.utils.process_pool import ProcessPoolMapper
from alebrew.data.data import AtomicStructure, AtomicData
from alebrew.utils.torch_geometric import Data, DataLoader
from alebrew.model.calculators import TorchCalculator


class BaseSampler:
    """Runs the defined atomistic simulation."""
    def run(self, folder: Union[str, Path]) -> AtomicStructures:
        """Runs the atomic simulation and stores results/progress in the folder.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.

        Returns: 
            AtomicStructures: Sampled/explored atomic structures.
        """
        raise NotImplementedError()


class ParallelEnsembleSampler(BaseSampler):
    """Run an ensemble of atomistic simulations in parallel.

    Args:
        sampler (List[BaseSampler]): List of atomistic simulations defined by the `BaseSampler` objects.
        n_threads (int): The number of parallel threads.
    """
    def __init__(self,
                 samplers: List[BaseSampler],
                 n_threads: int = 1,
                 **config: Any):
        self.samplers = samplers
        self.n_threads = n_threads

    def run(self, folder: Union[str, Path]) -> AtomicStructures:
        fs = [sampler.run for sampler in self.samplers]
        args_tuples = [(Path(folder) / f'sampler_{i}', ) for i in range(len(self.samplers))]
        mapper = ProcessPoolMapper(n_threads=min(len(self.samplers), self.n_threads), spawn=True)
        # returns a list of structures
        structures_list = mapper.map(fs, args_tuples)
        # concatenate all sampled structures
        structures = structures_list[0]
        for other_structures in structures_list[1:]:
            structures = structures + other_structures
        return structures


def strained_cell(strain: np.ndarray,
                  cell: np.ndarray) -> np.ndarray:
    """Computes the strained periodic cell.

    Args:
        strain (np.ndarray): Symmetric strain tensor.
        cell (np.ndarray): Periodic cell.

    Returns:
        np.ndarray: Strained periodic cell.
    """
    # to compute strained cell we use the fact that the distance between two points changes as
    # new_r ** 2 = old_r ** 2 + old_r_i * old_r_j * 2 * strain_matrix_ij
    # new_r = old_r @ sqrt(I + 2 * strain_matrix)
    # new_cell = old_cell @ sqrt(I + 2 * strain_matrix)
    # a similar implementation is provided by Psiflow (https://github.com/svandenhaute/psiflow)
    assert np.allclose(strain, strain.T)
    a = np.eye(3) + 2.0 * strain
    w, v = np.linalg.eigh(a)
    a_sqrt = v @ np.diag(np.sqrt(w)) @ v.T
    return cell @ a_sqrt


class RandomSampler(BaseSampler):
    """Randomly pre-samples atomic structures. Typically, it is used to prepare 
    an initial training data set.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object from which the simulation is 
                           initiated.
        seed (int, optional): Random seed. Defaults to 0.
        max_step (int, optional): The maximal number of random displacements. 
                                  Defaults to 10.
        mask (Optional[np.ndarray], optional): If np.eye(3) is used, then the cell is scaled 
                                               only along x-, y-, and z- axis. Defaults to None.
        wrap (bool, optional): Wrap positions back to the periodic cell. Defaults to False.
        to_triu (bool, optional): If True, periodic cell is transformed to an upper triangular. 
                                  Defaults to False.
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 seed: int = 0,
                 max_step: int = 10,
                 mask: Optional[np.ndarray] = None,
                 wrap: bool = False,
                 to_triu: bool = False,
                 **config: Any):
        self.atoms = atoms
        self.i_step = 0
        self.max_step = max_step

        # mask to strain cell along particular directions only
        assert mask is None or mask.shape == (3, 3)
        self.mask = mask

        # set random state
        self.random_state = np.random.RandomState(seed=seed)

        # wrap positions
        self.wrap = wrap
        # rotate axes such that unit cell is upper triangular
        self.to_triu = to_triu

    def random_shift(self, shape: Union[int, Tuple[int]]) -> np.ndarray:
        """Computes random shift vectors for atomic positions.

        Args:
            shape (Union[int, Tuple[int]]): Shape of atomic positions.

        Returns: 
            np.ndarray: Random shifts for atomic positions.

        """
        raise NotImplementedError()

    def random_strain(self, shape: Union[int, Tuple[int]]) -> np.ndarray:
        """Computes a random strain deformation matrix.

        Args:
            shape (Union[int, Tuple[int]]): Shape of the periodic cell.

        Returns: 
            np.ndarray: Random strain deformation matrix.
        """
        raise NotImplementedError()

    def run(self, folder: Union[str, Path]) -> AtomicStructures:
        # manage directories
        folder = Path(folder)
        if not os.path.exists(folder):
            # create directory if it does not exist
            os.makedirs(folder)
        else:
            # otherwise, load the last state of the sampler
            self._load_state(folder)
        # write sampled atoms into trajectory
        traj = ase.io.Trajectory(folder / 'sampler.traj', 'a')
        try:
            while True:
                # run a single step
                self._irun(traj, folder)
        except StopIteration:
            traj.close()
            pass
        # load trajectory and transform to structures
        traj = read(folder / 'sampler.traj', ':')
        structures = AtomicStructures.from_traj(traj, wrap=self.wrap)
        if self.to_triu:
            structures = structures.to_triu()
        return structures

    def _irun(self,
              traj: ase.io.Trajectory,
              folder: Union[str, Path]):
        """Runs the atomistic simulation for a single step.

        Args:
            traj (ase.io.Trajectory): Trajectory in which `ase.Atoms` are stored.
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        if self.i_step >= self.max_step:
            raise StopIteration()
        else:
            self.i_step += 1
            # create random displaced structures around the initial one
            atoms = self.atoms.copy()
            positions = atoms.get_positions()
            # shift positions by a random vector
            positions += self.random_shift(positions.shape)
            if np.any(atoms.get_pbc()):
                cell = np.asarray(atoms.get_cell())
                # compute scaled positions (fractional coordinates)
                scaled_positions = positions @ np.linalg.inv(cell)
                strain = self.random_strain(cell.shape)
                # symmetrize strain matrix
                strain += strain.T
                strain /= 2.0
                if self.mask is not None:
                    strain *= self.mask
                # compute new strained cell
                cell = strained_cell(strain, cell)
                atoms.set_cell(cell)
                positions = scaled_positions @ cell
            atoms.set_positions(positions)
            # write random structure
            traj.write(atoms)
            # save current state of the sampler
            self._save_state(folder)

    def _load_state(self, folder: Union[str, Path]):
        """Loads the sampler's state to restart the atomistic simulation.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        if os.path.exists(folder / 'sampler_state.pkl'):
            state_dict = load_object(folder / 'sampler_state.pkl')
            self.i_step = state_dict['i_step']
            self.random_state.set_state(state_dict['random_state'])

    def _save_state(self, folder: Union[str, Path]):
        """Stores the sampler's state to restart the atomistic simulation.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        to_save = {'i_step': self.i_step,
                   'random_state': self.random_state.get_state()}
        save_object(folder / f'sampler_state.pkl', to_save)


class UniformRandomSampler(RandomSampler):
    """Generates random samples using uniform atomic positions shifts and cell strain deformations.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` objects to initiate the simulation.
        amplitude_shift (float): Amplitude of the position shifts.
        amplitude_strain (float): Amplitude of the cell strain.
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 amplitude_shift: float = 0.1,
                 amplitude_strain: float = 0.05,
                 **config: Any):
        super().__init__(atoms=atoms, **config)

        self.amplitude_shift = amplitude_shift
        self.amplitude_strain = amplitude_strain

    def random_shift(self, shape: Union[int, Tuple[int]]) -> np.ndarray:
        return self.random_state.uniform(-self.amplitude_shift, self.amplitude_shift, shape)

    def random_strain(self, shape: Union[int, Tuple[int]]) -> np.ndarray:
        return self.random_state.uniform(-self.amplitude_strain, self.amplitude_strain, shape)


class AdversarialLoss:
    """Computes adversarial loss as proposed in https://www.nature.com/articles/s41467-021-25342-8.

    Args:
        calc (TorchCalculator): The `TorchCalculator` object which predicts energies and uncertainties.
        adversarial_temperature (float, optional): Temperature in kT units. It is used restrict the 
                                                   range of sampled energies. Defaults to 1.0.
    """
    def __init__(self, 
                 calc: TorchCalculator, 
                 adversarial_temperature: float = 1.0):
        self.calc = calc
        self.adversarial_temperature = adversarial_temperature
        
        assert self.calc.partition_function is not None
        
    def __call__(self, 
                 graph: Data, 
                 **kwargs: Any) -> torch.Tensor:
        """Calculates the adversarial loss for atomic data in the batch.

        Args:
            graph (Data): Atomic structures represented as a graph.

        Returns:
            torch.Tensor: Loss value.
        """
        # compute results on the graph
        results = self.calc(graph, **kwargs)
        boltzmann_probability = torch.exp(-results['energy'] / self.adversarial_temperature) / self.calc.partition_function
        # define as -u(x)*p(x) to make steepest ascent
        return -1.0 * results['uncertainty'] * boltzmann_probability


class AdversarialSampler(BaseSampler):
    """Performs adversarial attacks as proposed in: https://www.nature.com/articles/s41467-021-25342-8.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object from which the simulation is initiated.
        calc (Calculator): The `Calculator` object which predicts energies and uncertainties.
        max_step (int, optional): The maximal number of steps. Defaults to 1000.
        eval_step (int, optional): Defines how often sampled structures are saved. Defaults to 10.
        adversarial_shift (float, optional): Amplitude of the position shifts. Defaults to 0.02.
        adversarial_temperature (float, optional): Temperature in kT units used restrict the range of 
                                                   sampled energies. Defaults to 1.0.
        adversarial_lr (float, optional): Learning rate. Defaults to 0.005.
        seed (int, optional): Random seed. Defaults to 0.
        unc_threshold (Optional[float], optional): Total uncertainty threshold, used to terminate simulations. 
                                                   If None, only atom-based uncertainty is used. Defaults to None.
        atomic_unc_threshold (Optional[float], optional): Atom-based uncertainty threshold, used to terminate simulations. 
                                                          If None, only total uncertainty is used. Defaults to None.
        use_softmax (bool, optional): If True, softmax is applied to atom-based uncertainties. Defaults to False.
        eps_forces_norm (float, optional): Small number used during computing forces norm for numerical stability. 
                                           Used if `use_softmax=True`. Defaults to 0.2.
        to_triu (bool, optional): If True, periodic cell is transformed into an upper triangular. Defaults to False.
        write_traj_details (bool, optional): If True, properties such as atom-based uncertainties, etc., are stored. 
                                             Defaults to False.
        write_traj_properties (List[str], optional): List with properties stored during simulations. Defaults to ['energy'].
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 calc: Calculator,
                 max_step: int = 1000,
                 eval_step: int = 10,
                 adversarial_shift: float = 0.02,
                 adversarial_temperature: float = 1.0,
                 adversarial_lr: float = 0.005,
                 seed: int = 0,
                 unc_threshold: Optional[float] = None,
                 atomic_unc_threshold: Optional[float] = None,
                 use_softmax: bool = False,
                 eps_forces_norm: float = 0.2,  # eV/Ang.
                 wrap: bool = False,
                 to_triu: bool = False,
                 write_traj_details: bool = False,
                 write_traj_properties: List[str] = ['energy'],
                 **config: Any):
        self.atoms = atoms
        self.atoms.set_calculator(calc)
        
        # define initial graph
        structure = AtomicStructure.from_atoms(atoms, wrap=calc.wrap, neighbors=calc.neighbors)
        structure = structure.to_type_names(calc.atomic_type_converter, check=True)
        dl = DataLoader([AtomicData(structure, r_cutoff=calc.r_cutoff, skin=calc.skin)], batch_size=1, shuffle=False,
                        drop_last=False)
        self.graph = next(iter(dl)).to(calc.device)
        
        # define adversarial loss, parameters, and optimizer
        self.loss_fn = AdversarialLoss(calc=calc.calc, adversarial_temperature=adversarial_temperature)
        self.delta = nn.Parameter(torch.zeros(self.graph.positions.shape, dtype=self.graph.positions.dtype, device=self.graph.positions.device))
        self.opt = torch.optim.Adam([self.delta], lr=adversarial_lr)
        
        # all samplers have similar state attributes
        self.i_step = 0

        # sampler specific parameters as the maximal amount of steps
        # and the step number after which the dynamics is evaluated
        self.max_step = max_step
        self.eval_step = eval_step
        
        # define random displacement
        self.random_state = np.random.RandomState(seed=seed)
        random_displacement = np.random.normal(0.0, adversarial_shift, self.graph.positions.shape)
        self.random_displacement = torch.as_tensor(random_displacement, dtype=self.graph.positions.dtype, device=self.graph.positions.device)

        # use uncertainty threshold to stop, if provided
        self.unc_threshold = unc_threshold
        self.atomic_unc_threshold = atomic_unc_threshold
        self.use_softmax = use_softmax
        assert self.unc_threshold is None or hasattr(self.atoms.calc, 'get_uncertainty')
        assert self.atomic_unc_threshold is None or hasattr(self.atoms.calc, 'get_atomic_uncertainties')
        self.eps_forces_norm = eps_forces_norm

        # wrap positions
        self.wrap = wrap
        # rotate axes such that unit cell is upper triangular
        self.to_triu = to_triu

        # change to True if details on the trajectory (forces, stress, bias, etc.) should be written into a file
        self.write_traj_details = write_traj_details
        self.write_traj_properties = write_traj_properties
        if self.write_traj_details:
            self.traj_details_writer = {}
        
    def run(self, folder: Union[str, Path]) -> AtomicStructures:
        # manage directories
        folder = Path(folder)
        if not os.path.exists(folder):
            # create directory if it does not exist
            os.makedirs(folder)
        else:
            # load the last state of sampler otherwise
            self._load_state(folder)
        # write sampled atoms and respective energy/temperature (for each eval step)
        traj = ase.io.Trajectory(folder / 'sampler.traj', 'a')
        try:
            while True:
                self._irun(traj, folder)
        except StopIteration:
            traj.close()
            pass
        if self.write_traj_details:
            for key in self.traj_details_writer:
                self.traj_details_writer[key].close()
        # load trajectory and transform to structures
        traj = read(folder / 'sampler.traj', ':')
        structures = AtomicStructures.from_traj(traj, wrap=self.wrap)
        if self.to_triu:
            structures = structures.to_triu()
        return structures
    
    def _irun(self,
              traj: ase.io.Trajectory,
              folder: Union[str, Path]):
        """Runs the atomistic simulation for a single step.

        Args:
            traj (ase.io.Trajectory): Trajectory in which `ase.Atoms` are stored.
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        if self.i_step >= self.max_step:
            raise StopIteration()
        else:
            # run adversarial sampling for one more step
            # we do not save the first frame as it may be a training point
            # also it skips repetitions in case active_learning is restarted
            graph = copy.deepcopy(self.graph).to(self.graph.positions.device)
            graph.positions = graph.positions + self.delta + self.random_displacement
            loss = self.loss_fn(graph, features=True, store_atomic_features=True, forces=True, create_graph=True)
            grads = torch.autograd.grad([loss], [self.delta], torch.ones_like(loss), create_graph=True)
            self.delta.grad = grads[0]
            self.opt.step()
            self.opt.zero_grad()
            self.i_step += 1
            # need to call calculate to update results for new atoms (for uncertainty evaluation)
            atoms = self.atoms.copy()
            positions = atoms.get_positions()
            # shift positions
            atoms.set_positions(positions + self.delta.detach().cpu().numpy() + self.random_displacement.detach().cpu().numpy())
            self.atoms.calc.calculate(atoms, properties=self.write_traj_properties)
            # write uncertainties to track progress of simulation
            if self.i_step % self.eval_step == 0:
                # write new atoms and save sampler state for re-starting
                traj.write(atoms)
                self._save_state(folder)
                # write each torch result to a .pkl file
                if self.write_traj_details:
                    self._write_torch_calc_results(folder)
            # check model's atomic uncertainties
            if self.atomic_unc_threshold is not None:
                unc = self.atoms.calc.get_atomic_uncertainties()
                if self.use_softmax:
                    forces_norm = np.sqrt(np.sum(self.atoms.get_forces() ** 2, -1))
                    unc = softmax(unc / (forces_norm + self.eps_forces_norm))
                if (unc > self.atomic_unc_threshold).any():
                    if self.i_step % self.eval_step != 0:
                        # store most uncertain atoms
                        traj.write(atoms)
                        self._save_state(folder)
                        # write each torch result to a .pkl file
                        if self.write_traj_details:
                            self._write_torch_calc_results(folder)
                    raise StopIteration()
            # check model's total uncertainty
            if self.unc_threshold is not None:
                unc = self.atoms.calc.get_uncertainty()
                if unc > self.unc_threshold:
                    if self.i_step % self.eval_step != 0:
                        # store most uncertain atoms
                        traj.write(atoms)
                        self._save_state(folder)
                        # write each torch result to a .pkl file
                        if self.write_traj_details:
                            self._write_torch_calc_results(folder)
                    raise StopIteration()
                
    def _write_torch_calc_results(self, folder: Union[str, Path]):
        """Writes the predicted values (energies, forces, etc.) into a file during the simulation.

        Args:
            folder (Union[str, Path]):  Folder in which sampling progress/simulation results are stored.
        """
        torch_calc_results = self.atoms.calc.get_torch_calc_results()
        for key in torch_calc_results:
            if key not in self.traj_details_writer:
                self.traj_details_writer[key] = PickleStreamWriter(folder / (key + '.pkl'), mode='ab')
            self.traj_details_writer[key].append(torch_calc_results[key])

    def _load_state(self, folder: Union[str, Path]):
        """Loads the sampler's state to restart the atomistic simulation.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        if os.path.exists(folder / 'sampler_state.pkl'):
            state_dict = load_object(folder / 'sampler_state.pkl')
            self.i_step = state_dict['i_step']
            self.delta = state_dict['delta']
            self.random_state.set_state(state_dict['random_state'])

    def _save_state(self, folder: Union[str, Path]):
        """Stores the sampler's state to restart the atomistic simulation.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        to_save = {'i_step': self.i_step,
                   'delta': self.delta,
                   'random_state': self.random_state.get_state()}
        save_object(folder / f'sampler_state.pkl', to_save)


def softmax(x: np.ndarray) -> np.ndarray:
    """Applies softmax function to the input.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Softmax applied to the input array.
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class DynamicSampler(BaseSampler):
    """Explores atomic structures by running molecular dynamics (MD) simulations.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object from which the simulation is initiated.
        dyn (ase.md.md.MolecularDynamics): MD object which performs the time integration.
        max_step (int, optional): The maximal number of MD steps. Defaults to 1000.
        eval_step (int, optional): Defines how often sampled structures are saved. Defaults to 10.
        unc_threshold (Optional[float], optional): Total uncertainty threshold, used to terminate simulations. 
                                                   If None, only atom-based uncertainty is used. Defaults to None.
        atomic_unc_threshold (Optional[float], optional): Atom-based uncertainty threshold, used to terminate simulations. 
                                                          If None, only total uncertainty is used. Defaults to None.
        use_softmax (bool, optional): If True, softmax is applied to atom-based uncertainties. Defaults to False.
        eps_forces_norm (float, optional): Small number used during computing forces norm for numerical stability. 
                                           Used if `use_softmax=True`. Defaults to 0.2.
        to_triu (bool, optional): If True, periodic cell is transformed into an upper triangular. Defaults to False.
        write_traj_details (bool, optional): If True, properties such as atom-based uncertainties, etc., are stored. 
                                             Defaults to False.
        write_traj_properties (List[str], optional): List with properties stored during simulations. Defaults to ['energy'].
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 dyn: ase.md.md.MolecularDynamics,
                 max_step: int = 1000,
                 eval_step: int = 10,
                 unc_threshold: Optional[float] = None,
                 atomic_unc_threshold: Optional[float] = None,
                 use_softmax: bool = False,
                 eps_forces_norm: float = 0.2,  # eV/Ang.
                 wrap: bool = False,
                 to_triu: bool = False,
                 write_traj_details: bool = False,
                 write_traj_properties: List[str] = ['energy'],
                 **config: Any):
        # atoms and dyn contain all relevant information
        # current step is stored in self.dyn.nsteps
        self.atoms = atoms
        self.dyn = dyn

        # all samplers have similar state attributes
        self.i_step = 0

        # sampler specific parameters as the maximal amount of steps
        # and the step number after which the dynamics is evaluated
        self.max_step = max_step
        self.eval_step = eval_step

        # use uncertainty threshold to stop, if provided
        self.unc_threshold = unc_threshold
        self.atomic_unc_threshold = atomic_unc_threshold
        self.use_softmax = use_softmax
        assert self.unc_threshold is None or hasattr(self.atoms.calc, 'get_uncertainty')
        assert self.atomic_unc_threshold is None or hasattr(self.atoms.calc, 'get_atomic_uncertainties')
        self.eps_forces_norm = eps_forces_norm

        # wrap positions
        self.wrap = wrap
        # rotate axes such that unit cell is upper triangular
        self.to_triu = to_triu

        # change to True if details on the trajectory (forces, stress, bias, etc.) should be written into a file
        self.write_traj_details = write_traj_details
        self.write_traj_properties = write_traj_properties
        if self.write_traj_details:
            self.traj_details_writer = {}

    def run(self, folder: Union[str, Path]) -> AtomicStructures:
        # manage directories
        folder = Path(folder)
        if not os.path.exists(folder):
            # create directory if it does not exist
            os.makedirs(folder)
        else:
            # load the last state of sampler otherwise
            self._load_state(folder)
        # write sampled atoms and respective energy/temperature (for each eval step)
        traj = ase.io.Trajectory(folder / 'sampler.traj', 'a')
        logger = MDLogger(dyn=self.dyn, atoms=self.atoms, logfile=folder / 'sampler.log', header=True, peratom=False,
                          mode='a')
        try:
            while True:
                self._irun(traj, logger, folder)
        except StopIteration:
            traj.close()
            pass
        if self.write_traj_details:
            for key in self.traj_details_writer:
                self.traj_details_writer[key].close()
        # load trajectory and transform to structures
        traj = read(folder / 'sampler.traj', ':')
        structures = AtomicStructures.from_traj(traj, wrap=self.wrap)
        if self.to_triu:
            structures = structures.to_triu()
        return structures

    def _irun(self,
              traj: ase.io.Trajectory,
              logger: MDLogger,
              folder: Union[str, Path]):
        """Runs the atomistic simulation for a single step.

        Args:
            traj (ase.io.Trajectory): Trajectory in which `ase.Atoms` are stored.
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        if self.i_step >= self.max_step:
            raise StopIteration()
        else:
            # run dynamics for one more step
            # we do not save the first frame as it may be a training point
            # also it skips repetitions in case active_learning is restarted
            self.dyn.run(1)
            self.i_step += 1
            # need to call calculate to update results for new atoms (for uncertainty evaluation)
            self.atoms.calc.calculate(self.atoms, properties=self.write_traj_properties)
            # write logger and uncertainties to track progress of simulation
            if self.i_step % self.eval_step == 0:
                # write new atoms and save sampler state for re-starting
                logger()
                traj.write(self.atoms)
                self._save_state(folder)
                # write each torch result to a .pkl file
                if self.write_traj_details:
                    self._write_torch_calc_results(folder)
            # check model's atomic uncertainties
            if self.atomic_unc_threshold is not None:
                unc = self.atoms.calc.get_atomic_uncertainties()
                if self.use_softmax:
                    forces_norm = np.sqrt(np.sum(self.atoms.get_forces() ** 2, -1))
                    unc = softmax(unc / (forces_norm + self.eps_forces_norm))
                if (unc > self.atomic_unc_threshold).any():
                    if self.i_step % self.eval_step != 0:
                        # store most uncertain atoms
                        logger()
                        traj.write(self.atoms)
                        self._save_state(folder)
                        # write each torch result to a .pkl file
                        if self.write_traj_details:
                            self._write_torch_calc_results(folder)
                    raise StopIteration()
            # check model's total uncertainty
            if self.unc_threshold is not None:
                unc = self.atoms.calc.get_uncertainty()
                if unc > self.unc_threshold:
                    if self.i_step % self.eval_step != 0:
                        # store most uncertain atoms
                        logger()
                        traj.write(self.atoms)
                        self._save_state(folder)
                        # write each torch result to a .pkl file
                        if self.write_traj_details:
                            self._write_torch_calc_results(folder)
                    raise StopIteration()

    def _write_torch_calc_results(self, folder: Union[str, Path]):
        """Writes the predicted values (energies, forces, etc.) into a file during the simulation.

        Args:
            folder (Union[str, Path]):  Folder in which sampling progress/simulation results are stored.
        """
        torch_calc_results = self.atoms.calc.get_torch_calc_results()
        for key in torch_calc_results:
            if key not in self.traj_details_writer:
                self.traj_details_writer[key] = PickleStreamWriter(folder / (key + '.pkl'), mode='ab')
            self.traj_details_writer[key].append(torch_calc_results[key])

    def _load_state(self, folder: Union[str, Path]):
        """Loads the sampler's state to restart the atomistic simulation.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        if os.path.exists(folder / 'sampler_state.pkl'):
            state_dict = load_object(folder / 'sampler_state.pkl')
            self.i_step = state_dict['i_step']
            self.dyn.nsteps = self.i_step
            # update atoms to the last stored state
            self.atoms = read(folder / f'sampler.traj')
            self.atoms.set_calculator(self.dyn.atoms.calc)
            self.dyn.atoms = self.atoms

    def _save_state(self, folder: Union[str, Path]):
        """Stores the sampler's state to restart the atomistic simulation.

        Args:
            folder (Union[str, Path]): Folder in which sampling progress/simulation results are stored.
        """
        to_save = {'i_step': self.i_step}
        save_object(folder / f'sampler_state.pkl', to_save)


class LangevinNVTSampler(DynamicSampler):
    """Molecular dynamics (MD) in the canonical (NVT) ensemble using Langevin dynamics.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object used to initiate the simulation.
        calc (Calculator): The `Calculator` objects used to predict energies, forces, 
                           and uncertainties.
        temperature_K (float): Simulation temperature in K.
        timestep (float): Integration time step.
        friction (float, optional): Friction coefficient. Defaults to 1e-2.
        init_random_velocities (bool, optional): If True, velocities are initialized by 
                                                 drawing values from Maxwell-Boltzmann distribution. 
                                                 Defaults to True.
        seed (int, optional): Random seed. Defaults to 1234.
        aux_cell (Optional[np.ndarray], optional): If not None, auxiliary cell is used to suppress rotations. 
                                                   Defaults to None.
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 calc: Calculator,
                 temperature_K: float,
                 timestep: float,
                 friction: float = 1e-2,
                 init_random_velocities: bool = True,
                 seed: int = 1234,
                 aux_cell: Optional[np.ndarray] = None,
                 **config: Any):
        # set an auxiliary cell to fix rotations of a molecule
        if aux_cell is not None:
            atoms.set_cell(aux_cell)
            atoms.center()
        # set calculator
        atoms.set_calculator(calc)
        # set velocity distribution
        if init_random_velocities:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
        # define dynamics
        dyn = Langevin(atoms, timestep=timestep, temperature_K=temperature_K, friction=friction,
                       rng=np.random.default_rng(seed), communicator=None)
        super().__init__(atoms=atoms, dyn=dyn, **config)


class NoseHooverNVTSampler(DynamicSampler):
    """Molecular dynamics (MD) in the canonical (NVT) ensemble using Nose--Hoover dynamics.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object used to initiate the simulation.
        calc (Calculator): The `Calculator` objects used to predict energies, forces, 
                           and uncertainties.
        temperature_K (float): Simulation temperature in K.
        timestep (float): Integration time step.
        ttime (float): Characteristic timescale of the thermostat. Defaults to 25.0*units.fs.
        init_random_velocities (bool, optional): If True, velocities are initialized by drawing 
                                                 values from Maxwell-Boltzmann distribution. 
                                                 Defaults to True.
        seed (int, optional): Random seed. Defaults to 1234.
        aux_cell (Optional[np.ndarray], optional): If not None, auxiliary cell is used to 
                                                   suppress rotations. Defaults to None.
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 calc: Calculator,
                 temperature_K: float,
                 timestep: float,
                 ttime: float = 25.0 * units.fs,
                 init_random_velocities: bool = True,
                 seed: int = 1234,
                 aux_cell: Optional[np.ndarray] = None,
                 **config: Any):
        # set an auxiliary cell to fix rotations of a molecule
        if aux_cell is not None:
            atoms.set_cell(aux_cell)
            atoms.center()
        # set calculator
        atoms.set_calculator(calc)
        # set velocity distribution
        if init_random_velocities:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
        # define dynamics
        dyn = NPT(atoms=atoms, timestep=timestep, temperature_K=temperature_K, ttime=ttime,
                  externalstress=1.01325 * units.bar, pfactor=None)
        super().__init__(atoms=atoms, dyn=dyn, **config)


class NoseHooverNPTSampler(DynamicSampler):
    """Molecular dynamics (MD) in the isothermal--isobaric (NpT) ensemble using 
    combined Nose--Hoover and Parrinello--Rahman dynamics.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object used to initiate the simulation.
        calc (Calculator): The `Calculator` objects used to predict energies, forces, 
                           and uncertainties.
        temperature_K (float): Simulation temperature in K.
        timestep (float): Integration time step.
        ttime (float): Characteristic timescale of the thermostat. 
                       Defaults to 25.0*units.fs.
        ptime (float): Characteristic timescale of the barostat. 
                       Defaults to 75.0*units.fs.
        bulk_modulus (float, optional): The bulk modulus. Defaults to 100e4*units.bar.
        externalstress (float, optional): The external stress in eV/Ang.^3. 
                                          Defaults to 1.01325*units.bar.
        mask (Optional[np.ndarray], optional): If np.eye(3) is used, then the cell is scaled only 
                                               along x-, y-, and z- axis. Defaults to np.eye(3).
        init_random_velocities (bool, optional): If True, velocities are initialized by drawing 
                                                 values from Maxwell-Boltzmann distribution. 
                                                 Defaults to True.
        seed (int, optional): Random seed. Defaults to 1234.
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 calc: Calculator,
                 temperature_K: float,
                 timestep: float,
                 ttime: float = 25.0 * units.fs,
                 ptime: float = 75.0 * units.fs,
                 bulk_modulus: float = 100e4 * units.bar,
                 externalstress: float = 1.01325 * units.bar,
                 mask: Optional[np.ndarray] = np.eye(3),
                 init_random_velocities: bool = True,
                 seed: int = 1234,
                 **config: Any):
        # set calculator
        atoms.set_calculator(calc)
        # set velocity distribution
        if init_random_velocities:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
        # define dynamics
        dyn = NPT(atoms=atoms, timestep=timestep, temperature_K=temperature_K, ttime=ttime,
                  externalstress=externalstress, pfactor=ptime*ptime*bulk_modulus, mask=mask)
        super().__init__(atoms=atoms, dyn=dyn, **config)


class BerendsenNPTSampler(DynamicSampler):
    """Molecular dynamics (MD) in the isothermal--isobaric (NpT) ensemble using Berendsen dynamics.

    Args:
        atoms (ase.Atoms): The `ase.Atoms` object used to initiate the simulation.
        calc (Calculator): The `Calculator` objects used to predict energies, forces, 
                           and uncertainties.
        temperature_K (float): Simulation temperature in K.
        timestep (float): Integration time step.
        ttime (float): Characteristic timescale of the thermostat. Defaults to 25.0*units.fs.
        ptime (float): Characteristic timescale of the barostat. Defaults to 75.0*units.fs.
        bulk_modulus (float, optional): The bulk modulus. Defaults to 100e4*units.bar.
        externalstress (float, optional): The external stress in eV/Ang.^3. Defaults to 1.01325*units.bar.
        mask (Tuple[int]): If (1, 1, 1) is used, then the cell is scaled only along x-, y-, and z- axis. 
                           Defaults to (1, 1, 1).
        inhomogeneous_berendsen (bool): If True, the shape of the simulation cell is altered. Defaults to True.
        init_random_velocities (bool, optional): If True, velocities are initialized by drawing values 
                                                 from Maxwell-Boltzmann distribution. Defaults to True.
        seed (int, optional): Random seed. Defaults to 1234.
    """
    def __init__(self,
                 atoms: ase.Atoms,
                 calc: Calculator,
                 temperature_K: float,
                 timestep: float,
                 ttime: float = 25.0 * units.fs,
                 ptime: float = 75.0 * units.fs,
                 bulk_modulus: float = 100e4 * units.bar,
                 externalstress: float = 1.01325 * units.bar,
                 mask: Tuple[int] = (1, 1, 1),
                 inhomogeneous_berendsen: bool = True,
                 init_random_velocities: bool = True,
                 seed: int = 1234,
                 **config: Any):
        # set calculator
        atoms.set_calculator(calc)
        # set velocity distribution
        if init_random_velocities:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
        # define dynamics
        if inhomogeneous_berendsen:
            dyn = Inhomogeneous_NPTBerendsen(atoms, timestep=timestep, temperature_K=temperature_K,
                                             taut=ttime, pressure_au=externalstress, taup=ptime,
                                             compressibility_au=1.0 / bulk_modulus, mask=mask)
        else:
            dyn = NPTBerendsen(atoms, timestep=timestep, temperature_K=temperature_K, taut=ttime,
                               pressure_au=externalstress, taup=ptime, compressibility_au=1.0 / bulk_modulus)
        dyn.communicator = None
        super().__init__(atoms=atoms, dyn=dyn, **config)


def get_sampler(sampling_method: str) -> Callable:
    """Provides the simulation/sampling method by name.

    Args:
        sampling_method (str): String with the sampling method name. 
                               Possible names: 'random', 'npt', 'berendsen-npt', 
                               'nvt', 'nh-nvt', and 'adversarial'.


    Returns:
        Callable: Sampling method.
    """
    if sampling_method == 'random':
        return UniformRandomSampler
    elif sampling_method == 'npt':
        return NoseHooverNPTSampler
    elif sampling_method == 'berendsen-npt':
        return BerendsenNPTSampler
    elif sampling_method == 'nvt':
        return LangevinNVTSampler
    elif sampling_method == 'nh-nvt':
        return NoseHooverNVTSampler
    elif sampling_method == 'adversarial':
        return AdversarialSampler
    else:
        raise NotImplementedError(f'{sampling_method=} is not implemented yet.')
