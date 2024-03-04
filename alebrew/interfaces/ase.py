"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     ase.py 
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
from pathlib import Path
from typing import List, Optional, Union, Any, Dict

import ase
import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System

from alebrew.data.data import AtomicTypeConverter, AtomicStructure, AtomicData
from alebrew.model.forward import ForwardAtomisticNetwork, find_last_ckpt
from alebrew.model.calculators import TorchCalculator, get_torch_calculator
from alebrew.utils.torch_geometric import DataLoader
from alebrew.utils.misc import get_default_device


class ASEWrapper(Calculator):
    """Wraps a `TorchCalculator` object into an ASE calculator. It is used to perform atomistic simulations within ASE.

    Args:
        calc (TorchCalculator): The `TorchCalculator` object which allows for energy, forces, etc. calculations.
        r_cutoff (float): Cutoff radius for computing the neighbor list.
        atomic_types (Optional[List[str]], optional): List of supported atomic numbers. Defaults to None.
        device (Optional[str], optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to None.
        skin (float, optional): Skin distance used to update the neighbor list. Defaults to 0.0.
        wrap (bool, optional): Wrap positions back to the periodic cell. Defaults to False.
        neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.
        energy_units_to_eV (float, optional): Energy conversion factor. Defaults to 1.0.
        length_units_to_A (float, optional): Length conversion factor. Defaults to 1.0.
        unc_method (Optional[str], optional): Method for evaluating uncertainty. Defaults to None.
        bias_method (Optional[str], optional): Method for computing bias potential. Defaults to None.
        compute_atomic_unc (bool, optional): If True, atom-based features are stored for computing atom-based 
                                             uncertainties. Defaults to False.
        compute_forces_unc (bool, optional): If True, force uncertainty is computed for an ensemble of models. 
                                             Defaults to False.
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self,
                 calc: TorchCalculator,
                 r_cutoff: float,
                 atomic_types: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 skin: float = 0.0,
                 wrap: bool = False,
                 neighbors: str = 'matscipy',
                 energy_units_to_eV: float = 1.0,
                 length_units_to_A: float = 1.0,
                 unc_method: Optional[str] = None,
                 bias_method: Optional[str] = None,
                 compute_atomic_unc: bool = False,
                 compute_forces_unc: bool = False,
                 **kwargs: Any):
        super().__init__()
        self.results = {}
        self.calc = calc
        # define device and move calc to it
        self.device = device or get_default_device()
        self.calc.to(self.device)
        # cutoff radius to compute neighbors
        self.r_cutoff = r_cutoff
        # skin to restore neighbors
        self.skin = skin
        # wrap atoms to the box
        self.wrap = wrap
        # convert atomic numbers to atomic types for internal use
        self.atomic_type_converter = AtomicTypeConverter.from_type_list(atomic_types)
        # method for calculating neighbor list
        self.neighbors = neighbors
        # re-scale energy and forces
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        # additional arguments specific to calculator set-up
        self.compute_features = unc_method is not None and unc_method in ['posterior', 'distance']
        self.create_graph = bias_method is not None
        self.compute_atomic_unc = compute_atomic_unc
        self.compute_forces_unc = compute_forces_unc
        # init last structure to check whether new neighbors have to be computed
        self._last_structure = None

        # storing calc results for external use
        self._torch_calc_results = {}

    def calculate(self,
                  atoms: Optional[ase.Atoms] = None,
                  properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes):
        """Calculates the total energy, atomic force, etc. for `ase.Atoms`.

        Args:
            atoms (Optional[ase.Atoms], optional): The `ase.Atoms` object. Defaults to None.
            properties (List[str], optional): List of properties to be computed. Defaults to ['energy'].
            system_changes (List[str], optional): Defaults to all_changes.
        """
        if self.calculation_required(atoms, properties):
            super().calculate(atoms)
            # prepare data
            structure = AtomicStructure.from_atoms(atoms, wrap=self.wrap, neighbors=self.neighbors)
            # check whether provided atom types are supported
            structure = structure.to_type_names(self.atomic_type_converter, check=True)
            # check if neighbors can be restored from the last structure
            if not structure.restore_neighbors_from_last(structure=self._last_structure,
                                                         r_cutoff=self.r_cutoff, skin=self.skin):
                self._last_structure = structure
            # prepare data
            dl = DataLoader([AtomicData(structure, r_cutoff=self.r_cutoff, skin=self.skin)], batch_size=1, shuffle=False,
                            drop_last=False)
            batch = next(iter(dl)).to(self.device)
            # predict
            out = self.calc(batch, forces='forces' in properties or self.compute_forces_unc, stress='stress' in properties,
                            features=self.compute_features, create_graph=self.create_graph or 'stress' in properties,
                            store_atomic_features=self.compute_features and self.compute_atomic_unc)
            # store calculator results for external use
            for key, val in out.items():
                if key != 'individual_results' and key != 'atomic_features':
                    if isinstance(val, float):
                        self._torch_calc_results[key] = val
                    else:
                        self._torch_calc_results[key] = val.detach().cpu()
            # store results for ase
            self.results = {'energy': out['energy'].detach().cpu().item() * self.energy_units_to_eV}
            if 'forces' in out:
                forces = out['forces'].detach().cpu().numpy() * (self.energy_units_to_eV / self.length_units_to_A)
                self.results['forces'] = forces
            if 'stress' in out:
                stress = out['stress'].detach().cpu().numpy()
                stress = stress.reshape(3, 3) * (self.energy_units_to_eV / self.length_units_to_A ** 3)
                stress_voigt = full_3x3_to_voigt_6_stress(stress)
                self.results['stress'] = stress_voigt
            if 'uncertainty' in out:
                self.results['uncertainty'] = out['uncertainty'].detach().cpu().item() * self.energy_units_to_eV
            if 'atomic_uncertainties' in out:
                atomic_uncertainties = out['atomic_uncertainties'].detach().cpu().numpy() * self.energy_units_to_eV
                self.results['atomic_uncertainties'] = atomic_uncertainties
    
    def get_uncertainty(self) -> float:
        """Provides the total uncertainty.

        Returns:
            float: Total uncertainty.
        """
        return self.results.get('uncertainty', None)

    def get_atomic_uncertainties(self) -> np.ndarray:
        """Provides the atom-based uncertainties.

        Returns:
            np.ndarray: Atom-based uncertainties.
        """
        return self.results.get('atomic_uncertainties', None)

    def get_torch_calc_results(self) -> Dict[str, Any]:
        """Provides results dictionary for the `TorchCalculator`. 
        Used during an atomistic simulation to store intermediate results.

        Returns:
            Dict[str, Any]: Torch calculator results.
        """
        return self._torch_calc_results

    @staticmethod
    def from_folder_list(folder_list: List[Union[Path, str]],
                         unc_method: Optional[str] = None,
                         bias_method: Optional[str] = None,
                         **kwargs: Any) -> Calculator:
        """Provides a wrapped ASE calculator from the list of models.

        Args:
            folder_list (List[Union[Path, str]]): List of folders containing trained models.
            unc_method (Optional[str], optional): Method for evaluating uncertainty. Defaults to None.
            bias_method (Optional[str], optional): Method for computing bias potential. Defaults to None.

        Returns:
            ase.Calculator: Wrapped ASE calculator.
        """
        if not isinstance(folder_list, list):
            raise RuntimeError(f'{folder_list=} has been provided, while a list is expected.')
        # load model from folder
        models = [ForwardAtomisticNetwork.from_folder(find_last_ckpt(f)) for f in folder_list]
        # check whether all model have the same cutoff radius
        if not all(model.config['r_cutoff'] == models[0].config['r_cutoff'] for model in models):
            raise RuntimeError(f'Cutoff radii for all models in the list must be the same. '
                               f'Got {[model.config["r_cutoff"] for model in models]}!')
        # check whether all model have same species
        if not all(model.config['atomic_types'] == models[0].config['atomic_types'] for model in models):
            raise RuntimeError(f'Atomic species for all models in the list must be the same. '
                               f'Got {[model.config["atomic_types"] for model in models]}!')
        # build torch calculator
        calc = get_torch_calculator(models, unc_method=unc_method, bias_method=bias_method, **kwargs)
        return ASEWrapper(calc, models[0].config['r_cutoff'], models[0].config['atomic_types'],
                          unc_method=unc_method, bias_method=bias_method, **kwargs)


class WeightedSumCalculator(Calculator):
    """A linear combination of two wrapped ASE calculators.

    Args:
        calcs (List[Calculator]): List of ASE calculators.
        weights (List[float]): List of weights coupling these calculators.
    """
    def __init__(self,
                 calcs: List[Calculator],
                 weights: List[float],
                 **config: Any):
        super().__init__()

        # some checks
        if not isinstance(calcs, list):
            raise ValueError(f'{calcs=} has been provided, while a list '
                             f'of calculators is expected!')

        if not len(calcs) == len(weights):
            raise ValueError(f'Provided {len(calcs)=} and {len(weights)=}!')

        for calc in calcs:
            if not isinstance(calc, Calculator):
                raise ValueError(f'{calc=} is not a {Calculator}!')

        self.results = {}

        # only properties shared by interfaces can be computed by the sum calculator
        setlist = [set(calc.implemented_properties) for calc in calcs]
        self.implemented_properties = list(set.intersection(*setlist))

        self.calcs = calcs
        self.weights = weights

    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        """Calculates the total energy, atomic force, etc. for `ase.Atoms`.

        Args:
            atoms (Optional[ase.Atoms], optional): The `ase.Atoms` object. Defaults to None.
            properties (List[str], optional): List of properties to be computed. Defaults to ['energy'].
            system_changes (List[str], optional): Defaults to all_changes.
        """
        super().calculate(atoms)

        for weight, calc in zip(self.weights, self.calcs):
            if calc.calculation_required(atoms, properties):
                calc.calculate(atoms, properties, system_changes)
            for prop in properties:
                if prop in self.results:
                    self.results[prop] += weight * calc.results[prop]
                else:
                    self.results[prop] = weight * calc.results[prop]

    def reset(self):
        """Reset results of all calculators.
        """
        super().reset()

        for calc in self.calcs:
            calc.reset()


class TorchMDCalculator(Calculator):
    """Wraps a torch-md calculator into an ASE calculator. It is used to perform atomistic simulations.

    Args:
        input_file (Union[str, Path]): Initial atomic positions, numbers.
        parm_file (Union[str, Path]): System topology.
        r_cutoff (float, optional): Cutoff radius. Defaults to 9.0.
        precision (optional): Data type. Defaults to torch.float.
        device (Optional[str], optional): Available device (e.g., 'cuda:0' or 'cpu'). Defaults to None.
        energy_units_to_eV (float, optional): Energy conversion factor. Defaults to units.kcal/units.mol.
        length_units_to_A (float, optional): Length conversion factor. Defaults to 1.0.
    """
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self,
                 input_file: Union[str, Path],
                 parm_file: Union[str, Path],
                 r_cutoff: float = 9.0,
                 precision=torch.float,
                 device: Optional[str] = None,
                 energy_units_to_eV: float = units.kcal / units.mol,
                 length_units_to_A: float = 1.0,
                 **config: Any):
        super().__init__()

        self.mol = Molecule(parm_file)  # Reading the system topology
        self.mol.read(input_file)   # Reading the initial simulation coordinates
        self.r_cutoff = r_cutoff

        self.precision = precision
        self.device = get_default_device() if device is None else device

        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

        self.ff = ForceField.create(self.mol, parm_file)
        self.params = Parameters(self.ff, self.mol, precision=self.precision, device=self.device)
        self.forces = Forces(self.params, cutoff=self.r_cutoff, rfa=True, switch_dist=7.5,
                             terms=["bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj"])

    def calculate(self,
                  atoms: Optional[ase.Atoms] = None,
                  properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes):
        """Calculates the total energy, atomic force, etc. for `ase.Atoms`.

        Args:
            atoms (Optional[ase.Atoms], optional): The `ase.Atoms` object. Defaults to None.
            properties (List[str], optional): List of properties to be computed. Defaults to ['energy'].
            system_changes (List[str], optional): Defaults to all_changes.
        """
        # remove the auxiliary cell
        if not atoms.get_pbc().any():
            atoms.set_cell(None)
        super().calculate(atoms)
        # prepare data
        system = System(len(atoms), nreplicas=1, precision=self.precision, device=self.device)
        system.set_positions(atoms.get_positions()[..., None])
        if np.any(self.mol.box != np.asarray(atoms.get_cell())):
            raise RuntimeError(f'Simulation box should not change: {self.mol.box=}, {np.asarray(atoms.get_cell())=}.')
        system.set_box(self.mol.box)
        # predict
        energy = self.forces.compute(system.pos, system.box, system.forces, returnDetails=False, explicit_forces=True)[0]
        # store results
        self.results = {'energy': energy * self.energy_units_to_eV,
                        'forces': system.forces[0].detach().cpu().numpy() * (self.energy_units_to_eV / self.length_units_to_A)}
