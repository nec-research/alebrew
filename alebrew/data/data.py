"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     data.py 
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

import h5py

import numpy as np
from typing import *
import ase
import torch
import ase.data
from ase.cell import Cell
from ase.io import write, read

from alebrew.data.neighbors import get_matscipy_neighbors, get_ase_neighbors, get_primitive_neighbors, get_torch_neighbors
from alebrew.utils.torch_geometric import Data


class AtomicTypeConverter:
    """Converts atomic numbers to internal types and vice versa.

    Args:
        to_atomic_numbers (np.ndarray): Array for mapping from internal types to atomic numbers.
        from_atomic_numbers (np.ndarray): Array for mapping from atomic numbers to internal types.
    """
    def __init__(self,
                 to_atomic_numbers: np.ndarray,
                 from_atomic_numbers: np.ndarray):
        self._to_atomic_numbers = to_atomic_numbers
        self._from_atomic_numbers = from_atomic_numbers

    def to_type_names(self,
                      atomic_numbers: np.ndarray,
                      check: bool = True) -> np.ndarray:
        """Converts an array with atomic numbers to an array with internal types.

        Args:
            atomic_numbers (np.ndarray): Array with atomic numbers.
            check (bool, optional): If True, check if atomic numbers are supported.

        Returns:
            np.ndarray: Array with internal types.
        """
        result = self._from_atomic_numbers[atomic_numbers]
        if check:
            assert np.all(result >= 0)
        return result

    def to_atomic_numbers(self, species: np.ndarray) -> np.ndarray:
        """Converts an array with internal types to an array with atomic numbers.

        Args:
            species (np.ndarray): Array with internal types.

        Returns:
            np.ndarray: Array with atomic numbers.
        """
        return self._to_atomic_numbers[species]

    def get_n_type_names(self) -> int:
        """

        Returns:
            int: The total number of species/elements.
        """
        return len(self._to_atomic_numbers)

    @staticmethod
    def from_type_list(atomic_types: Optional[List[Union[str, int]]] = None) -> 'AtomicTypeConverter':
        """Generates an object for converting atomic numbers to internal types and vice versa from the list of elements.

        Args:
            atomic_types (Optional[List[Union[str, int]]], optional): List of supported atomic numbers/elements. 
                                                                      Defaults to None.

        Returns:
            AtomicTypeConverter: Object for converting atomic numbers to internal types and vice versa.
        """
        if atomic_types is None:
            to_atomic_numbers = np.asarray(list(range(119)))
            from_atomic_numbers = to_atomic_numbers
        else:
            to_atomic_numbers = np.asarray(
                [ase.data.atomic_numbers[atomic_type] if isinstance(atomic_type, str) else int(atomic_type) for
                 atomic_type in atomic_types])
            max_entry = np.max(to_atomic_numbers)
            from_atomic_numbers = -np.ones(max_entry + 1, dtype=int)
            from_atomic_numbers[to_atomic_numbers] = np.arange(len(to_atomic_numbers))

        return AtomicTypeConverter(to_atomic_numbers, from_atomic_numbers)


class AtomicStructure:
    """Defines atomic structure using atomic numbers (species), atomic positions, and other features.

        Args:
            species (np.ndarray): Atomic numbers or atom types.
            positions (np.ndarray): Atomic positions.
            cell (Optional[np.ndarray], optional): Unit cell. Defaults to None.
            pbc (Optional[bool], optional): Periodic boundaries. Defaults to None.
            energy (Optional[float], optional): Total energy. Defaults to None.
            forces (Optional[np.ndarray], optional): Atomic forces. Defaults to None.
            stress (Optional[np.ndarray], optional): Stress tensor. Defaults to None.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.
        """
    def __init__(self,
                 species: np.ndarray,
                 positions: np.ndarray,
                 cell: Optional[np.ndarray] = None,
                 pbc: Optional[bool] = None,
                 energy: Optional[float] = None,
                 forces: Optional[np.ndarray] = None,
                 stress: Optional[np.ndarray] = None,
                 neighbors: str = 'matscipy'):
        # attributes should not be changed from outside,
        # because this might invalidate the computed edge_index (neighbor list) and shifts
        self.species = species
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self.energy = energy  # EnergyUnit
        self.forces = forces  # EnergyUnit/DistanceUnit
        self.stress = stress  # EnergyUnit/DistanceUnit**3
        # compute virials for training
        volume = np.abs(np.linalg.det(cell)) if cell is not None else None  # DistanceUnit**3
        self.virials = -1 * stress * volume if stress is not None and volume is not None else None  # EnergyUnit
        self.n_atoms = species.shape[0]

        if neighbors == 'matscipy':
            self.neighbors_fn = get_matscipy_neighbors
        elif neighbors == 'ase':
            self.neighbors_fn = get_ase_neighbors
        elif neighbors == 'primitive':
            self.neighbors_fn = get_primitive_neighbors
        elif neighbors == 'torch':
            self.neighbors_fn = get_torch_neighbors
        else:
            raise ValueError(f'{neighbors=} is not implemented yet! Use one of: "matscipy", "ase", "primitive",'
                             f' "torch".')

        self._r_cutoff = None
        self._skin = None
        self._edge_index = None
        self._shifts = None

        # check shapes
        assert tuple(positions.shape) == (self.n_atoms, 3)
        assert len(species.shape) == 1
        assert cell is None or tuple(cell.shape) == (3, 3)
        assert forces is None or tuple(forces.shape) == (self.n_atoms, 3)
        assert energy is None or isinstance(energy, float)
        assert stress is None or tuple(stress.shape) == (3, 3)

    def _compute_neighbors(self,
                           r_cutoff: float,
                           skin: float = 0.0):
        """Computes neighbor list for the atomic structure.

        Args:
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            skin (float, optional): Skin distance for updating the neighbor list. Defaults to 0.0.
        """
        if (self._r_cutoff is not None and self._r_cutoff == r_cutoff) and \
                (self._skin is not None and self._skin == skin):
            return  # neighbors have already been computed for the same cutoff and skin radius
        self._r_cutoff = r_cutoff
        self._skin = skin

        self._edge_index, self._shifts = self.neighbors_fn(r_cutoff=r_cutoff, skin=skin, **vars(self))

        assert self._edge_index.shape[0] == 2 and len(self._edge_index.shape) == 2
        assert self._shifts.shape[1] == 3 and len(self._shifts.shape) == 2

    def get_edge_index(self,
                       r_cutoff: float,
                       skin: float = 0.0) -> np.ndarray:
        """Computes edge indices.

        Args:
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            skin (float, optional): Skin distance for updating the neighbor list. Defaults to 0.0.

        Returns:
            np.ndarray: Edge indices (neighbor list) containing the central (out[0, :]) and neighboring (out[1, :]) atoms.
        """
        self._compute_neighbors(r_cutoff, skin)
        return self._edge_index

    def get_shifts(self,
                   r_cutoff: float,
                   skin: float = 0.0) -> np.ndarray:
        """Computes shift vectors.

        Args:
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            skin (float, optional): Skin distance for updating the neighbor list. Defaults to 0.0.

        Returns:
            np.ndarray: Shift vector, i.e., the number of cell boundaries crossed by the bond between atoms.
        """
        self._compute_neighbors(r_cutoff, skin)
        return self._shifts

    def to_type_names(self,
                      converter: AtomicTypeConverter,
                      check: bool = False) -> 'AtomicStructure':
        """Convert atomic numbers to internal types in the atomic structure.

        Args:
            converter (AtomicTypeConverter): Object for converting atomic numbers to internal types and vice versa.
            check (bool, optional): If True, check if atomic numbers are supported by `AtomicTypeConverter`. Defaults to False.

        Returns:
            AtomicStructure: Atomic structure with internal types instead of atomic numbers.
        """
        return AtomicStructure(species=converter.to_type_names(self.species, check=check),
                               positions=self.positions,
                               cell=self.cell,
                               pbc=self.pbc,
                               forces=self.forces,
                               energy=self.energy,
                               stress=self.stress)

    def to_atomic_numbers(self, converter: AtomicTypeConverter) -> 'AtomicStructure':
        """Convert internal types to atomic numbers in the atomic structure.

        Args:
            converter (AtomicTypeConverter): Object for converting atomic numbers to internal types and vice versa.

        Returns:
            AtomicStructure: Atomic structure with atomic numbers instead of internal types.
        """
        return AtomicStructure(species=converter.to_atomic_numbers(self.species),
                               positions=self.positions,
                               cell=self.cell,
                               pbc=self.pbc,
                               forces=self.forces,
                               energy=self.energy,
                               stress=self.stress)

    def to_triu(self) -> 'AtomicStructure':
        """Converts the unit cell of the atomic structure to an upper triangular form.

        Returns:
            AtomicStructure: Atomic structure with an upper triangular unit cell.
        """
        if self._is_triu():
            return self
        if self.cell is None:
            raise RuntimeError(f'No cell is provided to transform to an upper triangular form: {self.cell=}.')
        if np.any(self.forces is not None) or np.any(self.stress is not None):
            raise RuntimeError(f'Make sure to transform cell before computing forces or stress. '
                               f'Otherwise corresponding transformations have to be applied to them, too.'
                               f'Provided: {self.forces=} and {self.stress=}.')
        # first, rotate axes such that unit cell is lower triangular
        # then, swap x and z axes, then swap x and z coordinates to get an upper triangular cell
        # alternatively, one may apply Q = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) matrix to the cell, i.e.,
        # cell = (Q @ cell) @ Q
        cell, Q = Cell(self.cell).standard_form()
        cell[[0, 2], :] = cell[[2, 0], :]
        cell[:, [0, 2]] = cell[:, [2, 0]]
        # for positions one needs to apply transformation matrix, and swap x and z coordinates
        positions = self.positions @ Q.T
        positions[:, [0, 2]] = positions[:, [2, 0]]
        return AtomicStructure(species=self.species,
                               positions=positions,
                               cell=cell,
                               pbc=self.pbc,
                               forces=self.forces,
                               energy=self.energy,
                               stress=self.stress)

    def _is_triu(self) -> bool:
        """Checks if the unit cell of the atomic structure is in the upper triangular form.

        Returns:
            bool: True, if the unit cell of the atomic structure is in the upper triangular form.
        """
        return self.cell[1, 0] == self.cell[2, 0] == self.cell[2, 1] == 0.0

    def to_atoms(self) -> ase.Atoms:
        """Converts the atomic structure to `ase.Atoms`.

        Returns:
            ase.Atoms: The `ase.Atoms` object.
        """
        atoms = ase.Atoms(positions=self.positions, numbers=self.species, cell=self.cell, pbc=self.pbc)
        if self.forces is not None:
            atoms.arrays['forces'] = self.forces
        if self.energy is not None:
            atoms.info['energy'] = self.energy
        if self.stress is not None:
            atoms.info['stress'] = self.stress
        return atoms

    @staticmethod
    def from_atoms(atoms: ase.Atoms,
                   wrap: bool = False,
                   neighbors: str = 'matscipy',
                   **kwargs: Any) -> 'AtomicStructure':
        """Converts `ase.Atoms` to `AtomicStructure`.

        Args:
            atoms (ase.Atoms): The `ase.Atoms` object.
            wrap (bool, optional): If True, wrap atomic positions back to the unit cell. Defaults to False.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructure: The `AtomicStructure` object which allows for convenient calculation of the 
                             neighbor list and transformations between atomic numbers and internal types.
        """
        return AtomicStructure(species=atoms.get_atomic_numbers(),
                               positions=atoms.get_positions(wrap=wrap),
                               cell=np.asarray(atoms.get_cell()),
                               pbc=atoms.get_pbc(),
                               forces=atoms.arrays.get('forces', None),
                               energy=atoms.info.get('energy', None),
                               stress=atoms.info.get('stress', None),
                               neighbors=neighbors)

    def restore_neighbors_from_last(self,
                                    r_cutoff: float,
                                    structure: Optional['AtomicStructure'] = None,
                                    skin: float = 0.) -> bool:
        """Restores the neighbor list from the last atomic structure. Used together with the skin distance 
        to identify when neighbors have to be re-computed.

        Args:
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            structure (Optional[AtomicStructure], optional): The `AtomicStructure` object from which neighbors 
                                                             are re-used if possible. Defaults to None.
            skin (float, optional): Skin distance for updating the neighbor list. Defaults to 0.0.

        Returns:
            bool: True, if neighbors of the last atomic structure can be re-used.
        """
        if structure is None or skin <= 0.:
            # no reference structure has been provided or skin <= 0. has been provided
            return False

        if r_cutoff != structure._r_cutoff or skin != structure._skin or np.any(self.pbc != structure.pbc) \
                or np.any(self.cell != structure.cell):
            # cutoff radius, skin radius, periodic boundaries, or periodic cell have been changed
            return False

        max_dist_sq = ((self.positions - structure.positions) ** 2).sum(-1).max()
        if max_dist_sq > (skin / 2.0) ** 2:
            # atoms moved out of the skin (r_cutoff += skin)
            return False

        # structure has not been changed considerably such that we may restore neighbors from last structure
        self._r_cutoff = structure._r_cutoff
        self._skin = structure._skin
        self._edge_index = structure._edge_index
        self._shifts = structure._shifts

        return True


class AtomicStructures:
    """Atomic structures to deal with a list of `AtomicStructure` objects (atomic structures).

    Args:
        structures (List[AtomicStructure]): List of `AtomicStructure` objects.
    """
    def __init__(self, structures: List[AtomicStructure]):
        self.structures = structures

        self._n_species = None
        self._EperA_mean = None
        self._EperA_stdev = None
        self._EperA_regression = None

    def __len__(self) -> int:
        """Provides the total number of atomic structures in the list.

        Returns:
            int: Total number of atomic structures.
        """
        return len(self.structures)

    # def __iter__(self):
    #     return iter(self.structures)

    def get_EperA_mean(self, n_species: int) -> np.ndarray:
        """Computes the mean energy per atom.

        Args:
            n_species (int): Total number of atom species/types.

        Returns:
            np.ndarray: Atomic energy shift parameter computed as the mean energy per atom.
        """
        self._compute_energy_stats(n_species)
        return self._EperA_mean

    def get_EperA_stdev(self, n_species: int) -> np.ndarray:
        """Computes the root-mean-square error (RMSE) per atom of `EperA_mean`.

        Args:
            n_species (int): Total number of atom species.

        Returns:
            np.ndarray: Atomic energy scale parameter computed as the RMSE per atom of `EperA_mean`.
        """
        self._compute_energy_stats(n_species)
        return self._EperA_stdev

    def get_EperA_regression(self, n_species: int) -> np.ndarray:
        """Computes the energy per atom by solving a linear regression problem.

        Args:
            n_species (int): Total number of atom species.

        Returns:
            np.ndarray: A species dependent array with atomic energy shift parameters computed by solving a linear regression problem.
        """
        self._compute_energy_stats(n_species)
        return self._EperA_regression

    def save_npz(self, file_path: Union[Path, str]):
        """Saves atomic structures to an `.npz` file.

        Args:
            file_path (Union[Path, str]): Path to the `.npz` file.
        """
        if not str(file_path)[-4:] == '.npz':
            raise ValueError(f'{file_path} has been provided, while an .npz file is expected.')

        atomic_dict = {}
        for structure in self.structures:
            for key, val in structure.__dict__.items():
                if not key.startswith('_'):
                    if key in atomic_dict:
                        atomic_dict[key].append(val)
                    else:
                        atomic_dict[key] = [val]

        # zero padding atomic properties such as positions, atomic numbers, and atomic forces
        for key, vals in atomic_dict.items():
            if key in ['positions', 'forces']:
                pad = len(max(vals, key=len))
                padded_val = [np.pad(val, ((0, (pad - len(val))), (0, 0))) for val in vals]
                atomic_dict[key] = padded_val
            if key in ['species']:
                pad = len(max(vals, key=len))
                padded_val = [np.pad(val, (0, (pad - len(val)))) for val in vals]
                atomic_dict[key] = padded_val

        np.savez(file_path, **atomic_dict)

    def save_extxyz(self, file_path: Union[Path, str]):
        """Saves atomic structures to an `.extxyz` file.

        Args:
            file_path (Union[Path, str]): Path to the `.extxyz` file.
        """
        if not str(file_path)[-7:] == '.extxyz':
            raise ValueError(f'{file_path} has been provided, while an .extxyz file is expected.')

        for structure in self.structures:
            atoms = ase.Atoms(numbers=structure.species, positions=structure.positions,
                              cell=structure.cell, pbc=structure.pbc)
            if structure.energy is not None:
                atoms.info.update({'energy': structure.energy})
            if structure.forces is not None:
                atoms.arrays.update({'forces': structure.forces})
            if structure.stress is not None:
                atoms.info.update({'stress': structure.stress})
            write(file_path, atoms, format='extxyz', append=True)

    @staticmethod
    def from_npz(file_path: Union[Path, str],
                 key_mapping: Optional[dict] = None,
                 neighbors: str = 'matscipy',
                 **kwargs: Any) -> 'AtomicStructures':
        """Loads atomic structures from an `.npz` file.

        Args:
            file_path (Union[Path, str]): Path to the `.npz` file.
            key_mapping (Optional[dict], optional): Dictionary mapping custom to default keys ('positions', 'cell',
                                                    'numbers', 'energy', forces', 'stress', 'n_atoms'). Defaults to None.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if not str(file_path)[-4:] == '.npz':
            raise ValueError(f'{file_path} has been provided, while an .npz file is expected.')

        if key_mapping is None:
            # default key mapping
            key_mapping = {'R': 'positions',
                           'C': 'cell',
                           'Z': 'numbers',
                           'E': 'energy',
                           'F': 'forces',
                           'W': 'stress',
                           'N': 'n_atoms'}

        atomic_dict = {}
        with np.load(file_path) as data:
            for key, value in key_mapping.items():
                atomic_dict[value] = data.get(key, None)

        structures = []
        for i_mol in range(len(atomic_dict['energy'])):
            n_atoms = atomic_dict['n_atoms'][i_mol]
            positions = atomic_dict['positions'][i_mol, :n_atoms]
            if atomic_dict['cell'] is not None:
                cell = atomic_dict['cell'][i_mol]
                pbc = True  # todo: what if pbc along a subset of axes?
            else:
                cell = np.asarray([0., 0., 0.])
                pbc = False
            numbers = atomic_dict['numbers'][i_mol, :n_atoms]
            energy = float(atomic_dict['energy'][i_mol])
            forces = atomic_dict['forces'][i_mol, :n_atoms]

            atoms = ase.Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
            atoms.arrays.update({'forces': forces})
            atoms.info.update({'energy': energy})
            if 'stress' in atomic_dict:
                stress = atomic_dict['stress'][i_mol]
                atoms.info.update({'stress': stress})

            structures.append(AtomicStructure.from_atoms(atoms, neighbors=neighbors, **kwargs))

        return AtomicStructures(structures)

    @staticmethod
    def from_extxyz(file_path: Union[Path, str],
                    range_str: str = ':',
                    neighbors: str = 'matscipy',
                    **kwargs: Any) -> 'AtomicStructures':
        """Loads atomic structures from an `.extxyz` file.

        Args:
            file_path (Union[Path, str]): Path to the `.extxyz` file.
            range_str (str): Range of the atomic structures, i.e. ':10' to chose the first ten atomic structures.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if not str(file_path)[-7:] == '.extxyz':
            raise ValueError(f'{file_path} has been provided, while an .extxyz file is expected.')

        traj = read(file_path, format='extxyz', index=range_str)

        structures = []
        for atoms in traj:
            structures.append(AtomicStructure.from_atoms(atoms, neighbors=neighbors, **kwargs))

        return AtomicStructures(structures)

    @staticmethod
    def from_traj(traj: List[ase.Atoms],
                  neighbors: str = 'matscipy',
                  **kwargs: Any) -> 'AtomicStructures':
        """Loads atomic structures from a list of `ase.Atoms`.

        Args:
            traj (List[ase.Atoms]): List of `ase.Atoms`.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        return AtomicStructures([AtomicStructure.from_atoms(a, neighbors=neighbors, **kwargs) for a in traj])
    
    @staticmethod
    def from_hdf5(file_path: Union[Path, str],
                  key_mapping: Optional[dict] = None,
                  energy_unit: float = ase.units.Hartree,
                  length_unit: float = ase.units.Bohr,
                  neighbors: str = 'matscipy',
                  **kwargs: Any) -> 'AtomicStructures':
        """Loads atomic structures from an `.hdf5` file.

        Args:
            file_path (Union[Path, str]): Path to the `.hdf5` file.
            key_mapping (Optional[dict], optional): Dictionary mapping custom to default keys ('positions', 'cell', 
                                                    'numbers', 'energy', forces', 'stress', 'n_atoms'). Defaults to None.
            neighbors (str, optional): Method for computing the neighbor list. Defaults to 'matscipy'.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if not str(file_path)[-5:] == '.hdf5':
            raise ValueError(f'{file_path} has been provided, while an .hdf5 file is expected.')

        if key_mapping is None:
            # default key mapping
            key_mapping = {'positions': 'conformations',
                           'numbers': 'atomic_numbers',
                           'energy': 'formation_energy',
                           'forces': 'dft_total_gradient'}

        structures = []
        with h5py.File(file_path, 'r') as f:
            for grp in f.values():
                numbers = grp[key_mapping['numbers']][()]
                positions = grp[key_mapping['positions']][()] * length_unit
                energy = grp[key_mapping['energy']][()] * energy_unit
                forces = grp[key_mapping['forces']][()] * energy_unit / length_unit
                
                for i in range(len(positions)):
                    atoms = ase.Atoms(numbers=numbers, positions=positions[i])
                    atoms.info.update({'energy': energy[i]})
                    atoms.arrays.update({'forces': forces[i]})
                    structures.append(AtomicStructure.from_atoms(atoms, neighbors=neighbors, **kwargs))
                    
        return AtomicStructures(structures)

    @staticmethod
    def from_file(file_path: Union[Path, str],
                  **config: Any) -> 'AtomicStructures':
        """Loads atomic structures from a file.

        Args:
            file_path (Union[Path, str]): Path to the (`.npz`, `.extxyz`, or `.hdf5`) file.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if str(file_path)[-4:] == '.npz':
            return AtomicStructures.from_npz(file_path, **config)
        elif str(file_path)[-7:] == '.extxyz':
            return AtomicStructures.from_extxyz(file_path, **config)
        elif str(file_path)[-5:] == '.hdf5':
            return AtomicStructures.from_hdf5(file_path, **config)
        else:
            raise ValueError(f'Provided wrong data format for {file_path=}. Use ".extxyz", ".npz", or ".hdf5" instead!')

    def to_type_names(self,
                      converter: AtomicTypeConverter,
                      check: bool = False) -> 'AtomicStructures':
        """Converts atomic numbers to internal types for all atomic structures in the list.

        Args:
            converter (AtomicTypeConverter): Object for converting atomic numbers to internal types and vice versa.
            check (bool, optional): If True, check if atomic numbers are supported. Defaults to False.

        Returns:
            AtomicStructures: The `AtomicStructures` object with internal types instead of atomic numbers.
        """
        return AtomicStructures([s.to_type_names(converter, check=check) for s in self.structures])

    def to_atomic_numbers(self, converter: AtomicTypeConverter) -> 'AtomicStructures':
        """Converts internal types to atomic numbers for all atomic structures in the list.

        Args:
            converter (AtomicTypeConverter): Object for converting atomic numbers to internal types and vice versa.

        Returns:
            AtomicStructures: The `AtomicStructures` object with atomic numbers instead of internal types.
        """
        return AtomicStructures([s.to_atomic_numbers(converter) for s in self.structures])

    def to_triu(self) -> 'AtomicStructures':
        """Converts units cells to their upper triangular form.

        Returns:
            AtomicStructures: The `AtomicStructures` object with upper triangular unit cells.
        """
        return AtomicStructures([s.to_triu() for s in self.structures])

    def to_data(self, r_cutoff: float) -> List['AtomicData']:
        """Converts `AtomicStructures` to a list of `AtomicData` used by implemented models and algorithms.
        `AtomicData` handles atomic structures as graphs.

        Args:
            r_cutoff (float): Cutoff radius for computing neighbor lists.

        Returns:
            List[AtomicData]: List of `AtomicData`, handling atomic structures as graphs.
        """
        return [AtomicData(s, r_cutoff=r_cutoff) for s in self.structures]

    def random_split(self,
                     sizes: Dict[str, int],
                     seed: int = None) -> Dict[str, 'AtomicStructures']:
        """Splits atomic structures using a random seed.
        
        Args:
            sizes (Dict[str, int]): Dictionary containing names and sizes of data splits.
            seed (int): Random seed. Defaults to None.

        Returns:
            Dict[str, AtomicStructures]: Dictionary of `AtomicStructures` splits.
        """
        random_state = np.random.RandomState(seed=seed)
        idx = random_state.permutation(np.arange(len(self.structures)))
        sub_idxs = {}
        for key, val in sizes.items():
            sub_idxs.update({key: idx[0:val]})
            idx = idx[val:]
        if len(idx) > 0:
            sub_idxs.update({"test": idx})
        return {name: self[si] for name, si in sub_idxs.items()}

    def split_by_indices(self, idxs: List[int]) -> Tuple[Union['AtomicStructures', AtomicStructure], Union['AtomicStructures', AtomicStructure]]:
        """Splits atomic structures using provided indices.
        
        Args:
            idxs (List[int]): Indices with which atomic structures are split.

        Returns:
            Tuple: Atomic structures defined by `idxs`, and those which remain.
        """
        remaining_idxs = list(set(range(len(self.structures))).difference(set(idxs)))
        remaining_idxs.sort()
        return self[idxs], self[remaining_idxs]

    def __getitem__(self, idxs: int) -> 'AtomicStructures':
        """Provides atomic structures defined by indices or slices.

        Args:
            idxs (int): Indices or slice to extract a portion from atomic structures.

        Returns:
            AtomicStructures: The `AtomicStructures` object.
        """
        if isinstance(idxs, int):
            return self.structures[idxs]
        elif isinstance(idxs, slice):
            return AtomicStructures(self.structures[idxs])
        else:
            # assume idxs is array_like
            return AtomicStructures([self.structures[i] for i in idxs])

    def __add__(self, other: 'AtomicStructures') -> 'AtomicStructures':
        """Combines atomic structures to a single `AtomicStructures` object.

        Args:
            other (AtomicStructures): Atomic structures to be added to `self`.

        Returns:
            AtomicStructures: The combined `AtomicStructures` object.
        """
        return AtomicStructures(self.structures + other.structures)
    
    def _compute_energy_stats(self, n_species: int):
        """Computes mean and std of energy used to shift and re-scale the energy predicted by implemented models.

        Args:
            n_species (int): Total number of atom species.
        """
        if self._n_species is not None and self._n_species == n_species:
            return  # already computed

        E_sum = 0.0
        atoms_sum = 0
        for i in range(len(self)):
            E_sum += self.structures[i].energy
            atoms_sum += self.structures[i].n_atoms
        EperA_mean = E_sum / atoms_sum

        # compute regression from (n_per_species_1, ..., n_per_species_119) to E - N_at * EperA_mean
        # The reason that we subtract EperA_mean is that we don't want to regularize the mean
        mean_err_sse = 0.0
        XTy = np.zeros(n_species)
        XTX = np.zeros(shape=(n_species, n_species), dtype=np.int64)
        for i in range(len(self)):
            Z_counts = np.zeros(n_species, dtype=np.int64)
            for z in self.structures[i].species:
                Z_counts[int(z)] += 1
            err = self.structures[i].energy - self.structures[i].n_atoms * EperA_mean
            XTy += err * Z_counts
            XTX += Z_counts[None, :] * Z_counts[:, None]
            mean_err_sse += err ** 2 / self.structures[i].n_atoms

        lam = 1.0  # regularization, should be a float such that the integer matrix XTX is converted to float
        EperA_regression = np.linalg.solve(XTX + lam * np.eye(n_species), XTy) + EperA_mean
        EperA_std = np.sqrt(mean_err_sse / atoms_sum)

        # convert it to a vector to have the same shape as EperA_regression
        EperA_mean = EperA_mean * np.ones(len(EperA_regression))
        EperA_std = EperA_std * np.ones(len(EperA_regression))

        self._EperA_mean, self._EperA_stdev, self._EperA_regression = EperA_mean, EperA_std, EperA_regression


def to_optional_tensor(arr: Optional[Union[np.ndarray, float, int]],
                       dtype=torch.float32) -> Optional[torch.Tensor]:
    """Converts a numpy array to a torch tensor.

    Args:
        arr (Optional[Union[np.ndarray, float, int]]): Numpy array.
        dtype: Data type. Defaults to torch.float32.

    Returns:
        Optional[torch.Tensor]: Torch tensor.
    """
    return None if arr is None else torch.as_tensor(arr, dtype=dtype)


class AtomicData(Data):
    """Converts atomic structures to graphs.

    Args:
        structure (AtomicStructure): The `AtomicStructure` object.
        r_cutoff (float): Cutoff radius for computing the neighbor list.
        skin (float, optional): Skin distance for updating neighbor list, if necessary. Defaults to 0.0.
    """
    def __init__(self,
                 structure: AtomicStructure,
                 r_cutoff: float,
                 skin: float = 0.0):
        # aggregate data
        data = {
            'num_nodes': to_optional_tensor(structure.n_atoms, dtype=torch.long),
            # duplicate, but num_nodes is not directly provided in the batch
            'n_atoms': to_optional_tensor(structure.n_atoms, dtype=torch.long),
            'edge_index': to_optional_tensor(structure.get_edge_index(r_cutoff, skin), dtype=torch.long),
            'positions': to_optional_tensor(structure.positions),
            'shifts': to_optional_tensor(structure.get_shifts(r_cutoff, skin)),
            'cell': to_optional_tensor(structure.cell).unsqueeze(0),
            'species': to_optional_tensor(structure.species, dtype=torch.long),
            'energy': to_optional_tensor(structure.energy),
            'forces': to_optional_tensor(structure.forces),
            'stress': to_optional_tensor(structure.stress) if structure.stress is None else to_optional_tensor(
                structure.stress).unsqueeze(0),
            'virials': to_optional_tensor(structure.virials) if structure.virials is None else to_optional_tensor(
                structure.virials).unsqueeze(0),
            # strain is required to compute stress
            'strain': to_optional_tensor(np.zeros_like(structure.cell)).unsqueeze(0),
        }
        super().__init__(**data)
