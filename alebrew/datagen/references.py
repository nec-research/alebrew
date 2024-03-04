"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     references.py 
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
from pathlib import Path

import ase
from ase import Atoms
from ase.calculators.general import Calculator
from ase.calculators.cp2k import CP2K
from ase.calculators.orca import ORCA

from typing import List, Dict, Any, Union, Tuple

from ase.io import read, write

from alebrew.data.data import AtomicStructures
from alebrew.utils.process_pool import ProcessPoolMapper


class BaseReference:
    """Computes reference energy, force, and stress values.

    Args:
        force_threshold (float, optional): Atomic force threshold to prevent the generation of 
                                           badly converged reference values. Defaults to 1e6.
    """
    def __init__(self, force_threshold: float = 1e6):
        self.force_threshold = force_threshold

    def label_traj(self,
                   traj: List[Atoms],
                   folder: Union[str, Path]) -> Tuple[List[Atoms], List[Atoms]]:
        """Computes reference values for a list of `ase.Atoms`.

        Args:
            traj (List[Atoms]): List of `ase.Atoms`.
            folder (Union[str, Path]): Folder where the progress of reference calculations is stored.

        Returns:
            Tuple[List[Atoms], List[Atoms]]: List of `ase.Atoms` with completed and failed reference calculations.
        """
        raise NotImplementedError()

    def label_structures(self,
                         structures: AtomicStructures,
                         folder: Union[str, Path]) -> Tuple[AtomicStructures, AtomicStructures]:
        """Computes reference values for `AtomicStructures`.

        Args:
            structures (AtomicStructures): The `AtomicStructures` object.
            folder (Union[str, Path]): Folder where the progress of reference calculations is stored.

        Returns:
            Tuple[AtomicStructures, AtomicStructures]: `AtomicStructures` with completed and failed reference calculations.
        """
        folder = Path(folder)
        # convert structures to atoms trajectory
        traj = [s.to_atoms() for s in structures]
        completed, failed = self.label_traj(traj, folder)
        # convert traj to atomic structures
        completed = AtomicStructures.from_traj(completed)
        failed = AtomicStructures.from_traj(failed)
        # store completed structures
        if not os.path.exists(folder):
            os.makedirs(folder)
        completed.save_extxyz(folder / 'completed_structures.extxyz')
        return completed, failed

    def clean_reference_folder(self, folder: Union[str, Path]):
        """Cleans the folder with the progress of reference calculations. Used to remove, e.g., stored wave functions as 
        they require a lot of space.

        Args:
            folder (Union[str, Path]): Folder where the progress of reference calculations is stored.
        """
        pass


class ProcessPoolReference(BaseReference):
    """Computes reference values for multiple atomic structures in parallel.

    Args:
        n_threads (int): The number of parallel threads.
        force_threshold (float, optional): Atomic force threshold to prevent the generation of 
                                           badly converged reference values. Defaults to 1e6.
    """
    def __init__(self, n_threads: int, force_threshold: float = 1e6):
        super().__init__(force_threshold=force_threshold)
        self.n_threads = n_threads

    def label_traj(self,
                   traj: List[Atoms],
                   folder: Union[str, Path]) -> Tuple[List[Atoms], List[Atoms]]:
        # if folder is None,
        fs = [self._calculate] * len(traj)
        args_tuples = [(atoms, Path(folder), i) for i, atoms in enumerate(traj)]
        mapper = ProcessPoolMapper(n_threads=min(len(traj), self.n_threads))
        labeled_atoms = mapper.map(fs, args_tuples)
        return [a['atoms'] for a in labeled_atoms if a['status'] == 'completed'], \
            [a['atoms'] for a in labeled_atoms if a['status'] == 'failed']

    def _calculate(self, 
                   atoms: ase.Atoms, 
                   folder: Path, 
                   index: int) -> Dict[str, Any]:
        """Calculates reference values for a single `ase.Atoms` object.

        Args:
            atoms (ase.Atoms): The `ase.Atoms` object.
            folder (Path): Folder where the progress of reference calculations is stored.
            index (int): `ase.Atoms` index in the list of `ase.Atoms`.

        Returns:
            Dict[str, Any]: Dictionary containing ase.Atoms and the report about 
                            the convergence of the reference calculation.
        """
        raise NotImplementedError()


class XTBReference(ProcessPoolReference):
    """Computes reference values with XTB.

    Args:
        method (str, optional): XTB method. Defaults to 'GFN2-xTB'.
        n_threads (int, optional): The number of parallel threads. Defaults to 1.
        force_threshold (float, optional): Atomic force threshold to prevent the generation of 
                                           badly converged reference values. Defaults to 1e6.
    """
    def __init__(self,
                 method: str = 'GFN2-xTB',
                 n_threads: int = 1,
                 force_threshold: float = 1e6):
        super().__init__(n_threads=n_threads, force_threshold=force_threshold)
        try:
            from xtb.ase.calculator import XTB
        except Exception as e:
            raise RuntimeError(e)
        self.calc = XTB(method=method)

    def _calculate(self,
                   atoms: ase.Atoms,
                   folder: Union[str, Path],
                   index: int) -> Dict[str, Any]:
        try:
            atoms.set_calculator(self.calc)
            atoms.arrays['forces'] = atoms.get_forces()
            atoms.info['energy'] = atoms.get_potential_energy()
            return {'atoms': atoms, 'status': 'completed', 'message': 'none'}
        except Exception as e:
            return {'atoms': atoms, 'status': 'failed', 'message': f'{e=}'}


class CP2KReference(ProcessPoolReference):
    """Computes reference values with CP2K.

    Args:
        cp2k_input_file (Union[str, Path], optional): CP2K input file. Defaults to 'cp2k_input.txt'.
        command (str, optional): The command used by the calculator to launch the CP2K-shell. 
                                 Defaults to 'mpirun -n 48 cp2k_shell.popt'.
        label (str, optional): Prefix of output files. Defaults to 'cp2k'.
        n_threads (int, optional): The number of parallel threads. Defaults to 1.
        force_threshold (float, optional): Atomic force threshold to prevent the generation of 
                                           badly converged reference values. Defaults to 1e6.
        compute_stress (bool, optional): If True, stress tensor is computed. Defaults to False.
    """
    def __init__(self,
                 cp2k_input_file: Union[str, Path] = 'cp2k_input.txt',
                 command: str = 'mpirun -n 48 cp2k_shell.popt',
                 label: str = 'cp2k',
                 n_threads: int = 1,
                 force_threshold: float = 1e6,
                 compute_stress: bool = False):
        super().__init__(n_threads=n_threads, force_threshold=force_threshold)
        # read CP2K input file
        with open(cp2k_input_file, 'r') as f:
            self.cp2k_input = f.read()
        self.command = command
        self.label = label
        self.compute_stress = compute_stress

    def clean_reference_folder(self, folder: Union[str, Path]):
        folder = Path(folder)
        try:
            for d in os.listdir(folder):
                if d.startswith('structure'):
                    for file in os.listdir(folder / d):
                        if file.startswith(self.label):
                            # shutil.rmtree(folder / d / file)
                            os.remove(folder / d / file)
        except Exception as e:
            print(e)
            pass

    def _calculate(self,
                   atoms: ase.Atoms,
                   folder: Union[str, Path],
                   index: int) -> Dict[str, Any]:
        old_wd = os.getcwd()
        new_folder = folder / f'structure_{index}'
        # we first check if the new folder exists and a calculation has been finished (converged SCF run or not)
        if os.path.exists(new_folder / 'COMPLETED'):
            atoms = read(new_folder / 'structure.extxyz')
            return {'atoms': atoms, 'status': 'completed', 'message': 'none'}
        elif os.path.exists(new_folder / 'FAILED'):
            atoms = read(new_folder / 'structure.extxyz')
            return {'atoms': atoms, 'status': 'failed', 'message': 'SCF run NOT converged'}
        else:
            # if none of the cases we run DFT on the respective atoms
            try:
                # create the current working directory
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                # find all running and finished calculation folders
                other_folders = [f for f in folder.iterdir() if f.is_dir() and f != new_folder]
                # check if any calculation in other folders has been finished and the respective wave function
                # can be used to restart (may lead to a faster SCF convergence)
                for other_folder in other_folders:
                    files = [file for file in other_folder.iterdir() if file.is_file()]
                    if other_folder / 'COMPLETED' in files and other_folder / f'{self.label}-RESTART.wfn' in files:
                        shutil.copy(other_folder / f'{self.label}-RESTART.wfn', new_folder / f'{self.label}-RESTART.wfn')
                        break
                # perform calculation in the new folder
                os.chdir(new_folder)
                # set the new CP2K input
                cp2k_input_new = self.cp2k_input.replace('inputs/', str(Path(old_wd)) + '/inputs/')
                # define calculator
                calc = CP2K(basis_set=None, basis_set_file=None, max_scf=None, cutoff=None,
                            force_eval_method=None, potential_file=None, poisson_solver=None,
                            pseudo_potential=None, stress_tensor=False, xc=None, label=self.label,
                            inp=cp2k_input_new, command=self.command)
                atoms.set_calculator(calc)
                # perform calculation
                atoms.arrays['forces'] = atoms.get_forces()
                atoms.info['energy'] = atoms.get_potential_energy()
                if self.compute_stress:
                    atoms.info['stress'] = atoms.get_stress(voigt=False)
                # remove calculator
                atoms.set_calculator()
                # manually delete CP2KShell as otherwise it may be not removed
                del calc._shell
                del calc
                # store atoms to restart DFT run
                write('structure.extxyz', atoms, format='extxyz')
                # check if SCF run has converged
                with open(self.label + '.out', 'r') as file:
                    content = file.read()
                    if 'SCF run NOT converged' in content:
                        raise RuntimeError('SCF run NOT converged')
                # check if a force threshold has been exceeded
                if (abs(atoms.arrays['forces']) > self.force_threshold).any():
                    raise RuntimeError('Force threshold has been exceeded')
                # store a file indicating that a calculation has been finished and SCF run has converged
                with open('COMPLETED', 'w') as file:
                    file.write('')
                # move back to the main folder
                os.chdir(old_wd)
                return {'atoms': atoms, 'status': 'completed', 'message': 'none'}
            except Exception as e:
                print(e)
                # store a file indicating that a calculation has been finished and SCF run has NOT converged
                with open('FAILED', 'w') as file:
                    file.write('')
                # move back to the main folder
                os.chdir(old_wd)
                return {'atoms': atoms, 'status': 'failed', 'message': f'{e=}'}


class ORCAReference(ProcessPoolReference):
    """Computes reference values with ORCA.

    Args:
        orcasimpleinput (str, optional): ORCA input string. Defaults to 'rks ma-def2-svp def2/j rijcosx wb97x-d4'.
        orcablocks (str, optional): ORCA SCF details. Defaults to '%scf ConvForced 1 Maxiter 1000 end %pal nprocs 8 end'.
        command (str, optional): The command used by the calculator to launch ORCA calculation. 
                                 Defaults to '/opt/bwhpc/common/chem/orca/5.0.4_static/orca PREFIX.inp > PREFIX.out'.
        charge (int, optional): Total charge. Defaults to 0.
        mult (int, optional): Spin multiplicity. Defaults to 1.
        task (str, optional): ORCA task. Defaults to 'gradient'.
        label (str, optional): Prefix of output files. Defaults to 'orca'.
        n_threads (int, optional): The number of parallel threads. Defaults to 1.
        force_threshold (float, optional): Atomic force threshold to prevent the generation of 
                                           badly converged reference values. Defaults to 1e6.
    """
    def __init__(self,
                 orcasimpleinput: str = 'rks ma-def2-svp def2/j rijcosx wb97x-d4',
                 orcablocks: str = '%scf ConvForced 1 Maxiter 1000 end %pal nprocs 8 end',
                 command: str = '/opt/bwhpc/common/chem/orca/5.0.4_static/orca PREFIX.inp > PREFIX.out',
                 charge=0,
                 mult=1,
                 task='gradient',
                 label: str = 'orca',
                 n_threads: int = 1,
                 force_threshold: float = 1e6):
        super().__init__(n_threads=n_threads, force_threshold=force_threshold)
        os.environ['ASE_ORCA_COMMAND'] = command
        self.orcasimpleinput = orcasimpleinput
        self.orcablocks = orcablocks
        self.charge = charge
        self.mult = mult
        self.task = task
        self.label = label

    def clean_reference_folder(self, folder: Union[str, Path]):
        folder = Path(folder)
        try:
            for d in os.listdir(folder):
                if d.startswith('structure'):
                    for file in os.listdir(folder / d):
                        if file.startswith(self.label):
                            shutil.rmtree(folder / d / file)
        except Exception as e:
            print(e)
            pass

    def _calculate(self,
                   atoms: ase.Atoms,
                   folder: Union[str, Path],
                   index: int) -> Dict[str, Any]:
        old_wd = os.getcwd()
        new_folder = folder / f'structure_{index}'
        # we first check if the new folder exists and a calculation has been finished (converged SCF run or not)
        if os.path.exists(new_folder / 'COMPLETED'):
            atoms = read(new_folder / 'structure.extxyz')
            return {'atoms': atoms, 'status': 'completed', 'message': 'none'}
        elif os.path.exists(new_folder / 'FAILED'):
            atoms = read(new_folder / 'structure.extxyz')
            return {'atoms': atoms, 'status': 'failed', 'message': 'SCF run NOT converged'}
        else:
            # if none of the cases we run DFT on the respective atoms
            try:
                # create the current working directory
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                # perform calculation in the new folder
                os.chdir(new_folder)
                # define calculator
                calc = ORCA(label=self.label, orcasimpleinput=self.orcasimpleinput, orcablocks=self.orcablocks,
                            charge=self.charge, mult=self.mult, task=self.task)
                atoms.set_calculator(calc)
                # perform calculation
                atoms.arrays['forces'] = atoms.get_forces()
                atoms.info['energy'] = atoms.get_potential_energy()
                # remove calculator
                atoms.set_calculator()
                # store atoms to restart DFT run
                write('structure.extxyz', atoms, format='extxyz')
                # check if a force threshold has been exceeded
                if (abs(atoms.arrays['forces']) > self.force_threshold).any():
                    raise RuntimeError('Force threshold has been exceeded')
                # store a file indicating that a calculation has been finished and SCF run has converged
                with open('COMPLETED', 'w') as file:
                    file.write('')
                # move back to the main folder
                os.chdir(old_wd)
                return {'atoms': atoms, 'status': 'completed', 'message': 'none'}
            except Exception as e:
                # store a file indicating that a calculation has been finished and SCF run has NOT converged
                with open('FAILED', 'w') as file:
                    file.write('')
                # move back to the main folder
                os.chdir(old_wd)
                return {'atoms': atoms, 'status': 'failed', 'message': f'{e=}'}


class SingleCalculatorReference(ProcessPoolReference):
    """Calculates reference values using a single calculator (unlike some DFT codes which require 
    separate calculators for each structure, separate folders to run calculation, etc.). 
    Can be used with interatomic potentials.
    
    Args:
        calc (Calculator): The `Calculator` object which provides reference values.
        n_threads (int, optional): The number of parallel threads. Defaults to 1.
        force_threshold (float, optional): Atomic force threshold to prevent the generation of 
                                           badly converged reference values. Defaults to 1e6.
        compute_stress (bool, optional): If True, stress tensor is computed. Defaults to False.
    """
    def __init__(self,
                 calc: Calculator,
                 n_threads: int = 1,
                 force_threshold: float = 1e6,
                 compute_stress: bool = False):
        super().__init__(n_threads=n_threads, force_threshold=force_threshold)
        self.calc = calc
        self.compute_stress = compute_stress

    def _calculate(self,
                   atoms: ase.Atoms,
                   folder: Union[str, Path],
                   index: int) -> Dict[str, Any]:
        try:
            atoms.set_calculator(self.calc)
            atoms.arrays['forces'] = atoms.get_forces()
            atoms.info['energy'] = atoms.get_potential_energy()
            if self.compute_stress:
                atoms.info['stress'] = atoms.get_stress(voigt=False)
            # remove calculator
            atoms.set_calculator()
            # check if a force threshold has been exceeded
            if (abs(atoms.arrays['forces']) > self.force_threshold).any():
                raise RuntimeError('Force threshold has been exceeded')
            return {'atoms': atoms, 'status': 'completed', 'message': 'none'}
        except Exception as e:
            return {'atoms': atoms, 'status': 'failed', 'message': f'{e=}'}
