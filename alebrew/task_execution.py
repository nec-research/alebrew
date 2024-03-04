"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     task_execution.py 
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
from pathlib import Path
from typing import List, Union, Optional, Any

import numpy as np
from ase import units
from ase.io import read

from alebrew.data.data import AtomicStructures
from alebrew.datagen.references import BaseReference, CP2KReference, ORCAReference, SingleCalculatorReference
from alebrew.interfaces.ase import TorchMDCalculator
from alebrew.learning import OnlineLearning
from alebrew.strategies import TrainingStrategy, SamplingStrategy, SelectionStrategy, EvaluationStrategy
from alebrew.utils.misc import get_default_device, save_object


class CustomPaths:
    """Custom paths to organize results, data sets, and plots for experiments.

    Args:
        folder (Union[str, Path], optional): Folder where results, data sets, and plots are stored. Defaults to `./`.
    """
    def __init__(self, folder: Union[str, Path] = './'):
        folder = Path(folder)
        self.data_path = folder / 'datasets'
        self.results_path = folder / 'results'
        self.plots_path = folder / 'plots'

    def get_data_path(self) -> Path:
        """

        Returns:
            Path: Folder with data sets.
        """
        return self.data_path

    def get_results_path(self) -> Path:
        """

        Returns:
            Path: Folder with results.
        """
        return self.results_path

    def get_plots_path(self) -> Path:
        """

        Returns:
            Path: Folder with plots.
        """
        return self.plots_path


class Task:
    """Defines a task to systematically evaluate different online active learning approaches for learning interatomic 
    potentials, i.e., biased and unbiased ones. A task comprises structures used to initialize learning and a setup 
    for atomistic simulations (temperatures, pressures, integration time steps, cutoff radii, etc.).

    Args:
        task_name (str): The task's name, typically defined by the atomic systems or the material class. Also, the task 
                         name can reflect the physical conditions under which the corresponding atomic system is 
                         investigated or the reference method used to compute energies, atomic force, etc.
        structures (AtomicStructures): These structures are used to run the pre-sampling step and generate the initial 
                                       (training) data set. However, a pre-defined data set instead of pre-sampling can be
                                       used for some applications.
        init_structures (AtomicStructures): Initial structures are used to initialize atomistic simulations because 
                                            pre-sampling often generates highly distorted configurations with high energy 
                                            and stress values, affecting the results of active learning experiments. 
                                            However, the same structures as those provided by 'structures' can be used
                                            or `init_structures` can be set to None, which implies the same.
        reference (BaseReference): Classical, ab initio, or first-principles method for calculating reference energy, 
                                   atomic force, and stress values.
        r_cutoff (float): Cutoff radius defining the maximal range of modeled atomic interactions.
        atomic_types (List[str]): List of atomic types, typically defined by atomic names. 
        timestep (float): The integration time step employed during atomistic simulations. For adversarial attacks, 
                          the learning rate is defined by the learner.
        temperatures (List[float]): List of temperatures at which atomistic simulations are run.
                                    Note that for experiments with uncertainty bias, lower temperatures can be used.
        externalstresses (List[float]): List of external stress values at which atomistic simulations are run. 
                                        Note that for experiments with uncertainty bias, lower external stresses can be used.
        use_precomp_labels (bool, optional): If True, pre-computed labels for `init_structures` are used for the first step of 
                                             active learning instead of evaluating them with the reference method. Defaults to False.
        mask (Optional[np.ndarray], optional): If `np.eye(3)` is used, then the cell is scaled only along x-, y-, and z- axis during 
                                               atomistic simulations. Defaults to None.
        test_structures (Optional[AtomicStructures], optional): These structures are used to evaluate the performance of interatomic 
                                                                potentials trained with active learning. Defaults to None.
        train_with_virials (bool, optional): If True, models are trained with virials in addition to energies and atomic forces. 
                                             Defaults to False.
    """
    def __init__(self,
                 task_name: str,
                 structures: AtomicStructures,
                 init_structures: AtomicStructures,
                 reference: BaseReference,
                 r_cutoff: float,
                 atomic_types: List[str],
                 timestep: float,
                 temperatures: List[float],
                 externalstresses: List[float],
                 use_precomp_labels: bool = False,
                 mask: Optional[np.ndarray] = None,
                 test_structures: Optional[AtomicStructures] = None,
                 train_with_virials: bool = False, ):
        self.task_name = task_name
        self.structures = structures
        self.init_structures = init_structures
        self.reference = reference
        self.r_cutoff = r_cutoff
        self.atomic_types = atomic_types
        self.timestep = timestep
        self.temperatures = temperatures
        self.externalstresses = externalstresses
        self.use_precomp_labels = use_precomp_labels
        self.mask = mask
        self.test_structures = test_structures
        self.train_with_virials = train_with_virials

    @staticmethod
    def create(task_name: str,
               force_threshold: float = 1e6,
               n_threads: int = 1,
               n_cores: int = 48,
               custom_paths: CustomPaths = CustomPaths()) -> 'Task':
        """Creates tasks used to generate results for https://arxiv.org/abs/2312.01416v1, but can also be expanded to 
        any other task.

        Args:
            task_name (str): The task's name.
            force_threshold (float, optional): Atomic forces threshold value used by the reference method to 
                                               identify atomic structures with poor SCF convergence. 
                                               Defaults to 1e6.
            n_threads (int, optional): Number of parallel threads for reference evaluations. Defaults to 1.
            n_cores (int, optional): Number of cores used for a single reference evaluation. Defaults to 48.
            custom_paths (CustomPaths, optional): Custom paths for results, data sets, and plots. 
                                                  Defaults to `CustomPaths()`.

        Returns:
            Task: Task specific to the provided 'task_name'.
        """
        if task_name == 'ala2-300K':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_init.extxyz')
            reference = ORCAReference(label='orca', orcasimpleinput='rks ma-def2-svp def2/j rijcosx wb97x-d4',
                                      orcablocks=f'%scf ConvForced 1 Maxiter 1000 end %pal nprocs {n_cores} end',
                                      charge=0, mult=1, task='gradient', n_threads=n_threads,
                                      force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=5.0,
                        atomic_types=['H', 'C', 'N', 'O'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures)
        elif task_name == 'ala2-1200K':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_init.extxyz')
            reference = ORCAReference(label='orca', orcasimpleinput='rks ma-def2-svp def2/j rijcosx wb97x-d4',
                                      orcablocks=f'%scf ConvForced 1 Maxiter 1000 end %pal nprocs {n_cores} end',
                                      charge=0, mult=1, task='gradient', n_threads=n_threads,
                                      force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=5.0,
                        atomic_types=['H', 'C', 'N', 'O'], timestep=0.5, temperatures=[1200.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures)
        elif task_name == 'ala2-300K-ffs':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_init.extxyz')
            calc = TorchMDCalculator(input_file=os.path.join(str(custom_paths.data_path), 'input.pdb'),
                                     parm_file=os.path.join(str(custom_paths.data_path), 'parm7.prmtop'),
                                     r_cutoff=9.0)
            reference = SingleCalculatorReference(calc=calc, n_threads=n_threads,
                                                  force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=5.0,
                        atomic_types=['H', 'C', 'N', 'O'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures)
        elif task_name == 'ala2-600K-ffs':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_init.extxyz')
            calc = TorchMDCalculator(input_file=os.path.join(str(custom_paths.data_path), 'input.pdb'),
                                     parm_file=os.path.join(str(custom_paths.data_path), 'parm7.prmtop'),
                                     r_cutoff=9.0)
            reference = SingleCalculatorReference(calc=calc, n_threads=n_threads,
                                                  force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=5.0,
                        atomic_types=['H', 'C', 'N', 'O'], timestep=0.5, temperatures=[600.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures)
        elif task_name == 'ala2-1200K-ffs':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_init.extxyz')
            calc = TorchMDCalculator(input_file=os.path.join(str(custom_paths.data_path), 'input.pdb'),
                                     parm_file=os.path.join(str(custom_paths.data_path), 'parm7.prmtop'),
                                     r_cutoff=9.0)
            reference = SingleCalculatorReference(calc=calc, n_threads=n_threads,
                                                  force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'ala2_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=5.0,
                        atomic_types=['H', 'C', 'N', 'O'], timestep=0.5, temperatures=[1200.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures)
        elif task_name == 'uio66-300K-0bar':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'uio66_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.popt', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'uio66_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Zr'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures)
        elif task_name == 'uio66-600K-500bar':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'uio66_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.popt', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'uio66_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Zr'], timestep=0.5, temperatures=[600.0],
                        externalstresses=[500.0, -500.0], mask=None, test_structures=test_structures)
        elif task_name == 'mil53_cp-300K-0bar-v1':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-300K-0bar-v2':
            # read Sander's training structures to initialize online learning
            # exclude Sander's random displaced structures as they may have
            # large internal pressures and energy which biases exploration
            traj = read(custom_paths.data_path / 'mil53_train.extxyz', ':')
            vols = np.array([a.get_volume() / (len(a) / 76) for a in traj])
            idxs = np.where(vols < 1200.0)[0]
            structures = AtomicStructures.from_traj(traj)
            structures = structures[idxs]
            # read init structures
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=structures, use_precomp_labels=True,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-300K-2500bar-v1':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[2500.0, -2500.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-300K-2500bar-v2':
            # read Sander's training structures to initialize online learning
            # exclude Sander's random displaced structures as they may have
            # large internal pressures and energy which biases exploration
            traj = read(custom_paths.data_path / 'mil53_train.extxyz', ':')
            vols = np.array([a.get_volume() / (len(a) / 76) for a in traj])
            idxs = np.where(vols < 1200.0)[0]
            structures = AtomicStructures.from_traj(traj)
            structures = structures[idxs]
            # read init structures
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=structures, use_precomp_labels=True,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[300.0],
                        externalstresses=[2500.0, -2500.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-600K-0bar-v1':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[600.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-600K-0bar-v2':
            # read Sander's training structures to initialize online learning
            # exclude Sander's random displaced structures as they may have
            # large internal pressures and energy which biases exploration
            traj = read(custom_paths.data_path / 'mil53_train.extxyz', ':')
            vols = np.array([a.get_volume() / (len(a) / 76) for a in traj])
            idxs = np.where(vols < 1200.0)[0]
            structures = AtomicStructures.from_traj(traj)
            structures = structures[idxs]
            # read init structures
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=structures, use_precomp_labels=True,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[600.0],
                        externalstresses=[0.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-600K-2500bar-v1':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[600.0],
                        externalstresses=[2500.0, -2500.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'mil53_cp-600K-2500bar-v2':
            # read Sander's training structures to initialize online learning
            # exclude Sander's random displaced structures as they may have
            # large internal pressures and energy which biases exploration
            traj = read(custom_paths.data_path / 'mil53_train.extxyz', ':')
            vols = np.array([a.get_volume() / (len(a) / 76) for a in traj])
            idxs = np.where(vols < 1200.0)[0]
            structures = AtomicStructures.from_traj(traj)
            structures = structures[idxs]
            # read init structures
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_cp_init.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'mof_base_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.psmp', label='cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold, compute_stress=True)
            test_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'mil53_test.extxyz')
            return Task(task_name=task_name, structures=structures, use_precomp_labels=True,
                        init_structures=init_structures, reference=reference, r_cutoff=6.0,
                        atomic_types=['H', 'C', 'O', 'Al'], timestep=0.5, temperatures=[600.0],
                        externalstresses=[2500.0, -2500.0], mask=None, test_structures=test_structures,
                        train_with_virials=True)
        elif task_name == 'KCl-1650K-0bar':
            init_structures = AtomicStructures.from_extxyz(custom_paths.data_path / 'KCl_1650K.extxyz')
            # make upper triangular (necessary for ASE NTP molecular dynamics)
            init_structures = init_structures.to_triu()
            reference = CP2KReference(cp2k_input_file=custom_paths.data_path / 'KCl_cp2k_input.txt',
                                      command=f'mpirun -n {n_cores} cp2k_shell.popt', label='KCl_cp2k',
                                      n_threads=n_threads, force_threshold=force_threshold)
            return Task(task_name=task_name, structures=init_structures, use_precomp_labels=False,
                        init_structures=init_structures, reference=reference, r_cutoff=5.5,
                        atomic_types=['K', 'Cl'], timestep=2.0, temperatures=[1650.0], externalstresses=[0.0],
                        mask=None)
        else:
            raise NotImplementedError(f'{task_name=} is not implemented yet.')


class Learner:
    """Defines a learner, which comprises pre-sampling, sampling, selection, uncertainty quantification, 
    and (uncertainty-)biasing methods. 

    Args:
        method_name (str): The method's name typically comprises all methods involved and the random seed 
                           used to split the data, initialize models, and set up atomistic simulations.
        sampling_method (str): Sampling method employed to sample/explore new atomic configurations: 
                               'random', 'nvt', 'nh-nvt', 'npt', 'berendsen-npt', or 'adversarial'.
        sel_method (str): Selection method employed to select the most uncertain (and diverse, depending 
                          on chosen method) structures from the candidate pools generated with the sampling 
                          method/atomistic simulation: 'random', 'max_diag', 'max_det', 'max_dist', or 'lcmd'.
        unc_method (Optional[str], optional): Uncertainty quantification method: None, 'posterior', 'distance', 
                                              'ensemble', or 'random'. Defaults to None.
        bias_method (Optional[str], optional): The method employed to introduce a bias to the potential energy surface 
                                               and, thus, bias the respective atomistic simulation: None or 'linear_unc'. 
                                               Defaults to None.
        relative_atom_weights (Optional[np.ndarray], optional): Array containing relative atom weights, used to 
                                                                define relative atom-based biasing strength values. 
                                                                Defaults to None.
        data_seed (int, optional): Random seed used to split data sets. Defaults to 1234.
        sampling_seed (int, optional): Random seed used to set up atomistic simulations. Defaults to 1234.
        model_seeds (List[int], optional): List of random seeds used to initialize models. Defaults to [1234].
        neighbors (str, optional): The method used to compute neighbor lists: 'matscipy', 'ase', 'primitive', or 'torch'. 
                                   Defaults to 'matscipy'.
        max_sampling_step (int, optional): Maximal number of iterations employed during an atomistic simulation 
                                           (defined by the sampling method). 
                                           Each parallel sampler runs for the defined number of steps. Defaults to 1000.
        eval_sampling_step (int, optional): Frequency of storing atomic structures sampled/explored during 
                                            an atomistic simulation. Defaults to 10.
        n_samplers (int, optional): Number of parallel samplers/parallel atomistic simulations. Defaults to 1.
        custom_paths (CustomPaths, optional): Custom paths for results, data sets, and plots. 
                                              Defaults to `CustomPaths()`.
    """
    def __init__(self,
                 method_name: str,
                 sampling_method: str,
                 sel_method: str,
                 unc_method: Optional[str] = None,
                 bias_method: Optional[str] = None,
                 relative_atom_weights: Optional[np.ndarray] = None,
                 data_seed: int = 1234,
                 sampling_seed: int = 1234,
                 model_seeds: List[int] = [1234],
                 neighbors: str = 'matscipy',
                 max_sampling_step: int = 1000,
                 eval_sampling_step: int = 10,
                 n_samplers: int = 1,
                 custom_paths: CustomPaths = CustomPaths(),
                 **config: Any):
        self.method_name = method_name
        # sampling, selection, uncertainty, and biasing methods
        self.sampling_method = sampling_method
        self.sel_method = sel_method
        self.unc_method = unc_method
        self.bias_method = bias_method
        self.relative_atom_weights = relative_atom_weights
        # all seeds
        self.data_seed = data_seed
        self.sampling_seed = sampling_seed
        self.model_seeds = model_seeds
        # some other method specific parameters
        self.neighbors = neighbors
        self.max_sampling_step = max_sampling_step
        self.eval_sampling_step = eval_sampling_step
        self.n_samplers = n_samplers
        # managing directories
        self.custom_paths = custom_paths
        # all other parameters
        self.config = config.copy()

    def _create_learning(self, task: Task) -> OnlineLearning:
        """Crates learner given the task.

        Args:
            task (Task): Task to which learner is applied.

        Returns:
            OnlineLearning: The `OnlineLearning` object.
        """
        if 'device' in self.config:
            device = self.config['device']
        else:
            device = get_default_device()

        eval_losses = [{'type': 'energy_per_atom_rmse'},
                       {'type': 'energy_per_atom_mae'},
                       {'type': 'forces_rmse'},
                       {'type': 'forces_mae'}]

        train_loss = {
            'type': 'weighted_sum',
            'losses': [
                {'type': 'energy_sse'},
                {'type': 'forces_sse'}
            ],
            'weights': [
                1.0,
                4.0
            ]
        }

        early_stopping_loss = {
            'type': 'weighted_sum',
            'losses': [
                {'type': 'energy_mae'},
                {'type': 'forces_mae'}
            ],
            'weights': [
                1.0,
                1.0
            ]
        }

        if task.train_with_virials:
            eval_losses += [{'type': 'virials_rmse'},
                            {'type': 'virials_mae'}]
            train_loss['losses'] += [{'type': 'virials_sse'}]
            train_loss['weights'] += [0.01]
            early_stopping_loss['losses'] += [{'type': 'virials_mae'}]
            early_stopping_loss['weights'] += [1.0]
    
        train_config = dict(device=device, r_cutoff=task.r_cutoff, n_radial=7, n_basis=9,
                            atomic_types=task.atomic_types, neighbors=self.neighbors,
                            eval_losses=eval_losses, train_loss=train_loss,
                            early_stopping_loss=early_stopping_loss)

        temp_stress_pairs = []
        for i in range(len(task.temperatures)):
            for j in range(len(task.externalstresses)):
                temp_stress_pairs.append((task.temperatures[i], task.externalstresses[j]))

        sampling_configs = [{'temperature_K': temperature_K, 'externalstress': externalstress * units.bar,
                             'timestep': task.timestep * units.fs, 'mask': task.mask}
                            for temperature_K, externalstress in temp_stress_pairs]

        train = TrainingStrategy(train_config)

        # create atom-based rescaling if relative atom weights have been provided
        if self.relative_atom_weights is not None:
            atom_based_rescaling = np.take(self.relative_atom_weights, task.init_structures[0].species)
        else:
            atom_based_rescaling = None
        sample = SamplingStrategy(sampling_method=self.sampling_method, sampling_configs=sampling_configs,
                                  unc_method=self.unc_method, bias_method=self.bias_method, n_samplers=self.n_samplers,
                                  eval_sampling_step=self.eval_sampling_step, max_sampling_step=self.max_sampling_step,
                                  atom_based_rescaling=atom_based_rescaling, **self.config)

        max_sampled_structures = int(self.n_samplers * self.max_sampling_step / self.eval_sampling_step)
        select = SelectionStrategy(sel_method=self.sel_method, unc_method=self.unc_method,
                                   max_sampled_structures=max_sampled_structures,
                                   **self.config)

        evaluate = EvaluationStrategy(train_config)

        return OnlineLearning(training_strategy=train, sampling_strategy=sample, selection_strategy=select,
                              reference=task.reference, evaluation_strategy=evaluate, **self.config)

    def run_on_task(self, task: Task):
        """Evaluates the specified learner on the task.

        Args:
            task (Task): Specified task.
        """
        results_folder = self.custom_paths.results_path / task.task_name / self.method_name
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # store config parameters
        params = {'task_name': task.task_name,
                  'method_name': self.method_name,
                  'sampling_method': self.sampling_method,
                  'sel_method': self.sel_method,
                  'unc_method': self.unc_method,
                  'bias_method': self.bias_method,
                  'data_seed': self.data_seed,
                  'sampling_seed': self.sampling_seed,
                  'model_seeds': self.model_seeds,
                  'max_sampling_step': self.max_sampling_step,
                  'eval_sampling_step': self.eval_sampling_step,
                  'n_samplers': self.n_samplers}
        params.update({key: val for key, val in self.config.items()})
        save_object(results_folder / 'config.yaml', params, use_yaml=True)

        # create learning and run on task
        learning = self._create_learning(task)
        structures, _ = learning.run(structures=task.structures, use_precomp_labels=task.use_precomp_labels,
                                     init_structures=task.init_structures, data_seed=self.data_seed,
                                     sampling_seed=self.sampling_seed, model_seeds=self.model_seeds,
                                     folder=results_folder, test_structures=task.test_structures)
        structures.save_extxyz(results_folder / 'all_structures.extxyz')
