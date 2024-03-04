"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     forward.py 
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

import torch
import torch.nn as nn
import numpy as np

from typing import *

import yaml

from alebrew.data.data import AtomicStructures
from alebrew.nn.layers import LinearLayer, RescaledSiLULayer
from alebrew.nn.local_reps import GaussianMoments
from alebrew.nn.transformations import AtomicScaleShift, RepsBatchNorm

from alebrew.utils.torch_geometric import Data
from alebrew.utils.misc import load_object, save_object


def build_model(atomic_structures: Optional[AtomicStructures] = None,
                **config: Any):
    """Builds feed-forward atomistic neural network from `config`.

    Args:
        atomic_structures (Optional[AtomicStructures], optional): Atomic structures, typically those from the training data sets, 
                                                                  used to compute energy statistics. Defaults to None.
        
    Returns:
        ForwardAtomisticNetwork: Atomistic neural network.
    """
    torch.manual_seed(config['model_seed'])
    np.random.seed(config['model_seed'])

    representation = GaussianMoments(**config)

    representation_tfms = []
    for tfm in config['representation_tfms'] or []:
        if tfm == 'batch_norm':
            representation_tfms.append(RepsBatchNorm(n_features=representation.get_n_features(), **config))
        else:
            raise ValueError(f'Unknown transform for representation: {tfm}')

    layers = []
    for in_size, out_size in zip([representation.get_n_features()] + config['hidden_sizes'],
                                 config['hidden_sizes'] + [1]):
        layers.append(LinearLayer(in_size, out_size))
        layers.append(RescaledSiLULayer())

    mlp = nn.Sequential(*layers[:-1])

    if atomic_structures is None:
        shift_params = np.ones(config['n_species'])
        scale_params = np.ones(config['n_species'])
    else:
        shift_params = atomic_structures.get_EperA_regression(n_species=config['n_species'])
        scale_params = atomic_structures.get_EperA_stdev(n_species=config['n_species'])

    output_tfms = []
    for tfm in config['output_tfms'] or []:
        if tfm == 'atomic_scale_shift':
            output_tfms.append(AtomicScaleShift(shift_params=shift_params, scale_params=scale_params))
        else:
            raise ValueError(f'Unknown transform for output: {tfm}')

    return ForwardAtomisticNetwork(representation=representation, mlp=mlp, representation_tfms=representation_tfms,
                                   output_tfms=output_tfms, config=config)


class ForwardAtomisticNetwork(nn.Module):
    """An atomistic model based on feed-forward neural networks.

    Args:
        representation (nn.Module): Local atomic representation (currently only Gaussian moments are implemented).
        mlp (nn.Module): Multilayer perceptron (i.e., readout layer).
        representation_tfms (List[nn.Module]): Transformations applied to the local atomic representation (i.e., 
                                               batch normalization useful for QM9).
        output_tfms (List[nn.Module]): Transformations applied to the output (i.e., energy re-scaling and shift).
    """
    def __init__(self,
                 representation: nn.Module,
                 mlp: nn.Module,
                 representation_tfms: List[nn.Module],
                 output_tfms: List[nn.Module],
                 config: Dict[str, Any]):
        super().__init__()
        # all necessary modules
        self.representation = representation
        self.representation_tfms = nn.ModuleList(representation_tfms)
        self.mlp = mlp
        self.output_tfms = nn.ModuleList(output_tfms)
        # provide config file to store it
        self.config = config

    def forward(self, graph: Data) -> torch.Tensor:
        """Computes atomic energies for the provided batch.

        Args:
            graph (Data): Atomic data graph.

        Returns:
            torch.Tensor: Atomic energies.
        """
        # compute representation
        x = self.representation(graph)
        # apply representation transformations
        for m in self.representation_tfms:
            x = m(x, graph)
        # apply multilayer perceptron to transformed representation
        x = self.mlp(x)
        # apply output transformations
        for m in self.output_tfms:
            x = m(x, graph)
        # squeeze atomic properties predicted by the network
        atomic_energies = x.squeeze(-1)
        return atomic_energies

    def get_device(self) -> str:
        """Provides device on which calculations are performed.
        
        Returns: 
            str: Device on which calculations are performed.
        """
        return list(self.mlp.parameters())[0].device

    def load_params(self, file_path: Union[str, Path]):
        """Loads network parameters from the file.

        Args:
            file_path (Union[str, Path]): Path to the file where network parameters are stored.
        """
        self.load_state_dict(load_object(file_path))

    def save_params(self, file_path: Union[str, Path]):
        """Stores network parameters to the file.

        Args:
            file_path (Union[str, Path]): Path to the file where network parameters are stored.
        """
        save_object(file_path, self.state_dict())

    def save(self, folder_path: Union[str, Path]):
        """Stores config and network parameters to the file.

        Args:
            folder_path (Union[str, Path]): Path to the folder where network parameters are stored.
        """
        (Path(folder_path) / 'config.yaml').write_text(str(yaml.safe_dump(self.config)))
        self.save_params(Path(folder_path) / 'params.pkl')

    @staticmethod
    def from_folder(folder_path: Union[str, Path]) -> 'ForwardAtomisticNetwork':
        """Loads model from the defined folder.

        Args:
            folder_path (Union[str, Path]): Path to the folder where network parameters are stored.

        Returns:
            ForwardAtomisticNetwork: The `ForwardAtomisticNetwork` object.
        """
        config = yaml.safe_load((Path(folder_path) / 'config.yaml').read_text())
        nn = build_model(None, **config)
        nn.load_params(Path(folder_path) / 'params.pkl')
        return nn


def find_last_ckpt(folder: Union[Path, str]):
    """Finds the last/best checkpoint to load the model from.

    Args:
        folder (Union[Path, str]): Path to the folder where checkpoints are stored.

    Returns:
        Last checkpoint to load the model from.
    """
    # if no checkpoint exists raise an error
    files = list(Path(folder).iterdir())
    if len(files) == 0:
        raise RuntimeError(f'Provided {folder} which is empty.')
    if len(files) >= 2:
        folders = [f for f in files if f.name.startswith('ckpt_')]
        file_epoch_numbers = [int(f.name[5:]) for f in folders]
        newest_file_idx = np.argmax(np.asarray(file_epoch_numbers))
        return folders[newest_file_idx]
    else:
        return files[0]


def load_models_from_folder(model_path: Union[str, Path],
                            n_models: int,
                            key: str = 'best') -> List[ForwardAtomisticNetwork]:
    """Loads model (an ensemble of models) from the provided folder.

    Args:
        model_path (Union[str, Path]): Path to the model.
        n_models (int): Number of models.
        key (str, optional): Choose which model to select, the best or last stored one: 'best' and 'log'. 
                             Defaults to 'best'.

    Returns:
        List[ForwardAtomisticNetwork]: List of model (model ensemble if more than one model).
    """
    paths = [os.path.join(model_path, str(i), key) for i in range(n_models)]
    models = []
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f'Provided path to the {key} model does not exist: {path=}.')
        models.append(ForwardAtomisticNetwork.from_folder(find_last_ckpt(path)))
    return models
