"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     transformations.py 
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
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from alebrew.utils.torch_geometric import Data


class RepsBatchNorm(nn.Module):
    """Computes a custom version of Batch Normalization. Used only for models trained without force information (e.g., QM9).

    Args:
        n_features (int): Number of input features.
        norm_scale_factor (float, optional): Empirically introduced rescaling. Defaults to 0.4.
        bn_momentum (float, optional): Momentum of Batch Normalization. Defaults to 0.9.
        bn_eps (float, optional): Small number to avoid division by zero. Defaults to 1e-8.
    """
    def __init__(self,
                 n_features: int,
                 norm_scale_factor: float = 0.4,
                 bn_momentum: float = 0.9,
                 bn_eps: float = 1e-8,
                 **config: Any):
        super(RepsBatchNorm).__init__()
        self.n_features = n_features
        self.bn_momentum = bn_momentum
        self.eps = bn_eps
        self.norm_scale_factor = norm_scale_factor

        # don't use bias in this BatchNorm variant
        self.register_buffer('running_mean', torch.zeros(n_features))
        self.register_buffer('running_var', torch.as_tensor(0.0))
        self.register_buffer('missing_fraction', torch.as_tensor(1.0))

    def forward(self,
                x: torch.Tensor,
                graph: Data) -> torch.Tensor:
        """Computes the normalized features for the provided batch.

        Args:
            x (torch.Tensor): Input tensor.
            graph (Data): Atomic data graph.

        Returns:
            torch.Tensor: Output of the normalized layer.
        """
        if self.training:
            mean = torch.mean(x, 0)
            var = torch.mean((x-mean[None, :])**2)

            # update statistics
            self.running_mean.lerp_(mean, 1.0-self.bn_momentum)
            self.running_var.lerp_(var, 1.0-self.bn_momentum)
            self.missing_fraction *= self.bn_momentum

        correction_factor = 1.0 / (1.0 - self.missing_fraction + self.eps)
        corrected_running_mean = correction_factor * self.running_mean
        corrected_running_var = correction_factor * (self.running_var + self.eps)

        running_normalized = self.norm_scale_factor * (x - corrected_running_mean) / torch.sqrt(corrected_running_var)

        return running_normalized


class AtomicScaleShift(nn.Module):
    """Re-scales and shifts the energy of the specified model.


    Args:
        shift_params (np.ndarray): Parameters by which atomic energies should be shifted.
        scale_params (np.ndarray): Parameters by which atomic energies should be rescaled.
    """
    def __init__(self,
                 shift_params: np.ndarray,
                 scale_params: np.ndarray):
        # Takes the output of an atomistic model and then re-scales and shifts them using variables that are
        # initialized using shift_params and scale_params. Parameters can be scale and shift for atomic energies or
        # the respective values for magnetic anisotropy tensors. The parameters depend on the atomic species.
        super().__init__()

        # shape of scale, shift, and factor parameters: n_species
        # we allow different scale_parameters for different atoms
        # at initialization, different to previous implementation
        self.factors = nn.Parameter(torch.as_tensor(scale_params, dtype=torch.float32), requires_grad=False)
        # re-scale parameters to aid training process
        self.scale_params = nn.Parameter(torch.ones(*scale_params.shape))
        self.shift_params = nn.Parameter(torch.as_tensor(shift_params, dtype=torch.float32) / self.factors)

    def forward(self,
                x: torch.Tensor,
                graph: Data) -> torch.Tensor:
        """Computes the re-scaled and shifted energies for the provided batch.

        Args:
            x (torch.Tensor): Inputs to the model.
            graph (Data): Atomic data graph.

        Returns:
            torch.Tensor: Rescaled and shifted energies of the specified model.
        """
        factors_species = self.factors.index_select(0, graph.species)
        scale_species = self.scale_params.index_select(0, graph.species)
        shift_species = self.shift_params.index_select(0, graph.species)
        return factors_species[:, None] * (scale_species[:, None] * x + shift_species[:, None])
