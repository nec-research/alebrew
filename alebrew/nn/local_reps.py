"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     local_reps.py 
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
from torch import nn

from alebrew.nn.cutoff_fns import CosineCutoff
from alebrew.nn.radial_fns import GaussianRBF
from alebrew.utils.math import segment_sum
from alebrew.utils.torch_geometric import Data


class GaussianMoments(nn.Module):
    """Gaussian moment representation of local atomic environments.

	Args:
		r_cutoff (float): Cutoff radius (maximal radius for atomic interactions).
		n_radial (int, optional): Number of radial functions. Defaults to 7.
		n_contr (int, optional): Number of contractions. Defaults to 8.
		n_basis (int, optional): Number of basis functions. Defaults to 7.
		r_min (float, optional):  Minimal radius for atomic interactions. Defaults to 0.5.
		n_species (int, optional): Number of atom species. Defaults to 119.
		dtype (_type_, optional): Data type. Defaults to torch.float32.
		emb_init (str, optional): Method to initialize embeddings. Defaults to 'uniform'.
	"""
    def __init__(self,
                 r_cutoff: float,
                 n_radial: int = 7,
                 n_contr: int = 8,
                 n_basis: int = 7,
                 r_min: float = 0.5,
                 n_species: int = 119,
                 dtype=torch.float32,
                 emb_init: str = 'uniform',
                 **config: Any):
        super(GaussianMoments, self).__init__()
        self.n_radial = n_radial
        self.n_radial_sq = self.n_radial * n_radial
        self.n_radial_cb = self.n_radial_sq * n_radial
        self.n_contr = n_contr
        self.n_basis = n_basis
        self.r_min = r_min
        self.r_cutoff = r_cutoff
        self.n_species = n_species
        self._dtype = dtype
        self.emb_init = emb_init

        # triu indices for tensors of rank 2 and 3
        triu_indices_rank_two = torch.combinations(torch.arange(0, self.n_radial), r=2, with_replacement=True)
        self.register_buffer('triu_indices_rank_two_flat',
                             self.n_radial * triu_indices_rank_two[:, 0] + triu_indices_rank_two[:, 1])

        self.n_symm01_features = triu_indices_rank_two.shape[0] * self.n_radial

        triu_indices_rank_three = torch.combinations(torch.arange(0, self.n_radial), r=3, with_replacement=True)
        # todo: define as parameters with requires_grad=False
        self.register_buffer('triu_indices_rank_three_flat',
                             self.n_radial_sq * triu_indices_rank_three[:, 0] +
                             self.n_radial * triu_indices_rank_three[:, 1] + triu_indices_rank_three[:, 2])

        self.radial_fn = GaussianRBF(self.r_cutoff, self.n_radial, self.n_basis, self.r_min,
                                     self.emb_init, self.n_species, self._dtype)
        self.cutoff_fn = CosineCutoff(self.r_cutoff)

    def get_n_features(self) -> int:
        """Provides the total number of invariant features.
  
		Returns:
			int: Number of invariant features.
		"""
        return int(self.n_radial + (3 + 2 * self.n_radial) * len(self.triu_indices_rank_two_flat) +
                   len(self.triu_indices_rank_three_flat) + self.n_radial ** 3)

    def forward(self, graph: Data) -> torch.Tensor:
        """Computes Gaussian moment features for the provided atomic data graph.

		Args:
			graph (Data): Atomic data graph.

		Returns:
			torch.Tensor: Local atomic features invariant to rotations.
		"""
    	# r_ij has shape n_neighbors x 3
        # idx_i has shape n_neighbors
        # idx_j has shape n_neighbors
        # species has shape n_atoms
        idx_i, idx_j = graph.edge_index[0, :], graph.edge_index[1, :]
        r_ij = graph.positions.index_select(0, idx_i) - graph.positions.index_select(0, idx_j) - graph.shifts
        species = graph.species

        # calculate moments of the distance distribution
        r_ij_len = torch.norm(r_ij, dim=-1)
        r_ij_vec = r_ij / (r_ij_len[..., None] + 1e-12)

        e = self.cutoff_fn(r_ij_len)[:, None] * self.radial_fn(species, r_ij_len, idx_i, idx_j)

        # zero_moment = e
        # first_moment = torch.einsum('ar,aj->arj', zero_moment, r_ij_vec)
        # second_moment = torch.einsum('ari,aj->arij', first_moment, r_ij_vec)
        # third_moment = torch.einsum('arij,ak->arijk', second_moment, r_ij_vec)
        #
        # zero_moment = segment_sum(zero_moment, idx_i, species.shape[0], 0)
        # first_moment = segment_sum(first_moment, idx_i, species.shape[0], 0)
        # second_moment = segment_sum(second_moment, idx_i, species.shape[0], 0)
        # third_moment = segment_sum(third_moment, idx_i, species.shape[0], 0)

        # compute outer products (a more efficient implementation)
        # shape: n_neighbors x n_radial x (3)^(moment_number)
        xyz = e[:, :, None] * r_ij_vec[:, None, :]

        x_xyz = xyz[:, :, :] * r_ij_vec[:, None, 0:1]
        y_yz = xyz[:, :, 1:] * r_ij_vec[:, None, 1:2]
        # z_z = e - x_x - y_y can be computed after segment_sum

        x_x_xyz = x_xyz * r_ij_vec[:, None, 0:1]
        xy_y_yz = r_ij_vec[:, None, 0:2, None] * y_yz[:, :, None, :]

        # now reduce
        e = segment_sum(e, idx_i, species.shape[0], 0)
        xyz = segment_sum(xyz, idx_i, species.shape[0], 0)
        x_xyz = segment_sum(x_xyz, idx_i, species.shape[0], 0)
        y_yz = segment_sum(y_yz, idx_i, species.shape[0], 0)
        x_x_xyz = segment_sum(x_x_xyz, idx_i, species.shape[0], 0)
        xy_y_yz = segment_sum(xy_y_yz, idx_i, species.shape[0], 0)

        # now compute missing elements
        z_z = e - x_xyz[:, :, 0] - y_yz[:, :, 0]  # z_z = e - x_x - y_y
        x_z_z = xyz[:, :, 0] - x_x_xyz[:, :, 0] - xy_y_yz[:, :, 0, 0]  # x_z_z = x - x_x_x - x_y_y
        y_z_z = xyz[:, :, 1] - x_x_xyz[:, :, 1] - xy_y_yz[:, :, 1, 0]  # y_z_z = y - x_x_y - y_y_y
        z_z_z = xyz[:, :, 2] - x_x_xyz[:, :, 2] - xy_y_yz[:, :, 1, 1]  # z_z_z = z - x_x_z - y_y_z

        # now put it together
        y_xyz = torch.cat([x_xyz[:, :, 1:2], y_yz], -1)
        z_xyz = torch.cat([x_xyz[:, :, 2:3], y_yz[:, :, 1:2], z_z[:, :, None]], -1)
        xyz_xyz = torch.stack([x_xyz, y_xyz, z_xyz], -2)

        x_y_xyz = torch.cat([x_x_xyz[:, :, 1:2], xy_y_yz[:, :, 0, 0:2]], -1)
        x_z_xyz = torch.cat([x_x_xyz[:, :, 2:3], xy_y_yz[:, :, 0, 1:2], x_z_z[:, :, None]], -1)
        x_xyz_xyz = torch.stack([x_x_xyz, x_y_xyz, x_z_xyz], -2)

        y_y_xyz = torch.cat([xy_y_yz[:, :, 0, 0:1], xy_y_yz[:, :, 1, :]], -1)
        y_z_xyz = torch.cat([xy_y_yz[:, :, 0, 1:2], xy_y_yz[:, :, 1, 1:2], y_z_z[:, :, None]], -1)

        z_z_xyz = torch.stack([x_z_z, y_z_z, z_z_z], -1)

        y_xyz_xyz = torch.stack([x_y_xyz, y_y_xyz, y_z_xyz], -2)
        z_xyz_xyz = torch.stack([x_z_xyz, y_z_xyz, z_z_xyz], -2)
        xyz_xyz_xyz = torch.stack([x_xyz_xyz, y_xyz_xyz, z_xyz_xyz], -3)

        zero_moment = e
        first_moment = xyz
        second_moment = xyz_xyz
        third_moment = xyz_xyz_xyz

        # einsum = scripted_einsum
        einsum = torch.einsum

        # calculate contractions
        # convention: a corresponds to n_atoms, m to n_models, r/s/t to n_radial, i/j/k/l to 3
        contr_0 = zero_moment
        contr_1 = einsum('ari, asi -> ars', first_moment, first_moment).reshape(-1, self.n_radial_sq)
        contr_2 = einsum('arij, asij -> ars', second_moment, second_moment).reshape(-1, self.n_radial_sq)
        contr_3 = einsum('arijk, asijk -> ars', third_moment, third_moment).reshape(-1, self.n_radial_sq)
        contr_4 = einsum('arij, asik, atjk -> arst', second_moment, second_moment,
                         second_moment).reshape(-1, self.n_radial_cb)
        contr_5 = einsum('ari, asj, atij -> arst', first_moment, first_moment,
                         second_moment).reshape(-1, self.n_radial_sq, self.n_radial)
        contr_6 = einsum('arijk, asijl, atkl -> arst', third_moment, third_moment,
                         second_moment).reshape(-1, self.n_radial_sq, self.n_radial)
        contr_7 = einsum('arijk, asij, atk -> arst', third_moment, second_moment, first_moment)

        gaussian_moments = [contr_0,
                            contr_1.index_select(1, self.triu_indices_rank_two_flat),
                            contr_2.index_select(1, self.triu_indices_rank_two_flat),
                            contr_3.index_select(1, self.triu_indices_rank_two_flat),
                            contr_4.index_select(1, self.triu_indices_rank_three_flat),
                            contr_5.index_select(1, self.triu_indices_rank_two_flat).reshape(-1,
                                                                                             self.n_symm01_features),
                            contr_6.index_select(1, self.triu_indices_rank_two_flat).reshape(-1,
                                                                                             self.n_symm01_features),
                            contr_7.reshape(-1, self.n_radial * self.n_radial * self.n_radial)
                            ]

        gaussian_moments = torch.cat(gaussian_moments[:self.n_contr], -1)

        return gaussian_moments
