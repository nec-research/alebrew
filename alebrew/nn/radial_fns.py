"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     radial_fns.py 
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
import torch
import torch.nn as nn


def uniform_embeddings(n_species: int,
                       n_radial: int,
                       n_basis: int,
                       dtype=torch.float32) -> nn.Parameter:
    """Provides uniform embeddings of atomic species used for the radial basis.

    Args:
        n_species (int): Number of atom species.
        n_radial (int): Number of radial functions.
        n_basis (int): Number of basis functions.
        dtype (optional): Data type. Defaults to torch.float32.

    Returns:
        nn.Parameter: Atomic species embeddings initialized uniformly between -1.0 and 1.0.
    """
    embeddings = 2.0 * torch.rand([n_species, n_species, n_radial, n_basis], dtype=dtype) - 1.0
    return embeddings


def constant_embeddings(n_species: int,
                        n_radial: int,
                        n_basis: int,
                        dtype=torch.float32) -> nn.Parameter:
    """Provides constant embeddings of atomic species used for the radial basis.

    Args:
        n_species (int): Number of atom species.
        n_radial (int): Number of radial functions.
        n_basis (int): Number of basis functions.
        dtype (optional): Data type. Defaults to torch.float32.

    Returns:
        nn.Parameter: Atomic species embeddings initialized to 1.0 if i=j and 0.0 else.
    """
    comb_matrix = torch.as_tensor([[max(0.0, 1.0 - torch.abs(i - j * (n_radial - 1) / (n_basis - 1)))
                                    for j in range(n_basis)] for i in range(n_radial)], dtype=dtype)
    emb = torch.tile(comb_matrix[None, None, :, :], (n_species, n_species, 1, 1))
    return torch.sqrt(n_radial) * emb


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions.

    Args:
        r_cutoff (float, optional): Cutoff radius. Defaults to 5.0.
        n_radial (int, optional): Number of radial functions. Defaults to 5.
        n_basis (int, optional): Number of basis functions. Defaults to 7.
        r_min (float, optional): Minimal radius for atomic interactions. Defaults to 0.5.
        emb_init (str, optional): Method to initialize embeddings. Defaults to 'uniform'.
        n_species (int, optional): Number of atom species. Defaults to 119.
        dtype (optional): Data type. Defaults to torch.float32.
    """
    def __init__(self,
                 r_cutoff: float = 5.0,
                 n_radial: int = 5,
                 n_basis: int = 7,
                 r_min: float = 0.5,
                 emb_init: str = 'uniform',
                 n_species: int = 119,
                 dtype=torch.float32):
        super(GaussianRBF, self).__init__()
        self.r_cutoff = r_cutoff
        self.n_basis = n_basis
        self.n_radial = n_radial
        self.r_min = r_min
        self.emb_init = emb_init
        self.n_species = n_species
        self._dtype = dtype

        self.betta = self.n_basis ** 2 / self.r_cutoff ** 2
        self.norm = (2.0 * self.betta / torch.pi) ** 0.25
        self.scale_emb = 1.0 / torch.sqrt(torch.as_tensor(self.n_basis))

        # shape: n_shifts
        self.register_buffer('shift_parameter', torch.as_tensor(
            [self.r_min + i * (self.r_cutoff - self.r_min) / self.n_basis for i in range(self.n_basis)]))

        # shape: n_species x n_species x n_radial x n_basis
        embeddings = uniform_embeddings(self.n_species, self.n_radial, self.n_basis, dtype=self._dtype) \
            if emb_init == 'uniform' else \
            constant_embeddings(self.n_species, self.n_radial, self.n_basis, dtype=self._dtype)

        # shape: n_species**2 x n_radial x n_basis
        self.embeddings_flat = nn.Parameter(embeddings.view(
            self.n_species * self.n_species, self.n_radial, self.n_basis))

    def forward(self,
                species: torch.Tensor,
                r_ij_len: torch.Tensor,
                idx_i: torch.Tensor,
                idx_j: torch.Tensor) -> torch.Tensor:
        """Computes the values of the radial basis for provided atomic distances.

        Args:
            species (torch.Tensor): Atom species (their types).
            r_ij_len (torch.Tensor): Atomic pair-distances.
            idx_i (torch.Tensor): Indices of central atoms.
            idx_j (torch.Tensor): Indices of neighboring atoms.

        Returns:
            torch.Tensor: Values of the radial basis functions.
        """
        # R_ij_len has shape n_neighbors
        # idx_i has shape n_neighbors
        # idx_j has shape n_neighbors
        # species has shape n_atoms

        # calculate species dependent coefficients
        # species_i, species_j have shape n_neighbors x 1
        species_i = species.index_select(0, idx_i)
        species_j = species.index_select(0, idx_j)

        # gather embeddings
        # shape: n_neighbors x n_radial x n_basis
        species_pair_coefficients = self.embeddings_flat.index_select(0, self.n_species * species_i + species_j)
        species_pair_coefficients = self.scale_emb * species_pair_coefficients

        # shape: n_neighbors x n_basis
        basis_values = self.norm * torch.exp(-self.betta * (self.shift_parameter[None, :] - r_ij_len[:, None]) ** 2)

        # shape: n_neighbors x n_models x n_radial  ('nrb,nb->nr')
        radial_function = torch.einsum('ari, ai -> ar', species_pair_coefficients, basis_values)

        return radial_function
