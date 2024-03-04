"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     neighbors.py 
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
import numpy as np
from typing import *

import torch

import ase
from ase import Atoms

from matscipy.neighbours import neighbour_list

from alebrew.utils.misc import get_default_device


def get_matscipy_neighbors(positions: np.ndarray,
                           cell: np.ndarray,
                           pbc: Union[List[bool], bool],
                           r_cutoff: float,
                           skin: float = 0.,
                           eps: float = 1e-8,
                           buffer: float = 1.0,
                           **config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Computes neighbor lists using 'matscipy'.

    Args:
        positions (np.ndarray): Atom positions.
        cell (np.ndarray): Unit cell.
        pbc (Union[List[bool], bool]): Periodic boundaries.
        r_cutoff (float): Cutoff radius.
        skin (float, optional): Skin distance. Defaults to 0.
        eps (float, optional): Small number to check if no cell is provided. Defaults to 1e-8.
        buffer (float, optional): Small buffer for stability if no cell is provided. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """

    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)

    # Add cell if none is present (volume = 0)
    if atoms.cell.volume < eps:
        # max values - min values along xyz augmented by small buffer for stability
        new_cell = np.ptp(atoms.positions, axis=0) + buffer
        # Set cell and center
        atoms.set_cell(new_cell, scale_atoms=False)
        atoms.center()

    # Compute neighborhood
    idx_i, idx_j, S = neighbour_list("ijS", atoms, r_cutoff + skin)
    edge_idx = np.stack([idx_i, idx_j])
    offset = S @ cell

    return edge_idx, offset


def get_ase_neighbors(positions: np.ndarray,
                      cell: np.ndarray,
                      pbc: Union[List[bool], bool],
                      r_cutoff: float,
                      skin: float = 0.,
                      self_interaction: bool = False,
                      **config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Computes neighbor lists with ase.

    Args:
        positions (np.ndarray): Atomic positions.
        cell (np.ndarray): Unit cell.
        pbc (Union[List[bool], bool]): Periodic boundaries.
        r_cutoff (float): Cutoff radius.
        skin (float, optional): Skin distance. Defaults to 0.
        self_interaction (bool, optional): If False, an atom  does not return itself as a neighbor. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """
    idx_i, idx_j, S = ase.neighborlist.primitive_neighbor_list(quantities="ijS", pbc=pbc, cell=cell,
                                                               positions=positions, cutoff=r_cutoff + skin,
                                                               self_interaction=self_interaction,
                                                               use_scaled_positions=False)

    edge_idx = np.stack([idx_i, idx_j])
    offset = S @ cell

    return edge_idx, offset


def get_primitive_neighbors(positions: np.ndarray,
                            cell: np.ndarray,
                            pbc: Union[List[bool], bool],
                            r_cutoff: float,
                            skin: float,
                            **config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Computes primitive neighbor lists (using minimum image convention for periodic atomic structures with orthorhombic boxes).

    Args:
        positions (np.ndarray): Atomic positions.
        cell (np.ndarray): Unit cell.
        pbc (Union[List[bool], bool]): Periodic boundaries.
        r_cutoff (float): Cutoff radius.
        skin (float, optional): Skin distance. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """
    # define device:
    device = get_default_device()
    # move tensors to devices
    positions = torch.as_tensor(positions, device=device)
    pbc = torch.as_tensor(pbc, device=device)
    cell = torch.as_tensor(cell, device=device)
    # compute neighbors
    if torch.all(pbc):
        # does not need skin as distances are computed within r_cutoff
        return primitive_pbc(positions, cell, r_cutoff)
    else:
        return primitive_nopbc(positions)


def primitive_pairs(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes neighbor pairs.

    Args:
        positions (torch.Tensor): Atomic positions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """
    n_atoms = positions.shape[0]

    idx_i_uni, idx_j_uni = torch.triu_indices(n_atoms, n_atoms, 1, device=positions.device)

    idx_i_bi = torch.cat([idx_i_uni, idx_j_uni])
    idx_j_bi = torch.cat([idx_j_uni, idx_i_uni])

    sort_idx = torch.argsort(idx_i_bi)

    idx_i_bi = idx_i_bi.index_select(0, sort_idx)
    idx_j_bi = idx_j_bi.index_select(0, sort_idx)
    offset = torch.zeros((idx_i_bi.shape[0], 3), device=positions.device)

    edge_idx = torch.stack([idx_i_bi, idx_j_bi])

    return edge_idx, offset


def primitive_nopbc(positions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the primitive neighbor list for a non-periodic atomic system.

    Args:
        positions (torch.Tensor): Atomic positions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (zeros in this case).
    """
    edge_idx, offset = primitive_pairs(positions)
    return edge_idx.detach().cpu().numpy(), offset.detach().cpu().numpy()


def primitive_pbc(positions: torch.Tensor,
                  cell: torch.Tensor,
                  r_cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the primitive neighbor list for a periodic atomic structure using minimum image convention
    (applicable only for orthorhombic boxes).

    Args:
        positions (np.ndarray): Atomic positions.
        cell (np.ndarray): Unit cell.
        r_cutoff (float): Cutoff radius.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """
    edge_idx, offset = primitive_pairs(positions)

    if is_diag(cell):
        # minimum image convention works only for orthorhombic boxes, for non-orthorhombic distances may be incorrect
        if cell.diag().min() < 2. * r_cutoff:
            raise ValueError(f'Cutoff radius must be at most half the length of the shortest side of '
                             f'the orthorhombic box. Provided {r_cutoff=} and {cell.diag().min()=}!')
        r_ij_vec = positions.index_select(0, edge_idx[0]) - positions.index_select(0, edge_idx[1])
        offset = np.rint(r_ij_vec / np.diag(cell)) * np.diag(cell)
    else:
        raise ValueError(f'Minimum image convention works well only for orthorhombic boxes. '
                         f'Provided a non-orthorhombic box {cell=}!')

    return edge_idx.detach().cpu().numpy(), offset.detach().cpu().numpy()


def is_diag(input: torch.Tensor) -> torch.Tensor:
    """Checks if a tensor has non-zero elements only on its diagonal.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: True, if only diagonal elements are not zeros.
    """
    i, j = torch.nonzero(input).t()
    return torch.all(i == j)


def get_torch_neighbors(positions: np.ndarray,
                        cell: np.ndarray,
                        pbc: Union[List[bool], bool],
                        r_cutoff: float,
                        skin: float = 0.,
                        **config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Verlet neighbor list using pytorch.

    Args:
        positions (np.ndarray): Atomic positions.
        cell (np.ndarray): Unit cell.
        pbc (Union[List[bool], bool]): Periodic boundaries.
        r_cutoff (float): Cutoff radius.
        skin (float, optional): Skin distance. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """
    # define device:
    device = get_default_device()
    # add skin to cutoff:
    r_cutoff += skin
    # move tensors to devices
    positions = torch.as_tensor(positions, device=device)
    pbc = torch.as_tensor(pbc, device=device)
    cell = torch.as_tensor(cell, device=device)
    # compute neighbors
    if torch.any(pbc):
        return neighbor_pairs_pbc(positions, cell, pbc, r_cutoff)
    else:
        return neighbor_pairs_nopbc(positions, r_cutoff)


def neighbor_pairs_nopbc(positions: torch.Tensor,
                         r_cutoff: float,
                         **config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Computes neighbor pairs for a non-periodic system. 
    
    Copyright 2018- Xiang Gao and other ANI developers (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).

    Args:
        positions (torch.Tensor): Atomic positions.
        r_cutoff (float): Cutoff radius.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (zeros).
    """
    n_atoms = positions.shape[0]

    idx_i_uni, idx_j_uni = torch.triu_indices(n_atoms, n_atoms, 1, device=positions.device)
    r_ij_vec = positions.index_select(0, idx_i_uni) - positions.index_select(0, idx_j_uni)
    r_ij_len = r_ij_vec.norm(2, -1)
    in_cutoff = torch.nonzero(r_ij_len < r_cutoff, as_tuple=False)

    pair_index = in_cutoff.squeeze()
    idx_i_uni = idx_i_uni.index_select(0, pair_index)
    idx_j_uni = idx_j_uni.index_select(0, pair_index)

    idx_i_bi = torch.cat([idx_i_uni, idx_j_uni])
    idx_j_bi = torch.cat([idx_j_uni, idx_i_uni])

    sort_idx = torch.argsort(idx_i_bi)
    idx_i_bi = idx_i_bi.index_select(0, sort_idx)
    idx_j_bi = idx_j_bi.index_select(0, sort_idx)
    offset = torch.zeros((idx_i_bi.shape[0], 3), device=positions.device)

    edge_idx = torch.stack([idx_i_bi, idx_j_bi])

    return edge_idx.detach().cpu().numpy(), offset.detach().cpu().numpy()


def neighbor_pairs_pbc(positions: torch.Tensor,
                       cell: torch.Tensor,
                       pbc: torch.Tensor,
                       r_cutoff: float,
                       **config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Computes neighbor pairs for a periodic system.
    
    Copyright 2018- Xiang Gao and other ANI developers (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).

    Args:
        positions (np.ndarray): Atomic positions.
        cell (np.ndarray): Unit cell.
        pbc (Union[List[bool], bool]): Periodic boundaries.
        r_cutoff (float): Cutoff radius.
        skin (float, optional): Skin distance. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Neighbors (edge indices) and shift vectors (the number of cell boundaries crossed by the bond between atoms).
    """
    n_atoms = positions.shape[0]
    all_atoms = torch.arange(n_atoms, device=cell.device)

    # step 1: compute shifts
    shifts = get_shifts(cell, pbc, r_cutoff)

    # step 2: center cell
    # shape: 2 x n_atoms * (n_atoms - 1) // 2
    idx_i_center, idx_j_center = torch.triu_indices(n_atoms, n_atoms, 1, device=cell.device)
    shifts_center = shifts.new_zeros((idx_i_center.shape[0], 3))

    # step 3: cells with shifts
    n_shifts = shifts.shape[0]
    all_shifts = torch.arange(n_shifts, device=cell.device)
    # prod has shape 3 x n_shifts * n_atoms ** 2
    shift_index, idx_i_outside, idx_j_outside = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t()
    shifts_outside = shifts.index_select(0, shift_index)

    # step 4: combine results for all cells
    shifts_all = torch.cat([shifts_center, shifts_outside])
    idx_i_all = torch.cat([idx_i_center, idx_i_outside])
    idx_j_all = torch.cat([idx_j_center, idx_j_outside])

    # step 5: compute distances, and find all pairs within cutoff
    shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
    r_ij_vec = positions.index_select(0, idx_i_all) - positions.index_select(0, idx_j_all) + shift_values
    r_ij_len = r_ij_vec.norm(2, -1)
    in_cutoff = torch.nonzero(r_ij_len < r_cutoff, as_tuple=False)

    pair_index = in_cutoff.squeeze()
    idx_i_uni = idx_i_all.index_select(0, pair_index)
    idx_j_uni = idx_j_all.index_select(0, pair_index)
    offset = shift_values.index_select(0, pair_index)

    # step 6: prepare bidirectional indices and offsets
    offset = torch.cat([-offset, offset])
    idx_i_bi = torch.cat([idx_i_uni, idx_j_uni])
    idx_j_bi = torch.cat([idx_j_uni, idx_i_uni])

    sort_idx = torch.argsort(idx_i_bi)
    idx_i_bi = idx_i_bi.index_select(0, sort_idx)
    idx_j_bi = idx_j_bi.index_select(0, sort_idx)
    offset = offset.index_select(0, sort_idx)

    edge_idx = torch.stack([idx_i_bi, idx_j_bi])

    return edge_idx.detach().cpu().numpy(), offset.detach().cpu().numpy()


def get_shifts(cell: torch.Tensor,
               pbc: torch.Tensor,
               r_cutoff: float) -> torch.Tensor:
    """Computes shifts vectors.
    
    Copyright 2018- Xiang Gao and other ANI developers (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).

    Args:
        cell (np.ndarray): Unit cell.
        pbc (Union[List[bool], bool]): Periodic boundaries.
        r_cutoff (float): Cutoff radius.

    Returns:
        torch.Tensor: All shift vectors.
    """
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    n_repeats = torch.ceil(r_cutoff * inv_distances).to(torch.long)
    n_repeats = torch.where(pbc, n_repeats, n_repeats.new_zeros(()))
    r1 = torch.arange(1, n_repeats[0].item() + 1, device=cell.device)
    r2 = torch.arange(1, n_repeats[1].item() + 1, device=cell.device)
    r3 = torch.arange(1, n_repeats[2].item() + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)
    return torch.cat([
        torch.cartesian_prod(r1, r2, r3),
        torch.cartesian_prod(r1, r2, o),
        torch.cartesian_prod(r1, r2, -r3),
        torch.cartesian_prod(r1, o, r3),
        torch.cartesian_prod(r1, o, o),
        torch.cartesian_prod(r1, o, -r3),
        torch.cartesian_prod(r1, -r2, r3),
        torch.cartesian_prod(r1, -r2, o),
        torch.cartesian_prod(r1, -r2, -r3),
        torch.cartesian_prod(o, r2, r3),
        torch.cartesian_prod(o, r2, o),
        torch.cartesian_prod(o, r2, -r3),
        torch.cartesian_prod(o, o, r3),
    ])
