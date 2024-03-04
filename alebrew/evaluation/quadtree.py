"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     quadtree.py 
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
import itertools
from typing import List, Tuple

import torch


class _Interval:
    def __init__(self,
                 min: float,
                 max: float):
        self._min = min
        self._max = max

    def __contains__(self,
                     item: float):
        return self._min <= item <= self._max

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    def __str__(self):
        return f'({self.min}, {self.max})'


class Hypercube:
    def __init__(self,
                 boundaries: List[Tuple[float, float]]):
        """Initializes the hypercube with a list of boundaries [(min, max),...]
        for each dimension

        Args:
            boundaries (List[Tuple[float, float]]): A list of tuple of float 
                                                    specifying the boundaries
                                                    in each dimension.
        """
        self._dimensions = len(boundaries)
        self._boundaries = [_Interval(min=b[0], max=b[1]) for b in boundaries]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __contains__(self,
                     item: torch.Tensor) -> bool:
        """Checks if a point lies in the hypercube.
        
        Args: 
            item (torch.Tensor): A d-dimensional torch vector.
            
        Returns:
            bool: True, if a point lies in the hypercube.
        """
        for d in range(self._dimensions):
            if float(item[d]) not in self._boundaries[d]:
                return False
        return True

    def __getitem__(self,
                    boundary_idx: int) -> _Interval:
        """Returns a specific boundary in the list.

        Args:
            boundary_idx (int): The index of the boundary in the list.

        Returns: 
            _Interval: An `_Interval` object.
        """
        return self._boundaries[boundary_idx]

    def __str__(self):
        return f'{[str(interval) for interval in self._boundaries]}'


class WeightedSpacePartitioner:
    """Implements a regular and recursive partitioning of a d-dimensional
    Euclidean space induced by a tree structure. Each node is assigned a 
    weight depending on its height (but for the root), 
    which is 1/(ariety^height). The ariety of the tree depends on how we
    partition the space and the dimensionality (e.g, 2^d if we split each
    dimension in half each time). The weight is used to keep track of which
    nodes we visit during a search: if a node has already been visited, we do
    not add up its weight, and we consider it otherwise. This helps us to keep
    track of how good an active learning framework is exploring the space of
    the variables of interest.
    """
    def __init__(self,
                 num_splits: int,
                 depth: int,
                 boundaries: List[Tuple[float, float]],
                 height: int = 0):
        """Initializes the root node with its boundaries [(min, max),...] 
        and recursively creates children.

        Args:
            num_splits (int): How to split each dimension. For example, 3 means we
                              split each side of the space into 3 parts recursively. 
                              In 2 dimensions, a split of 2 generates a quadtree.
            depth (int): A list of tuple of float specifying the boundaries
                         in each dimension
            boundaries (List[Tuple[float, float]]): A list of tuple of float specifying 
                                                    the initial boundaries in each 
                                                    dimension.
            height (int, optional): Parameter used internally, do not touch. Defaults to 0.
        """
        self.num_splits = num_splits
        self.depth = depth
        self.height = height
        self.root_hypercube = Hypercube(boundaries)
        self.ariety = num_splits**self.root_hypercube.dimensions

        # Persistent value needed to compute the total weight as more points
        # in the space are observed
        self.visited = False

        if height == 0:
            self.root_weight = 0.0
        else:
            self.root_weight = 1.0 / (self.ariety**height)

        self.children_hypercubes = []

        # do not create children if we have reached the desired level of
        # granularity
        if depth == 0:
            return

        # first, create partitions for each dimension
        partitions = []
        for d in range(self.root_hypercube.dimensions):
            boundary = self.root_hypercube[d]
            partitions_d = self._partition(boundary)
            partitions.append(partitions_d)

        # compute all possible combinations of partitions along the d dims.
        # each combination becomes the hypercube of a children
        combinations = itertools.product(
            list(range(self.num_splits)), repeat=self.root_hypercube.dimensions
        )

        # use the combinations to create all possible hypercube boundaries
        # each element of this list specifies a different hypercube
        children_boundaries = [
            [
                (partitions[dim][spl][0], partitions[dim][spl][1])
                for dim, spl in enumerate(c)
            ]
            for c in combinations
        ]

        for child in children_boundaries:
            self.children_hypercubes.append(
                WeightedSpacePartitioner(
                    num_splits=self.num_splits,
                    depth=self.depth - 1,
                    boundaries=child,
                    height=self.height + 1,
                )
            )

    def search(self,
               point: torch.Tensor) -> float:
        """Search the path to the leaf which represents the smallest hypercube
        represented by the data structure. When a node (hypercube) is visited
        for the first time, its weight is added to a counter. When the search
        is finished, the total weight, which represents the virtual "gains" of
        exploring that part of the subspace, is returned.

        Args:
            point (torch.Tensor): A torch tensor of d dimensions.

        Returns: 
            float: Total weight gained.
        """
        counter = 0.0
        if point not in self.root_hypercube:
            if self.height == 0.0:
                raise Exception(
                    "the point is out of boundaries, please double check your code"
                )
            return counter

        if self.height != 0 and not self.visited:
            counter += self.root_weight
            self.visited = True

        for child in self.children_hypercubes:
            counter += child.search(point)

        return counter

    @property
    def num_partitions(self) -> int:
        """Returns the number of multi-scale partitions created.
        """
        return (self.num_splits**self.root_hypercube.dimensions) ** self.depth

    @property
    def total_weight(self) -> int:
        """Returns the maximum possible weight compoundable by many searches.
        """
        return self.depth

    def _partition(self,
                   boundary: _Interval) -> List[Tuple[float, float]]:
        """Partition an interval into parts depending on the parameters of the
        class.

        Args:
            boundary (_Interval): The _Interval object.

        Returns: 
            List[Tuple[float, float]]: A list of boundaries that can be used to 
                                       initialize hypercubes.
        """
        min, max = boundary.min, boundary.max
        stepsize = (max - min) / self.num_splits

        boundaries = []
        for i in range(self.num_splits):
            if i < (self.num_splits - 1):
                boundaries.append((min + i * stepsize, min + (i + 1) * stepsize))
            else:
                # ensure we cover all space regardless of numerical errors
                boundaries.append((min + i * stepsize, max))
        return boundaries

    def __str__(self):
        root_str = ""

        num_tabs = self.height
        for _ in range(num_tabs):
            root_str += "\t"

        root_str = f"Root Hypercube: {self.root_hypercube} \n"

        children_str = ""

        for i in range(len(self.children_hypercubes)):
            for _ in range(num_tabs + 1):
                children_str += "\t"

            children_str += f"Child {i+1}: {str(self.children_hypercubes[i])} \n"

        return root_str + children_str
