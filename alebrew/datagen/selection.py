"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     selection.py 
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

from typing import Optional, Union, List, Tuple, Callable, Any

from bmdal_reg.bmdal.selection import *
from bmdal_reg.bmdal.features import Features

from alebrew.data.data import AtomicStructures
from alebrew.model.calculators import UncertaintyAndPropertyCalculator
from alebrew.utils.torch_geometric import Data


class BaseSelection:
    """Selects a batch (N > 1) atomic structures at once."""
    def select_from_features(self,
                             pool_features: Features,
                             train_features: Features,
                             sel_batch_size: int) -> torch.Tensor:
        """Selects a batch using precomputed (gradient) features.

        Args:
            pool_features (Features): Features evaluated for the pool of candidates.
            train_features (Features): Features evaluated for the training data set.
            sel_batch_size (int): The size of the selected batch.

        Returns:
            torch.Tensor: Indices of the selected atomic structures.
        """
        raise NotImplementedError()

    def select_from_structures(self,
                               calc: UncertaintyAndPropertyCalculator,
                               pool_structures: AtomicStructures,
                               train_structures: AtomicStructures,
                               sel_batch_size: int,
                               r_cutoff: float,
                               eval_batch_size: int = 32,
                               retain_neighbors: bool = True) -> torch.Tensor:
        """Selects a batch using atomic structures, with features computed explicitly. 
        Allows setting retain_neighbors=False, if RAM is overloaded.

        Args:
            calc (UncertaintyAndPropertyCalculator): `Calculator` which supports the calculation of features.
            pool_structures (AtomicStructures): Atomic structures from the pool of candidates.
            train_structures (AtomicStructures): Atomic structures from the training data.
            sel_batch_size (int): The size of the selected batch.
            r_cutoff (float): Cutoff radius for computing the neighbor list.
            eval_batch_size (int, optional): Batch size used during evaluation. Defaults to 32.
            retain_neighbors (bool, optional): If True, neighbors are retained after calculation. It might lead 
                                               to overloading the RAM. Defaults to True.

        Returns:
            torch.Tensor: Indices of the selected atomic structures.
        """
        assert 'features' in calc.features_keys
        # allows to set retain_neighbors to False in case RAM is overloaded by neighbor list calculation
        pool_features = calc.get_tfm_features_on_structures(pool_structures, r_cutoff=r_cutoff,
                                                            eval_batch_size=eval_batch_size,
                                                            retain_neighbors=retain_neighbors)
        train_features = calc.get_tfm_features_on_structures(train_structures, r_cutoff=r_cutoff,
                                                             eval_batch_size=eval_batch_size,
                                                             retain_neighbors=retain_neighbors)
        return self.select_from_features(pool_features['tfm_features'], train_features['tfm_features'], sel_batch_size)

    def select_from_ds(self,
                       calc: UncertaintyAndPropertyCalculator,
                       pool_ds: List[Data],
                       train_ds: List[Data],
                       sel_batch_size: int,
                       eval_batch_size: int = 32) -> torch.Tensor:
        """Selects a batch using atomic data, with features computed explicitly.

        Args:
            calc (UncertaintyAndPropertyCalculator): `Calculator` which supports the calculation of features.
            pool_ds (List[Data]): Atomic data from the pool of candidates.
            train_ds (List[Data]): Atomic data from the training data.
            sel_batch_size (int): The size of the selected batch.
            eval_batch_size (int, optional): Batch size used during evaluation. Defaults to 32.

        Returns:
            torch.Tensor: Indices of the selected atomic structures.
        """
        assert 'features' in calc.features_keys
        pool_features = calc.get_tfm_features_on_ds(pool_ds, eval_batch_size=eval_batch_size)
        train_features = calc.get_tfm_features_on_ds(train_ds, eval_batch_size=eval_batch_size)
        return self.select_from_features(pool_features['tfm_features'], train_features['tfm_features'], sel_batch_size)


class RandomSelection(BaseSelection):
    """Selects candidates randomly."""
    def select_from_features(self,
                             pool_features: Features,
                             train_features: Features,
                             sel_batch_size: int) -> torch.Tensor:
        alg = RandomSelectionMethod(pool_features, verbosity=0)
        batch_idxs = alg.select(sel_batch_size)
        return batch_idxs


class MaxDiagSelection(BaseSelection):
    """Greedily selects candidates with the largest uncertainty."""
    def select_from_features(self,
                             pool_features: Features,
                             train_features: Features,
                             sel_batch_size: int) -> torch.Tensor:
        alg = MaxDiagSelectionMethod(pool_features, verbosity=0)
        batch_idxs = alg.select(sel_batch_size)
        return batch_idxs


class MaxDetSelection(BaseSelection):
    """Greedily selects candidates that maximize the determinant of the covariance matrix.

    Args:
        sel_with_train (bool, optional): If True, features evaluated for the training data set
                                         are used along with those evaluated for the pool of 
                                         candidates. Defaults to False.
        noise_sigma (float, optional): `noise_sigma ** 2` is added to the kernel diagonal for 
                                        the determinant maximization. Defaults to 0.0.
    """
    def __init__(self,
                 sel_with_train: bool = False,
                 noise_sigma: float = 0.0):
        self.sel_with_train = sel_with_train
        self.noise_sigma = noise_sigma

    def select_from_features(self,
                             pool_features: Features,
                             train_features: Features,
                             sel_batch_size: int) -> torch.Tensor:
        alg = MaxDetSelectionMethod(pool_features, train_features=train_features if self.sel_with_train else None,
                                    sel_with_train=self.sel_with_train, noise_sigma=self.noise_sigma)
        batch_idxs = alg.select(sel_batch_size)
        return batch_idxs


class MaxDistSelection(BaseSelection):
    """Greedily selects candidates with the maximal distance to the previously selected points 
    (also to the training data, for sel_with_train=True).

    Args:
        sel_with_train (bool, optional): If True, features evaluated for the training data set are 
                                         used along with those evaluated for the pool of candidates. 
                                         Defaults to True.
    """
    def __init__(self, sel_with_train: bool = True):
        self.sel_with_train = sel_with_train

    def select_from_features(self,
                             pool_features: Features,
                             train_features: Features,
                             sel_batch_size: int) -> torch.Tensor:
        alg = MaxDistSelectionMethod(pool_features, train_features=train_features if self.sel_with_train else None,
                                     sel_with_train=self.sel_with_train, verbosity=0)
        batch_idxs = alg.select(sel_batch_size)
        return batch_idxs


class LargestClusterMaxDistSelection(BaseSelection):
    """Greedily selects candidates with the maximal distance within the largest cluster 
    (including training points, for sel_with_train=True).

    Args:
        sel_with_train (bool, optional): If True, features evaluated for the training data set are 
                                         used along with those evaluated for the pool of candidates. 
                                         Defaults to True.
    """
    def __init__(self, sel_with_train: bool = True):
        self.sel_with_train = sel_with_train

    def select_from_features(self,
                             pool_features: Features,
                             train_features: Features,
                             sel_batch_size: int) -> torch.Tensor:
        alg = LargestClusterMaxDistSelectionMethod(pool_features,
                                                   train_features=train_features if self.sel_with_train else None,
                                                   sel_with_train=self.sel_with_train, verbosity=0)
        batch_idxs = alg.select(sel_batch_size)
        return batch_idxs


class BatchSizeScaling:
    """Increases the selected batch size with, e.g., the growing training data set size.

    Args:
        min_sel_batch_size (int, optional): Minimal size of the selected batch. Defaults to 1.
        max_sel_batch_size (int, optional): Maximal size of the selected batch. Defaults to 1024.
        max_sampled_structures (Optional[int], optional): Maximal length of the sampled trajectory. 
                                                          Defaults to None.
        scale_sel_batch_size (float, optional): Scaling factor by which the size of the selected batch grows. 
                                                Defaults to 1.0.
    """
    def __init__(self,
                 min_sel_batch_size: int = 1,
                 max_sel_batch_size: int = 1024,
                 max_sampled_structures: Optional[int] = None,
                 scale_sel_batch_size: float = 1.0,
                 **config: Any):
        self.min_sel_batch_size = min_sel_batch_size
        self.max_sel_batch_size = max_sel_batch_size
        if max_sampled_structures is not None:
            # linearly growing batch size with the number of sampled structures
            self.f = lambda x, _: self.max_sel_batch_size / max_sampled_structures * x
        else:
            # growing batch size with the number of labeled structures (exponentially, for scale_sel_batch_size=1.0)
            self.f = lambda _, x: scale_sel_batch_size * x

    def __call__(self,
                 n_samples: int,
                 n_labeled: int) -> int:
        """Provides the new selected batch size.

        Args:
            n_samples (int): The number of candidates.
            n_labeled (int): The number of labeled atomic structures.

        Returns:
            int: New selected batch size.
        """
        return min(min(max(int(self.f(n_samples, n_labeled)), self.min_sel_batch_size), n_samples),
                   min(self.max_sel_batch_size, n_labeled))


def get_sel_method(sel_method: str) -> BaseSelection:
    """Provides the selection method by name.

    Args:
        sel_method (str): String with the selection method name. 
                          Possible names: 'random', 'max_diag', 
                          'max_det', 'max_dist', and 'lcmd'.


    Returns:
        BaseSelection: Batch selection method.
    """
    if sel_method == 'random':
        return RandomSelection()
    elif sel_method == 'max_diag':
        return MaxDiagSelection()
    elif sel_method == 'max_det':
        return MaxDetSelection()
    elif sel_method == 'max_dist':
        return MaxDistSelection(sel_with_train=True)
    elif sel_method == 'lcmd':
        return LargestClusterMaxDistSelection(sel_with_train=True)
    else:
        raise NotImplementedError(f'{sel_method=} is not implemented yet.')
