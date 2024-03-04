"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     features.py 
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

from alebrew.nn.layers import LinearLayer


class FeatureComputation(nn.Module):
    """Computes last-layer or sketched gradient features.

    Args:
        model (nn.Module): Models for which last-layer or sketched gradient features are computed. 
                           Currently MLPs are supported.
        use_grad_features (bool): If True, sketched gradient features are computed.
        n_random_projections (int, optional): The number of random projection. If -1, 
                                              last-layer features are computed. Defaults to -1.
        seed (int, optional): Random seed. Defaults to 1234.
    """
    def __init__(self,
                 model: nn.Module,
                 use_grad_features: bool,
                 n_random_projections: int = -1,
                 seed: int = 1234,
                 **config: Any):
        assert not use_grad_features or n_random_projections >= 1

        super().__init__()
        self.use_grad_features = use_grad_features
        self.random_projections = None
        self.n_random_projections = n_random_projections
        # don't use nn.ModuleList since we don't want them to be registered here
        self.linear_layers = [m for m in model.modules() if isinstance(m, LinearLayer)]

        if not self.use_grad_features:
            self.linear_layers = self.linear_layers[-1:]
        if n_random_projections > 0:
            generator = torch.Generator('cpu')
            generator.manual_seed(seed)
            self.random_projections = nn.ModuleList([nn.ParameterList(
                [nn.Parameter(torch.randn(sz, n_random_projections, generator=generator), requires_grad=False)
                                        for sz in [l.in_features+1, l.out_features]]) for l in self.linear_layers])

        self.input_hooks = [None] * len(self.linear_layers)
        self.grad_output_hooks = [None] * len(self.linear_layers)
        self.inputs = [None] * len(self.linear_layers)
        # self.outputs = [None] * len(self.linear_layers)
        self.grad_outputs = [None] * len(self.linear_layers)

    def set_input_(self,
                   inp: torch.Tensor,
                   out: torch.Tensor,
                   layer_idx: int):
        # this is used to have a method to call in the hooks
        self.grad_output_hooks[layer_idx] = out.register_hook(lambda grad, idx=layer_idx, s=self:
                                                              s.set_grad_output_(grad, layer_idx))
        self.inputs[layer_idx] = inp
        # self.outputs[layer_idx] = out

    def set_grad_output_(self,
                         x: torch.Tensor,
                         layer_idx: int):
        # this is used to have a method to call in the hooks
        self.grad_outputs[layer_idx] = x

    def before_forward(self):
        """Callback that is called before the data is passed through the model. It is used to set up hooks 
        that grab input data.
        """
        # sets up hooks that store the input
        self.input_hooks = [l.register_forward_hook(lambda layer, inp, output, s=self, idx=i:
                                                    s.set_input_(inp[0], output, idx))
                            for i, l in enumerate(self.linear_layers)]
        # self.grad_output_hooks = [l.register_full_backward_hook(lambda layer, grad_input, grad_output, s=self, idx=i:
        #                                                         s.set_grad_output_(grad_output[0], idx))
        #                           for i, l in enumerate(self.linear_layers)]

    def pop_features(self, y: torch.Tensor) -> torch.Tensor:
        """Computes last-layer or sketched gradient features.

        Args:
            y (torch.Tensor): Input tensor, e.g., atomic energies in our case.

        Returns:
            torch.Tensor: Last-layer or sketched gradient features.
        """
        # TODO: could use self.outputs[0], that would be a small optimization
        _ = torch.autograd.grad(y, self.inputs[0], torch.ones_like(y), create_graph=True)[0]

        for i in range(len(self.linear_layers)):
            self.input_hooks[i].remove()
            self.input_hooks[i] = None
            self.grad_output_hooks[i].remove()
            self.grad_output_hooks[i] = None

        ext_inputs = [torch.cat([l.weight_factor * inp,
                                 l.bias_factor * torch.ones(inp.shape[0], 1, device=inp.device)], dim=1)
                      for inp, l in zip(self.inputs, self.linear_layers)]

        if self.n_random_projections > 0:
            results = []
            for i in range(len(self.linear_layers)):
                if self.grad_outputs[i].shape[-1] == 1:
                    # don't apply random projection to grad_output since it is one-dimensional
                    results.append((self.grad_outputs[i] * ext_inputs[i]) @ self.random_projections[i][0])
                else:
                    results.append((ext_inputs[i] @ self.random_projections[i][0])
                                   * (self.grad_outputs[i] @ self.random_projections[i][1]))
            result = sum(results)
        else:
            # must be in the last-layer case
            result = self.grad_outputs[0] * ext_inputs[0]

        self.inputs = [None] * len(self.linear_layers)
        # self.outputs = [None] * len(self.linear_layers)
        self.grad_outputs = [None] * len(self.linear_layers)

        return result
