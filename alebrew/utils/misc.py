"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     misc.py 
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
import json
import pickle
from pathlib import Path
from typing import List, Union, Any, Dict, Tuple

import torch
import yaml
from yaml import Dumper, Loader


def save_object(filename: Union[str, Path],
                obj: Any,
                use_json: bool = False,
                use_yaml: bool = False):
    file = open(filename, 'w' if (use_json or use_yaml) else 'wb')
    if use_json:
        json.dump(obj, file)
    elif use_yaml:
        yaml.dump(obj, file, Dumper=Dumper)
    else:
        pickle.dump(obj, file, protocol=3)
    file.close()


def load_object(filename: Union[str, Path],
                use_json: bool = False,
                use_yaml: bool = False) -> Any:
    file = open(filename, 'r' if (use_json or use_yaml) else 'rb')
    if use_json:
        result = json.load(file)
    elif use_yaml:
        result = yaml.load(file, Loader=Loader)
    else:
        result = pickle.load(file)
    file.close()
    return result


def padded_str(strs: List[str],
               lens: List[int]) -> str:
    # strs should be a list of strings and lens should be a list of integers of the same length.
    # This function concatenates the strings in strs but fills them up with whitespaces at the end such that they have
    # (at least) the corresponding lengths
    result = ''
    for s, l in zip(strs, lens):
        result = result + s + (' ' * max(0, l-len(s)))
    return result


def create_header(file: str):
    f = open(file, 'a+')
    f.write('********************************************************************************************************** \n')
    f.write('   ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials    \n')
    f.write('   Please, include this reference in published work with ALEBREW: https://arxiv.org/abs/2312.01416v1       \n')
    f.write('********************************************************************************************************** \n')
    f.write(' \n')
    f.write(' \n')
    f.close()


def get_default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def get_available_devices():
    if torch.cuda.is_available():
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        return ['cpu']


def append_results(ds_results: List[Dict[str, List[torch.Tensor]]],
                   batch_results: List[Dict[str, torch.Tensor]]) -> List[Dict[str, List[torch.Tensor]]]:
    if not ds_results:
        return [{key: [val.detach()] for key, val in batch_r.items()} for batch_r in batch_results]
    else:
        return [{key: ds_r[key] + [val.detach()] for key, val in batch_r.items()} for ds_r, batch_r in zip(ds_results, batch_results)]


def cat_results(results: List[Dict[str, List[torch.Tensor]]]) -> List[Dict[str, torch.Tensor]]:
    return [{key: torch.cat(val) for key, val in r.items()} for r in results]


def recursive_detach(inputs: Any) -> Any:
    if isinstance(inputs, list):
        return [recursive_detach(input) for input in inputs]
    elif isinstance(inputs, dict):
        return {key: recursive_detach(value) for key, value in inputs.items()}
    return inputs.detach()


def recursive_cat(inputs: List,
                  dim: int):
    if isinstance(inputs[0], list):
        # we want to concatenate along the outer list, returning a list over the inner list dimension
        return [recursive_cat([input[i] for input in inputs], dim) for i in range(len(inputs[0]))]
    elif isinstance(inputs[0], dict):
        # we want to concatenate along the outer list, returning a dict with the "shape" of the inner dicts
        return {key: recursive_cat([input[key] for input in inputs], dim) for key in inputs[0].keys()}
    return torch.cat(inputs, dim=dim)


def harmonic_mean(input: torch.Tensor,
                  dim=-1) -> torch.Tensor:
    return 1.0 / ((1.0 / input).mean(dim))


class PickleStreamWriter:
    def __init__(self,
                 filename: Union[str, Path],
                 mode: str = 'wb'):
        self.f = open(filename, mode)  # or append or so

    def append(self, obj):
        pickle.dump(obj, self.f)

    def close(self):
        self.f.close()


def readPickleStream(filename: Union[str, Path]):
    objs = []
    f = open(filename, 'rb')
    try:
        while True:
            objs.append(pickle.load(f))
    except EOFError:
        f.close()
        return objs


def get_batch_intervals(n_total: int,
                        batch_size: int) -> List[Tuple[int, int]]:
    boundaries = [i * batch_size for i in range(1 + n_total // batch_size)]
    if boundaries[-1] != n_total:
        boundaries.append(n_total)
    return [(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])]
