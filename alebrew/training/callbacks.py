"""
       ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials
	  
  File:     callbacks.py 
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
from typing import Dict

from alebrew.utils.misc import create_header, padded_str


class TrainingCallback:
    """Generates training callbacks."""
    def before_fit(self, trainer: 'Trainer'):
        """Callbacks before training.

        Args:
            trainer (Trainer): The `Trainer` object which performs training of the model.
        """
        raise NotImplementedError()

    def after_epoch(self,
                    trainer: 'Trainer',
                    train_avg_metrics: Dict[str, float],
                    avg_metrics: Dict[str, float],
                    epoch_time: float):
        """Callbacks after a single epoch.

        Args:
            trainer (Trainer): The `Trainer` object which performs training of the model.
            train_avg_metrics (Dict[str, float]): Dictionary of running average error metrics evaluated 
                                                  on the training data set.
            avg_metrics (Dict[str, float]): Dictionary of running average error metrics evaluated 
                                            on the validation data set.
            epoch_time (float): Runtime of the epoch.
        """
        raise NotImplementedError()

    def after_fit(self,
                  trainer: 'Trainer',
                  session_time: float):
        """Callback after training.

        Args:
            trainer (Trainer): The `Trainer` object which performs training of the model.
            session_time (float): The total runtime for training the model.
        """
        raise NotImplementedError()


class FileLoggingCallback(TrainingCallback):
    """Generates file logging callbacks."""
    def __init__(self):
        self.train_out = None
        self.column_widths = None

        self.test_out = None
        self.test_column_widths = None

    def before_fit(self, trainer: 'Trainer'):
        # define files to save the progress of training
        self.train_out = os.path.join(trainer.model_path, 'train.out')

        if trainer.epoch > 0:
            # restored checkpoint
            f = open(self.train_out, "a+")
            f.write('============= \n')
            f.write('Training is restarted from epoch {} \n'.format(trainer.epoch))

        else:
            # start new session
            create_header(self.train_out)
            f = open(self.train_out, "a+")

        headings = [metric + ' (train/valid/best_valid)' for metric in trainer.eval_loss_fns]
        # last column does not need to be whitespace-padded because it does not matter visually
        self.column_widths = [17] + [len(heading) + 2 for heading in headings] + [0]
        headings = ['Epoch'] + headings + ['Time']

        f.write('Best checkpoints for model 0 can be found in           '
                '............. {} \n'.format(trainer.best_dirs[0]))
        f.write('Checkpoints for restart for model 0 can be found in    '
                '............. {} \n'.format(trainer.log_dirs[0]))
        f.write(' \n')
        f.write(padded_str(headings, self.column_widths) + '\n')
        f.write("".ljust(sum(self.column_widths) + 9, "=") + "\n")
        f.close()

    def after_epoch(self,
                    trainer: 'Trainer',
                    train_avg_metrics: Dict[str, float],
                    avg_metrics: Dict[str, float],
                    epoch_time: float):
        f = open(self.train_out, "a+")
        strs = [f'Epoch {trainer.epoch}/{trainer.max_epoch}: ']
        for metric_name in train_avg_metrics:
            vals = [train_avg_metrics[metric_name], avg_metrics[metric_name],
                    trainer.best_avg_metrics[metric_name]]
            strs.append('/'.join([f'{val:6.3f}' for val in vals]) + ' ')
        strs.append(f'[{epoch_time:5.2f} s]')
        f.write(padded_str(strs, self.column_widths) + '\n')
        f.close()

    def after_fit(self,
                  trainer: 'Trainer',
                  session_time: float):
        f = open(self.train_out, "a+")
        f.write("".ljust(sum(self.column_widths) + 9, "=") + "\n")
        f.write('Timing report \n')
        f.write("".ljust(13, "-") + "\n")
        f.write(f"Total time                    ............. {session_time:g} s \n")
        f.write('Best model report \n')
        f.write("".ljust(17, "-") + "\n")
        f.write(f'Best epochs from the training  ............. {trainer.best_epochs} \n')
        f.close()
