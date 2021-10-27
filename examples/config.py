
# from salad packege, !!!!! todos
""" Experiment Configurations for ``salad``

This file contains classes to easily configure experiments for different solvers
available in ``salad``.
"""


import sys
import argparse

class BaseConfig(argparse.ArgumentParser):

    """ Basic configuration with arguments for most deep learning experiments
    """
    
    def __init__(self, description, log = './log'):
        super().__init__(description=description)

        self.log = log
        self._init()


    def _init(self):

        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
        # ---
        self.add_argument('--gpu', default=0,
            help='Specify GPU', type=int)
        self.add_argument('--cpu', action='store_true',
            help="Use CPU Training")
        self.add_argument('--njobs', default=4,
            help='Number of processes per dataloader', type=int)
        self.add_argument('--log', default=self.log,
            help="Log directory. Will be created if non-existing")
        self.add_argument('--epochs', default="100",
            help="Number of Epochs (Full passes through the unsupervised training set)", type=int)
        self.add_argument('--checkpoint', default="",
            help="Checkpoint path")
        self.add_argument('--learningrate', default=1e-3, type=float,
            help="Learning rate for Adam. Defaults to Karpathy's constant ;-)")
        self.add_argument('--dryrun', action='store_true',
            help="Perform a test run, without actually training a network.")
        self.add_argument('--save_frequency', default=1, type=int,
            help="save_frequency for model saving, default 1")
        # ---
        # --- for sto ALM
        self.add_argument('--fix_weight', default=True, type=str2bool,
            help="Fix the weight of additional loss function of not.")
        self.add_argument('--loss_func_weight', default=1.0, type=float,
            help="Weight of additional loss function.")
        self.add_argument('--stoalm_optim', default='mmd', type=str,
            help="Loss function for the stoalm optimizer.")
        self.add_argument('--org_optim', default='adam', type=str,
            help="Loss function for the stoalm optimizer.")
        self.add_argument('--lamb', default=10.0, type=float,
            help="Initial value for lambda of stoalm.")
        self.add_argument('--lamb_max', default=1.0e20, type=float,
            help="Initial value for max lambda of stoalm.")
        self.add_argument('--rho', default=1.0, type=float,
            help="Initial value for rho of stoalm.")
        self.add_argument('--epsilon', default=10.0, type=float,
            help="Initial value for epsilon of stoalm.")
        self.add_argument('--sigma', default=1.0e-4, type=float,
            help="Initial value for sigma of stoalm.")
        self.add_argument('--tau', default=0.9, type=float,
            help="Initial value for tau of stoalm.")
        self.add_argument('--gamma', default=1.01, type=float,
            help="Initial value for gamma of stoalm.")
        self.add_argument('--subfolder', default='new_bigv_lambda1', type=str,
            help="Subfolder for storing results under log.")
        
        
        
        
        

    def print_config(self):
        print("Start Experiments")


class DomainAdaptConfig(BaseConfig):
    """ Base Configuration for Unsupervised Domain Adaptation Experiments
    """

    def _init(self):
        super()._init()

        self.add_argument('--source', default="svhn", choices=['mnist', 'svhn', 'usps', 'synth', 'synth-small'],
                            help="Source Dataset. Choose mnist or svhn")
        self.add_argument('--target', default="mnist", choices=['mnist', 'svhn', 'usps', 'synth', 'synth-small'],
                            help="Target Dataset. Choose mnist or svhn")
        self.add_argument('--sourcebatch', default=128, type=int,
                            help="Batch size of Source")
        self.add_argument('--targetbatch', default=128, type=int,
                            help="Batch size of Target")