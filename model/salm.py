# -*- coding: utf-8 -*-

"""Script for the Stochastic Augmented Lagrangian Method."""

import numpy as np
import torch
import typing

# Type aliases.
StoalmDict = typing.Dict[str, typing.Union[float, None]]
TupleDict = typing.Tuple[StoalmDict, typing.Dict[str, torch.Tensor]]

def stochatic_augmented_lagrangian(
    stoalm_dict: StoalmDict,
    loss_dict: typing.Dict[str, torch.Tensor],
    epoch: int,
    small_constant_1: float = 1.0e-5,
    small_constant_2: float = 1.0e-4,
    constant: float = 1.0e3,
) -> TupleDict:
    """Update the external parameters and losses using SALM.

    Args:
        stoalm_dict: parameter dictionary for the SALM.
        loss_dict: label loss and other regularization loss (e.g.
            "MMD", "associative", etc.)

    Returns:
        Updated parameter dictionary and loss dictionary.
    """
    for n in loss_dict.keys():
        org_loss = loss_dict[n]  # initial loss, which can be 'mmd', 'assoc', etc.
        c_theta = org_loss - max(small_constant_1, stoalm_dict['epsilon']/np.sqrt(epoch + 1))
        stoalm_dict['add_loss_func'].append(float(loss_dict[n]))  
        stoalm_dict['bv'].append(
            float(
                max(-stoalm_dict['lamb'][-1]/stoalm_dict['rho'][-1], c_theta)
            )
        )
        max_lambda = max(stoalm_dict['rho'][-1]*c_theta+stoalm_dict['lamb'][-1], small_constant_1)
        stoalm_dict['lamb'].append(
            float(
                min(stoalm_dict['lamb_max'], max_lambda)
            ) 
        )
        #  Update \rho or not.
        if len(stoalm_dict['bv']) == 1:
            stoalm_dict['rho'].append(float(stoalm_dict['rho'][-1])) 
        elif abs(stoalm_dict['bv'][-1]) <= stoalm_dict['tau'] * abs(stoalm_dict['bv'][-2]):
            stoalm_dict['rho'].append(
                max(small_constant_2, float(stoalm_dict['rho'][-1])/np.sqrt(stoalm_dict['gamma']))
            )
        else:
            stoalm_dict['rho'].append(
                min(constant, float(stoalm_dict['rho'][-1]*stoalm_dict['gamma']))
            )
        ## -------------------------------------------------------------------------------------
        # TODO: What is the loss_weight here? It wasn't defined at all.
        # -- loss weight, as loss is changed by 'stoalm'
        # loss_weights[n] = 1.0
        ## -------------------------------------------------------------------------------------

        # Update the additonal loss, override the MMD measure.
        # min() is used to constrain the absolute value of the addtitional term, 
        # to avoid to large additional loss term, which could mess the 'ce' loss.
        loss_dict[n] = stoalm_dict['rho'][-1]/2.0*torch.max(
                       torch.Tensor([0]),
                       (c_theta+stoalm_dict['lamb'][-1]/stoalm_dict['rho'][-1])
                    )**2
        if not isinstance (loss_dict[n], torch.Tensor):
            raise TypeError(f"'loss_dict[n]' has wrong type.")
    return stoalm_dict, loss_dict
