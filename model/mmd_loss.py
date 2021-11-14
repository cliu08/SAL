"""Script for calculating the MMD loss."""

import torch

def gaussian_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the Gaussian kernel between two tensors.

    Args:
        a: input tensor.
        b: input tensor.

    Returns:
        Gaussian kernel.
    """
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the MMD loss.

    Args:
        a: input tensor.
        b: input tensor.

    Returns:
        Maximum mean discrepancy.
    """
    gau_a = gaussian_kernel(a, a).mean()
    gau_b = gaussian_kernel(b, b).mean()
    gau_ab = gaussian_kernel(a, b).mean()
    mmd_loss = gau_a + gau_b - 2*gau_ab
    return mmd_loss
