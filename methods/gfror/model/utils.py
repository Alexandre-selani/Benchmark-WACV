"""
    Tensor helpers used by the GFROR autoencoder pipeline.

    Source of the functions kept here:
        clamp_to_unit_sphere, to_img, to_4d
        https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/vector.py

    This is the subset of the original `model/utils.py` that the TinyImageNet
    path actually needs. The Wide-ResNet parameter helpers (cast, conv_params,
    linear_params, bnparams, data_parallel, flatten, batch_norm,
    print_tensor_dict), taken from Sergey Zagoruyko's wide-residual-networks
    (http://arxiv.org/abs/1605.07146), were dropped together with `wrn28-10.py`,
    which also removes the `nested_dict` dependency from the benchmark.
"""

import torch


def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace = subspace / norm.expand(1, -1).t()  # + epsilon
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)


def to_img(x, size=32):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, size, size)
    return x


def to_4d(x):
    return x.unsqueeze(-1).unsqueeze(-1)
