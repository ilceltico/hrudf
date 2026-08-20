from math import sqrt

from torch import nn

from .deepsdf import DeepSDF

def get_model(network, **kwargs):
    if network.lower() == "deepsdf":
        return DeepSDF(**kwargs)
    else:
        raise NotImplementedError(f"Unkown model \"{network}\"")


def get_latents(n_shapes, dim, max_norm=None):
    """Create and initialize latent vectors as embeddings."""
    latents = nn.Embedding(n_shapes, dim, max_norm=max_norm).cuda()
    nn.init.normal_(latents.weight.data, 0., 1 / sqrt(dim))
    return latents