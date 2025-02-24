import torch
import torch.nn.functional as F


def reduce(x, reduction="none"):
    if reduction == "none":
        return x
    elif reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    else:
        raise TypeError("invalid reduction: {}".format(reduction))


def kl_divergence_loss(
    mu1: torch.Tensor,
    var1: torch.Tensor,
    mu2: torch.Tensor,
    var2: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    # mu1 = F.normalize(mu1, p=2, dim=1)
    # mu2 = F.normalize(mu2, p=2, dim=1)
    # var1 = F.normalize(var1, p=2, dim=1)
    # var2 = F.normalize(var2, p=2, dim=1)
    kl = 0.5 * (var2.log() - var1.log() + (var1 + (mu1 - mu2) ** 2) / var2 - 1)
    kl = kl.sum(dim=1)
    return reduce(kl, reduction)


def compute_cosine_similarity(tensor):
    # Normalize the input tensor along the last dimension (d)
    normalized_tensor = F.normalize(tensor, p=2, dim=1)

    # Compute the dot product between each pair of normalized vectors
    similarity_matrix = torch.matmul(normalized_tensor, normalized_tensor.t())

    return similarity_matrix
