import torch
import torch.nn as nn
from einops import rearrange


'''
Just took this all from the github that I found and am using it now
'''

def stochastic_depth(input: torch.tensor, p: float, mode: str, training: bool = True):
    if p < 0.0 or p > 1.0:
        raise ValueError("drop probability has to be between 0 and 1, but got {}".format(p))
    if mode not in ["batch", "row"]:
        raise ValueError("mode has to be either 'batch' or 'row', but got {}".format(mode))
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise

class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """
    def __init__(self, p: float, mode: str) -> None:
        # torchvision.ops.StochasticDepth could replace this in torchvision>=0.11
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'p=' + str(self.p)
        tmpstr += ', mode=' + str(self.mode)
        tmpstr += ')'
        return tmpstr

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X