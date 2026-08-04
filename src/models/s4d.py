import math
import torch
import torch.nn as nn
from einops import repeat

from src.models.dropout import DropoutNd


'''
Implementing a state space model
- The equations that we are given/start with: 
- x'(t) = A * x(t) + B * u(t) <-- states evolve on itself and input
- y(t) = C * x(t) + D * u(t) <-- output reads off state 
- After doing a time step
- x(t) = A * x[t-1] + B * u[t] <-- recurrence?
- y(t) = C * x[t] + D * u[t] 

'''


class S4DKernel(nn.Module):
    def __init__(self, d_model, N = 64, dt_min = 0.001, dt_max = 0.1, lr = None):
        super().__init__()
        H = d_model
        
        #Implementing with the same learning rate


        '''
        -There are three variables that we are dealing with
        dynamics (A), input (B), and output (C)

        - A is a matrix exponential and we want to make it diagonal
        - log_A_real is negative and keeps system stable
        - log_dt are the time steps
        - C is the complex output matrix with size (H, N//2)
        
        '''
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min) #are we just storing the log because it must be positive?
        #Just giving us a range for the time steps 

        C = torch.randn(H, N // 2, dtype = torch.cfloat) #gives the shape of the matrix
        self.C = nn.Parameter(torch.view_as_real(C)) 

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2)) #Controlling the real part of A
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h = H) #controlling the imaginary part of A
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag) #storing these variables as tensors that need to be updated
    
    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)


        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)

        #Vandermonde multiplication using code from given github
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device = A.device)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K
    
    def register(self, name, tensor, lr = None):
        if lr == 0.0:
            self.register_buffer(name, tensor)

        else:
            self.register_parameter(name, nn.Parameter(tensor))


            optim = {"weight_decay": 0.0}
            if lr is not None: 
                optim["lr"] = lr
            
            setattr(getattr(self, name), "_optim", optim)

class S4D(nn.Module):
    def __init__(self, d_model, d_state = 64, dropout = 0.0, transposed = True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed


        self.D = nn.Parameter(torch.randn(self.h))

        self.kernel = S4DKernel(self.h, N = self.n, **kernel_args)
        self.activation = nn.GELU() #need to use GELU since we have a deeper learning model not just going to use RELU

        dropout_function = DropoutNd
        self.dropout = dropout_function(dropout) if dropout > 0.0 else nn.Identity()

        self.output_linear = nn.Sequential(

            nn.Conv1d(self.h, 2*self.h, kernel_size = 1),
            nn.GLU(dim =- 2)
        )


    def forward(self, u, **kwargs):

        #need to check if we have data that is in the right orientation
        if not self.transposed:
            u = u.transpose(-1, -2)

        L = u.size(-1) #need to build a matrix of size from the last dimension

        #Need to get our kernel
        Kernel = self.kernel(L = L)
        #need to run convolution

        k_f = torch.fft.rfft(Kernel, n = 2 * L)
        u_f = torch.fft.rfft(u, n = 2 * L)
        y = torch.fft.irfft(k_f * u_f, n = 2 * L)[..., :L]

        #determining D in our state space equation
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed: 
            y = y.transpose(-1, -2)
        return y, None 

def masked_token_pool(x, mask, n_tokens, with_std=False):
    """Split the time axis of x (B, L, D) into n_tokens ORDERED consecutive blocks of
    L//n_tokens cadences and take the MASKED mean within each block -> (B, n_tokens, D).
    A fully-masked (all missing) block returns a ZERO token, never NaN, because the
    denominator is clamped to >=1. Missing cadences never contribute to a token. With
    with_std=True the per-block std is concatenated (mean_std readout)."""
    B, L, D = x.shape
    N = int(n_tokens)
    xr = x.reshape(B, N, L // N, D)                       # (B, N, block, D) -- consecutive blocks
    if mask is not None:
        mf = mask.reshape(B, N, L // N, 1).to(x.dtype)
        denom = mf.sum(dim=2).clamp(min=1)               # empty block -> denom 1, numerator 0 -> zero token
        mean = (xr * mf).sum(dim=2) / denom
        if with_std:
            var = (((xr - mean.unsqueeze(2)) ** 2) * mf).sum(dim=2) / denom
            return torch.cat([mean, (var + 1e-6).sqrt()], dim=-1)
        return mean
    mean = xr.mean(dim=2)
    if with_std:
        return torch.cat([mean, xr.std(dim=2)], dim=-1)
    return mean


class S4Model(nn.Module):
    def __init__(
            self, 
            d_input,
            d_output = 10,
            d_model = 256,
            d_state = 64,
            n_layers = 4,
            dropout = 0.2,
            prenorm = False,
            lr = None,
            dt_min = 0.001,
            dt_max = 0.1,
            n_tokens = 4,
            token_dim = 16,
            readout = "mean"
    ):

        super().__init__()

        self.prenorm = prenorm
        self.readout = readout

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers= nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, d_state, dropout, True, dt_min = dt_min, dt_max = dt_max, lr = lr)
            )

            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        

        self.n_tokens = n_tokens
        # "mean_std" concatenates per-segment mean AND std before the decoder, so
        # sub-segment oscillation amplitude (high-freq class signal that the mean
        # averages to ~0) survives the pooling. "mean" = original behaviour.
        pool_dim = d_model * (2 if readout == "mean_std" else 1)
        self.decoder = nn.Linear(pool_dim, token_dim)

    def forward(self, x, mask = None):
        #not sure how to implement this properly

        x = self.encoder(x)

        x = x.transpose(-1, -2)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):

            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1,-2)

            z, _ = layer(z)
            z = dropout(z)
            x = z + x

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1,-2)
        
        x = x.transpose(-1,-2)                                    # (B, L, D) time-resolved features

        pooled = masked_token_pool(x, mask, self.n_tokens,
                                   with_std=(self.readout == "mean_std"))
        tokens = self.decoder(pooled)                            # shared Linear(d_model -> token_dim) per token
        return tokens

        
class S4Classifier(nn.Module):

    def __init__(self, d_input = 1, d_model = 256, n_layers = 4, dropout = 0.2, n_tokens = 4, token_dim = 16, n_classes = 8, readout = "mean"):
        super().__init__()
        self.backbone = S4Model(d_input =d_input, d_model = d_model, n_layers = n_layers, dropout = dropout, n_tokens = n_tokens, token_dim = token_dim, readout = readout)

        self.head = nn.Linear(n_tokens * token_dim, n_classes)
    
    def forward(self, x, mask = None):

        tokens = self.backbone(x, mask)
        z = tokens.flatten(1)
        return self.head(z)