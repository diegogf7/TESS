import torch
import torch.nn as nn
class Predictor(nn.Module):
    def __init__(self, n_tokens = 4, token_dimension = 4, hidden = 256):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dimension = token_dimension

        flat_dimension = n_tokens * token_dimension
        
        self.net = nn.Sequential(
            nn.Linear(flat_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, flat_dimension)
        )
    
    def forward(self, context):
        B = context.shape[0]

        x = context.reshape(B, self.n_tokens * self.token_dimension)
        x = self.net(x)
        x = x.reshape(B, self.n_tokens, self.token_dimension)

        return x