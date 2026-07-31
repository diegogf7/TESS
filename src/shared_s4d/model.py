import os
import torch.nn as nn

from src.models.s4d import S4Model

GRID = int(os.environ.get("GRID_LEN", "1024"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
D_MODEL = int(os.environ.get("S4_DMODEL", "256"))
N_LAYERS = int(os.environ.get("S4_NLAYERS", "4"))
DROPOUT = float(os.environ.get("S4_DROPOUT", "0.2"))

class SharedS4DSystematics(nn.Module):

    def __init__(self, grid = GRID, latent_dim = LATENT_DIM, d_model = D_MODEL, n_layers = N_LAYERS, dropout = DROPOUT):
        super().__init__()
        self.grid = int(grid)
        self.latent_dim = int(latent_dim)


        self.encoder = S4Model(d_input = 1, d_model = d_model, n_layers =n_layers, dropout = dropout, n_tokens = 1, token_dim = latent_dim, readout = "mean")

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.grid)
        )

    
    def encode(self, x, mask = None):
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        tokens = self.encoder(x, mask)
        return tokens.reshape(x.shape[0], self.latent_dim)
    
    def forward(self, x, mask = None):

        latent = self.encode(x, mask)
        return self.decoder(latent), latent
    
def build_model():
    return SharedS4DSystematics()


def preprocessing_config():
    return {"grid_len": GRID, "latent_dim": LATENT_DIM, "d_model": D_MODEL,
            "n_layers": N_LAYERS, "dropout": DROPOUT, "input": "per-curve median/MAD-normalzied raw aperture flux on the shared 1024-cadence Sector 14 grid",
            "cleaned_curve": "x_clean = x - s_hat (normalized space)"
            }