import os
import torch.nn as nn

from src.models.s4d import S4Model

GRID = int(os.environ.get("GRID_LEN", "1024"))
# Time-resolved latent: N_TOKENS ordered temporal tokens, each TOKEN_DIM wide.
# N_TOKENS=1 reproduces the original single pooled 32-D latent (old checkpoints load).
N_TOKENS = int(os.environ.get("N_TOKENS", "1"))
TOKEN_DIM = int(os.environ.get("TOKEN_DIM", "32"))
LATENT_DIM = N_TOKENS * TOKEN_DIM              # flattened latent width == decoder input
D_MODEL = int(os.environ.get("S4_DMODEL", "256"))
N_LAYERS = int(os.environ.get("S4_NLAYERS", "4"))
DROPOUT = float(os.environ.get("S4_DROPOUT", "0.2"))


class SharedS4DSystematics(nn.Module):
    """One curve -> S4D time-resolved features (B,1024,256) -> split into n_tokens
    ordered temporal blocks -> masked mean per block -> shared Linear(256->token_dim)
    per token -> flatten ordered tokens (B, n_tokens*token_dim) -> shared MLP -> (B,1024)
    correction. Every curve uses the SAME encoder/projection/decoder (shared weights);
    single-curve inference works with no neighbours. n_tokens=1 == the original model."""

    def __init__(self, grid=GRID, n_tokens=N_TOKENS, token_dim=TOKEN_DIM,
                 d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.grid = int(grid)
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.latent_dim = self.n_tokens * self.token_dim
        if self.grid % self.n_tokens != 0:
            raise ValueError(f"grid {self.grid} not divisible by n_tokens {self.n_tokens}")

        # S4Model owns: encoder Linear(1->d_model), the S4D stack, the per-block masked
        # pooling (zero-token-safe via clamp(min=1)), and the shared Linear(d_model->token_dim)
        # projection applied to EVERY token. It returns ordered tokens (B, n_tokens, token_dim).
        self.encoder = S4Model(d_input=1, d_model=d_model, n_layers=n_layers, dropout=dropout,
                               n_tokens=self.n_tokens, token_dim=self.token_dim, readout="mean")

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.grid),
        )

    def encode(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        tokens = self.encoder(x, mask)                       # (B, n_tokens, token_dim)
        return tokens.reshape(x.shape[0], self.latent_dim)   # ordered flatten, tokens NOT averaged

    def forward(self, x, mask=None):
        latent = self.encode(x, mask)
        return self.decoder(latent), latent


def build_model(n_tokens=N_TOKENS, token_dim=TOKEN_DIM):
    return SharedS4DSystematics(n_tokens=n_tokens, token_dim=token_dim)


def preprocessing_config():
    return {"grid_len": GRID, "n_tokens": N_TOKENS, "token_dim": TOKEN_DIM, "latent_dim": LATENT_DIM,
            "d_model": D_MODEL, "n_layers": N_LAYERS, "dropout": DROPOUT,
            "input": "per-curve median/MAD-normalzied raw aperture flux on the shared 1024-cadence Sector 14 grid",
            "cleaned_curve": "x_clean = x - s_hat (normalized space)"}
