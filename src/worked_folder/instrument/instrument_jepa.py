import os


import torch
import torch.nn as nn
import numpy as np
from src.models.s4d import S4Model

import torch.nn.functional as Functional
from src.worked_folder.physics.latent_jepa import LatentPredictor

class InstrumentJEPA(nn.Module):
    def __init__(
            self,
            n_tokens = 16,
            token_dimension = 16,
            d_model = 256,
            n_layers = 4,
            dropout = 0.2,
            momentum = 0.996,
            readout = "mean",
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.momentum = momentum
        self.context_encoder = S4Model(
            d_input =1, d_model =d_model, n_layers = n_layers, dropout = dropout, n_tokens = n_tokens, token_dim = token_dimension, readout = readout
        )

        self.target_encoder = S4Model(
            d_input =1, d_model =d_model, n_layers = n_layers, dropout = dropout, n_tokens = n_tokens, token_dim = token_dimension, readout = readout
        )

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = LatentPredictor(n_tokens = n_tokens, token_dimension = token_dimension)

    def forward(self, context_flux, context_mask, target_flux, target_mask):
        context_tokens = self.context_encoder(context_flux.unsqueeze(-1), context_mask)
        prediction = self.predictor(context_tokens)

        with torch.no_grad():

            target = self.target_encoder(target_flux.unsqueeze(-1), target_mask)
            target = Functional.layer_norm(target, (target.shape[-1],))

        return prediction, target
    

    @torch.no_grad()
    def encode(self, flux, observed_mask = None):

        return self.target_encoder(flux.unsqueeze(-1), observed_mask)
    
    @torch.no_grad()
    def update_target(self):
        m = self.momentum

        for online_p, target_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data = m * target_p.data + (1.0 - m) * online_p.data

def instrument_jepa_loss(prediction, target, target_mask = None, var_weight = 0.0):

    per_token = Functional.smooth_l1_loss(prediction, target, reduction = "none").mean(dim = -1)

    if target_mask is None:

        loss = per_token.mean()
    
    else:
        B, N = per_token.shape
        observed = target_mask.reshape(B, N, -1).mean(dim = 2)

        loss = (per_token * observed).sum() / observed.sum().clamp(min = 1e-6)

    if var_weight > 0.0:
        flat = prediction.reshape(-1, prediction.shape[-1])

        standard_deviation = flat.std(dim = 0)

        loss = loss + var_weight * torch.relu(1.0 - standard_deviation).mean()
        

    return loss



def build_instrument_jepa():

    return InstrumentJEPA(
        n_tokens = int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dimension = int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model = 256, n_layers = 4, dropout = 0.2, momentum = 0.996,
        readout = os.environ.get("JEPA_READOUT", "mean")
    )

