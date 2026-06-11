import torch
import torch.nn as nn

from s4d import S4Model


#going to define our physics model and then our instrument model
class PhysicsEncoder(nn.Module):
    def __init__(self, d_model = 256, n_layers= 4, dropout = 0.2):
        super().__init__()
        self.model = S4Model(d_input = 1, d_output = 64, d_model = d_model, n_layers = n_layers, dropout = dropout)

    def forward(self, x, mask = None):
        x = self.model.forward(x, mask)
        return x
    
class InstrumentEncoder(nn.Module):
    def __init__(self, d_model = 256, n_layers = 4, dropout = 0.2):
        super().__init__()
        self.model = S4Model(d_input = 1, d_output = 16, d_model = d_model, n_layers = n_layers, dropout = dropout)

    def forward(self, x, mask = None):
        x = self.model.forward(x, mask)
        return x
    
class Pair_Decoder(nn.Module):
    def __init__(self, grid_length = 1024):
        super().__init__()
        self.layer1 = nn.Linear(80, 256)
        self.layer2 = nn.Linear(256, 1024)

    def forward(self, x):
        x = self.layer1(x)
        x = x.relu()
        x = self.layer2(x)

        return x

class DisentanglementModel(nn.Module):
    def __init__(self, d_model = 256, n_layers = 4, dropout = 0.2):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(d_model, n_layers, dropout)
        self.instrument_encoder = InstrumentEncoder(d_model, n_layers, dropout)
        self.decoder = Pair_Decoder()

    def forward(self, anchor, anchor_mask, same_star, same_star_mask, same_sector, same_sector_mask):
        #For all three pairs
        #1. encode 2. decode 3. return latents and reconstructions

        #we need x_physics, x_instrument then we need x_physics_same_star adn x_instrument_same sector

        x_physics = self.physics_encoder(anchor, anchor_mask)
        x_instrument = self.instrument_encoder(anchor, anchor_mask)

        x_physics_same_star = self.physics_encoder(same_star, same_star_mask)

        x_instrument_same_sector = self.instrument_encoder(same_sector, same_sector_mask)

        concatenate = torch.cat([x_physics, x_instrument], dim= -1)
        decode = self.decoder(concatenate)

        return x_physics, x_instrument, x_physics_same_star, x_instrument_same_sector, decode


def reconstruction_loss(reconstruction, anchor_flux):
    return nn.functional.mse_loss(reconstruction, anchor_flux)

