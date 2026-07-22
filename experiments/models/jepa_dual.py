import torch
import torch.nn as nn

from src.models.disentangle import PhysicsEncoder, InstrumentEncoder
from src.models.jepa_1 import Predictor, jepa_loss

class DualJEPA(nn.Module):

    def __init__(self, d_model =256, n_layers = 4, dropout = 0.2, momentum = 0.996):
        super().__init__()

        self.momentum = momentum

        #need the online encoders
        self.physics_encoder = PhysicsEncoder(d_model, n_layers, dropout)
        self.instrument_encoder = InstrumentEncoder(d_model, n_layers, dropout)

        #need the predictor
        self.predictor = Predictor(20, 4, 256)

        #EMA target encoders
        self.target_physics = PhysicsEncoder(d_model, n_layers, dropout)
        self.target_instrument = InstrumentEncoder(d_model, n_layers, dropout)

        self.target_physics.load_state_dict(self.physics_encoder.state_dict())
        self.target_instrument.load_state_dict(self.instrument_encoder.state_dict())

        for p in self.target_physics.parameters():
            p.requires_grad = False
        
        for p in self.target_instrument.parameters():
            p.requires_grad = False

    def forward(self, anchor_flux, anchor_mask, same_star_flux, same_star_mask, same_sector_flux, same_sector_mask):
        z_physics = self.physics_encoder(same_star_flux, same_star_mask)
        z_instrument = self.instrument_encoder(same_sector_flux, same_sector_mask)

        context = torch.cat([z_physics, z_instrument], dim = 1)
        prediction = self.predictor(context)

        with torch.no_grad():

            z_physics_anchor = self.target_physics(anchor_flux, anchor_mask)
            z_instrument_anchor = self.target_instrument(anchor_flux, anchor_mask)

            target = torch.cat([z_physics_anchor, z_instrument_anchor], dim = 1)

        
        return prediction, target
    
    @torch.no_grad()
    def update_target(self):
        m = self.momentum
        for online_p, target_p in zip(self.physics_encoder.parameters(), self.target_physics.parameters()):
            
            target_p.data = m * target_p.data + (1.0 - m) * online_p.data

        
        for online_p, target_p in zip(self.instrument_encoder.parameters(), self.target_instrument.parameters()):
            
            target_p.data = m * target_p.data + (1.0 - m) * online_p.data