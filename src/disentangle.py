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
    

class Velocity_Net(nn.Module):
    def __init__(self, length = 1024, concatenate_dim = 80, hidden = 512, t_dim = 64):
        super().__init__()

        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),

        )

        self.net = nn.Sequential(
            nn.Linear(length + t_dim + concatenate_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, length),

        )
    
    def forward(self, xt, t, concatenate):
        t_emb = self.t_embed(t)
        h = torch.cat([xt, t_emb, concatenate], dim = -1)

        return self.net(h)



class DisentanglementModel(nn.Module):
    def __init__(self, d_model = 256, n_layers = 4, dropout = 0.2):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(d_model, n_layers, dropout)
        self.instrument_encoder = InstrumentEncoder(d_model, n_layers, dropout)
        self.velocity_net = Velocity_Net(length = 1024, concatenate_dim = 80)

    def forward(self, same_star, same_star_mask, same_sector, same_sector_mask):
        #For the physics side we are examining the same star as the anchor but at a different sector
        #on the instrument side we are examining a different star in the same sector as the achor
        x_physics = self.physics_encoder(same_star, same_star_mask)

        x_instrument = self.instrument_encoder(same_sector, same_sector_mask)

        return x_physics, x_instrument

        

def flow_matching_loss(velocity_net, x1, concatenate):
    #x1 is the anchor_flux and concatenate is the concatenation of our latent spaces

    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device = x1.device)

    xt = (1 - t) * x0 + (t * x1) #to the straight noise and data path
    target_velocity = x1 - x0

    prediction_velocity = velocity_net(xt, t, concatenate)
    return nn.functional.mse_loss(prediction_velocity, target_velocity)



    
    
    #return nn.functional.mse_loss(reconstruction, anchor_flux)