import torch
import torch.nn as nn

from s4d import S4Model


#going to define our physics model and then our instrument model
class PhysicsEncoder(nn.Module):
    def __init__(self, d_model = 256, n_layers= 4, dropout = 0.2):
        super().__init__()
        self.model = S4Model(d_input = 1, d_model = d_model, n_layers = n_layers, dropout = dropout, n_tokens = 16, token_dim = 4)

    def forward(self, x, mask = None):
        x = self.model.forward(x, mask)
        return x
    
class InstrumentEncoder(nn.Module):
    def __init__(self, d_model = 256, n_layers = 4, dropout = 0.2):
        super().__init__()
        self.model = S4Model(d_input = 1, d_model = d_model, n_layers = n_layers, dropout = dropout, n_tokens = 4, token_dim = 4)

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

    def __init__(self, length = 1024, patch_size = 16, token_dim = 4, d = 128, n_heads = 4, t_dim = 64):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = length // patch_size

        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, d)
        )

        self.patch_projection = nn.Linear(patch_size, d)
        self.position = nn.Parameter(torch.randn(1, self.num_patches, d) * 0.02)

        self.context_projection = nn.Linear(token_dim, d)

        self.norm_sa = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(d, n_heads, batch_first = True)

        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(d, n_heads, batch_first = True)

        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4*d, d))
        
        self.out_projection = nn.Linear(d, patch_size)

    def forward(self, xt, t, context):
        B, L = xt.shape

        q = xt.reshape(B, self.num_patches, self.patch_size)
        q = self.patch_projection(q) + self.position
        q = q + self.t_embed(t).unsqueeze(1)

        kv = self.context_projection(context)

        s = self.norm_sa(q)
        q = q + self.self_attention(s,s,s)[0]

        q = q + self.cross_attention(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))[0]
        q = q + self.ff(self.norm_ff(q))

        v = self.out_projection(q).reshape(B, L)
        return v


class DisentanglementModel(nn.Module):
    def __init__(self, d_model = 256, n_layers = 4, dropout = 0.2):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(d_model, n_layers, dropout)
        self.instrument_encoder = InstrumentEncoder(d_model, n_layers, dropout)
        self.velocity_net = Velocity_Net(length = 1024, token_dim = 4)

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


def info_nce(z_a, z_b, temperature = 0.2):
    z_a = nn.functional.normalize(z_a, dim = 1)
    z_b = nn.functional.normalize(z_b, dim = 1)
    logits = z_a @ z_b.t() / temperature

    labels = torch.arange(z_a.shape[0], device = z_a.device)
    return 0.5 * (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.t(), labels))


    
    
    #return nn.functional.mse_loss(reconstruction, anchor_flux)


def contrastive_flow_matching_loss(velocity_net, x1, concatenate, lam = 0.05):
    x0 = torch.randn_like(x1)

    t = torch.rand(x1.shape[0], 1, device = x1.device)

    xt = (1 - t) * x0 + t * x1
    target_velocity = x1 - x0

    permutation = torch.randperm(x1.shape[0], device = x1.device)

    negative_velocity = target_velocity[permutation]

    prediction_velocity = velocity_net(xt, t, concatenate)
    positive = nn.functional.mse_loss(prediction_velocity, target_velocity)
    negative = nn.functional.mse_loss(prediction_velocity, negative_velocity)
    
    return positive - (lam * negative)
    