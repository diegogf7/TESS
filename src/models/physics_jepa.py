import torch
import torch.nn as nn

from src.models.disentangle import PhysicsEncoder

class Predictor(nn.Module):

    def __init__(self, n_tokens = 16, token_dimension = 4, hidden = 256):
        super().__init__()
        self.n_tokens = n_tokens

        self.token_dimension = token_dimension
        
        flat_dimension = n_tokens * token_dimension

        self.net = nn.Sequential(
            nn.Linear(flat_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, flat_dimension),

        )
    def forward(self, context):

        B = context.shape[0]
        x = context.reshape(B, self.n_tokens * self.token_dimension)
        x = self.net(x)

        x = x.reshape(B, self.n_tokens, self.token_dimension)
        return x
    
class PhysicsJEPA(nn.Module):

    def __init__(self, d_model = 256, n_layers = 4, dropout = 0.2, momentum = 0.996, patch = 16, mask_ratio = 0.5, amplitude_min = 0.5, amplitude_max = 2.0, noise_min = 0.1, noise_max = 0.6):
        super().__init__()

        self.momentum = momentum
        self.patch = patch

        self.mask_ratio = mask_ratio

        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.noise_min = noise_min
        self.noise_max = noise_max

        self.online_encoder = PhysicsEncoder(d_model = d_model, n_layers = n_layers, dropout = dropout)
        self.predictor = Predictor(16, 4, 256)

        self.target_encoder = PhysicsEncoder(d_model = d_model, n_layers = n_layers, dropout = dropout)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def mask_input(self, flux, mask):
        B, L = flux.shape
        n_patches = L // self.patch

        keep_patch = (torch.rand(B, n_patches, device = flux.device) > self.mask_ratio).float()
        keep = keep_patch.repeat_interleave(self.patch, dim = 1)
        
        keep = keep[:, :L]

        view_flux = flux * keep
        view_mask = mask * keep
        return view_flux, view_mask
    
    def augment(self, flux, mask):

        B, L = flux.shape

        scale = torch.empty(B, 1, device = flux.device).uniform_(self.amplitude_min, self.amplitude_max)
        flux = flux * scale

        #need to add random noise to our curve
        curve_scale = (flux * mask).std(dim = 1, keepdim = True) + 1e-6
        noise_level = torch.empty(B, 1, device = flux.device).uniform_(self.noise_min, self.noise_max)
        flux = flux + torch.randn_like(flux) * noise_level * curve_scale * mask

        return self.mask_input(flux, mask)
    
    def forward(self, flux, mask):

        online_flux, online_mask = self.mask_input(flux, mask)
        z_online = self.online_encoder(online_flux.unsqueeze(-1), online_mask)
        prediction = self.predictor(z_online)



        with torch.no_grad():
            z_target = self.target_encoder(flux.unsqueeze(-1), mask)

        
        return prediction, z_target
    
    @torch.no_grad()
    def update_target(self):
        m = self.momentum

        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data = m * target_p.data + (1.0 - m) * online_p.data


def jepa_loss(prediction, z_target):

    prediction = nn.functional.normalize(prediction, dim = -1)
    z_target = nn.functional.normalize(z_target, dim = -1)
    return nn.functional.mse_loss(prediction, z_target)

        