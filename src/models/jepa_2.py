import torch
import torch.nn as nn

from src.models.disentangle import InstrumentEncoder, PhysicsEncoder


class Predictor(nn.Module):
    def __init__(self, n_tokens = 16, token_dimension = 4, hidden = 256):
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
    
class JEPA_2(nn.Module):
    def __init__(self, d_model = 256, n_layers= 4, dropout = 0.2, momentum = 0.996): #just searched it up and found this momentum value to be the best
        super().__init__()
        self.momentum = momentum

        #the encoder that we're giving our same sector different star to 
        self.online_encoder = PhysicsEncoder(d_model = d_model, n_layers = n_layers, dropout = dropout)

        #the predictor that takes the output from the online encoder
        self.predictor = Predictor(16, 4, 256)

        #our ema encoder
        self.target_encoder = PhysicsEncoder(d_model = d_model, n_layers = n_layers, dropout = dropout)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict()) #start at our same state

        #for the target encoder we do not need to have a gradient at all
        #since we're going with the moving average
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, context, context_mask, target, target_mask): #context for online and the target for EMA

        z_context = self.online_encoder(context, context_mask)
        prediction = self.predictor(z_context)

        #target curve does not have a gradient going on

        with torch.no_grad():
            z_target = self.target_encoder(target, target_mask)
        
        return prediction, z_target, z_context
    
    @torch.no_grad()
    def update_target(self):
        m = self.momentum
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            
            target_p.data = m * target_p.data + (1.0 - m) * online_p.data



def jepa_loss(prediction, z_target):
    prediction = nn.functional.normalize(prediction, dim = -1)
    z_target = nn.functional.normalize(z_target, dim = -1)

    return nn.functional.mse_loss(prediction, z_target)

"""def variance_loss(z, gamma = 1.0, eps = 0.0001):
    z = z.reshape(z.shape[0], -1)
    std = torch.sqrt(z.var(dim = 0) + eps)
    return torch.mean(torch.relu(gamma - std))"""