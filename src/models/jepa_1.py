import torch
import torch.nn as nn

from src.models.disentangle import InstrumentEncoder


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
    
class JEPA_1(nn.Module):
    def __init__(self, d_model = 256, n_layers= 4, dropout = 0.2, momentum = 0.996): #just searched it up and found this momentum value to be the best
        super().__init__()
        self.momentum = momentum

        #the encoder that we're giving our same sector different star to 
        self.online_encoder = InstrumentEncoder(d_model = d_model, n_layers = n_layers, dropout = dropout)

        #the predictor that takes the output from the online encoder
        self.predictor = Predictor(4, 4, 256)

        #our ema encoder
        self.target_encoder = InstrumentEncoder(d_model = d_model, n_layers = n_layers, dropout = dropout)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict()) #start at our same state

        #for the target encoder we do not need to have a gradient at all
        #since we're going with the moving average
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    

        

