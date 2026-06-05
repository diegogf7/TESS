import torch
import torch.nn as nn

class Patches(nn.Module):
    def __init__(self, patch_size = 16, dimension_vector = 128): #going to divy up into 64 patches of 16 frames. 
        #Each patch is projected as a 128 piece vector
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(self.patch_size, dimension_vector) #doing our transformation for our vector
        
    def forward(self, x):
        x = x.reshape(-1, 64, self.patch_size) #making sure that we get the right size
        return self.projection(x)
    

class Positions(nn.Module):
    def __init__(self, patches = 64, dimension_vector = 128):
        super(Positions, self).__init__()
        self.patches = patches
        self.embedding = nn.Embedding(patches, dimension_vector)

    def forward(self, k):
        tensors_position = torch.arange(self.patches, device= k.device)
        shape = self.embedding(tensors_position)
        return (k + shape) 
    

class Transformer_Encoder(nn.Module):
    def __init__(self, dimension_vector = 128, number_heads = 4, mlp_dim = 256):
        super(Transformer_Encoder, self).__init__()
        self.num_layers = 4
        self.Encoder_Layer = nn.TransformerEncoderLayer(d_model = dimension_vector, nhead = number_heads, dim_feedforward = mlp_dim, batch_first = True)
        self.Encoder = nn.TransformerEncoder(self.Encoder_Layer, num_layers = 4)

    def forward(self, k):
        k = self.Encoder(k)
        return k
    
class Transformer_Decoder(nn.Module):
    def __init__(self, dimension_vector = 128, patch_size = 16):
        super(Transformer_Decoder, self).__init__()

        self.Decoder = nn.Linear(dimension_vector, patch_size)
    
    def forward(self, k):
        k = self.Decoder(k)
        return k
