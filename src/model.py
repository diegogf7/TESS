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
    
class masking_autoencoder(nn.Module):
    def __init__(self, patch_size = 16, dimension_vector = 128, number_patches=64, mask_ratio= 0.75):
        super(masking_autoencoder, self).__init__()
        self.patch_size = patch_size
        self.dimension_vector = dimension_vector
        self.number_patches = number_patches
        self.mask_ratio = mask_ratio
        #need to do all four functions
        self.patches = Patches(patch_size, dimension_vector)
        self.positions = Positions(number_patches, dimension_vector)
        self.encoder = Transformer_Encoder(dimension_vector, 4, 256)
        self.decoder = Transformer_Decoder(dimension_vector, patch_size)
        

        self.mask_token = nn.Parameter(torch.zeros(1,1, dimension_vector))

    def forward(self, x):
        x = self.patches(x)
        x = self.positions(x)

        batch_size = x.shape[0]
        number_visible = int(self.number_patches * (1 - self.mask_ratio))

        noise_to_observe = torch.rand(batch_size, self.number_patches, device = x.device) #what does this line do?
        shuffle = torch.argsort(noise_to_observe, dim=1)

        #needed to use claude code for this part it was getting kind of complicated
        indexes_to_keep = shuffle[:, :number_visible]
        indexes = indexes_to_keep.unsqueeze(-1).expand(-1,-1, self.dimension_vector)
        x_visible = torch.gather(x, 1, indexes)

        x_encoder = self.encoder(x_visible)


        x_full = self.mask_token.expand(batch_size, self.number_patches, -1).clone()
        x_full.scatter_(1, indexes, x_encoder)

        x_decoded = self.decoder(x_full)
        return x_decoded, shuffle