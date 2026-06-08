import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

'''
Want to train a transformer
- Need to give a set number of patches to train on (say 1000)
- Normalize?

'''


def resample_to_grid(time, flux, grid_length = 1024): #
    
    time_grid = np.linspace(time[0], time[len(time) - 1], grid_length)

    interpolar = interp1d(time, flux)
    grid_flux = interpolar(time_grid)

    #need a way to create a list that shows that we have gaps in data
    observed = np.isfinite(grid_flux).astype(np.float32)
    grid_flux = np.nan_to_num(grid_flux, nan=0.0)

    return grid_flux, observed

def normalize(flux, clip_sigma = 6.0):
    median = np.median(flux)

    if median == 0:
        median = 1.0 

    flux = flux / median #need to scale it around 1

    mad = np.median(np.abs(flux - np.median(flux))) #making sure that we got the std but immune to outliers
    scale_to_use = mad * 1.4826 #changing to our std
    if scale_to_use >0:
        #flux = np.clip(flux, -clip_sigma * scale_to_use, clip_sigma * scale_to_use) #I think this is the error
        flux = np.clip(flux, 1.0 - clip_sigma * scale_to_use, 1.0 + clip_sigma * scale_to_use )
    return flux


class LightCurveDataset(Dataset):
    def __init__(self, parquet, grid_length = 1024):
        self.df = pd.read_parquet(parquet)
        self.grid_length = grid_length
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        time = np.array(row["time"])
        flux = np.array(row["flux"])
        flux = normalize(flux)
        grid_flux, observed = resample_to_grid(time, flux, self.grid_length)

        grid_flux = torch.tensor(grid_flux, dtype = torch.float32)
        observed = torch.tensor(observed, dtype = torch.float32)
        
        return grid_flux, observed
    

#code from claude code to make sure I get the right file names for the training
if __name__ == "__main__":
    path = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/tess_classification.parquet"
    ds = LightCurveDataset(path, grid_length=1024)
    print(f"Dataset size: {len(ds)}")
    flux, mask = ds[0]
    print(f"Flux shape: {flux.shape}, dtype: {flux.dtype}")
    print(f"Mask shape: {mask.shape}, sum: {mask.sum()}")





