import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import random

'''
Want to train a transformer
- Need to give a set number of patches to train on (say 1000)
- Normalize?

'''

CLASSES = ['APERIODIC', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE','GDOR_SPB', 'INSTRUMENT/JUNK', 'RRLYR_CEPH', 'SOLARLIKE']

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

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
    flux = flux / (median - 1.0)

    mean_absolute_deviation = np.median(np.abs(flux))
    scale = mean_absolute_deviation * 1.4826
    if scale > 0:
        flux = np.clip(flux, -clip_sigma * scale, clip_sigma * scale)
    
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
    

class ClassificationDataset(Dataset):
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

        label = CLASS_TO_IDX[row['label']]
        
        return grid_flux, observed, torch.tensor(label)


    
class DisentanglementDataset(Dataset):
    def __init__(self, parquet, grid_length = 1024,multi_sector_only = False):
        self.df = pd.read_parquet(parquet)
        self.grid_length = grid_length
        

        #I don't know how to do this properly
        self.tic_to_indices = {tic: list(indices) for tic, indices in self.df.groupby('TIC').groups.items()}

        self.sector_to_indices = {sector: list(indices) for sector, indices in self.df.groupby('sector').groups.items()}

        self.tics = self.df['TIC'].to_numpy()
        self.sectors = self.df['sector'].to_numpy()
        

        if multi_sector_only:
            self.anchor_indices = [i for i in range(len(self.df))
                if len({self.sectors[j] for j in self.tic_to_indices[self.tics[i]]}) >= 2]
        else:
            self.anchor_indices = list(range(len(self.df)))

    def __len__(self):
        return len(self.anchor_indices)
    
    def load_curve(self, idx):

        row = self.df.iloc[idx]
        flux = normalize(np.array(row["flux"]))
        grid_flux, mask = resample_to_grid(np.array(row['time']), flux, self.grid_length)
        flux_tensor = torch.tensor(grid_flux, dtype = torch.float32)
        mask_tensor = torch.tensor(mask, dtype = torch.float32)
        return flux_tensor, mask_tensor
    
    def __getitem__(self, idx):
        idx = self.anchor_indices[idx]
        anchor_row = self.df.iloc[idx]
        anchor_tic = anchor_row['TIC']
        anchor_sector = anchor_row['sector']

        #now we have our TIC and sector 
        #we need to choose a random index that has a different sector than anchor

        index = self.tic_to_indices[anchor_tic]

        potential_sectors = [i for i in index if self.sectors[i] != anchor_sector]

        if len(potential_sectors) == 0:
            new_sector = idx
        else:
            new_sector = random.choice(potential_sectors)


        sector = self.sector_to_indices[anchor_sector]
        potential_TICs = [i for i in sector if self.tics[i] != anchor_tic]

        if len(potential_TICs) == 0:
            new_TIC = idx
        else:
            new_TIC = random.choice(potential_TICs)

        anchor_flux, anchor_mask = self.load_curve(idx)
        second_flux, second_mask = self.load_curve(new_sector)
        second_sector_flux, second_sector_mask = self.load_curve(new_TIC)

        return anchor_flux, anchor_mask, second_flux, second_mask, second_sector_flux, second_sector_mask

class SectorDataset(Dataset):
    def __init__(self, parquet, grid_length = 1024):
        self.df = pd.read_parquet(parquet)
        self.grid_length = grid_length
    

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        time = np.array(row["time"])

        flux = normalize(np.array(row["flux"]))
        grid_flux, observed = resample_to_grid(time, flux, self.grid_length)

        grid_flux = torch.tensor(grid_flux, dtype = torch.float32)
        observed = torch.tensor(observed, dtype = torch.float32)

        sector = torch.tensor(int(row["sector"])) #
        
        return grid_flux, observed, sector 
        




#code from claude code to make sure I get the right file names for the training
if __name__ == "__main__":
    path = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/tess_classification.parquet"
    ds = LightCurveDataset(path, grid_length=1024)
    print(f"Dataset size: {len(ds)}")
    flux, mask = ds[0]
    print(f"Flux shape: {flux.shape}, dtype: {flux.dtype}")
    print(f"Mask shape: {mask.shape}, sum: {mask.sum()}")
