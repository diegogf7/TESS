import numpy as np
import torch
from torch.utils.data import Dataset


class NpzCurveSource:
    """Expects arrays: flux (N, L), observed (N, L), region (N,)."""

    def __init__(self, path, seq_len=1024):
        d = np.load(path)
        self.flux = d["flux"].astype(np.float32)
        self.observed = d["observed"].astype(np.uint8)
        self.region = d["region"].astype(np.int64)
        assert self.flux.shape[1] == seq_len, f"expected {seq_len} cadences"


class SyntheticCurveSource:
    """Region-shared systematics + per-star physics + noise. For smoke tests."""

    def __init__(self, n_regions=16, per_region=64, seq_len=1024, seed=0):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, seq_len, dtype=np.float32)
        flux, region = [], []
        for r in range(n_regions):
            # shared instrument trend: a ramp plus a slow sinusoid
            common = (rng.normal() * t
                      + rng.normal() * np.sin(2 * np.pi * rng.uniform(1, 3) * t))
            for _ in range(per_region):
                period = rng.uniform(4, 40)
                phys = rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * seq_len / period * t)
                flux.append(common * rng.uniform(0.7, 1.3) + phys
                            + rng.normal(0, 0.1, seq_len))
                region.append(r)
        self.flux = np.stack(flux).astype(np.float32)
        self.region = np.asarray(region, dtype=np.int64)
        self.observed = np.ones_like(self.flux, dtype=np.uint8)
        # punch realistic gaps
        for i in range(len(self.flux)):
            s = rng.integers(0, seq_len - 80)
            self.observed[i, s:s + rng.integers(20, 80)] = 0


def normalise(flux, observed):
    """Zero-median, unit-MAD per curve over observed cadences only."""
    m = observed.astype(bool)
    out = np.zeros_like(flux)
    for i in range(len(flux)):
        v = flux[i][m[i]]
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-8
        out[i] = (flux[i] - med) / mad
    return out.astype(np.float32)


class RegionGroupDataset(Dataset):
    """Instruction 1.3: yields `group_size` curves from one region."""

    def __init__(self, source, group_size=32, groups_per_epoch=2000, seed=0):
        self.flux = normalise(source.flux, source.observed)
        self.observed = source.observed
        self.group_size = group_size
        self.groups_per_epoch = groups_per_epoch
        self.rng = np.random.default_rng(seed)
        self.by_region = {}
        for r in np.unique(source.region):
            idx = np.flatnonzero(source.region == r)
            if len(idx) >= group_size:
                self.by_region[int(r)] = idx
        if not self.by_region:
            raise ValueError(f"no region has >= {group_size} curves")
        self.regions = sorted(self.by_region)

    def __len__(self):
        return self.groups_per_epoch

    def __getitem__(self, _):
        r = self.regions[self.rng.integers(len(self.regions))]
        pick = self.rng.choice(self.by_region[r], self.group_size, replace=False)
        return (torch.from_numpy(self.flux[pick]),
                torch.from_numpy(self.observed[pick]).float())


class SingleCurveDataset(Dataset):
    """Part 2: one curve at a time."""

    def __init__(self, source):
        self.flux = normalise(source.flux, source.observed)
        self.observed = source.observed

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, i):
        return (torch.from_numpy(self.flux[i]),
                torch.from_numpy(self.observed[i]).float())


def random_time_mask(flux, observed, ratio_min=0.30, ratio_max=0.50,
                     generator=None):
    """Instruction 2.2 / spec 4.1. Fresh mask every call -- never cached, reused.

    Hides 30-50% of each curve's CURRENTLY OBSERVED cadences (not 30-50% of the
    1024 slots): on real curves only 79-95% of slots carry data, so sampling
    over all slots would hide a smaller, coverage-dependent share of the real
    signal and make the task easier for well-covered stars.

    Returns `visible` (B, L): 1 where the encoder may read the value.
    """
    B, L = flux.shape
    obs = observed > 0
    n_obs = obs.sum(dim=1)                                        # (B,)
    dev = flux.device
    if generator is not None:
        # fixed masking seed: draw on the generator's device, then move
        gdev = generator.device
        ratios = torch.rand(B, device=gdev, generator=generator).to(dev)
        noise = torch.rand(B, L, device=gdev, generator=generator).to(dev)
    else:
        ratios = torch.rand(B, device=dev)
        noise = torch.rand(B, L, device=dev)
    ratios = ratios * (ratio_max - ratio_min) + ratio_min
    k = (ratios * n_obs.float()).long()                           # per-row hide count
    noise = noise.masked_fill(~obs, float("inf"))                 # gaps sort last
    order = torch.argsort(noise, dim=1)
    rank = torch.argsort(order, dim=1)
    hide = rank < k.unsqueeze(1)                                  # only observed slots
    return observed * (~hide).float()
