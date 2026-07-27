import os
import tempfile
import numpy as np
import pandas as pd
import torch

from src.instrument_v2.eval_phyts_instrument_ab import(

    assert_classes_present,
    instrument_cleaned_curve,
    matched_split,
    ordered_hash,
    physics_grid,
)
from src.data.data import CLASSES, DualEvalDataset
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad

from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import grid_curve_shared

T0, T1 = 1683, 1710.0

def _tiny_instrument():

    torch.manual_seed(0)
    m = FixedTeacherInstrumentJEPA(n_tokens = 16, token_dim = 16, d_model = 32, n_layers = 1, readout = "mean", predictor_type = "mlp")

    m.eval()
    for p in m.trainable_parameters():
        p.requires_grad_(False)
    
    return m

def test_arm_a_matches_dualeval_dataset():

    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):

        time = np.linspace(1683.0, 1710.0, 400)
        rows.append({"TIC": f"T{i}", "sector": 14, "label": CLASSES[i % len(CLASSES)], "time": time, "flux": rng.normal(1000, 30, 400)})

        path = os.path.join(tempfile.mkdtemp(), "phyts.parquet")
        pd.DataFrame(rows).to_parquet(path)

        ds = DualEvalDataset(path, 1024)

        for i in range(len(ds)):
            gf, ob, _, _ = ds[i]

            a_x, a_m = physics_grid(ds.df["time"].iloc[i]. ds.df["flux"].iloc[i])
            assert np.allclose(a_x, gf.numpy(), atol = 1e-5)
            assert np.allclose(a_m, ob.numpy())