import numpy as np
import torch

from src.instrument_v2.inspect_phyts_aperiodic_eclipse import pick_first_by_tic, predict_arm

from src.instrument_v2.eval_phyts_instrument_ab import matched_split, instrument_cleaned_curve

from src.data.data import CLASSSE, CLASS_TO_IDX
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash


AP = CLASS_TO_IDX["APERIODIC"]
EC = CLASS_TO_IDX["ECLIPSE"]

def test_pick_first_by_tic_selects_exactly_one_sorted():

    test_tics = np.array(["T9", "T2", "T5", "T2", "T7"])
    test_y = np.array([AP, AP, EC, EC, AP])
    ap = pick_first_by_tic(test_tics, test_y, AP)

    ec = pick_first_by_tic(test_tics, test_y, EC)
    assert ap == 1 and ec == 3
    assert test_y[ap] == AP and test_y[ec] == EC
    assert isinstance(ap, int) and isinstance(ec, int)
    assert pick_first_by_tic(test_tics, np.full(5, AP), EC) is None

