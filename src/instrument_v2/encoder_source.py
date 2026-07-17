# All this code is from Claude
"""Online (context) vs EMA (target) encoder extraction for the encoder audit.

The abl1 benchmark evaluated `model.encode()`, which ALWAYS uses the EMA
target encoder -- but SupCon/hybrid gradients optimize the ONLINE context
encoder directly. This helper makes the source explicit and auditable.

encode_features() batches encoder(flux.unsqueeze(-1), observed_mask) directly
on the extracted module; it never calls model.encode(), so the online path
cannot silently fall back to the EMA copy.
"""

import numpy as np
import torch

BATCH = 256
SOURCES = ("online", "ema")


def extract_encoder(model, source):
    if source == "online":
        return model.context_encoder
    if source == "ema":
        return model.target_encoder
    raise ValueError(f"unknown encoder source {source!r} (use 'online' or 'ema')")


def assert_same_architecture(model):
    """Online and EMA must be the same architecture with identical shapes."""
    online = extract_encoder(model, "online").state_dict()
    ema = extract_encoder(model, "ema").state_dict()
    assert online.keys() == ema.keys(), "encoder state dict keys differ"
    for k in online:
        assert online[k].shape == ema[k].shape, f"shape mismatch at {k}"


def encoders_identical(model):
    """True iff every online parameter equals its EMA counterpart (random init)."""
    online = extract_encoder(model, "online").state_dict()
    ema = extract_encoder(model, "ema").state_dict()
    return all(torch.equal(online[k], ema[k]) for k in online)


def param_distance(model):
    """Relative L2 distance ||online - ema|| / ||ema|| over all parameters."""
    online = extract_encoder(model, "online").state_dict()
    ema = extract_encoder(model, "ema").state_dict()
    num = sum(float(((online[k].double() - ema[k].double()) ** 2).sum()) for k in online)
    den = sum(float((ema[k].double() ** 2).sum()) for k in ema)
    return float(np.sqrt(num / max(den, 1e-30)))


def encode_features(model, source, X, M, device, batch=BATCH):
    """Frozen features from the chosen encoder. Identical preprocessing for
    both sources: the SAME gridded X/M go through the SAME call convention,
    encoder(flux.unsqueeze(-1), observed_mask). model.encode() is never used."""
    encoder = extract_encoder(model, source)
    encoder.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            f = torch.tensor(X[start:start + batch]).to(device)
            m = torch.tensor(M[start:start + batch]).to(device)
            z = encoder(f.unsqueeze(-1), m)
            pieces.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(pieces)
