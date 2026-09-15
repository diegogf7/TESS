from dataclasses import dataclass


@dataclass
class Part1Config:
    """Common-mode (systematics) curve encoder."""
    seq_len: int = 1024
    group_size: int = 32          # curves per region, Instruction 1.3
    latent_dim: int = 32          # Instruction 1.1
    n_tokens: int = 4             # 32 = 4 time blocks x 8 features
    d_model: int = 256
    d_state: int = 64
    n_layers: int = 4
    dropout: float = 0.0          # frozen later; keep inference-consistent
    var_gamma: float = 1.0
    var_weight: float = 1.0       # anti-collapse on the 32 dims
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 4000
    batch_groups: int = 4         # regions per batch
    log_every: int = 20
    ckpt: str = "artifacts/vicreg_jepa/part1_encoder.pt"


@dataclass
class Part2Config:
    """VICReg-based JEPA."""
    seq_len: int = 1024
    physics_dim: int = 64         # Instruction 2.1
    sys_dim: int = 32             # must match Part1Config.latent_dim
    n_tokens: int = 4             # 64 = 4 time blocks x 16 features
    d_model: int = 256
    d_state: int = 64
    n_layers: int = 4
    dropout: float = 0.0

    mask_ratio_min: float = 0.30  # Instruction 2.2
    mask_ratio_max: float = 0.50

    phi: float = 1.0              # Instruction 2.8
    lam: float = 25.0
    mu: float = 1.0               # measured: too weak. With lam=25/mu=1 the latent
                                  # std only reaches ~0.12 after 80 steps (target
                                  # gamma=1.0) because shrinking the latent is the
                                  # cheapest way to satisfy L_inv and L_cor_sys.
                                  # mu=25 reaches ~1.35 and cor_sys still falls to
                                  # 0.14. See test_pipeline.py::loss_weight_balance.
    nu: float = 0.01
    gamma: float = 1.0            # Instruction 2.6 target std
    tau: float = 0.999            # Instruction 2.9

    # 32 samples < 64 dims makes the covariance matrix rank-deficient and L_cov
    # a meaningless estimate. VICReg uses ~2048. 256 is the practical floor.
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    steps: int = 20000
    log_every: int = 20
    ckpt: str = "artifacts/vicreg_jepa/part2_jepa.pt"
