# All this code is from Claude
"""Instance-to-Subspace instrument JEPA.

Fixes two defects of the group-mean JEPA:
  1. the old loss constrained only the group AVERAGE while downstream uses
     INDIVIDUAL star embeddings -- here every context star individually
     predicts the held-out group target (no averaging before the loss);
  2. mean pooling discards multi-dimensional detector structure -- the
     instance_mean_var / instance_cov arms make the target a fixed compressed
     code of the target set's mean+variance / mean+covariance.

Arms: mean_to_mean (current group-mean baseline), instance_mean,
instance_mean_var, instance_cov. Architecture (S4D online + EMA + trainable
instrument projector) is identical across all arms and the random control.

The projector output IS the downstream instrument representation:
encode_instrument(flux, mask, source) -- frozen probes and fine-tuning must
use it (never a bare encoder view).

All random projections are deterministic, registered, non-trainable buffers.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.s4d import S4Model

ARMS = ("mean_to_mean", "instance_mean", "instance_mean_var", "instance_cov")
PROJ_SEED = 20260717           # deterministic seed for the fixed projections


class InstanceSubspaceJEPA(nn.Module):
    def __init__(self, arm, n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                 momentum=0.996, readout="mean", subspace_dim=16,
                 var_weight=1.0, cov_weight=0.01):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.momentum = float(momentum)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.embed_dim = n_tokens * token_dim
        self.subspace_dim = int(subspace_dim)

        encoder_kwargs = dict(d_input=1, d_model=d_model, n_layers=n_layers,
                              dropout=0.0, n_tokens=n_tokens, token_dim=token_dim,
                              readout=readout)
        self.context_encoder = S4Model(**encoder_kwargs)
        self.target_encoder = S4Model(**encoder_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # trainable instrument projector: its output is the downstream representation
        self.instrument_projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim))

        # deterministic non-trainable projections (registered buffers)
        gen = torch.Generator().manual_seed(PROJ_SEED)
        d, s = self.embed_dim, self.subspace_dim
        self.register_buffer("proj_subspace",
                             torch.randn(d, s, generator=gen) / d ** 0.5)
        self.register_buffer("proj_meanvar",
                             torch.randn(2 * d, d, generator=gen) / (2 * d) ** 0.5)
        cov_stat = s + s * (s + 1) // 2
        self.register_buffer("proj_cov",
                             torch.randn(cov_stat, d, generator=gen) / cov_stat ** 0.5)
        triu = torch.triu_indices(s, s)
        self.register_buffer("triu_rows", triu[0])
        self.register_buffer("triu_cols", triu[1])

    # ------------------------------------------------------------- pieces
    @staticmethod
    def _encode_set(encoder, flux, observed_mask):
        if flux.ndim != 3:
            raise ValueError(f"expected flux (batch,set,time), got {tuple(flux.shape)}")
        batch, group, length = flux.shape
        tokens = encoder(flux.reshape(batch * group, length).unsqueeze(-1),
                         observed_mask.reshape(batch * group, length))
        return tokens.reshape(batch, group, -1)          # (B, K, 256)

    def group_target(self, per_star_z):
        """Arm-specific fixed target code from TARGET-set per-star embeddings
        (B, K, 256) -> layer-normalized (B, 256). Deterministic, no learning."""
        mean = per_star_z.mean(dim=1)                    # (B, 256)
        if self.arm in ("mean_to_mean", "instance_mean"):
            code = mean
        elif self.arm == "instance_mean_var":
            var = per_star_z.var(dim=1, unbiased=False)
            code = torch.cat([mean, var], dim=-1) @ self.proj_meanvar
        else:                                            # instance_cov
            sub = per_star_z @ self.proj_subspace        # (B, K, s)
            sub_mean = sub.mean(dim=1)
            centered = sub - sub_mean[:, None, :]
            cov = centered.transpose(1, 2) @ centered / sub.shape[1]
            triu = cov[:, self.triu_rows, self.triu_cols]
            code = torch.cat([sub_mean, triu], dim=-1) @ self.proj_cov
        return F.layer_norm(code, (code.shape[-1],))

    def student_codes(self, per_star_z):
        """Trainable instrument codes for every individual star (B, K, 256)."""
        return self.instrument_projector(per_star_z)

    def prediction_loss_from_embeddings(self, ctx_per_star, target_code):
        """mean_i SmoothL1(student_i, stopgrad(target)) from context per-star
        EMBEDDINGS (B, K, 256). mean_to_mean projects the context MEAN instead
        (the baseline being fixed); instance arms never average before the loss."""
        target = target_code.detach()
        if self.arm == "mean_to_mean":
            student = self.instrument_projector(ctx_per_star.mean(dim=1))
            return F.smooth_l1_loss(student, target)
        students = self.student_codes(ctx_per_star)
        return F.smooth_l1_loss(students, target[:, None, :].expand_as(students))

    def vicreg_terms(self, students_flat):
        """Variance hinge + off-diagonal covariance penalty over batch*stars."""
        z = students_flat - students_flat.mean(dim=0)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        var_term = torch.relu(1.0 - std).mean()
        n = max(z.shape[0] - 1, 1)
        cov = (z.T @ z) / n
        off = cov - torch.diag(torch.diag(cov))
        cov_term = (off ** 2).sum() / z.shape[1]
        return var_term, cov_term

    # ------------------------------------------------------------- forward
    def forward(self, context_flux, context_mask, target_flux, target_mask):
        ctx_per_star = self._encode_set(self.context_encoder, context_flux, context_mask)
        with torch.no_grad():
            tgt_per_star = self._encode_set(self.target_encoder, target_flux, target_mask)
            target_code = self.group_target(tgt_per_star)

        if self.arm == "mean_to_mean":
            students = self.instrument_projector(ctx_per_star.mean(dim=1))  # (B, 256)
            pred_loss = F.smooth_l1_loss(students, target_code.detach())
            students_flat = students
        else:
            students = self.student_codes(ctx_per_star)                     # (B, K, 256)
            pred_loss = F.smooth_l1_loss(
                students, target_code.detach()[:, None, :].expand_as(students))
            students_flat = students.reshape(-1, students.shape[-1])

        var_term, cov_term = self.vicreg_terms(students_flat)
        loss = pred_loss + self.var_weight * var_term + self.cov_weight * cov_term
        return {"loss": loss, "pred_loss": pred_loss, "var_loss": var_term,
                "cov_loss": cov_term, "students": students_flat,
                "target_code": target_code, "ctx_per_star": ctx_per_star}

    # ------------------------------------------------------ downstream API
    def encode_instrument(self, flux, observed_mask=None, source="online"):
        """Individual-star instrument representation = projector(encoder(x)).
        This is the ONLY sanctioned downstream embedding."""
        if source not in ("online", "ema"):
            raise ValueError("source must be 'online' or 'ema'")
        encoder = self.context_encoder if source == "online" else self.target_encoder
        with torch.no_grad():
            tokens = encoder(flux.unsqueeze(-1), observed_mask)
            return self.instrument_projector(tokens.flatten(1))

    @torch.no_grad()
    def target_cosine(self, target_flux_a, target_mask_a, target_flux_b, target_mask_b):
        """Cosine similarity between the target codes of two disjoint sets."""
        za = self.group_target(self._encode_set(self.target_encoder, target_flux_a, target_mask_a))
        zb = self.group_target(self._encode_set(self.target_encoder, target_flux_b, target_mask_b))
        return float(F.cosine_similarity(za, zb, dim=-1).mean())

    @torch.no_grad()
    def update_target(self):
        m = self.momentum
        for online, target in zip(self.context_encoder.parameters(),
                                  self.target_encoder.parameters()):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)


def build_instance_subspace(arm):
    return InstanceSubspaceJEPA(
        arm,
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dim=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=int(os.environ.get("JEPA_DMODEL", "256")),
        n_layers=int(os.environ.get("JEPA_NLAYERS", "4")),
        momentum=float(os.environ.get("JEPA_MOMENTUM", "0.996")),
        readout=os.environ.get("JEPA_READOUT", "mean"),
        subspace_dim=int(os.environ.get("SUBSPACE_DIM", "16")),
        var_weight=float(os.environ.get("VIC_VAR_WEIGHT", "1.0")),
        cov_weight=float(os.environ.get("VIC_COV_WEIGHT", "0.01")),
    )
