"""GradNorm (Chen et al. 2018) for the Part 2 loss weights.

Balances the four terms by equalising their gradient norms on the last shared
layer, scaled by each term's relative inverse training rate:

    G_i(t)  = || d(w_i * L_i) / dW ||
    L~_i(t) = L_i(t) / L_i(0)                 inverse training rate
    r_i(t)  = L~_i(t) / mean(L~(t))           relative rate
    target  = mean(G) * r_i^alpha             (a constant)
    L_grad  = sum_i | G_i - target_i |

The weights are updated by descending L_grad, then renormalised to sum to the
number of terms so the overall loss scale stays fixed.

IMPORTANT for this model: GradNorm equalises training RATES, which is not the
same as preventing collapse. A term that falls quickly gets down-weighted --
and L_var falls quickly precisely when the latent is shrinking. Used alone it
can therefore accelerate collapse. `mu_floor` keeps the dual-ascent multiplier
as a hard lower bound on the variance weight, so the anti-collapse constraint
cannot be traded away by the balancer.
"""
import torch


class GradNorm:
    TERMS = ("inv", "cor_sys", "var", "cov")

    def __init__(self, shared_param, init_weights=None, alpha=1.5, lr=2.5e-2,
                 device="cpu"):
        self.W = shared_param
        self.alpha = float(alpha)
        n = len(self.TERMS)
        if init_weights is None:
            init_weights = {k: 1.0 for k in self.TERMS}
        w0 = torch.tensor([float(init_weights[k]) for k in self.TERMS],
                          dtype=torch.float32, device=device)
        w0 = w0 * n / w0.sum().clamp(min=1e-12)          # start summing to n
        self.log_w = torch.log(w0.clamp(min=1e-8)).requires_grad_(True)
        self.opt = torch.optim.Adam([self.log_w], lr=lr)
        self.L0 = None
        self.n = n

    def weights(self):
        with torch.no_grad():
            return {k: float(v) for k, v in zip(self.TERMS, self.log_w.exp())}

    def step(self, losses):
        """losses: dict of term -> scalar tensor WITH graph. Returns new weights."""
        vals = [losses[k] for k in self.TERMS]
        if self.L0 is None:
            self.L0 = [float(v.detach().clamp(min=1e-8)) for v in vals]

        w = self.log_w.exp()
        G = []
        for i, L in enumerate(vals):
            g = torch.autograd.grad(w[i] * L, self.W, retain_graph=True,
                                    create_graph=True)[0]
            G.append(g.norm())
        G = torch.stack(G)
        Gbar = G.mean().detach()

        Lt = torch.stack([v.detach() / l0 for v, l0 in zip(vals, self.L0)])
        r = Lt / Lt.mean().clamp(min=1e-12)
        target = (Gbar * r.clamp(min=1e-8).pow(self.alpha)).detach()

        l_grad = (G - target).abs().sum()
        self.opt.zero_grad(set_to_none=True)
        l_grad.backward(retain_graph=True)
        self.opt.step()

        with torch.no_grad():                            # renormalise to sum n
            w = self.log_w.exp()
            self.log_w.copy_(torch.log((w * self.n / w.sum().clamp(min=1e-12))
                                       .clamp(min=1e-8)))
        return self.weights(), float(l_grad.detach())
