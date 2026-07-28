"""Gradient sanity check: PyTorch autograd vs. finite differences.

Runs the real PARADIS calibration pipeline (steps 1-4: presence threshold,
carrying capacity, calibration-site selection, posteriors) for a single
species (*Castor fiber*, European beaver) using real data, then verifies
that the autograd gradients of ``cost_function`` w.r.t. (n, r, Tg) match
finite-difference gradients computed by perturbing each parameter and
re-evaluating the cost.

This does NOT run the full Adam optimisation loop — it evaluates the cost
function's gradient at a single fixed parameter point, which is the
information that actually matters for verifying autograd correctness.

Usage
-----
    python examples/gradient_check_castor_fiber.py
"""

import pathlib

import numpy as np
import torch
from PIL import Image

from paradis._device import device
from paradis.calibration.presence import calibrate_presence_threshold
from paradis.calibration.carrying_capacity import estimate_carrying_capacity
from paradis.calibration.sites import sample_calibration_sites
from paradis.calibration.ratios import compute_all_posteriors
from paradis.core.adjacency import adjacency_matrix_torch
from paradis.calibration.carrying_capacity import logistic as _logistic
from paradis.learning.optimizer import cost_function

# ---------------------------------------------------------------------------
# 1. Load real data for Castor fiber
# ---------------------------------------------------------------------------

DATA = pathlib.Path(r"C:\Users\hoare\OneDrive\Bureau\Python VScode\PARADIS\learn_params_AVE_MAM")

hs   = np.array(Image.open(DATA / "HS_mammals" / "Castor_fiber.tif")).astype(float)
obs  = np.array(Image.open(DATA / "Obs_mammals" / "Castor fiber.tif")).astype(float)
cr   = np.array(Image.open(DATA / "Current_Range_mammals" / "Castor fiberbinary_50.tif")).astype(float)
taxa_ref = np.array(Image.open(DATA / "sampling_effort_small_mammalia.tif")).astype(float)

hs[hs == np.nanmax(hs)] = 0.0
hs = hs / np.nanmax(hs)
cr[cr == np.nanmin(cr)] = 0.0
cr = cr / np.nanmax(cr)
obs[obs < 0] = 0.0

mdd_val = 3.7513465009  # Castor fiber, from Dispersal_mammals.csv
hmean = float(hs[cr > 0].mean())

print(f"[data] hs shape={hs.shape}  hmean={hmean:.4f}  mdd={mdd_val}")

# ---------------------------------------------------------------------------
# 2. Presence threshold + carrying capacity
# ---------------------------------------------------------------------------

print("\n[step 1] Calibrating presence threshold ...")
pres_thresh = calibrate_presence_threshold(cr, obs, taxa_ref, plot=False, verbose=False)

print("[step 2] Fitting carrying capacity K=f(HS) ...")
L, k, x0 = estimate_carrying_capacity(
    hs, obs, taxa_ref, cr, plot=False, presence_threshold=pres_thresh,
)
print(f"  L={L:.4f}  k={k:.4f}  x0={x0:.4f}")

# ---------------------------------------------------------------------------
# 3. Calibration sites + posteriors
# ---------------------------------------------------------------------------

print("\n[step 3] Selecting calibration sites (this may take a moment) ...")
calib_sites = sample_calibration_sites(
    hs, obs, taxa_ref,
    n_samples=800, window_size=70, distmin=80,
    time_budget=30.0, plot=False, verbose=False,
)
print(f"  {len(calib_sites)} calibration sites found.")

print("[step 4] Computing observation-ratio posteriors ...")
posts_masks = compute_all_posteriors(calib_sites, verbose=False)

# ---------------------------------------------------------------------------
# 4. Precompute adjacency matrices + K_is (same as learn_dispersal_parameters)
# ---------------------------------------------------------------------------

print("\n[precompute] Building adjacency matrices and K_is ...")
adj_mats, K_is_list = [], []
for hs_map in calib_sites.hs_maps:
    hs_t = torch.tensor(hs_map, dtype=torch.float32, device=device)
    adj_mats.append(adjacency_matrix_torch(hs_t))
    K_is_list.append(torch.tensor(
        _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
        dtype=torch.float32,
    ))

carrying_capacity_params = (L, k, x0)
batch_indices = list(range(len(calib_sites)))  # use ALL sites (deterministic cost, no mini-batch noise)

# ---------------------------------------------------------------------------
# 5. Evaluate cost_function at a fixed point and get autograd gradients
# ---------------------------------------------------------------------------

# A reasonable interior point for (n, r, Tg) — not at the parameter bounds,
# so finite differences on both sides stay valid.
n0, r0, tg0 = 200.0, 0.01, 3.0

n_param  = torch.tensor(n0,  requires_grad=True, dtype=torch.float32, device=device)
r_param  = torch.tensor(r0,  requires_grad=True, dtype=torch.float32, device=device)
tg_param = torch.tensor(tg0, requires_grad=True, dtype=torch.float32, device=device)

print(f"\n[autograd] Evaluating cost_function at n={n0}, r={r0}, Tg={tg0} ...")
loss = cost_function(
    mdd_val, posts_masks, calib_sites,
    (n_param, r_param, tg_param), batch_indices,
    carrying_capacity_params, hmean, adj_mats, K_is_list,
    plot=False, verbose=False,
)
loss.backward()

grad_autograd = {
    "n":  n_param.grad.item(),
    "r":  r_param.grad.item(),
    "Tg": tg_param.grad.item(),
}
loss_val = loss.item()
print(f"  cost = {loss_val:.6f}")
print(f"  autograd grads: n={grad_autograd['n']:+.6f}  "
      f"r={grad_autograd['r']:+.6f}  Tg={grad_autograd['Tg']:+.6f}")

# ---------------------------------------------------------------------------
# 6. Finite-difference gradients (central differences)
# ---------------------------------------------------------------------------

def eval_cost(n_val: float, r_val: float, tg_val: float) -> float:
    with torch.no_grad():
        n_t  = torch.tensor(n_val,  dtype=torch.float32, device=device)
        r_t  = torch.tensor(r_val,  dtype=torch.float32, device=device)
        tg_t = torch.tensor(tg_val, dtype=torch.float32, device=device)
        c = cost_function(
            mdd_val, posts_masks, calib_sites,
            (n_t, r_t, tg_t), batch_indices,
            carrying_capacity_params, hmean, adj_mats, K_is_list,
            plot=False, verbose=False,
        )
    return c.item()

# Step sizes chosen relative to each parameter's natural scale.
eps_n, eps_r, eps_tg = 1.0, 1e-4, 0.05

print("\n[finite-diff] Evaluating cost at perturbed points ...")
fd_grad_n  = (eval_cost(n0 + eps_n,  r0, tg0) - eval_cost(n0 - eps_n,  r0, tg0)) / (2 * eps_n)
fd_grad_r  = (eval_cost(n0, r0 + eps_r,  tg0) - eval_cost(n0, r0 - eps_r,  tg0)) / (2 * eps_r)
fd_grad_tg = (eval_cost(n0, r0, tg0 + eps_tg) - eval_cost(n0, r0, tg0 - eps_tg)) / (2 * eps_tg)

grad_fd = {"n": fd_grad_n, "r": fd_grad_r, "Tg": fd_grad_tg}

# ---------------------------------------------------------------------------
# 6b. Eps sweep for r: the equilibrium solve is a nonlinear iterative system,
# so finite-difference accuracy is sensitive to step size. If the
# finite-difference estimate converges toward the autograd value as eps
# shrinks, the discrepancy is truncation error, not a gradient bug.
# ---------------------------------------------------------------------------

print("\n[eps sweep] Checking finite-difference convergence for r ...")
print(f"{'eps_r':>10}{'fd_grad_r':>15}{'rel diff to autograd':>25}")
for eps_r_test in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    fd_r_test = (eval_cost(n0, r0 + eps_r_test, tg0) - eval_cost(n0, r0 - eps_r_test, tg0)) / (2 * eps_r_test)
    rel = abs(fd_r_test - grad_autograd["r"]) / (abs(grad_autograd["r"]) + 1e-12)
    print(f"{eps_r_test:>10.0e}{fd_r_test:>15.6f}{rel:>24.4%}")

# ---------------------------------------------------------------------------
# 7. Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print(f"{'param':<6}{'autograd':>15}{'finite-diff':>15}{'abs diff':>15}{'rel diff':>12}")
print("-" * 70)
for p in ["n", "r", "Tg"]:
    ag = grad_autograd[p]
    fd = grad_fd[p]
    abs_diff = abs(ag - fd)
    rel_diff = abs_diff / (abs(fd) + 1e-12)
    print(f"{p:<6}{ag:>15.6f}{fd:>15.6f}{abs_diff:>15.6f}{rel_diff:>12.4%}")
print("=" * 70)

all_ok = all(
    abs(grad_autograd[p] - grad_fd[p]) / (abs(grad_fd[p]) + 1e-8) < 0.05
    for p in ["n", "r", "Tg"]
)
if all_ok:
    print("\n[PASS] Autograd gradients match finite differences (< 5% relative error).")
else:
    print("\n[FAIL] Autograd gradients diverge from finite differences by > 5%.")
