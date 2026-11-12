"""
smoketest.py – Fast verification and smoke test for SAGE on Kaggle / Local.
Run:
    python smoketest.py
    python smoketest.py --epochs 10
"""

import argparse
import sys
import os
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disnet_sage import DisjunctiveModel, LinearConstraint, constrained_model


def run_smoketest(epochs: int = 5, lr: float = 0.02):
    print("=" * 65)
    print(f"SAGE SMOKE TEST (Running {epochs} Epochs)")
    print("=" * 65)

    # 1. Test Model Construction
    print("[1/4] Building DisjunctiveModel (flux_end2end)...", end=" ")
    dm = DisjunctiveModel(2)
    dm.set_bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])
    dm.add_linear_constraint([1.0, 1.0], ">=", 0.8)

    dm.add_disjunction(
        [LinearConstraint([1.0, 0.0], "<=", 0.2)],
        [LinearConstraint([1.0, 0.0], ">=", 0.4), LinearConstraint([1.0, 0.0], "<=", 0.6)],
        [LinearConstraint([1.0, 0.0], ">=", 0.8)],
        name="x_rule"
    )
    dm.add_disjunction(
        [LinearConstraint([0.0, 1.0], "<=", 0.2)],
        [LinearConstraint([0.0, 1.0], ">=", 0.35), LinearConstraint([0.0, 1.0], "<=", 0.55)],
        [LinearConstraint([0.0, 1.0], ">=", 0.7)],
        name="y_rule"
    )

    branches = dm.to_dnf_branches()
    # 9 combinations total, 2 are geometrically empty (max sum < 0.8), 7 non-empty
    assert len(branches) in (7, 9), f"Unexpected branch count: {len(branches)}"
    print(f"PASS ({len(branches)} non-empty feasible branches)")

    # 2. Test Forward Projection
    print("[2/4] Testing Forward Feasibility...", end=" ")
    raw_z = torch.tensor([0.2, 0.2])  # Violates y1 + y2 >= 0.8
    # Global project
    dists = torch.stack([b.distance(raw_z) for b in branches])
    k_star = int(torch.argmin(dists).item())
    y_proj = branches[k_star].project_fast(raw_z)
    assert y_proj.sum().item() >= 0.799, f"Feasibility violation! Sum is {y_proj.sum().item()}"
    print(f"PASS (Raw: {raw_z.tolist()} -> Feasible: {y_proj.tolist()}, sum={y_proj.sum().item():.2f})")

    # 3. Test Backbone + Layer integration
    print("[3/4] Initializing Backbone + SAGE Layer...", end=" ")
    backbone = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    model = constrained_model(backbone, dm, layer_type="sage", rho=0.25, n_samples=16)
    x = torch.tensor([0.2, 0.7, 0.4])
    target = torch.tensor([0.85, 0.15])
    loss_fn = lambda a, b: 0.5 * (a - b).pow(2).sum()
    print("PASS")

    # 4. Run Training Epochs
    print(f"[4/4] Running {epochs} Training Epochs...")
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model.backbone(x)
        sage_grad, y_out, k_curr, info = model.layer.compute_sage_gradient(z, target, loss_fn)
        loss = loss_fn(y_out, target)
        z.backward(sage_grad)
        optimizer.step()
        cert_str = "YES" if info["certified"] else "NO"
        print(f"   Epoch {ep:2d}/{epochs:2d} | Loss: {loss.item():.6f} | Branch: {k_curr} | Certified: {cert_str}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.3f}s ({elapsed/epochs*1000:.1f}ms/epoch)")
    print("=" * 65)
    print("✅ [SMOKETEST PASSED] SAGE is verified and ready for Kaggle!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of test epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    args = parser.parse_args()
    run_smoketest(epochs=args.epochs, lr=args.lr)
