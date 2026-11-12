"""
demo_disnet_sage.py – Official end-to-end demo replicating DisNet's flux_end2end.jl
Runs DisNet vs SAGE on the exact 2-rule, 9-branch disjunctive constraint model:
- Global constraint: y1 + y2 >= 0.8
- Rule 1 (x_rule): y1 <= 0.2 OR 0.4 <= y1 <= 0.6 OR y1 >= 0.8
- Rule 2 (y_rule): y2 <= 0.2 OR 0.35 <= y2 <= 0.55 OR y2 >= 0.7
"""

import sys
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure local import works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disnet_sage import DisjunctiveModel, LinearConstraint, constrained_model


def build_disnet_model():
    """Builds the exact DisjunctiveModel from DisjunctiveNet.jl's flux_end2end.jl."""
    dm = DisjunctiveModel(2)
    dm.set_bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])

    # Global constraint: y1 + y2 >= 0.8
    dm.add_linear_constraint([1.0, 1.0], ">=", 0.8)

    # Rule 1 (3 disjuncts on y1):
    # y1 <= 0.2 OR 0.4 <= y1 <= 0.6 OR y1 >= 0.8
    dm.add_disjunction(
        [LinearConstraint([1.0, 0.0], "<=", 0.2)],
        [LinearConstraint([1.0, 0.0], ">=", 0.4), LinearConstraint([1.0, 0.0], "<=", 0.6)],
        [LinearConstraint([1.0, 0.0], ">=", 0.8)],
        name="x_rule"
    )

    # Rule 2 (3 disjuncts on y2):
    # y2 <= 0.2 OR 0.35 <= y2 <= 0.55 OR y2 >= 0.7
    dm.add_disjunction(
        [LinearConstraint([0.0, 1.0], "<=", 0.2)],
        [LinearConstraint([0.0, 1.0], ">=", 0.35), LinearConstraint([0.0, 1.0], "<=", 0.55)],
        [LinearConstraint([0.0, 1.0], ">=", 0.7)],
        name="y_rule"
    )

    return dm


def main():
    print("=" * 70)
    print("DEMO: SAGE on DisjunctiveNet.jl Official Benchmark (flux_end2end.jl)")
    print("=" * 70)

    torch.manual_seed(42)
    dm = build_disnet_model()
    branches = dm.to_dnf_branches()
    print(f"Generated {len(branches)} feasible DNF branches.")

    # 1. Inputs & Target (matching flux_end2end.jl)
    x = torch.tensor([0.2, 0.7, 0.4])
    # Target chosen in branch: y1 >= 0.8 and y2 >= 0.0 (sum >= 0.8)
    target = torch.tensor([0.85, 0.15])

    loss_fn = lambda y, y_gt: 0.5 * (y - y_gt).pow(2).sum()

    # 2. Two identical backbones initialized in a suboptimal branch
    def make_backbone():
        torch.manual_seed(123)
        return nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    bb_disnet = make_backbone()
    bb_sage = make_backbone()

    model_disnet = constrained_model(bb_disnet, dm, layer_type="disnet")
    model_sage = constrained_model(bb_sage, dm, layer_type="sage", rho=0.25, n_samples=32)

    # Check initial projection
    with torch.no_grad():
        init_disnet = model_disnet(x)
        init_sage = model_sage(x)
    print(f"Initial raw DisNet prediction : {init_disnet.numpy().round(4)} | Sum: {init_disnet.sum().item():.4f}")
    print(f"Initial raw SAGE prediction   : {init_sage.numpy().round(4)} | Sum: {init_sage.sum().item():.4f}")

    epochs = 100
    lr = 0.02
    opt_disnet = torch.optim.Adam(bb_disnet.parameters(), lr=lr)
    opt_sage = torch.optim.Adam(bb_sage.parameters(), lr=lr)

    history_disnet = []
    history_sage = []

    print("\nTraining DisNet baseline vs SAGE...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        # --- DisNet Step ---
        opt_disnet.zero_grad()
        y_d = model_disnet(x)
        l_d = loss_fn(y_d, target)
        l_d.backward()
        opt_disnet.step()
        history_disnet.append(l_d.item())

        # --- SAGE Step ---
        opt_sage.zero_grad()
        # SAGE uses compute_sage_gradient on backbone(x)
        z_s = bb_sage(x)
        sage_grad, y_s, k_s, info = model_sage.layer.compute_sage_gradient(z_s, target, loss_fn)
        l_s = loss_fn(y_s, target)
        z_s.backward(sage_grad)
        opt_sage.step()
        history_sage.append(l_s.item())

        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:3d} | DisNet Loss: {l_d.item():.6f} | SAGE Loss: {l_s.item():.6f} | SAGE Branch: {k_s} | Certified: {info['certified']}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f} seconds ({elapsed/epochs*1000:.2f} ms/epoch).")

    # Evaluation
    with torch.no_grad():
        final_d = model_disnet(x)
        final_s = model_sage(x)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION:")
    print(f"Target          : {target.numpy().round(4)}")
    print(f"DisNet Final    : {final_d.numpy().round(4)} | Final Loss: {history_disnet[-1]:.6f} | Valid: {final_d.sum().item() >= 0.7999}")
    print(f"SAGE Final      : {final_s.numpy().round(4)} | Final Loss: {history_sage[-1]:.6f} | Valid: {final_s.sum().item() >= 0.7999}")

    improvement = (history_disnet[-1] - history_sage[-1]) / max(history_disnet[-1], 1e-6) * 100
    print(f"Loss Improvement by SAGE: {improvement:.2f}%")
    print("=" * 70)

    # Plot results
    plt.figure(figsize=(9, 4.5))
    plt.plot(history_disnet, 'r--', label='DisNet (Original)', lw=2)
    plt.plot(history_sage, 'g-', label='SAGE (Proposed)', lw=2.5)
    plt.title('DisNet vs SAGE on Official DisjunctiveNet Benchmark (9 Branches)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disnet_vs_sage_comparison.png")
    plt.savefig(out_png, dpi=150)
    print(f"✅ Comparison plot saved to {out_png}")


if __name__ == "__main__":
    main()

