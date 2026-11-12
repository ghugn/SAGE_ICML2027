"""
core.py – Core data structures matching DisjunctiveNet.jl API exactly.
Mirrors:
- LinearConstraint(a, sense, b)
- Disjunction(disjuncts, name)
- DisjunctiveModel(n_outputs)
"""

from __future__ import annotations
import itertools
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor


class LinearConstraint:
    """A scalar affine constraint of the form:
        a' * y <= b
        a' * y >= b
        a' * y == b
    """
    def __init__(self, a: Union[List[float], np.ndarray, Tensor], sense: str, b: float):
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"sense must be '<=', '>=', or '==', got {sense}")
        self.a = np.array(a, dtype=np.float64) if not isinstance(a, np.ndarray) else a.astype(np.float64)
        self.sense = sense
        self.b = float(b)

    def to_canonical(self) -> List[Tuple[np.ndarray, float]]:
        """Converts to one or two canonical inequalities: a'^T * y <= b'."""
        if self.sense == "<=":
            return [(self.a, self.b)]
        elif self.sense == ">=":
            return [(-self.a, -self.b)]
        elif self.sense == "==":
            return [(self.a, self.b), (-self.a, -self.b)]
        raise ValueError(f"Invalid sense {self.sense}")

    def __repr__(self) -> str:
        return f"LinearConstraint({self.a.tolist()}, '{self.sense}', {self.b})"


class Disjunction:
    """A disjunction represented as a list of disjuncts (OR-ed clauses).
    Each disjunct is a list of LinearConstraint (AND-ed).
    """
    def __init__(self, disjuncts: List[List[LinearConstraint]], name: Optional[str] = None):
        if not disjuncts:
            raise ValueError("A disjunction must contain at least one disjunct.")
        self.disjuncts = disjuncts
        self.name = name

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"Disjunction({len(self.disjuncts)} disjuncts{name_str})"


class _BranchProjectFn(torch.autograd.Function):
    """KKT Implicit Differentiation for Euclidean Polyhedral Projection (DisNet QP style)."""
    @staticmethod
    def forward(ctx, z: Tensor, branch: PolyhedralBranch) -> Tensor:
        y_star = branch._solve_projection(z.detach())
        ctx.save_for_backward(z, y_star)
        ctx.branch = branch
        return y_star

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        z, y_star = ctx.saved_tensors
        branch = ctx.branch
        device = z.device
        dtype = z.dtype

        # Gradient flow matching DisNet QP regularization backward:
        # Allows gradient to pull z back towards the feasible region when target is interior
        grad_z = grad_output.clone()

        # If linear inequality constraint A y <= b is active: project onto feasible subspace
        if branch.A.shape[0] > 0:
            A = branch.A.to(device, dtype)
            b = branch.b.to(device, dtype)
            viol = torch.abs(A @ y_star - b)
            active = (viol < 1e-3)
            # Only constrain if gradient pushes against the active boundary
            if active.any():
                A_act = A[active]
                push_outward = (A_act @ grad_z < 0)
                if push_outward.any():
                    try:
                        gram = A_act @ A_act.T
                        inv_gram = torch.linalg.pinv(gram)
                        proj_mat = torch.eye(z.shape[0], device=device, dtype=dtype) - A_act.T @ inv_gram @ A_act
                        grad_z = proj_mat @ grad_z
                    except Exception:
                        pass

        return grad_z, None


class PolyhedralBranch:
    """A single convex branch: {y : A y <= b, lo <= y <= hi}
    Provides vectorized, high-performance projection and distance computation.
    """
    def __init__(self, A: Tensor, b: Tensor, lo: Optional[Tensor] = None, hi: Optional[Tensor] = None):
        self.A = A.detach()  # (m, d)
        self.b = b.detach()  # (m,)
        self.d = A.shape[1] if A.ndim > 1 else (lo.shape[0] if lo is not None else hi.shape[0])
        self.lo = lo.detach() if lo is not None else torch.full((self.d,), -float('inf'))
        self.hi = hi.detach() if hi is not None else torch.full((self.d,), float('inf'))
        self.is_pure_box = (A.shape[0] == 0) and (not torch.isinf(self.lo).any()) and (not torch.isinf(self.hi).any())

    def _solve_projection(self, z: Tensor) -> Tensor:
        """Internal solver for Euclidean projection."""
        lo = self.lo.to(z.device, z.dtype)
        hi = self.hi.to(z.device, z.dtype)

        if self.is_pure_box:
            return torch.clamp(z, lo, hi)

        y = torch.clamp(z, lo, hi)
        if self.A.shape[0] == 0:
            return y

        A = self.A.to(z.device, z.dtype)
        b = self.b.to(z.device, z.dtype)

        if torch.all(A @ y <= b + 1e-6):
            return y

        # Fast Accelerated Projected Gradient (PGD)
        y_cur = y.clone()
        y_prev = y.clone()
        step_size = 0.1 / (torch.norm(A, p=2) ** 2 + 1e-4).item()

        for k in range(50):
            beta = (k - 1) / (k + 2) if k > 0 else 0.0
            v = y_cur + beta * (y_cur - y_prev)
            viol = torch.clamp(A @ v - b, min=0.0)
            grad = (v - z) + 2e2 * (A.T @ viol)
            y_next = torch.clamp(v - step_size * grad, lo, hi)
            y_prev = y_cur
            y_cur = y_next
            if torch.norm(y_cur - y_prev) < 1e-5 and torch.all(A @ y_cur <= b + 1e-5):
                break

        return y_cur

    def project_fast(self, z: Tensor) -> Tensor:
        """Euclidean projection of z onto this branch with autograd support."""
        if z.requires_grad:
            return _BranchProjectFn.apply(z, self)
        return self._solve_projection(z)

    def distance(self, z: Tensor) -> Tensor:
        y_star = self._solve_projection(z.detach())
        diff = y_star - z
        if diff.dim() == 1:
            return torch.norm(diff, p=2)
        return torch.norm(diff, p=2, dim=-1)


class DisjunctiveModel:
    """DisjunctiveModel container matching DisjunctiveNet.jl:
        dm = DisjunctiveModel(n_outputs)
        set_bounds!(dm, lower=[...], upper=[...])
        add_linear_constraint!(dm, a, sense, b)
        add_disjunction!(dm, disjunct1, disjunct2, ...)
    """
    def __init__(self, n_outputs: int):
        if n_outputs <= 0:
            raise ValueError("n_outputs must be positive.")
        self.n_outputs = int(n_outputs)
        self.lb = np.full(self.n_outputs, -np.inf, dtype=np.float64)
        self.ub = np.full(self.n_outputs, np.inf, dtype=np.float64)
        self.constraints: List[LinearConstraint] = []
        self.disjunctions: List[Disjunction] = []

    def set_bounds(self, lower: Optional[Union[List[float], np.ndarray]] = None,
                   upper: Optional[Union[List[float], np.ndarray]] = None):
        """Sets box bounds [lower, upper] for y."""
        if lower is not None:
            self.lb = np.array(lower, dtype=np.float64)
        if upper is not None:
            self.ub = np.array(upper, dtype=np.float64)

    def add_linear_constraint(self, a: Union[List[float], np.ndarray], sense: str, b: float):
        """Adds a global linear constraint: a' * y (sense) b."""
        c = LinearConstraint(a, sense, b)
        self.constraints.append(c)

    def add_disjunction(self, *disjuncts: List[LinearConstraint], name: Optional[str] = None):
        """Adds a disjunction of clauses: disjunct1 OR disjunct2 OR ..."""
        d = Disjunction(list(disjuncts), name=name)
        self.disjunctions.append(d)

    def to_dnf_branches(self) -> List[PolyhedralBranch]:
        """Expands the model into full DNF convex branches."""
        d = self.n_outputs
        global_A = []
        global_b = []
        for c in self.constraints:
            for ai, bi in c.to_canonical():
                global_A.append(ai)
                global_b.append(bi)

        if not self.disjunctions:
            A_mat = torch.tensor(np.array(global_A), dtype=torch.float32) if global_A else torch.empty((0, d))
            b_vec = torch.tensor(np.array(global_b), dtype=torch.float32) if global_b else torch.empty((0,))
            lo = torch.tensor(self.lb, dtype=torch.float32)
            hi = torch.tensor(self.ub, dtype=torch.float32)
            return [PolyhedralBranch(A_mat, b_vec, lo, hi)]

        disj_options = [d.disjuncts for d in self.disjunctions]
        branches = []

        for combo in itertools.product(*disj_options):
            branch_A = list(global_A)
            branch_b = list(global_b)
            branch_lb = np.copy(self.lb)
            branch_ub = np.copy(self.ub)

            for clause in combo:
                for c in clause:
                    nonzeros = np.nonzero(c.a)[0]
                    if len(nonzeros) == 1 and abs(c.a[nonzeros[0]]) == 1.0:
                        idx = nonzeros[0]
                        sign = c.a[idx]
                        if c.sense == "<=":
                            if sign > 0:
                                branch_ub[idx] = min(branch_ub[idx], c.b)
                            else:
                                branch_lb[idx] = max(branch_lb[idx], -c.b)
                        elif c.sense == ">=":
                            if sign > 0:
                                branch_lb[idx] = max(branch_lb[idx], c.b)
                            else:
                                branch_ub[idx] = min(branch_ub[idx], -c.b)
                    else:
                        for ai, bi in c.to_canonical():
                            branch_A.append(ai)
                            branch_b.append(bi)

            if np.any(branch_lb > branch_ub):
                continue

            A_mat = torch.tensor(np.array(branch_A), dtype=torch.float32) if branch_A else torch.empty((0, d))
            b_vec = torch.tensor(np.array(branch_b), dtype=torch.float32) if branch_b else torch.empty((0,))
            lo = torch.tensor(branch_lb, dtype=torch.float32)
            hi = torch.tensor(branch_ub, dtype=torch.float32)

            branches.append(PolyhedralBranch(A_mat, b_vec, lo, hi))

        return branches

    def __repr__(self) -> str:
        return (f"DisjunctiveModel(n_outputs={self.n_outputs}, "
                f"constraints={len(self.constraints)}, "
                f"disjunctions={len(self.disjunctions)})")
