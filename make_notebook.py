import json
from pathlib import Path

def build_standalone_notebook():
    cells = []

    def md(text):
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': [line + '\n' for line in text.strip().split('\n')]
        })

    def code(text):
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in text.strip().split('\n')]
        })

    # Header
    md("""# SAGE on DisjunctiveNet Official Benchmark (flux_end2end.jl)
### 100% Self-Contained Interactive Kaggle / Colab Notebook
- **Paper nền tảng:** DisjunctiveNet (ICML 2026, Shraman Pal & Can Li, Purdue University)
- **Phương pháp mới:** SAGE (Switching-Aware Gradients, ICML 2027 Proposal)
- **Đặc điểm:** Chạy độc lập hoàn toàn, chỉ cần 1 file `.ipynb` này là chạy từ đầu đến cuối trên Kaggle / Colab mà không cần cài thêm solver C++ hay upload thư mục ngoài!""")

    # Cell 1: Environment & Imports
    code("""import itertools
import time
from typing import List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device} | PyTorch version: {torch.__version__}")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')""")

    # Cell 2: Core DisNet Data Structures
    md(r"""## 1. Khung Cấu trúc Dữ liệu gốc của DisNet (Tương thích 1-1 với DisjunctiveNet.jl)
Bao gồm:
- `LinearConstraint(a, sense, b)`: Ràng buộc tuyến tính phẳng ($a^T y \le b$, $\ge$, $==$)
- `Disjunction(disjuncts, name)`: Mệnh đề tuyển chứa các luật HOẶC
- `DisjunctiveModel(n_outputs)`: Container chứa các luật và mở rộng sang dạng chuẩn tuyển DNF""")

    code("""class LinearConstraint:
    def __init__(self, a, sense: str, b: float):
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"sense must be '<=', '>=', or '==', got {sense}")
        self.a = np.array(a, dtype=np.float64)
        self.sense = sense
        self.b = float(b)

    def to_canonical(self) -> List[Tuple[np.ndarray, float]]:
        if self.sense == "<=":
            return [(self.a, self.b)]
        elif self.sense == ">=":
            return [(-self.a, -self.b)]
        elif self.sense == "==":
            return [(self.a, self.b), (-self.a, -self.b)]
        raise ValueError(f"Invalid sense {self.sense}")

class Disjunction:
    def __init__(self, disjuncts: List[List[LinearConstraint]], name: Optional[str] = None):
        self.disjuncts = disjuncts
        self.name = name

class _BranchProjectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, branch) -> Tensor:
        y_star = branch._solve_projection(z.detach())
        ctx.save_for_backward(z, y_star)
        ctx.branch = branch
        return y_star

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        z, y_star = ctx.saved_tensors
        branch = ctx.branch
        device, dtype = z.device, z.dtype
        grad_z = grad_output.clone()
        if branch.A.shape[0] > 0:
            A = branch.A.to(device, dtype)
            b = branch.b.to(device, dtype)
            viol = torch.abs(A @ y_star - b)
            active = (viol < 1e-3)
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
    def __init__(self, A: Tensor, b: Tensor, lo: Optional[Tensor] = None, hi: Optional[Tensor] = None):
        self.A = A.detach()
        self.b = b.detach()
        self.d = A.shape[1] if A.ndim > 1 else (lo.shape[0] if lo is not None else hi.shape[0])
        self.lo = lo.detach() if lo is not None else torch.full((self.d,), -float('inf'))
        self.hi = hi.detach() if hi is not None else torch.full((self.d,), float('inf'))
        self.is_pure_box = (A.shape[0] == 0) and (not torch.isinf(self.lo).any()) and (not torch.isinf(self.hi).any())

    def _solve_projection(self, z: Tensor) -> Tensor:
        lo = self.lo.to(z.device, z.dtype)
        hi = self.hi.to(z.device, z.dtype)
        if self.is_pure_box or self.A.shape[0] == 0:
            return torch.clamp(z, lo, hi)
        y = torch.clamp(z, lo, hi)
        A = self.A.to(z.device, z.dtype)
        b = self.b.to(z.device, z.dtype)
        if torch.all(A @ y <= b + 1e-6):
            return y
        if A.shape[0] == 1:
            A_row = A[0]
            b_val = b[0]
            lam_lo, lam_hi = 0.0, 50.0
            for _ in range(30):
                lam_mid = 0.5 * (lam_lo + lam_hi)
                y_cand = torch.clamp(z - lam_mid * A_row, lo, hi)
                if (A_row @ y_cand).item() > b_val.item():
                    lam_lo = lam_mid
                else:
                    lam_hi = lam_mid
            return torch.clamp(z - lam_hi * A_row, lo, hi)
        y_cur = y.clone()
        y_prev = y.clone()
        step_size = 0.1 / (torch.norm(A, p=2) ** 2 + 1e-4).item()
        for k in range(80):
            beta = (k - 1) / (k + 2) if k > 0 else 0.0
            v = y_cur + beta * (y_cur - y_prev)
            viol = torch.clamp(A @ v - b, min=0.0)
            grad = (v - z) + 1e3 * (A.T @ viol)
            y_next = torch.clamp(v - step_size * grad, lo, hi)
            y_prev = y_cur
            y_cur = y_next
            if torch.norm(y_cur - y_prev) < 1e-6 and torch.all(A @ y_cur <= b + 1e-5):
                break
        return y_cur

    def project_fast(self, z: Tensor) -> Tensor:
        if z.requires_grad:
            return _BranchProjectFn.apply(z, self)
        return self._solve_projection(z)

    def distance(self, z: Tensor) -> Tensor:
        y_star = self._solve_projection(z.detach())
        diff = y_star - z
        return torch.norm(diff, p=2) if diff.dim() == 1 else torch.norm(diff, p=2, dim=-1)

class DisjunctiveModel:
    def __init__(self, n_outputs: int):
        self.n_outputs = int(n_outputs)
        self.lb = np.full(self.n_outputs, -np.inf, dtype=np.float64)
        self.ub = np.full(self.n_outputs, np.inf, dtype=np.float64)
        self.constraints: List[LinearConstraint] = []
        self.disjunctions: List[Disjunction] = []

    def set_bounds(self, lower=None, upper=None):
        if lower is not None: self.lb = np.array(lower, dtype=np.float64)
        if upper is not None: self.ub = np.array(upper, dtype=np.float64)

    def add_linear_constraint(self, a, sense: str, b: float):
        self.constraints.append(LinearConstraint(a, sense, b))

    def add_disjunction(self, *disjuncts, name: Optional[str] = None):
        self.disjunctions.append(Disjunction(list(disjuncts), name=name))

    def to_dnf_branches(self) -> List[PolyhedralBranch]:
        d = self.n_outputs
        global_A = [ai for c in self.constraints for ai, _ in c.to_canonical()]
        global_b = [bi for c in self.constraints for _, bi in c.to_canonical()]
        if not self.disjunctions:
            A_m = torch.tensor(np.array(global_A), dtype=torch.float32) if global_A else torch.empty((0, d))
            b_v = torch.tensor(np.array(global_b), dtype=torch.float32) if global_b else torch.empty((0,))
            return [PolyhedralBranch(A_m, b_v, torch.tensor(self.lb, dtype=torch.float32), torch.tensor(self.ub, dtype=torch.float32))]
        branches = []
        for combo in itertools.product(*[d.disjuncts for d in self.disjunctions]):
            b_A = list(global_A)
            b_b = list(global_b)
            b_lb = np.copy(self.lb)
            b_ub = np.copy(self.ub)
            for clause in combo:
                for c in clause:
                    nz = np.nonzero(c.a)[0]
                    if len(nz) == 1 and abs(c.a[nz[0]]) == 1.0:
                        idx, s = nz[0], c.a[nz[0]]
                        if c.sense == "<=":
                            if s > 0: b_ub[idx] = min(b_ub[idx], c.b)
                            else: b_lb[idx] = max(b_lb[idx], -c.b)
                        elif c.sense == ">=":
                            if s > 0: b_lb[idx] = max(b_lb[idx], c.b)
                            else: b_ub[idx] = min(b_ub[idx], -c.b)
                    else:
                        for ai, bi in c.to_canonical():
                            b_A.append(ai); b_b.append(bi)
            if np.any(b_lb > b_ub):
                continue
            # Filter empty branches
            is_empty = False
            for ai, bi in zip(b_A, b_b):
                min_val = np.sum(np.where(ai >= 0, ai * b_lb, ai * b_ub))
                if min_val > bi + 1e-5:
                    is_empty = True
                    break
            if is_empty:
                continue
            A_m = torch.tensor(np.array(b_A), dtype=torch.float32) if b_A else torch.empty((0, d))
            b_v = torch.tensor(np.array(b_b), dtype=torch.float32) if b_b else torch.empty((0,))
            branches.append(PolyhedralBranch(A_m, b_v, torch.tensor(b_lb, dtype=torch.float32), torch.tensor(b_ub, dtype=torch.float32)))
        return branches""")

    # Cell 3: Differentiable Layers
    md("""## 2. Các Lớp Tối Ưu Hóa Khả Vi: DisNet gốc vs SAGE""")

    code("""class DisNetProjectionLayer(nn.Module):
    def __init__(self, model: DisjunctiveModel):
        super().__init__()
        self.branches = model.to_dnf_branches()

    def forward(self, z: Tensor) -> Tensor:
        dists = torch.stack([b.distance(z) for b in self.branches])
        k_star = int(torch.argmin(dists).item())
        z_diff = z.detach().clone().requires_grad_(True)
        y_diff = self.branches[k_star].project_fast(z_diff)
        return z + (y_diff - z.detach())

class SAGEProjectionLayer(nn.Module):
    def __init__(self, model: DisjunctiveModel, rho: float = 0.3, n_samples: int = 32):
        super().__init__()
        self.branches = model.to_dnf_branches()
        self.n_branches = len(self.branches)
        self.rho = rho
        self.n_samples = n_samples

    def forward(self, z: Tensor) -> Tensor:
        \"\"\"Forward projection onto nearest feasible DNF branch.\"\"\"
        dists = torch.stack([b.distance(z) for b in self.branches])
        k_star = int(torch.argmin(dists).item())
        return self.branches[k_star].project_fast(z)

    def compute_sage_gradient(self, z: Tensor, y_gt: Tensor, loss_fn: Callable):
        dists = torch.stack([b.distance(z) for b in self.branches])
        K = self.n_branches
        k0 = int(torch.argmin(dists).item())
        d_k0 = dists[k0]
        y_star = self.branches[k0].project_fast(z)

        others = torch.cat([dists[:k0], dists[k0+1:]]) if K > 1 else torch.tensor([float('inf')])
        margin = float((others - d_k0).min().item()) if K > 1 else float('inf')
        gaps = dists - d_k0
        competitors = (gaps <= 2.0 * self.rho).nonzero(as_tuple=True)[0].tolist()
        certified = (len(competitors) <= 1)

        z_loc = z.detach().clone().requires_grad_(True)
        y_loc = self.branches[k0].project_fast(z_loc)
        loss_loc = loss_fn(y_loc, y_gt)
        loss_loc.backward()
        grad_local = z_loc.grad.clone()

        local_is_dead = (grad_local.norm().item() < 1e-8)
        info = {'margin': margin, 'certified': certified, 'k0': k0}

        if certified and not local_is_dead:
            return grad_local, y_star, k0, info

        if local_is_dead and certified:
            competitors = list(range(K))
            info['certified'] = False
            rho = max(self.rho, float(margin) * 0.6 + 0.1)
        else:
            rho = self.rho

        d = z.shape[0]
        u = torch.randn(self.n_samples, d, device=z.device, dtype=z.dtype)
        R0_vals = torch.zeros(self.n_samples, device=z.device, dtype=z.dtype)

        for i in range(self.n_samples):
            z_p = z.detach() + rho * u[i]
            y_k0_p = self.branches[k0].project_fast(z_p)
            d_k0_p = (y_k0_p - z_p).norm()
            L0 = loss_fn(y_k0_p, y_gt).detach()

            best_d = d_k0_p
            winner_k = k0
            y_win = y_k0_p

            for j in competitors:
                if j == k0: continue
                y_j_p = self.branches[j].project_fast(z_p)
                d_j = (y_j_p - z_p).norm()
                if d_j < best_d:
                    best_d = d_j
                    winner_k = j
                    y_win = y_j_p

            R0_vals[i] = (loss_fn(y_win, y_gt).detach() - L0) if winner_k != k0 else 0.0

        cross_grad = (1.0 / rho) * (R0_vals.unsqueeze(-1) * u).mean(dim=0)
        return grad_local + cross_grad, y_star, k0, info

class ConstrainedModel(nn.Module):
    def __init__(self, backbone: nn.Module, layer: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.layer = layer
    def forward(self, x: Tensor) -> Tensor:
        return self.layer(self.backbone(x))

def constrained_model(backbone, model, layer_type='sage', rho=0.25, n_samples=32):
    layer = SAGEProjectionLayer(model, rho=rho, n_samples=n_samples) if layer_type == 'sage' else DisNetProjectionLayer(model)
    return ConstrainedModel(backbone, layer)""")

    # Cell 4: DisNet Official flux_end2end.jl Problem Setup
    md("""## 3. Khởi tạo Bài toán Chính thức của DisNet (flux_end2end.jl)
- **Global constraint:** $y_1 + y_2 \\ge 0.8$
- **Rule 1 ($x\\text{\\_rule}$):** $y_1 \\le 0.2$ OR $0.4 \\le y_1 \\le 0.6$ OR $y_1 \\ge 0.8$
- **Rule 2 ($y\\text{\\_rule}$):** $y_2 \\le 0.2$ OR $0.35 \\le y_2 \\le 0.55$ OR $y_2 \\ge 0.7$""")

    code("""dm = DisjunctiveModel(2)
dm.set_bounds(lower=[0.0, 0.0], upper=[1.0, 1.0])
dm.add_linear_constraint([1.0, 1.0], \">=\", 0.8)

dm.add_disjunction(
    [LinearConstraint([1.0, 0.0], \"<=\", 0.2)],
    [LinearConstraint([1.0, 0.0], \">=\", 0.4), LinearConstraint([1.0, 0.0], \"<=\", 0.6)],
    [LinearConstraint([1.0, 0.0], \">=\", 0.8)],
    name=\"x_rule\"
)

dm.add_disjunction(
    [LinearConstraint([0.0, 1.0], \"<=\", 0.2)],
    [LinearConstraint([0.0, 1.0], \">=\", 0.35), LinearConstraint([0.0, 1.0], \"<=\", 0.55)],
    [LinearConstraint([0.0, 1.0], \">=\", 0.7)],
    name=\"y_rule\"
)

branches = dm.to_dnf_branches()
print(f'✅ Khởi tạo thành công {len(branches)} nhánh DNF khả thi!')
for i, b in enumerate(branches):
    print(f'  • Nhánh {i}: y1 in [{b.lo[0]:.2f}, {b.hi[0]:.2f}], y2 in [{b.lo[1]:.2f}, {b.hi[1]:.2f}]')""")

    # Cell 5: Training & Comparison
    md("""## 4. Huấn luyện Đối sánh Trực tiếp: DisNet gốc vs SAGE""")

    code("""x = torch.tensor([0.2, 0.7, 0.4])
target = torch.tensor([0.85, 0.15])
loss_fn = lambda y, y_gt: 0.5 * (y - y_gt).pow(2).sum()

def make_backbone():
    torch.manual_seed(123)
    return nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))

model_disnet = constrained_model(make_backbone(), dm, layer_type=\"disnet\")
model_sage = constrained_model(make_backbone(), dm, layer_type=\"sage\", rho=0.25, n_samples=32)

epochs = 100
lr = 0.02
opt_d = torch.optim.Adam(model_disnet.backbone.parameters(), lr=lr)
opt_s = torch.optim.Adam(model_sage.backbone.parameters(), lr=lr)

hist_d, hist_s = [], []
print('Bắt đầu huấn luyện...')
t0 = time.time()

for ep in range(1, epochs + 1):
    # DisNet
    opt_d.zero_grad()
    y_d = model_disnet(x)
    l_d = loss_fn(y_d, target)
    l_d.backward()
    opt_d.step()
    hist_d.append(l_d.item())

    # SAGE
    opt_s.zero_grad()
    z_s = model_sage.backbone(x)
    s_grad, y_s, k_s, info = model_sage.layer.compute_sage_gradient(z_s, target, loss_fn)
    l_s = loss_fn(y_s, target)
    z_s.backward(s_grad)
    opt_s.step()
    hist_s.append(l_s.item())

    if ep % 20 == 0 or ep == 1:
        print(f'Epoch {ep:3d} | DisNet Loss: {l_d.item():.6f} | SAGE Loss: {l_s.item():.6f} | SAGE Branch: {k_s} | Certified: {info[\"certified\"]}')

print(f'Huấn luyện xong trong {time.time()-t0:.2f} giây!')

# Trực quan hóa
plt.figure(figsize=(9, 4.5))
plt.plot(hist_d, 'r--', label='DisNet (Original KKT)', lw=2)
plt.plot(hist_s, 'g-', label='SAGE (Proposed)', lw=2.5)
plt.title('Training Convergence on DNF Branches (flux_end2end.jl)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

with torch.no_grad():
    res_d = model_disnet(x)
    res_s = model_sage(x)

print('=' * 65)
print(f'Target mong muốn   : {target.numpy()}')
print(f'DisNet Nghiệm cuối : {res_d.numpy().round(4)} | Loss: {hist_d[-1]:.6f} | Hợp lệ (y1+y2>=0.8): {res_d.sum().item() >= 0.7999}')
print(f'SAGE Nghiệm cuối   : {res_s.numpy().round(4)} | Loss: {hist_s[-1]:.6f} | Hợp lệ (y1+y2>=0.8): {res_s.sum().item() >= 0.7999}')
print('=' * 65)""")

    nb = {
        'cells': cells,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
        'nbformat': 4,
        'nbformat_minor': 4
    }

    out_p = Path(r'D:\Downloads\Lab_resource\Task5-MHoang\DisNet\SAGE_DisNet_Official_Benchmark.ipynb')
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f'✅ Rebuilt {out_p} as 100% STANDALONE notebook!')

if __name__ == '__main__':
    build_standalone_notebook()
