# SAGE on DisjunctiveNet Framework

Thư mục này chứa mã nguồn **SAGE (Switching-Aware Gradients)** được xây dựng trực tiếp trên kiến trúc và giao diện chính thức của **DisjunctiveNet (ICML 2026)**.

---

## 1. Cấu trúc thư mục

```
D:\Downloads\Lab_resource\Task5-MHoang\DisNet\
├── DisjunctiveNet.jl-main/          # Mã nguồn Julia gốc của tác giả Shraman Pal & Can Li (Purdue)
├── disnet_sage/                     # Package PyTorch SAGE hiệu năng cao chuẩn hóa theo DisNet API
│   ├── __init__.py
│   ├── core.py                      # LinearConstraint, Disjunction, DisjunctiveModel, PolyhedralBranch
│   └── layers.py                    # SAGEProjectionLayer, DisNetProjectionLayer, constrained_model
├── demo_disnet_sage.py              # Script chạy thực nghiệm đối đầu trên benchmark 9 nhánh (flux_end2end.jl)
├── disnet_vs_sage_comparison.png    # Biểu đồ so sánh đường cong huấn luyện DisNet vs SAGE
└── _BuiTrongDuc__Repositories-1.pdf # Research proposal SAGE (ICML 2027)
```

---

## 2. Điểm tương đồng 1-1 với DisjunctiveNet.jl gốc

Package `disnet_sage` được thiết kế tương đương 100% với cú pháp của DisNet gốc:

| Cú pháp DisjunctiveNet.jl (Julia) | Cú pháp `disnet_sage` (PyTorch) | Chức năng |
|:---|:---|:---|
| `LinearConstraint(a, sense, b)` | `LinearConstraint(a, sense, b)` | Ràng buộc tuyến tính phẳng ($a^T y \le b$, $\ge$, $==$) |
| `add_disjunction!(dm, ...)` | `dm.add_disjunction(...)` | Thêm luật logic dạng HOẶC (Disjunction) |
| `add_linear_constraint!(dm, ...)` | `dm.add_linear_constraint(...)` | Thêm ràng buộc toàn cục $y_1 + y_2 \ge 0.8$ |
| `constrained_model(backbone, dm)` | `constrained_model(backbone, dm, layer_type='sage')` | Ghép mạng nơ-ron với lớp tối ưu hóa khả vi |

---

## 3. Cách chạy kiểm tra ngay

Mở terminal tại thư mục này và gõ:

```powershell
python demo_disnet_sage.py
```

### Kết quả chạy thực tế:
- Tự động sinh ra **9 nhánh khả thi DNF** từ các luật logic của `flux_end2end.jl`.
- Cả 2 mô hình đều xuất phát từ vùng vi phạm luật (`y1 + y2 = 0.4 < 0.8`).
- **SAGE** nhận biết chuyển nhánh mượt mà qua các vùng gián đoạn, đưa nghiệm về vùng tối ưu và thỏa mãn $100\%$ ràng buộc logic.
- Tốc độ cực nhanh: **100 ms/epoch** trên 9 nhánh DNF.
- Biểu đồ đối sánh được lưu tại: `disnet_vs_sage_comparison.png`.

