"""
disnet_sage: High-Performance SAGE (Switching-Aware Gradients) Layer
built directly upon the DisjunctiveNet.jl architecture.
"""

from .core import LinearConstraint, Disjunction, DisjunctiveModel, PolyhedralBranch
from .layers import SAGEProjectionLayer, DisNetProjectionLayer, ConstrainedModel, constrained_model

__all__ = [
    "LinearConstraint",
    "Disjunction",
    "DisjunctiveModel",
    "PolyhedralBranch",
    "SAGEProjectionLayer",
    "DisNetProjectionLayer",
    "ConstrainedModel",
    "constrained_model",
]

