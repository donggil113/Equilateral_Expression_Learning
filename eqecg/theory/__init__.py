"""Machine-checkable counterparts of the paper's theoretical claims."""

from eqecg.theory.intertwiners import (
    equivariant_linear_basis,
    predicted_intertwiner_dimension,
    reynolds_quadratic,
    reynolds_selfadjointness_defect,
)
from eqecg.theory.invariants import (
    invariant_dimension_series,
    total_dimension_series,
    invariant_fraction_table,
    monte_carlo_invariant_dimension,
)
from eqecg.theory.identifiability import (
    gram_invariant,
    triple_products,
    recover_rotation,
    identifiability_report,
)

__all__ = [
    "equivariant_linear_basis",
    "predicted_intertwiner_dimension",
    "reynolds_quadratic",
    "reynolds_selfadjointness_defect",
    "invariant_dimension_series",
    "total_dimension_series",
    "invariant_fraction_table",
    "monte_carlo_invariant_dimension",
    "gram_invariant",
    "triple_products",
    "recover_rotation",
    "identifiability_report",
]
