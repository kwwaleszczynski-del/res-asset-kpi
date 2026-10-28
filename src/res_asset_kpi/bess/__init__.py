"""Battery storage: round-trip efficiency, LTSA availability, cycles and rainflow, capacity tests."""

from res_asset_kpi.bess.availability import (
    PRIORITY,
    TREATMENTS,
    interval_treatment,
    ltsa_availability,
    measured_availability,
)
from res_asset_kpi.bess.capacity import (
    CapacityTest,
    UncertaintyBudget,
    discharge_segment,
    evaluate_capacity_test,
    relative_uncertainty,
)
from res_asset_kpi.bess.cycles import (
    DEPTH_EDGES,
    depth_histogram,
    equivalent_full_cycles,
    rainflow,
    rainflow_efc,
    reversals,
    throughput,
)
from res_asset_kpi.bess.efficiency import (
    MIN_EFC,
    auto_eta,
    corrected_rte,
    interval_flows,
    stored_energy,
    window_keys,
    window_rte,
)

__all__ = [
    "DEPTH_EDGES",
    "MIN_EFC",
    "PRIORITY",
    "TREATMENTS",
    "CapacityTest",
    "UncertaintyBudget",
    "auto_eta",
    "corrected_rte",
    "depth_histogram",
    "discharge_segment",
    "equivalent_full_cycles",
    "evaluate_capacity_test",
    "interval_flows",
    "interval_treatment",
    "ltsa_availability",
    "measured_availability",
    "rainflow",
    "rainflow_efc",
    "relative_uncertainty",
    "reversals",
    "stored_energy",
    "throughput",
    "window_keys",
    "window_rte",
]
