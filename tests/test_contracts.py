"""Contract terms from YAML, damages methods, guarantee step-down, caps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from res_asset_kpi import contracts as c


@pytest.mark.parametrize("name", ["pv_om_contract", "wind_availability_warranty", "bess_ltsa"])
def test_examples_load(name: str) -> None:
    contract = c.load_contract(c.example_path(name))
    assert contract.guarantees
    text = c.example_path(name).read_text(encoding="utf-8")
    assert "ILLUSTRATIVE" in text and "Not legal advice" in text


def test_example_contents() -> None:
    b = c.load_contract(c.example_path("bess_ltsa"))
    assert b.period == "contract_year" and b.contract_start == "2025-01-01"
    assert b.event_treatments["force_majeure"] == "excluded" and b.planned_allowance_h == 16.0
    assert b.guarantees["capacity"].parameters["retention_soh"][1] == 0.97
    assert b.guarantees["rte"].damages.method == "rte_energy_cost"
    w = c.load_contract(c.example_path("wind_availability_warranty"))
    assert w.event_treatments["requested"] == "excluded"
    with pytest.raises(FileNotFoundError):
        c.example_path("nuclear")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"colour": "red"}, "unknown keys"),
        ({"asset": "hydro"}, "asset"),
        ({"period": "week"}, "period"),
        ({"period": "contract_year", "contract_start": None}, "needs contract_start"),
        ({"planned_maintenance": {"allowance_days": 1}}, "allowance_h"),
        ({"guarantees": {"x": {"basis": "time"}}}, "'guaranteed' is required"),
        ({"guarantees": {"x": {"guaranteed": 1, "damages": {"method": "vibes"}}}}, "unknown damages method"),
        ({"guarantees": {"x": {"guaranteed": 1, "damages": {"price": 1}}}}, "unknown keys"),
        ({"guarantees": {"x": {"guaranteed": 1, "damages": [1]}}}, "expected a mapping"),
    ],
)
def test_contract_errors(patch: dict[str, object], message: str) -> None:
    raw = yaml.safe_load(c.example_path("pv_om_contract").read_text(encoding="utf-8"))
    raw.update(patch)
    with pytest.raises(c.ContractError, match=message):
        c.load_contract(raw)


def test_contract_file_must_hold_a_mapping(tmp_path: Path) -> None:
    f = tmp_path / "list.yaml"
    f.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(c.ContractError, match="mapping"):
        c.load_contract(f)


def test_damages_methods_by_hand() -> None:
    m, g = np.array([0.95, 0.99, np.nan]), 0.97
    per_pp = c.damages_amount(c.Damages(method="per_point", rate_per_pp=20000.0, multiplier=50.0), m, g)
    assert per_pp[0] == pytest.approx(20000 * 2 * 50) and per_pp[1] == 0.0 and np.isnan(per_pp[2])
    rte = c.damages_amount(c.Damages(method="rte_energy_cost", price_per_mwh=90.0), [0.8, 0.9], 0.85,
                           e_out_mwh=[1000.0, 1000.0])
    assert rte.tolist() == pytest.approx([1000 * (1 / 0.8 - 1 / 0.85) * 90, 0.0])
    cap = c.damages_amount(c.Damages(method="per_mwh", rate_per_mwh=25000.0, cap=50000.0), [88.0, 95.0], 90.0)
    assert cap.tolist() == [50000.0, 0.0]
    with pytest.raises(ValueError, match="basis_energy_mwh"):
        c.damages_amount(c.Damages(method="energy"), [0.9], 0.95)
    with pytest.raises(ValueError, match="e_out_mwh"):
        c.damages_amount(c.Damages(method="rte_energy_cost"), [0.8], 0.85)


def test_assess_pr_with_step_down_and_data_share() -> None:
    pv = c.load_contract(c.example_path("pv_om_contract")).guarantees["performance_ratio"]
    pr = pd.Series({"CY1": 0.75, "CY2": 0.74, "CY3": 0.70})
    r = c.assess(pv, pr, contract_year=[1, 2, 3], basis_energy_mwh=[1000.0, 1000.0, 1000.0],
                 data_share=[0.9, 0.95, 0.5])
    g2 = 0.78 * 0.995 - 0.02
    assert r.loc["CY2", "guaranteed_effective"] == pytest.approx(g2)
    assert r.loc["CY1", "passed"] is False and r.loc["CY1", "damages"] == pytest.approx(0.01 * 1000 * 60)
    assert r.loc["CY2", "damages"] == pytest.approx((g2 - 0.74) * 1000 * 60)
    assert r.loc["CY3", "passed"] is None and np.isnan(r.loc["CY3", "damages"])


def test_annual_cap_and_retention() -> None:
    d = pd.Series([400.0, 0.0, 700.0])
    out = c.apply_annual_cap(d, ["CY1", "CY1", "CY2"], 500.0)
    assert out.tolist() == pytest.approx([400.0, 0.0, 500.0])
    assert c.apply_annual_cap(d, [1, 1, 1], None).tolist() == d.tolist()
    soh = c.guaranteed_retention([0, 1, 2, 3], [1.0, 0.97, 0.95, 0.935], [0.5, 2.5], efc_total=[100, 1000],
                                 efc_per_year=365)
    assert soh.tolist() == pytest.approx([0.985, np.interp(1000 / 365, [2, 3], [0.95, 0.935])])
    assert c.guaranteed_retention([0, 1], [1.0, 0.9], [0.5]).tolist() == pytest.approx([0.95])
