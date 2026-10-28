"""Numerical parity of res-asset-kpi with the projects its code was extracted from.

Every check runs a source project's own code and the library on the same inputs and reports the largest
absolute and relative difference. The source projects are public (github.com/kwwaleszczynski-del); run each
check with the Python environment of its source project, which has that project's dependencies, e.g.:

    <pv-budget-bridge>/.venv/Scripts/python validation/parity.py pv --source <pv-budget-bridge> --data <its data>

Checks and their sources:

    pv              pv-budget-bridge             PR family, availability, contract tests, energy bridge
    wind-avail      wind-downtime-availability   availability by IEC 61400-26 category (its result tables)
    power-curve     wind-power-curve-kelmarsh    air density, normalisation, binning, AEP on Kelmarsh SCADA
    reliability     wind-om-reliability          operating time, MTBF/MTTR, Crow-AMSAA, Laplace on Kelmarsh logs
    bess            bess-ops-warranty            RTE windows, LTSA availability, rainflow, capacity tests

No data is stored in this repository: ``--data`` points at the source project's own data folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parents[1] / "src"
log = logging.getLogger("parity")
ROWS: list[dict[str, object]] = []


def compare(check: str, source: Iterable[float] | float, library: Iterable[float] | float, note: str = "") -> None:
    """Record the largest absolute and relative difference of two sets of numbers (NaN must match NaN)."""
    a = np.atleast_1d(np.asarray(source, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(library, dtype=float)).ravel()
    if a.shape != b.shape:
        raise AssertionError(f"{check}: {a.shape} values from the source, {b.shape} from the library")
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    ok = ~np.isnan(a) & ~np.isnan(b)
    diff = np.abs(a[ok] - b[ok])
    rel = diff / np.maximum(np.abs(a[ok]), 1e-12)
    row = {"check": check, "values": int(ok.sum()), "nan_mismatch": nan_mismatch,
           "max_abs_diff": float(diff.max()) if ok.any() else 0.0,
           "max_rel_diff": float(rel.max()) if ok.any() else 0.0,
           "example_source": float(a[ok][0]) if ok.any() else float("nan"),
           "example_library": float(b[ok][0]) if ok.any() else float("nan"), "note": note}
    ROWS.append(row)
    log.info("%-60s n=%6d  max rel diff %.2e  NaN mismatch %d", check, row["values"], row["max_rel_diff"],
             nan_mismatch)


# ----------------------------------------------------------------------------------------------------------- PV

def parity_pv(data: Path) -> None:
    from pv_budget import cli, contract, kpi
    from pv_budget.losses import COMPONENTS, event_mask

    from res_asset_kpi import contracts, pv

    spec_all = contract.load(Path(cli.ROOT) / "configs" / "contract_example.yaml")
    for key in ("pl_demo", "pvdaq_34"):
        a = cli.analyze_example(key, data)
        cfg, fr, dt = a.cfg, a.loss.frame, a.data.dt_h
        t_typ = float(a.budget["T_typ"].iloc[0])
        local = fr.index.tz_convert(cfg.plant.timezone)
        month = np.array([f"{y}-{m:02d}" for y, m in zip(local.year, local.month, strict=True)])
        src, lib = [], []
        for per in pd.unique(month):
            sub = fr[month == per]
            v = kpi.pr_values(kpi.pr_terms(sub, cfg.plant, t_typ, dt))
            t = pv.pr_terms(sub["actual"], sub["poa"], cfg.plant.dc_kwp, dt, t_cell_c=sub["t_cell"],
                            gamma_per_k=cfg.plant.gamma_pdc, t_typ_c=t_typ)
            src += [v["pr"], v["pr_25"], v["pr_wc"]]
            lib += [t.pr, t.pr_temperature_corrected, t.pr_weather_corrected]
        compare(f"pv/{key}: monthly PR, PR_25, PR_wc", src, lib)

        # energy bridge: same weather ratios and loss fractions, bars recomputed by the library
        br = a.monthly
        weather = {"irradiance": br["r_irr"], "temperature": br["r_temp"], "degradation": br["r_deg"]}
        la = {c: br[f"frac_{c}"] for c in COMPONENTS}
        acc, lb = br["potential_budget"].to_numpy(float).copy(), {}
        for c in COMPONENTS:
            b = br[f"budget_{c}"].to_numpy(float)
            lb[c] = np.divide(b, acc, out=np.zeros_like(b), where=acc > 0)
            acc = acc - b
        bars = ["irradiance", "temperature", "degradation", *COMPONENTS, "residual"]
        seq = pv.energy_bridge(br["E_budget"], br["E_actual"], weather, la, lb, index=br.index)
        sh = pv.energy_bridge(br["E_budget"], br["E_actual"], weather, la, lb, method="shapley", index=br.index)
        compare(f"pv/{key}: bridge bars, sequential (kWh)", br[bars].to_numpy(), seq[bars].to_numpy())
        compare(f"pv/{key}: bridge bars, Shapley (kWh)", br[[f"shapley_{c}" for c in bars]].to_numpy(),
                sh[bars].to_numpy())

        # contract tests with the illustrative contract of the source project
        spec = spec_all
        periods = contract._periods(local, spec.get("period", "year"), None)
        excluded = list(spec.get("excluded_events", []))
        inv = list(a.loss.state.columns)
        lim = contract.expected_per_inverter(fr, cfg, inv)
        av = spec["availability"]
        allowance = spec.get("planned_maintenance_allowance_h", 0.0)
        s_av = contract.availability_test(av, a.loss.state, a.loss.weights, fr, a.data.events, dt, periods, excluded,
                                          allowance, lim)
        state = a.loss.state.replace({"up": 1.0, "night": 1.0, "down": 0.0, "gap": np.nan}).astype(float)
        excl = event_mask(a.data.events, fr.index, inv, [c for c in excluded if c != "planned_maintenance"])
        planned = event_mask(a.data.events, fr.index, inv, ["planned_maintenance"])
        l_av = pv.availability(state, dt, capacity=a.loss.weights[inv].to_numpy(), poa_wm2=fr["poa"],
                               daylight_wm2=av.get("daylight_wm2", 50), excluded=excl.to_numpy(),
                               planned=planned.to_numpy(), planned_allowance_h=allowance, expected_kw=lim,
                               periods=np.asarray(periods))
        l_av = l_av.loc[s_av["period"]]
        compare(f"pv/{key}: contract availability, time and energy",
                s_av[["availability_time", "availability_energy"]].to_numpy(),
                l_av[["availability_time", "availability_energy"]].to_numpy())
        guarantee = contracts.Guarantee(
            guaranteed=float(av["guaranteed"]),
            damages=contracts.Damages(method="energy", price_per_mwh=av["ld_price_per_mwh"], cap=av["ld_cap"]),
            bonus=contracts.Bonus(threshold=av["bonus_threshold"], share=av["bonus_share"],
                                  price_per_mwh=av["ld_price_per_mwh"]))
        basis = l_av["availability_energy" if av.get("basis") == "energy" else "availability_time"]
        assessed = contracts.assess(guarantee, basis, basis_energy_mwh=l_av["expected_kwh"].to_numpy() / 1000)
        compare(f"pv/{key}: availability damages and bonus", s_av[["damages", "bonus"]].to_numpy(),
                assessed[["damages", "bonus"]].to_numpy())

        prs = spec["performance_ratio"]
        s_pr = contract.pr_test(prs, fr, a.loss.state, a.data.events, cfg.plant, t_typ, dt, periods, excluded, None)
        st = a.loss.state.to_numpy()
        mask = (fr["poa"].to_numpy() >= prs["min_irradiance_wm2"]) & ~(st == "gap").any(axis=1) & \
            ~event_mask(a.data.events, fr.index, inv, excluded).to_numpy().any(axis=1) & ~fr["curtailed"].to_numpy()
        daylight = fr["poa"].to_numpy() >= prs["min_irradiance_wm2"]
        measured, ref, share = {}, [], []
        for per in pd.unique(np.asarray(periods)):
            pos = np.asarray(periods) == per
            t = pv.pr_terms(fr["actual"], fr["poa"], cfg.plant.dc_kwp, dt, t_cell_c=fr["t_cell"],
                            gamma_per_k=cfg.plant.gamma_pdc, t_typ_c=t_typ, mask=mask & pos)
            measured[per] = t.pr_weather_corrected
            ref.append(t.reference_wc_kwh / 1000)
            share.append((mask & pos).sum() / max((daylight & pos).sum(), 1))
        g = contracts.Guarantee(guaranteed=prs["guaranteed"], annual_degradation=prs["annual_degradation"],
                                tolerance=prs["tolerance"], min_data_share=prs["min_data_share"],
                                damages=contracts.Damages(method="energy", price_per_mwh=prs["ld_price_per_mwh"],
                                                          cap=prs["ld_cap"]))
        r = contracts.assess(g, pd.Series(measured), contract_year=np.arange(1, len(measured) + 1),
                             basis_energy_mwh=ref, data_share=share)
        compare(f"pv/{key}: PR test, measured PR_wc and effective guarantee",
                s_pr[["measured", "guaranteed_effective"]].to_numpy(),
                r[["measured", "guaranteed_effective"]].to_numpy())
        compare(f"pv/{key}: PR test damages", s_pr["damages"], r["damages"])


# --------------------------------------------------------------------------------------------------------- wind

def parity_wind_availability(source: Path) -> None:
    from wind_downtime import availability as src

    from res_asset_kpi import wind

    tot = pd.read_csv(source / "results" / "category_totals.csv")
    published = pd.read_csv(source / "results" / "availability_turbine_years.csv")
    keys = ["farm", "detector", "classifier", "group"]
    s_vals, l_vals = [], []
    for _, g in tot.groupby(keys, sort=False):
        for name in ("iec_operational", "oem_warranty"):
            allowance = src.MAINTENANCE_ALLOWANCE_H[name]
            s = src.kpis(g, src.CONTRACTS[name], allowance)
            r = wind.availability(g, name, allowance_h=allowance)
            s_vals += [s["time_availability"], s["energy_availability"], s["lost_unavailable_mwh"]]
            l_vals += [r.time, r.energy, r.lost_unavailable_kwh / 1000]
    compare("wind: time, energy availability and lost energy from category totals", s_vals, l_vals,
            f"{tot.groupby(keys).ngroups} turbine-years x 2 contracts, Kelmarsh, Penmanshiel, Hill of Towie")
    pub = published.copy()
    pub["group"] = pub["year"].astype(str) + "|" + pub["turbine"].astype(str)
    merged = pub.merge(tot[keys].drop_duplicates(), on=["farm", "detector", "classifier", "group"])
    lib = []
    for r in merged.itertuples():
        g = tot[(tot["farm"] == r.farm) & (tot["detector"] == r.detector) & (tot["classifier"] == r.classifier)
                & (tot["group"] == r.group)]
        lib.append(wind.availability(g, r.contract, allowance_h=src.MAINTENANCE_ALLOWANCE_H[r.contract]).time)
    compare("wind: time availability vs the published turbine-year table", merged["time_availability"], lib)


def parity_power_curve(data: Path) -> None:
    from wind_power_curve import power_curve as src

    from res_asset_kpi import wind

    scada = pd.read_parquet(data / "processed" / "scada_2020.parquet")
    scada = scada.reset_index()
    turbine_col = "turbine" if "turbine" in scada else scada.columns[0]
    s_rho = src.air_density(scada["temp_c"], elevation_m=100.0)
    l_rho = wind.air_density(scada["temp_c"], elevation_m=100.0)
    compare("power curve: air density, Kelmarsh 2020 (all records)", s_rho, l_rho)
    s_ws = src.normalise_wind_speed(scada["ws"].to_numpy(), s_rho)
    l_ws = wind.normalise_wind_speed(scada["ws"].to_numpy(), l_rho)
    compare("power curve: normalised wind speed", s_ws, l_ws)
    s_all, l_all, aep_s, aep_l = [], [], [], []
    for _, g in scada.assign(ws_n=s_ws).groupby(turbine_col):
        ok = g["power_kw"] > 0
        sc = src.bin_power_curve(g.loc[ok, "ws_n"], g.loc[ok, "power_kw"])
        lc = wind.bin_power_curve(g.loc[ok, "ws_n"], g.loc[ok, "power_kw"])
        s_all.append(sc[["ws", "power_kw", "std_kw", "n", "u_a_kw"]].to_numpy())
        l_all.append(lc[["ws", "power_kw", "std_kw", "n", "u_a_kw"]].to_numpy())
        for v in (5.0, 6.5, 8.0):
            aep_s.append(src.aep_rayleigh(sc["ws"], sc["power_kw"], v))
            aep_l.append(wind.aep_rayleigh(lc["ws"], lc["power_kw"], v))
    compare("power curve: bins (ws, power, std, n, u_A) per turbine", np.concatenate(s_all), np.concatenate(l_all))
    compare("power curve: AEP for Rayleigh 5, 6.5, 8 m/s per turbine (MWh)", aep_s, aep_l)


def parity_reliability(source: Path, data: Path) -> None:
    from wind_om import reliability as R
    from wind_om.analysis import Settings, analyse
    from wind_om.mapping import read_mapping
    from wind_om.sources import load_farm, log_window

    from res_asset_kpi import wind

    ev, meta = load_farm("kelmarsh", source / "data", kelmarsh_data=data)
    t0, t1 = log_window(ev, meta)
    a = analyse(ev, read_mapping(meta["mapping"]), t0, t1, settings=Settings(max_silence="7D"), origin=meta["cod"])
    up = wind.operating_time(a.expo, a.ep[["turbine_id", "start", "end"]])
    compare("reliability: operating hours per turbine", a.up.groupby("turbine_id")["hours"].sum(),
            up.groupby("turbine_id")["hours"].sum())
    hours = a.up.groupby("turbine_id")["hours"].sum()
    cols = ["failures", "operating_h", "mtbf_h", "mtbf_lo_h", "mtbf_hi_h", "mttr_h", "ttr_median_h", "ttr_p90_h",
            "downtime_h"]
    for by in ("turbine_id", "component"):
        s = R.mtbf_mttr(a.fail, a.up, by=[by]).set_index(by).sort_index()
        lib = wind.mtbf_mttr(a.fail, hours, by=by).set_index(by).sort_index()
        compare(f"reliability: MTBF, CI, MTTR, TTR by {by}", s[cols].to_numpy(), lib[cols].to_numpy())
    turbines = sorted(a.fail["turbine_id"].unique())
    groups = [("fleet", a.fail)] + [(t, a.fail[a.fail["turbine_id"] == t]) for t in turbines]
    groups += [(f"component:{c}", a.fail[a.fail["component"] == c]) for c in a.fail["component"].unique()]
    s_vals, l_vals = [], []
    for name, f in groups:
        expo = a.expo if not name.startswith("T") else a.expo[a.expo["turbine_id"] == name]
        s = R.crow_amsaa(f, expo, a.origin)
        r = wind.crow_amsaa(f, expo, a.origin)
        s_vals += [s.beta, s.beta_lo, s.beta_hi, s.lam, s.p_trend_lr, s.laplace_u, s.p_laplace]
        l_vals += [r.beta, r.beta_lo, r.beta_hi, r.lam, r.p_trend, r.laplace_u, r.p_laplace]
        assert s.verdict == r.verdict, (name, s.verdict, r.verdict)
    compare("reliability: Crow-AMSAA beta, bounds, lambda, LR p, Laplace (fleet, turbines, components)",
            s_vals, l_vals)
    pub = pd.read_csv(source / "results" / "kelmarsh" / "crow_amsaa.csv").set_index("group")
    compare("reliability: fleet Crow-AMSAA beta vs the published table", pub.loc["fleet", "crow_beta"],
            wind.crow_amsaa(a.fail, a.expo, a.origin).beta)


# --------------------------------------------------------------------------------------------------------- BESS

def parity_bess(source: Path) -> None:
    from bess_ops import availability as s_av
    from bess_ops import capacity as s_cap
    from bess_ops import cycles as s_cyc
    from bess_ops import energy as s_en
    from bess_ops.config import load_ltsa, load_site, load_warranty
    from bess_ops.pipeline import soc_series
    from bess_ops.schema import load_mapping, to_site_frame
    from bess_ops.synthetic import simulate

    from res_asset_kpi import bess, timeseries

    ex = source / "src" / "bess_ops" / "examples"
    sc = load_site(ex / "site.example.yaml")
    w = load_warranty(ex / "warranty.example.yaml", sc)
    lt = load_ltsa(ex / "ltsa.example.yaml", sc)
    sim = simulate(seed=1)
    site = to_site_frame(sim.long, load_mapping(ex / "mapping.example.yaml"))
    events = s_av.normalise_events(sim.events, sc["timezone"])
    tz = sc["timezone"]
    eta = float(s_en.window_rte(site, sc, "D")["eta"].iloc[0])
    cols = ["e_out", "e_in", "d_stored", "rte_raw", "rte", "rte_excl_aux", "efc", "valid"]
    for freq, lib_freq in (("D", "D"), ("M", "M"), ("LY", "CY")):
        s = s_en.window_rte(site, sc, freq, eta)
        lib = bess.window_rte(site, lib_freq, reference_energy=sc["soc_energy_mwh"], aux_metering=sc["aux_metering"],
                              soc_energy=sc["soc_energy_mwh"], tz=tz, contract_start=sc["cod"], eta=eta,
                              min_efc=s_en.MIN_EFC[freq])
        compare(f"bess: RTE windows {freq} (energy, dE, RTE with and without aux, EFC, valid)",
                s[cols].astype(float).to_numpy(), lib[cols].astype(float).to_numpy())
    compare("bess: one-way efficiency auto", eta,
            float(bess.window_rte(site, "Y", reference_energy=1.0, aux_metering=sc["aux_metering"],
                                  soc_energy=sc["soc_energy_mwh"], tz=tz)["eta"].iloc[0]))

    a = lt["availability"]
    iv = s_av.interval_availability(site, lt, events)
    meas = bess.measured_availability(site, power=lt["contract"]["power_mw"],
                                      charge_power=lt["contract"]["charge_power_mw"], formula=a["formula"],
                                      threshold=a["threshold"], directions=a["directions"], combine=a["combine"],
                                      energy=lt["contract"]["energy_mwh"])
    compare("bess: measured availability per interval", iv["measured"], meas)
    for period, freq in (("month", "M"), ("year", "CY")):
        s = s_av.period_availability(iv, lt, sc, period)
        labels = bess.window_keys(site.index, freq, tz, sc["cod"])
        lib = bess.ltsa_availability(meas, events=events, treatments=a["categories"],
                                     planned_allowance_h=a["planned_allowance_hours"], missing_data=a["missing_data"],
                                     periods=labels.to_numpy())
        c = ["availability", "hours_excluded", "hours_deemed_available", "hours_planned_in_allowance",
             "hours_planned_counted", "hours_missing", "equivalent_unavailable_hours"]
        compare(f"bess: LTSA availability per {period}", s[c].to_numpy(), lib.loc[s.index, c].to_numpy())

    soc = soc_series(site, sc)
    for h in (0.0, 0.01):
        s = s_cyc.rainflow(soc.to_numpy(), h)
        lib = bess.rainflow(soc.to_numpy(), h)
        compare(f"bess: rainflow cycles, hysteresis {h}", s[["depth", "mean", "count", "start", "end"]].to_numpy(),
                lib[["depth", "mean", "count", "start", "end"]].to_numpy())
        compare(f"bess: rainflow EFC and depth histogram, hysteresis {h}",
                np.r_[s_cyc.rainflow_efc(s), s_cyc.depth_histogram(s, 2.0).to_numpy().ravel()],
                np.r_[bess.rainflow_efc(lib), bess.depth_histogram(lib, 2.0).to_numpy().ravel()])
    dt = timeseries.interval_hours(site.index)
    for basis in ("discharge_meter", "half_throughput_meter", "discharge_dc"):
        compare(f"bess: throughput per interval, {basis}", s_cyc.throughput_energy(site, basis),
                bess.throughput(site, dt, basis))

    windows = s_cap.find_test_windows(site, lt, events)
    tests = s_cap.evaluate_tests(site, windows, lt, w, sc)
    t = lt["capacity"]["test"]
    rows_s, rows_l = [], []
    for r in tests.itertuples():
        sel = (site.index >= r.start) & (site.index < r.end)
        lo, hi, e_meas = bess.discharge_segment(site["p_meter"].to_numpy()[sel], dt, t["power_mw"],
                                                t["steady_tolerance"])
        res = bess.evaluate_capacity_test(e_meas, r.e_guaranteed_mwh, temp_c=r.temp_c, t_ref_c=t["t_ref_c"],
                                          temp_coeff_per_k=t["temp_coeff_per_k"],
                                          temp_coeff_u_per_k=t["temp_coeff_u_per_k"], temp_u_k=t["temp_u_k"],
                                          meter_class_pct=t["meter_class_pct"], ct_pct=t["ct_pct"],
                                          vt_pct=t["vt_pct"], coverage_k=t["coverage_k"],
                                          decision_rule=t["decision_rule"])
        rows_s.append([r.e_meas_mwh, r.temp_factor, r.e_corr_mwh, r.u_rel_combined, r.U_mwh, r.margin_mwh])
        rows_l.append([res.e_measured, res.temp_factor, res.e_corrected, res.u_rel_combined,
                       res.expanded_uncertainty, res.margin])
        assert r.verdict == res.verdict
    compare(f"bess: capacity tests ({len(tests)}): energy, correction, uncertainty, margin", rows_s, rows_l)
    from bess_ops.warranty import guaranteed_soh

    from res_asset_kpi.contracts import guaranteed_retention
    ret = w["retention"]
    years, efc = tests["elapsed_years"].to_numpy(), tests["efc_total"].to_numpy()
    compare("bess: guaranteed retention at equivalent age",
            guaranteed_soh(ret, years, efc, w["throughput"]["efc_per_year"]),
            guaranteed_retention(ret["years"], ret["soh"], years, efc, w["throughput"]["efc_per_year"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["pv", "wind-avail", "power-curve", "reliability", "bess"])
    ap.add_argument("--source", type=Path, required=True, help="root of the source project")
    ap.add_argument("--data", type=Path, help="data folder: of the source project (pv, power-curve) or the "
                    "Kelmarsh folder of wind-power-curve-kelmarsh (reliability)")
    ap.add_argument("--json", type=Path, help="append the results to this JSON file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("pv_budget", "wind_om", "wind_downtime", "wind_power_curve", "bess_ops", "py.warnings"):
        logging.getLogger(name).setLevel(logging.ERROR)
    sys.path[:0] = [str(LIB), str(args.source.resolve() / "src")]
    data = args.data.resolve() if args.data else args.source.resolve() / "data"
    {"pv": lambda: parity_pv(data), "wind-avail": lambda: parity_wind_availability(args.source.resolve()),
     "power-curve": lambda: parity_power_curve(data),
     "reliability": lambda: parity_reliability(args.source.resolve(), data),
     "bess": lambda: parity_bess(args.source.resolve())}[args.check]()
    if args.json:
        old = json.loads(args.json.read_text(encoding="utf-8")) if args.json.exists() else []
        old = [r for r in old if not str(r["check"]).startswith(tuple({r2["check"].split(":")[0] for r2 in ROWS}))]
        args.json.write_text(json.dumps(old + ROWS, indent=1), encoding="utf-8")
    worst = max((float(r["max_rel_diff"]) for r in ROWS), default=0.0)
    mism = sum(int(r["nan_mismatch"]) for r in ROWS)
    print(f"{len(ROWS)} checks, largest relative difference {worst:.2e}, NaN mismatches {mism}")
    return 0 if worst < 1e-9 and mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
