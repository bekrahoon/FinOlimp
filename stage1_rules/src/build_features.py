"""P0: row-level features for all 2165 discrepancy rows (Шаг 7 input).

Deterministic: same inputs -> byte-identical features.csv (sorted output,
rounded floats, no timestamps). Links rows back to acts/etm/registry via
parse.load_data + ticket tails / txn_id / ЦБ-С doc.
Registry-sum comparison (v2 rule as features): KGS-only pay_cell rows are
parsed (base + sf) -> reg_parseable/reg_sum/reg_eq_* (TOL 0.01); all-currency
rows are converted to KGS via per-row rates (P.parse_pay_kgs_total, anchors
S110/S112/S119) -> reg_parseable_conv/reg_sum_conv/reg_eq_*_conv with
CONV_TOL 2.0 (rate columns are rounded to 2dp: S110 +0.11, S119 +0.19).
Agent priors are strictly prior-to-row (date < row date) — no leakage.
Run: python3 build_features.py (cwd = stage1_rules/src/)
Out: ../reports/features.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

import os
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
# SRC = raw inputs (acts/etm/registry+yaml); OUT = reports (discrepancies in,
# features out). Bundle overrides both via env (match.py).
SRC = Path(os.environ.get("MATCH_SRC", HERE.parent / "input"))
OUT = Path(os.environ.get("MATCH_OUT", REPORTS))
sys.path.insert(0, str(HERE))
import parse as P  # noqa: E402
TOL = 0.01
CONV_TOL = 2.0  # okruglenie kursov v stroke reestra (S110: +0.11, S119: +0.19)
# DATA_END = max data strok (konets vygruzki): schitaetsya iz discrepancies.csv,
# ne zashit (rollover novogo mesyatsa bez pravok).


def parse_kgs_total(pay_cell: str) -> float:
    """Sum base+sf iff every chunk is KGS-denominated, else NaN."""
    if not isinstance(pay_cell, str) or not pay_cell.strip():
        return float("nan")
    total = 0.0
    for chunk in pay_cell.split("+"):
        # skobochnye pometki s bukvami ("(shtraf 50%)") — ne summy, vychischaem;
        # chisto-tsifrovye skobki ne trogaem (u parse_money_value eto minus).
        chunk = re.sub(r"\([^()]*[A-Za-zА-Яа-я][^()]*\)", "", chunk)
        cur = re.findall(r"[A-Za-zА-Яа-я]+", chunk)
        cur = [c for c in cur if c.lower() != "sf"]  # sf = marker sbora, ne valyuta
        if not cur or not all(c.lower() in ("kgs", "кгс", "сом", "сомов") for c in cur):
            return float("nan")
        v = P.parse_money_value(chunk, [])
        if pd.isna(v):
            return float("nan")
        total += v
    return round(total, 2)


def main():
    cfg = P.load_config(SRC / "reconcile.yaml")
    qlog = P.QualityLog()
    acts, etm, reg, _ = P.load_data(SRC / "acts.csv", SRC / "etm.csv",
                                    SRC / "registry.csv", cfg, qlog)
    disc = pd.read_csv(OUT / "discrepancies.csv")
    disc["date"] = pd.to_datetime(disc["date"], errors="coerce")
    DATA_END = disc["date"].max()
    if pd.isna(DATA_END):  # never-fail: bez dat — vozrast ne schitaem
        DATA_END = pd.Timestamp("2026-07-31")

    # label -> norm (discrepancies use canonical labels, acts_first)
    labels = P.canonical_labels(acts, etm, cfg)
    norm_of_label = {}
    for _, r in acts.iterrows():
        norm_of_label.setdefault(str(r.get("agent_label", "")), r["agent_norm"])
    for _, r in etm.iterrows():
        norm_of_label.setdefault(str(r.get("agent_label", "")), r["agent_norm"])

    # ticket tail -> row indices (same-file scope is enough: components are same-agent)
    etm_by_tail: dict = {}
    for i, r in etm.iterrows():
        for t in (r["ticket_set"] if isinstance(r["ticket_set"], set) else set()):
            etm_by_tail.setdefault(t, []).append(i)
    acts_by_doc: dict = {}
    for i, r in acts.iterrows():
        d = P.doc_ref(r.get("doc", ""), cfg)
        if d:
            acts_by_doc.setdefault(d, []).append(i)
    etm_by_txn = {str(r["txn_id"]): i for i, r in etm.iterrows()}

    reg["pay_kgs"] = reg["pay_cell"].map(parse_kgs_total)
    reg["pay_conv"] = reg.apply(
        lambda x: P.parse_pay_kgs_total(x["pay_cell"], x), axis=1)
    reg_by_tail: dict = {}
    for i, r in reg.iterrows():
        for t in (r["ticket_set"] if isinstance(r["ticket_set"], set) else set()):
            reg_by_tail.setdefault(t, []).append(i)

    rows = []
    for _, s in disc.iterrows():
        agent_label = str(s["agent"])
        agent_norm = norm_of_label.get(agent_label, "")
        toks = [t for t in str(s["identifier"]).split("|") if t]
        comment = str(s["comment"])
        prefix = comment.split(";")[0]
        f = {
            "agent": agent_label,
            "date": str(s["date"])[:10],
            "identifier": str(s["identifier"]),
            "sum_1c": round(float(s["sum_1c"]), 2),
            "sum_etm": round(float(s["sum_etm"]), 2),
            "diff": round(float(s["diff"]), 2),
            "source": str(s["source"]),
            "prefix": prefix,
            "abs_diff": round(abs(float(s["diff"])), 2),
        }
        base = max(abs(float(s["sum_1c"])), abs(float(s["sum_etm"])), 1e-9)
        f["rel_drift"] = round(abs(float(s["diff"])) / base, 6)
        f["one_sided_1c"] = int(abs(float(s["sum_etm"])) < TOL)
        f["one_sided_etm"] = int(abs(float(s["sum_1c"])) < TOL)
        ad = abs(float(s["diff"]))
        f["round_1k"] = int(abs(ad - 1000.0) < TOL)
        f["round_9k"] = int(abs(ad - 9000.0) < TOL)
        f["round_10k"] = int(abs(ad - 10000.0) < TOL)
        f["round_36k"] = int(abs(ad - 36000.0) < TOL)
        ratio = ad / abs(float(s["sum_1c"])) if abs(float(s["sum_1c"])) > TOL else -1
        f["pct_5"] = int(abs(ratio - 0.05) < 0.001)
        f["pct_10"] = int(abs(ratio - 0.10) < 0.001)
        age = (DATA_END - s["date"]).days if pd.notna(s["date"]) else -1
        f["age_days"] = int(age)
        f["fresh_31d"] = int(0 <= age <= 31)
        f["month"] = int(s["date"].month) if pd.notna(s["date"]) else 0
        mdays = s["date"].days_in_month if pd.notna(s["date"]) else 0
        f["month_edge"] = int(pd.notna(s["date"]) and min(s["date"].day, mdays - s["date"].day + 1) <= 3)
        f["n_tickets"] = len(toks)
        emps = re.search(r"\(реестр: ([^)]*)\)", comment)
        f["n_authors"] = len([a for a in (emps.group(1).split(",") if emps else []) if a.strip()])
        f["in_registry"] = int("; в реестре есть" in comment)

        # linked component rows
        e_idx: set = set()
        a_idx: set = set()
        r_idx: set = set()
        doc = P.payment_identifier(str(s["identifier"]))
        if doc.startswith("ЦБ-С"):
            a_idx.update(acts_by_doc.get(doc, []))
        if str(s["identifier"]) in etm_by_txn and "|" not in str(s["identifier"]):
            e_idx.add(etm_by_txn[str(s["identifier"])])
        for t in toks:
            e_idx.update(etm_by_tail.get(t, []))
            r_idx.update(reg_by_tail.get(t, []))
        # same-agent filter mirrors engine components
        e_idx = {i for i in e_idx if etm.loc[i, "agent_norm"] == agent_norm} if agent_norm else e_idx
        creators = {str(etm.loc[i, "creator"]) for i in e_idx} if e_idx else set()
        f["n_etm_rows"] = len(e_idx)
        f["creator_bot"] = int("etm-bot" in creators)
        f["creator_sub"] = int("субагент" in creators)
        f["creator_mixed"] = int(len(creators) > 1)
        f["creator_none"] = int(not creators)
        kinds = {str(etm.loc[i, "kind"]) for i in e_idx} if e_idx else set()
        for k in ("выкуп", "возврат", "войд", "оплата"):
            f[f"kind_{k}"] = int(k in kinds)
        f["n_reg_rows"] = len(r_idx)
        reg_sums = [reg.loc[i, "pay_kgs"] for i in r_idx
                    if pd.notna(reg.loc[i, "pay_kgs"])]
        f["reg_parseable"] = int(bool(r_idx) and len(reg_sums) == len(r_idx))
        if f["reg_parseable"] and reg_sums:
            rs = round(sum(reg_sums), 2)
            f["reg_sum"] = rs
            f["reg_eq_1c"] = int(abs(rs - abs(float(s["sum_1c"]))) < TOL)
            f["reg_eq_etm"] = int(abs(rs - abs(float(s["sum_etm"]))) < TOL)
        else:
            f["reg_sum"] = float("nan")
            f["reg_eq_1c"] = 0
            f["reg_eq_etm"] = 0
        # converted sums (all currencies, CONV_TOL guards rounded rates)
        reg_conv = [reg.loc[i, "pay_conv"] for i in r_idx
                    if pd.notna(reg.loc[i, "pay_conv"])]
        f["reg_parseable_conv"] = int(bool(r_idx) and len(reg_conv) == len(r_idx))
        if f["reg_parseable_conv"] and reg_conv:
            rc = round(sum(reg_conv), 2)
            f["reg_sum_conv"] = rc
            f["reg_eq_1c_conv"] = int(abs(rc - abs(float(s["sum_1c"]))) < CONV_TOL)
            f["reg_eq_etm_conv"] = int(abs(rc - abs(float(s["sum_etm"]))) < CONV_TOL)
        else:
            f["reg_sum_conv"] = float("nan")
            f["reg_eq_1c_conv"] = 0
            f["reg_eq_etm_conv"] = 0
        # short-party flag: any linked reg party strictly shorter (token-subset) than agent
        atoks = set(str(agent_norm).split())
        f["short_party"] = 0
        for i in r_idx:
            pnorm = P.apply_aliases(P.norm_agent(reg.loc[i, "party"], cfg), cfg)
            pt = set(pnorm.split())
            if pt and pt < atoks:
                f["short_party"] = 1
                break
        rows.append(f)

    feat = pd.DataFrame(rows)
    # prior-to-row agent priors (no leakage)
    feat["_d"] = pd.to_datetime(feat["date"], errors="coerce")
    feat = feat.sort_values(["agent", "_d", "identifier"]).reset_index(drop=True)
    feat["agent_prior_n"] = feat.groupby("agent").cumcount()
    feat["agent_prior_money"] = feat.groupby("agent")["abs_diff"].cumsum() - feat["abs_diff"]
    feat["agent_prior_money"] = feat["agent_prior_money"].round(2)
    feat = feat.drop(columns=["_d"])
    feat = feat.sort_values(["agent", "date", "identifier"]).reset_index(drop=True)
    feat.to_csv(OUT / "features.csv", index=False)
    print(f"features: {feat.shape} -> {OUT / 'features.csv'}")
    print("reg_parseable:", int(feat["reg_parseable"].sum()),
          "| reg_eq_1c:", int(feat["reg_eq_1c"].sum()),
          "| reg_eq_etm:", int(feat["reg_eq_etm"].sum()),
          "| short_party:", int(feat["short_party"].sum()))
    print("conv_parseable:", int(feat["reg_parseable_conv"].sum()),
          "| conv_eq_1c:", int(feat["reg_eq_1c_conv"].sum()),
          "| conv_eq_etm:", int(feat["reg_eq_etm_conv"].sum()))


if __name__ == "__main__":
    main()
