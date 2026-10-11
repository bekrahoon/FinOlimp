"""P3 (skeleton): agent + employee rankings on measured adjudication priors.

Risk = sum over rows of P(error|stratum): бот .98 / 1С .92 / payments .84 /
sum_diff .56 / other-выяснить .70 (adjudication/report.json, n=150).
Sort anchor #1 (decision C-2): old_money_30d+ = expected error money older
than 31d (not timing, won't self-resolve). Second view: density (err_rate).
Employees (decision C-1): suspect_rows (sum_diff authorship — numbers under
question) vs witness_rows (record verified correct: bot/1С classes) + denominator.
Fresh rows (<=31d) are shielded into a separate section, not the anchor.
Run: python3 rank.py (cwd = stage1_rules/src/)
Out: ../reports/agent_risk.csv, employee_risk.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

import os
HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
SRC = Path(os.environ.get("MATCH_SRC", HERE.parent / "input"))  # bundle override (match.py)
OUT = Path(os.environ.get("MATCH_OUT", REPORTS))
sys.path.insert(0, str(HERE))

PRIORS = {"бот": 0.98, "1С": 0.92}


def p_row(r):
    if r["source"] in PRIORS:
        return PRIORS[r["source"]]
    c = str(r["comment"])
    if c.startswith("Суммы операции отличаются"):
        return 0.56
    if c.startswith("Платеж"):
        return 0.84
    return 0.70


def build_employee_risk(edf: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Rank registry authors with all registry rows as the denominator.

    ``edf`` contains discrepancy-linked evidence; it is deliberately not used
    as the denominator because employees without a discrepancy would disappear
    and rates would be inflated.
    """
    registry_counts = {}
    if "employee" in registry:
        counts = {}
        for value in registry["employee"].fillna("").astype(str):
            # A registry row can name multiple authors. Count it once per author.
            authors = {name.strip() for name in value.split(",") if name.strip()}
            for author in authors:
                counts[author] = counts.get(author, 0) + 1
        registry_counts = counts

    if edf.empty:
        evidence = pd.DataFrame(columns=["employee", "suspect_rows", "witness_rows",
                                         "agents", "suspect_money"])
    else:
        evidence = edf.groupby("employee", as_index=False).agg(
            suspect_rows=("suspect", "sum"),
            witness_rows=("witness", "sum"),
            agents=("agent", "nunique"),
            suspect_money=("abs_diff", lambda s: round(
                float(s[edf.loc[s.index, "suspect"] == 1].sum()), 2)),
        )

    people = sorted(set(registry_counts) | set(evidence.get("employee", [])))
    out = pd.DataFrame({"employee": people})
    out["total_rows"] = out["employee"].map(registry_counts).fillna(0).astype(int)
    out = out.merge(evidence, on="employee", how="left")
    for col in ("suspect_rows", "witness_rows", "agents", "suspect_money"):
        out[col] = out[col].fillna(0)
    out["suspect_rows"] = out["suspect_rows"].astype(int)
    out["witness_rows"] = out["witness_rows"].astype(int)
    out["agents"] = out["agents"].astype(int)
    denominator = out["total_rows"].astype(float).replace(0, float("nan"))
    out["suspect_rate"] = (100 * out["suspect_rows"] / denominator).round(1)
    out = out.sort_values(["suspect_rows", "employee"],
                          ascending=[False, True]).reset_index(drop=True)
    return out


def main():
    feat = pd.read_csv(OUT / "features.csv")
    disc = pd.read_csv(OUT / "discrepancies.csv")
    df = feat.merge(disc[["agent", "date", "identifier", "comment", "diff"]],
                    on=["agent", "date", "identifier"], how="left")
    df["p_error"] = df.apply(p_row, axis=1)
    df["exp_money"] = (df["p_error"] * df["abs_diff"]).round(2)
    df["old"] = df["age_days"] > 31
    df["old_money"] = (df["exp_money"] * df["old"]).round(2)
    df["is_sumdiff"] = df["comment"].str.startswith("Суммы операции отличаются")
    df["emps"] = df["comment"].str.extract(r"\(реестр: ([^)]*)\)")[0].fillna("")

    # ---- agents ----
    g = df.groupby("agent")
    agent = g.agg(rows=("p_error", "count"),
                  exp_errors=("p_error", "sum"),
                  exp_money=("exp_money", "sum"),
                  old_money_30d=("old_money", "sum"),
                  fresh_rows=("fresh_31d", "sum"),
                  sum_abs_diff=("abs_diff", "sum")).reset_index()
    agent["err_rate"] = (100 * agent["exp_errors"] / agent["rows"]).round(1)
    for c in ("exp_errors", "exp_money", "old_money_30d", "sum_abs_diff"):
        agent[c] = agent[c].round(2)
    # top-3 factors: prefixes by expected errors
    pf = df.groupby(["agent", "prefix"])["p_error"].sum().reset_index()
    pf = pf.sort_values(["agent", "p_error"], ascending=[True, False])
    topf = pf.groupby("agent").head(3).groupby("agent")["prefix"].apply(
        lambda s: " | ".join(f"{p[:45]}" for p in s)).to_dict()
    agent["top_factors"] = agent["agent"].map(topf).fillna("")
    agent = agent.sort_values("old_money_30d", ascending=False).reset_index(drop=True)
    agent.to_csv(OUT / "agent_risk.csv", index=False)

    # ---- employees: discrepancy evidence + all registry rows as denominator ----
    erows = []
    for _, r in df.iterrows():
        authors = [a.strip() for a in str(r["emps"]).split(",") if a.strip()]
        for a in authors:
            suspect = int(r["is_sumdiff"])  # agent numbers under question
            witness = int(r["source"] in ("бот", "1С"))  # record verified correct
            erows.append({"employee": a, "suspect": suspect, "witness": witness,
                          "p_error": r["p_error"], "abs_diff": r["abs_diff"],
                          "agent": r["agent"], "date": r["date"],
                          "identifier": r["identifier"]})
    edf = pd.DataFrame(erows, columns=["employee", "suspect", "witness", "abs_diff", "agent"])
    import parse as P  # noqa: E402
    cfg = P.load_config(SRC / "reconcile.yaml")
    qlog = P.QualityLog()
    _acts, _etm, registry, _ignored = P.load_data(
        SRC / "acts.csv", SRC / "etm.csv", SRC / "registry.csv", cfg, qlog)
    emp = build_employee_risk(edf, registry)
    emp.to_csv(OUT / "employee_risk.csv", index=False)

    # ---- timing shield ----
    fresh = df[df["fresh_31d"] == 1]
    print(f"agents: {len(agent)} | top old_money: {agent.iloc[0]['agent']} "
          f"{agent.iloc[0]['old_money_30d']:.0f}")
    print(f"employees: {len(emp)}")
    print(emp[["employee", "total_rows", "suspect_rows", "witness_rows",
               "suspect_rate"]].to_string(index=False))
    print(f"shielded fresh rows: {len(fresh)} (money {fresh['abs_diff'].sum():.0f}) — "
          "recheck next month, not in anchor")
    print(f"file totals: exp_errors {df['p_error'].sum():.1f} / {len(df)}, "
          f"old_money {df['old_money'].sum():.0f}")


if __name__ == "__main__":
    main()
