"""Stratified sample of discrepancy rows for manual adjudication (Шаг 7 honesty).

Strata (seed 42): 50 бот + 50 1С + 50 выяснить (25 sum_diff + 25 one-sided
payments) = 150 rows. Each row gets: age_days (vs data end 2026-07-31),
agent volume, registry employees parsed from comment.
Outputs: matcher/adjudication/sample.csv + matcher/adjudication/context.jsonl
(linked acts/etm/registry records per sampled row for the adjudicator).
Run: python3 sample_adjudication.py (cwd = stage1_rules/src/)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

S1 = Path(__file__).resolve().parent.parent  # stage1 root
S1IN, S1REP = S1 / "input", S1 / "reports"
ADJ = S1.parent / "stage2_ml" / "train" / "adjudication"
sys.path.insert(0, str(S1 / "src"))  # parse (this dir; script is one-off, see below)
import parse as P  # noqa: E402

OUT = ADJ
DATA_END = pd.Timestamp("2026-07-31")
SEED = 42


def main():
    OUT.mkdir(exist_ok=True)
    disc = pd.read_csv(S1REP / "discrepancies.csv")
    disc["date"] = pd.to_datetime(disc["date"], errors="coerce")
    disc["age_days"] = (DATA_END - disc["date"]).dt.days
    disc["emps"] = disc["comment"].str.extract(r"\(реестр: ([^)]*)\)")[0].fillna("")

    bot = disc[disc["source"] == "бот"]
    c1c = disc[disc["source"] == "1С"]
    vy = disc[disc["source"] == "выяснить"]
    sumdiff = vy[vy["comment"].str.startswith("Суммы операции отличаются")]
    pay1s = vy[vy["comment"].str.startswith("Платеж")]

    samp = pd.concat([
        bot.sample(50, random_state=SEED),
        c1c.sample(50, random_state=SEED + 1),
        sumdiff.sample(25, random_state=SEED + 2),
        pay1s.sample(25, random_state=SEED + 3),
    ]).reset_index(drop=True)
    samp.insert(0, "sample_id", [f"S{i:03d}" for i in range(len(samp))])
    samp["stratum"] = (["бот"] * 50 + ["1С"] * 50 + ["выяснить/sum_diff"] * 25
                       + ["выяснить/payments"] * 25)
    samp.to_csv(OUT / "sample.csv", index=False)
    print(f"sample: {len(samp)} rows -> {OUT / 'sample.csv'}")
    print(samp["stratum"].value_counts().to_string())

    # ---- context: linked acts/etm/registry records per sampled row ----
    cfg = P.load_config(S1IN / "reconcile.yaml")
    qlog = P.QualityLog()
    acts, etm, reg, _ = P.load_data(S1IN / "acts.csv", S1IN / "etm.csv",
                                    S1IN / "registry.csv", cfg, qlog)

    with open(OUT / "context.jsonl", "w") as f:
        for _, s in samp.iterrows():
            tokens = [t for t in str(s["identifier"]).split("|") if t]
            # agent match on label column (discrepancies use canonical labels)
            a2 = acts[acts["folder"].astype(str).str.strip() == s["agent"]]
            e2 = etm[etm["agent"].astype(str).str.strip() == s["agent"]]
            a2 = a2[a2.apply(lambda r: any(
                t in f"{r.get('doc', '')} {r.get('ticket_cell', '')}" for t in tokens
            ), axis=1)] if len(a2) else a2
            e2 = e2[e2.apply(lambda r: any(
                t in f"{r.get('txn_id', '')} {r.get('tickets', '')}" for t in tokens
            ), axis=1)] if len(e2) else e2
            r2 = reg[reg.apply(lambda r: any(
                t in f"{r.get('tickets', '')} {r.get('party', '')} {r.get('employee', '')}"
                for t in tokens), axis=1)] if len(reg) else reg
            rec = {
                "sample_id": s["sample_id"], "agent": s["agent"],
                "date": str(s["date"])[:10], "identifier": s["identifier"],
                "sum_1c": s["sum_1c"], "sum_etm": s["sum_etm"],
                "diff": s["diff"], "comment": s["comment"], "source": s["source"],
                "acts": [({"doc": str(r.get("doc", "")), "date": str(r.get("date", "")),
                           "ticket_cell": str(r.get("ticket_cell", "")),
                           "debet": r.get("debet", ""), "credit": r.get("credit", ""),
                           "status": str(r.get("act_status", ""))}) for _, r in a2.iterrows()],
                "etm": [({"txn": str(r.get("txn_id", "")), "date": str(r.get("date", "")),
                          "kind": str(r.get("kind", "")), "tickets": str(r.get("tickets", "")),
                          "amount_kgs": r.get("amount_kgs", ""),
                          "creator": str(r.get("creator", ""))}) for _, r in e2.iterrows()],
                "reg": [({"date": str(r.get("date", "")), "employee": str(r.get("employee", "")),
                          "party": str(r.get("party", "")), "tickets": str(r.get("tickets", "")),
                          "pay_cell": str(r.get("pay_cell", ""))}) for _, r in r2.iterrows()],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"context: {len(samp)} records -> {OUT / 'context.jsonl'}")


if __name__ == "__main__":
    main()
