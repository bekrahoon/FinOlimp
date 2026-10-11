"""Шаг 6: action-лист по субагентам (agent_actions.csv).

Odna stroka na agenta: na skolko dvigat balans ETM, chto ispravit v 1С,
chto snachala vyyasnit — summy iz discrepancies.csv + kontrol iz
reconciliation_summary.csv (Σdiff == operation_diff u vseh 100 agentov).
Znak: diff = sum_1c - sum_etm; plus v action-column = uvelichit balans/saldo.
  move_etm_by = -Σ diff po strokam bot+agent (delta balansa ETM);
  fix_1c_by   = -Σ diff po strokam 1С (delta salda 1С);
  clarify_*   = strokam vyyasnit (pervichka/ сторно/singltony).
Deterministic: sortirovka po ubyvaniyu deneg deystviya, tay-brek agent.
Run: python3 build_actions.py (cwd = stage1_rules/src/)
Out: ../reports/agent_actions.csv (ili MATCH_OUT pri sborke match.py).
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
SRC = Path(os.environ.get("MATCH_SRC", REPORTS))  # bundle override (match.py)
OUT = Path(os.environ.get("MATCH_OUT", REPORTS))
sys.path.insert(0, str(HERE))


def build_actions(disc: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Build one accounting action row per agent.

    ``move_etm_by`` and ``fix_1c_by`` are balance deltas, not discrepancy
    amounts: a positive value increases the corresponding balance. Since
    ``diff = sum_1c - sum_etm``, correcting an ETM-side discrepancy changes
    the ETM balance by ``-diff``; correcting 1C changes its saldo by ``-diff``.
    """
    summ = summary.set_index("agent")
    rows = []
    for agent, gp in disc.groupby("agent"):
        b = gp[gp["source"].isin(["бот", "агент"])]["diff"].sum()
        c = gp[gp["source"] == "1С"]["diff"].sum()
        v = gp[gp["source"] == "выяснить"]
        op = round(float(summ.loc[agent, "operation_diff"]), 2) \
            if agent in summ.index else float("nan")
        rows.append({
            "agent": agent,
            "n_rows": len(gp),
            "n_bot": int((gp["source"] == "бот").sum()),
            "n_1c": int((gp["source"] == "1С").sum()),
            "n_agent": int((gp["source"] == "агент").sum()),
            "n_vyjasnit": int((gp["source"] == "выяснить").sum()),
            "move_etm_by": round(float(-b), 2),
            "fix_1c_by": round(float(-c), 2),
            "clarify_money": round(float(v["diff"].abs().sum()), 2),
            "clarify_n": len(v),
            "operation_diff": op,
            "residual": round(float(summ.loc[agent, "residual"]), 2)
            if agent in summ.index else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["action_money"] = (out["move_etm_by"].abs()
                           + out["fix_1c_by"].abs() + out["clarify_money"]).round(2)
    return out.sort_values(["action_money", "agent"],
                           ascending=[False, True]).reset_index(drop=True)


def main():
    disc = pd.read_csv(OUT / "discrepancies.csv")
    summ = pd.read_csv(OUT / "reconciliation_summary.csv")
    out = build_actions(disc, summ)
    out.to_csv(OUT / "agent_actions.csv", index=False)
    print(f"actions: {out.shape} -> {OUT / 'agent_actions.csv'}")
    print(f"  move_etm total: {out['move_etm_by'].sum():.2f} | "
          f"fix_1c total: {out['fix_1c_by'].sum():.2f} | "
          f"clarify total: {out['clarify_money'].sum():.2f}")


if __name__ == "__main__":
    main()
