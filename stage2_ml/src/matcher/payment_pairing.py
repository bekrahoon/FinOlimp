"""Payment pairing helper for run-local silver labels and fidelity audits."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FINOLIMP = HERE.parents[2]
sys.path.insert(0, str(FINOLIMP / "stage1_rules" / "src"))

import parse as R  # noqa: E402 (offline Stage 1 target generation)


def match_payments_with_pairs(acts: pd.DataFrame, etm: pd.DataFrame,
                              cfg: dict) -> list[tuple[int, int, str]]:
    """Apply Stage 1's deterministic payment matching and return chosen row pairs."""
    pcfg = cfg["payments"]
    window = pd.Timedelta(days=pcfg["date_window_days"])
    pairs = []
    agents = sorted(set(acts["agent_norm"]) | set(etm["agent_norm"]))
    for agent in agents:
        a = acts[(acts["agent_norm"] == agent) & acts["ticket_cell"].isna()].copy()
        e = etm[(etm["agent_norm"] == agent) & etm["tickets"].isna()].copy()
        a["_ref"] = a["doc"].map(lambda value: R.doc_ref(value, cfg))
        e["_ref"] = e["comment"].map(lambda value: R.doc_ref(value, cfg)) \
            if "comment" in e else ""
        free = set(e.index)
        a["_matched"] = False
        if pcfg["prefer_doc_match"]:
            by_ref = defaultdict(list)
            for idx in sorted(free):
                if e.loc[idx, "_ref"]:
                    key = (round(float(e.loc[idx, "sum_etm"]), 2), e.loc[idx, "_ref"])
                    by_ref[key].append(idx)
            for act_idx, act in a.iterrows():
                if act["_ref"]:
                    key = (round(float(act["sum_1c"]), 2), act["_ref"])
                    pool = [idx for idx in by_ref.get(key, []) if idx in free]
                    if pool:
                        chosen = min(pool, key=lambda idx: (
                            abs((e.loc[idx, "date_norm"] - act["date_norm"]).total_seconds()),
                            idx,
                        ))
                        free.remove(chosen)
                        a.loc[act_idx, "_matched"] = True
                        pairs.append((act_idx, chosen, "pay_doc"))
        for act_idx, act in a.loc[~a["_matched"]].sort_values("date_norm").iterrows():
            if pd.isna(act["date_norm"]):
                continue
            candidates = {
                idx for idx in free
                if abs(float(e.loc[idx, "sum_etm"]) - float(act["sum_1c"])) < cfg["tolerance"]
                and not pd.isna(e.loc[idx, "date_norm"])
                and abs(e.loc[idx, "date_norm"] - act["date_norm"]) <= window
            }
            if candidates:
                chosen = min(candidates, key=lambda idx: (
                    abs((e.loc[idx, "date_norm"] - act["date_norm"]).total_seconds()), idx,
                ))
                free.remove(chosen)
                pairs.append((act_idx, chosen, "pay_window"))
    return pairs
