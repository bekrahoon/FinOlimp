"""Additive review candidates from Final; these never change source labels.

The current discrepancy engine remains the accounting source of truth. This
module emits a separate audit queue for three contexts that require judgement:
missing-act balance blocks, registry voids without an ETM void, and likely FX
date shifts. Rows are candidates, not automatically attributed errors.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from parse import build_rate_table, rate_for, ticket_components


REVIEW_COLUMNS = [
    "issue", "agent", "date", "identifier", "amount_delta", "evidence", "source_rows",
]
FX_MAX_LAG_DAYS = 3
FX_RATIO_TOLERANCE = 0.0015


def _label(agent: str, labels: dict[str, str]) -> str:
    return labels.get(agent, agent)


def _missing_act_blocks(acts: pd.DataFrame, etm: pd.DataFrame,
                        labels: dict[str, str], tolerance: float) -> list[dict]:
    if acts.empty or "period_start_norm" not in acts or acts["period_start_norm"].isna().all():
        return []
    heads = (acts.dropna(subset=["period_start_norm"])
             .groupby(["agent_norm", "period_start_norm"], as_index=False)
             .agg(saldo_start=("saldo_start", "first"), saldo_end=("saldo_end", "first"))
             .sort_values(["agent_norm", "period_start_norm"]))
    rows = []
    for agent, group in heads.groupby("agent_norm", sort=True):
        group = group.reset_index(drop=True)
        for index in range(1, len(group)):
            previous, current = group.loc[index - 1], group.loc[index]
            previous_period = previous["period_start_norm"].to_period("M")
            current_period = current["period_start_norm"].to_period("M")
            gap = current_period.ordinal - previous_period.ordinal - 1
            if gap <= 0:
                continue
            missing = [previous_period + offset for offset in range(1, gap + 1)]
            et = etm[(etm["agent_norm"] == agent) & etm["date_norm"].notna()]
            et = et[et["date_norm"].dt.to_period("M").isin(missing)]
            observed = round(float(current["saldo_start"] - previous["saldo_end"]), 2)
            expected = round(float(et["sum_etm"].fillna(0).sum()), 2) if len(et) else 0.0
            unexplained = round(observed - expected, 2)
            if not len(et) and abs(observed) <= tolerance:
                continue
            month_text = ",".join(str(month) for month in missing)
            state = ("ETM net matches the change in 1C saldo" if abs(unexplained) <= tolerance
                     else "1C saldo change is not fully explained by ETM; request the missing act")
            txns = ",".join(et["txn_id"].astype(str).tolist()) if len(et) else ""
            rows.append({
                "issue": "missing_act_block",
                "agent": _label(agent, labels),
                "date": current["period_start_norm"].date().isoformat(),
                "identifier": month_text,
                "amount_delta": unexplained,
                "evidence": (f"1C saldo changed by {observed:.2f}; ETM net for the block is "
                             f"{expected:.2f} across {len(et)} rows; {state}."),
                "source_rows": f"acts:{previous_period}/{current_period};etm:{txns}",
            })
    return rows


def _registry_void_candidates(etm: pd.DataFrame, registry: pd.DataFrame,
                              labels: dict[str, str]) -> list[dict]:
    if etm.empty or registry.empty:
        return []
    void_kinds = {"войд", "void"}
    sale_kinds = {"выкуп", "sale"}
    sales: dict[tuple[str, str], list[str]] = defaultdict(list)
    etm_voids: set[tuple[str, str]] = set()
    for _, row in etm.iterrows():
        agent = str(row.get("agent_norm", ""))
        kind = str(row.get("kind", "")).strip().lower()
        tails = row.get("ticket_set", set())
        if not isinstance(tails, (set, frozenset)):
            continue
        for tail in tails:
            key = (agent, str(tail))
            if kind in sale_kinds:
                sales[key].append(str(row.get("txn_id", "")))
            elif kind in void_kinds:
                etm_voids.add(key)

    rows = []
    for reg_id, row in registry.iterrows():
        if str(row.get("kind", "")).strip().lower() not in void_kinds:
            continue
        agent = str(row.get("agent_norm", ""))
        tails = row.get("ticket_set", set())
        if not isinstance(tails, (set, frozenset)):
            continue
        missed = [str(tail) for tail in sorted(tails)
                  if (agent, str(tail)) in sales and (agent, str(tail)) not in etm_voids]
        if not missed:
            continue
        related_sales = sorted({txn for tail in missed for txn in sales[(agent, tail)]})
        rows.append({
            "issue": "registry_void_without_etm_void",
            "agent": _label(agent, labels),
            "date": (row["date_norm"].date().isoformat()
                     if pd.notna(row.get("date_norm")) else ""),
            "identifier": "|".join(missed),
            "amount_delta": np.nan,
            "evidence": (f"Registry void has no ETM void for these ticket tails; related ETM "
                         f"sale txn IDs: {', '.join(related_sales)}. Check cancellation and refund evidence."),
            "source_rows": f"registry:{reg_id + 2};etm:{','.join(related_sales)}",
        })
    return rows


def _fx_candidates(acts: pd.DataFrame, etm: pd.DataFrame, registry: pd.DataFrame,
                   cfg: dict, labels: dict[str, str], tolerance: float) -> list[dict]:
    if acts.empty or etm.empty or registry.empty:
        return []
    rates = build_rate_table(registry, cfg)
    fallback_rates = cfg.get("currencies", {}).get("fallback_rates", {})
    rows = []
    for agent in sorted(set(acts["agent_norm"]) & set(etm["agent_norm"])):
        a = acts[(acts["agent_norm"] == agent) & acts["ticket_cell"].notna()]
        e = etm[(etm["agent_norm"] == agent) & etm["tickets"].notna()]
        for act_ids, etm_ids in ticket_components(a, e):
            if not act_ids or not etm_ids:
                continue
            arows, erows = acts.loc[act_ids], etm.loc[etm_ids]
            act_docs = arows["doc"].fillna("").astype(str).str.strip().str.lower()
            etm_kinds = erows["kind"].fillna("").astype(str).str.strip().str.lower()
            # The normalized input has no operation-code column: classify only
            # explicit 1C realizations against ETM buyouts; returns, service-fee
            # documents, and other ticket-linked operations are not FX candidates.
            if (not act_docs.str.startswith("реализация").all()
                    or act_docs.str.contains("сервисный сбор").any()
                    or set(etm_kinds) != {"выкуп"}):
                continue
            a_dates = arows["date_norm"].dropna().dt.normalize().unique()
            e_dates = erows["date_norm"].dropna().dt.normalize().unique()
            if len(a_dates) != 1 or len(e_dates) != 1:
                continue
            act_date, etm_date = pd.Timestamp(a_dates[0]), pd.Timestamp(e_dates[0])
            lag = (etm_date - act_date).days
            if not 1 <= abs(lag) <= FX_MAX_LAG_DAYS:
                continue
            currencies = (erows.get("currency", pd.Series("KGS", index=erows.index))
                          .fillna("KGS").astype(str).str.strip().str.upper().unique())
            if len(currencies) != 1 or currencies[0] == "KGS":
                continue
            currency = currencies[0]
            rate_1c = rate_for(rates, fallback_rates, currency, act_date)
            rate_etm = rate_for(rates, fallback_rates, currency, etm_date)
            if not rate_1c or not rate_etm:
                continue
            sum_1c = float(arows["sum_1c"].sum())
            sum_etm = float(erows["sum_etm"].sum())
            if abs(sum_1c - sum_etm) <= tolerance or abs(sum_etm) <= tolerance:
                continue
            observed_ratio = abs(sum_1c / sum_etm)
            expected_ratio = float(rate_1c) / float(rate_etm)
            if abs(observed_ratio - expected_ratio) > FX_RATIO_TOLERANCE:
                continue
            tickets = sorted(set().union(*arows["ticket_set"].tolist(),
                                         *erows["ticket_set"].tolist()))
            direction = "after" if lag > 0 else "before"
            day_word = "day" if abs(lag) == 1 else "days"
            rows.append({
                "issue": "possible_fx_difference",
                "agent": _label(agent, labels),
                "date": etm_date.date().isoformat(),
                "identifier": "|".join(tickets),
                "amount_delta": round(sum_1c - sum_etm, 2),
                "evidence": (f"ETM is {abs(lag)} {day_word} {direction} 1C; currency {currency}, "
                             f"observed |1C/ETM|={observed_ratio:.6f}, "
                             f"rate ratio={expected_ratio:.6f}. Candidate only; verify invoice dates."),
                "source_rows": (f"acts:{','.join(str(i + 2) for i in act_ids)};"
                                f"etm:{','.join(erows['txn_id'].astype(str).tolist())}"),
            })
    return rows


def build_review_candidates(acts: pd.DataFrame, etm: pd.DataFrame, registry: pd.DataFrame,
                            cfg: dict, labels: dict[str, str]) -> pd.DataFrame:
    """Build non-authoritative review candidates without changing discrepancies."""
    candidates = (
        _missing_act_blocks(acts, etm, labels, float(cfg["tolerance"]))
        + _registry_void_candidates(etm, registry, labels)
        + _fx_candidates(acts, etm, registry, cfg, labels, float(cfg["tolerance"]))
    )
    if not candidates:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return (pd.DataFrame(candidates, columns=REVIEW_COLUMNS)
            .sort_values(["issue", "agent", "date", "identifier"], kind="mergesort")
            .reset_index(drop=True))
