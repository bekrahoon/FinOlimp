"""Inference-safe feature builder for the Stage 2 discrepancy-source model.

Only raw operation rows, raw registry rows and Stage 2's discrepancy grouping
are consumed. No Stage 1 parser, alias/link decisions, generated comments or
precomputed ``source`` fields are used as features.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "stage1_rules" / "src"))
import io_clean as C

SOURCE_FEATURES = [
    "row_is_ticket", "row_is_payment", "abs_diff", "rel_drift",
    "one_sided_1c", "one_sided_etm", "round_1k", "round_9k",
    "round_10k", "round_36k", "pct_5", "pct_10", "amount_sign_same",
    "month", "month_edge", "n_a_rows",
    "n_etm_rows", "n_tickets", "duplicate_ticket_occurrences",
    "creator_bot_frac", "creator_sub_frac", "creator_mixed",
    "creator_none", "kind_выкуп_frac", "kind_возврат_frac",
    "kind_войд_frac", "kind_оплата_frac", "n_raw_agents",
    "raw_agent_token_jaccard", "text_token_jaccard", "doc_ref_shared",
    "n_registry_rows", "n_registry_parties", "n_registry_employees",
    "n_registry_pnrs", "n_registry_routes", "registry_has_sale",
    "registry_has_return", "registry_date_gap_min",
    "registry_party_agent_jaccard_max", "registry_parseable_frac",
    "registry_amount_sum", "registry_min_record_diff_1c",
    "registry_min_record_diff_etm", "registry_total_diff_1c",
    "registry_total_diff_etm", "registry_eq_1c_001", "registry_eq_etm_001",
    "registry_eq_1c_2", "registry_eq_etm_2",
]
SOURCE_CATEGORICAL_FEATURES = [
    "raw_agent_category", "etm_creator_category", "etm_kind_category",
    "etm_currency_category", "registry_party_category",
    "registry_employee_category", "registry_kind_category",
    "registry_airline_category", "registry_currency_category",
]
MONEY_TOLERANCE = 0.01
DOC_REF = re.compile(r"ЦБ-С\d+")
WORD = re.compile(r"[A-ZА-Я0-9]{2,}")
MONEY_SEPARATORS = [" ", "\u00a0", "'", "’"]
RATE_FIELDS = {"USD": "rate_usd", "EUR": "rate_eur", "RUB": "rate_rub",
               "RUR": "rate_rub", "KZT": "rate_kzt"}
CURRENCY_NAMES = {"KGS": "KGS", "KGС": "KGS", "КГС": "KGS",
                  "СОМ": "KGS", "СОМОВ": "KGS", "USD": "USD",
                  "ДОЛЛАР": "USD", "ДОЛЛАРОВ": "USD", "EUR": "EUR",
                  "ЕВРО": "EUR", "RUB": "RUB", "RUR": "RUB",
                  "РУБ": "RUB", "РУБЛЬ": "RUB", "РУБЛЕЙ": "RUB",
                  "KZT": "KZT", "ТЕНГЕ": "KZT"}


def _texts(rows: pd.Series | list) -> set[str]:
    return set().union(*(set(WORD.findall(str(v).upper())) for v in rows)) if len(rows) else set()


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def _category(values) -> str:
    """Stable raw categorical summary; never resolves aliases or Stage 1 links."""
    normalized = sorted({re.sub(r"\s+", " ", str(value).strip()).casefold()
                         for value in values
                         if not pd.isna(value) and str(value).strip()})
    if not normalized:
        return "__unknown__"
    if len(normalized) > 1:
        return "__multiple__"
    return normalized[0]


def _currency_category(values) -> str:
    currencies = set()
    for value in values:
        if pd.isna(value):
            continue
        for word in re.findall(r"[A-Za-zА-Яа-я]+", str(value).upper()):
            currency = CURRENCY_NAMES.get(word)
            if currency:
                currencies.add(currency)
    return "|".join(sorted(currencies)) if currencies else "__unknown__"


def _registry_amount_kgs(row: pd.Series) -> float:
    """Mechanically total raw pay_cell chunks in KGS, without party linking rules."""
    value = row.get("pay_cell", "")
    if pd.isna(value) or not str(value).strip():
        return float("nan")
    text = re.sub(r"\([^()]*[A-Za-zА-Яа-я][^()]*\)", "", str(value))
    chunks = [chunk.strip() for chunk in text.split("+") if chunk.strip()]
    total = 0.0
    for chunk in chunks:
        words = [word.upper() for word in re.findall(r"[A-Za-zА-Яа-я]+", chunk)
                 if word.lower() != "sf"]
        currencies = {CURRENCY_NAMES[word] for word in words if word in CURRENCY_NAMES}
        if len(currencies) != 1 or any(word not in CURRENCY_NAMES for word in words):
            return float("nan")
        currency = next(iter(currencies))
        amount = C.parse_money_value(chunk, MONEY_SEPARATORS)
        if pd.isna(amount):
            return float("nan")
        if currency == "KGS":
            rate = 1.0
        else:
            rate = row.get(RATE_FIELDS[currency], float("nan"))
            if pd.isna(rate):
                return float("nan")
            rate = float(rate)
        total += float(amount) * rate
    return round(total, 2)


def prepare_registry(registry: pd.DataFrame | None) -> pd.DataFrame:
    if registry is None:
        return pd.DataFrame(columns=["date_norm", "ticket_set", "party", "employee",
                                     "pnr", "route", "kind"])
    out = registry.copy()
    for col in ("party", "employee", "pnr", "route", "kind", "tickets",
                "airline", "pay_cell"):
        if col not in out:
            out[col] = ""
    out["date_norm"] = _parse_dates(out.get("date", pd.Series(index=out.index)))
    out["ticket_set"] = out["tickets"].map(lambda v: C.ticket_numbers(v, None))
    if "pay_cell" in out:
        out["amount_kgs_raw"] = out.apply(_registry_amount_kgs, axis=1)
    else:
        out["amount_kgs_raw"] = float("nan")
    return out


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.notna()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y %H:%M:%S",
                "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"):
        if not remaining.any():
            break
        attempt = pd.to_datetime(series[remaining], format=fmt, errors="coerce")
        hit = attempt.notna()
        parsed.loc[attempt[hit].index] = attempt[hit]
        remaining &= parsed.isna()
    if remaining.any():
        attempt = pd.to_datetime(series[remaining], errors="coerce", dayfirst=True)
        parsed.loc[attempt[attempt.notna()].index] = attempt[attempt.notna()]
    return parsed


def build_source_features(discrepancies: pd.DataFrame, acts: pd.DataFrame,
                          etm: pd.DataFrame,
                          registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one raw-safe feature vector per Stage 2 discrepancy component."""
    reg = prepare_registry(registry)
    reg_by_tail: dict[str, set[int]] = defaultdict(set)
    for idx, row in reg.iterrows():
        for tail in row["ticket_set"]:
            reg_by_tail[tail].add(int(idx))
    output = []
    for _, discrepancy in discrepancies.iterrows():
        a_ids = json.loads(discrepancy.get("a_indices", "[]") or "[]")
        e_ids = json.loads(discrepancy.get("e_indices", "[]") or "[]")
        arows = acts.loc[a_ids] if a_ids else acts.iloc[0:0]
        erows = etm.loc[e_ids] if e_ids else etm.iloc[0:0]
        sum_a, sum_e = float(discrepancy["sum_1c"]), float(discrepancy["sum_etm"])
        diff = float(discrepancy["diff"])
        abs_diff = abs(diff)
        denom = max(abs(sum_a), abs(sum_e), 1.0)
        date = pd.to_datetime(discrepancy.get("date", ""), errors="coerce")
        tails = set()
        for value in arows.get("ticket_cell", pd.Series(dtype=object)):
            tails.update(C.ticket_numbers(value, None))
        for value in erows.get("tickets", pd.Series(dtype=object)):
            tails.update(C.ticket_numbers(value, None))
        if not tails:
            tails.update(t for t in str(discrepancy.get("identifier", "")).split("|") if t)

        a_agent_names = set(arows.get("folder", pd.Series(dtype=object)).dropna().astype(str))
        e_agent_names = set(erows.get("agent", pd.Series(dtype=object)).dropna().astype(str))
        agent_tokens = _texts(list(a_agent_names))
        agent_tokens_e = _texts(list(e_agent_names))
        agent_overlap = _jaccard(agent_tokens, agent_tokens_e)
        a_texts = arows.get("doc", pd.Series(dtype=object)).fillna("").astype(str)
        e_texts = erows.get("comment", pd.Series(dtype=object)).fillna("").astype(str)
        a_tokens, e_tokens = _texts(a_texts), _texts(e_texts)
        doc_refs_a = set(DOC_REF.findall(" ".join(a_texts.tolist())))
        doc_refs_e = set(DOC_REF.findall(" ".join(e_texts.tolist())))

        reg_ids = set().union(*(reg_by_tail.get(t, set()) for t in tails)) if tails else set()
        reg_rows = reg.loc[sorted(reg_ids)] if reg_ids else reg.iloc[0:0]
        # Use only registry records available on/before this discrepancy date.
        # A future registry row must not reveal a validation target's resolution.
        if pd.notna(date) and len(reg_rows):
            reg_rows = reg_rows[
                reg_rows["date_norm"].notna() & (reg_rows["date_norm"] <= date)]
        else:
            reg_rows = reg.iloc[0:0]
        registry_pay_cells = reg_rows.get("pay_cell", pd.Series(dtype=object))
        reg_amounts = (reg_rows["amount_kgs_raw"].dropna().astype(float)
                       if len(reg_rows) and "amount_kgs_raw" in reg_rows else pd.Series(dtype=float))
        reg_total = float(reg_amounts.sum()) if len(reg_amounts) else -1.0
        abs_a, abs_e = abs(sum_a), abs(sum_e)
        reg_dates = reg_rows["date_norm"].dropna() if len(reg_rows) else pd.Series(dtype="datetime64[ns]")
        gap_days = (reg_dates - date).abs().dt.total_seconds() / 86400.0 \
            if pd.notna(date) and len(reg_dates) else pd.Series(dtype=float)
        party_sims = [_jaccard(_texts([party]), agent_tokens | agent_tokens_e)
                      for party in reg_rows.get("party", pd.Series(dtype=object)).fillna("")]
        kinds = erows.get("kind", pd.Series(dtype=object)).fillna("").astype(str).str.lower()
        creators = (erows.get("creator", pd.Series(dtype=object)).fillna("")
                    .astype(str).str.lower())
        creator_counts = {name: int(creators.eq(name).sum())
                          for name in ("etm-bot", "субагент")}
        n_etm = len(erows)
        n_a = len(arows)
        ticket_occurrences = sum(len(C.ticket_numbers(v, None))
                                 for v in arows.get("ticket_cell", pd.Series(dtype=object))) \
            + sum(len(C.ticket_numbers(v, None))
                  for v in erows.get("tickets", pd.Series(dtype=object)))
        month_days = date.days_in_month if pd.notna(date) else 0
        kind_lower = kinds
        row_type = str(discrepancy.get("row_type", ""))
        output.append({
            "row_is_ticket": int(row_type == "ticket"),
            "row_is_payment": int(row_type == "payment"),
            "abs_diff": abs_diff,
            "rel_drift": abs_diff / denom,
            "one_sided_1c": int(abs(sum_e) < MONEY_TOLERANCE),
            "one_sided_etm": int(abs(sum_a) < MONEY_TOLERANCE),
            "round_1k": int(abs(abs_diff - 1000) < MONEY_TOLERANCE),
            "round_9k": int(abs(abs_diff - 9000) < MONEY_TOLERANCE),
            "round_10k": int(abs(abs_diff - 10000) < MONEY_TOLERANCE),
            "round_36k": int(abs(abs_diff - 36000) < MONEY_TOLERANCE),
            "pct_5": int(abs(abs_diff / max(abs(sum_a), MONEY_TOLERANCE) - 0.05) < 0.001),
            "pct_10": int(abs(abs_diff / max(abs(sum_a), MONEY_TOLERANCE) - 0.10) < 0.001),
            "amount_sign_same": int(np.sign(sum_a) == np.sign(sum_e)),
            "month": int(date.month) if pd.notna(date) else 0,
            "month_edge": int(pd.notna(date) and min(date.day, month_days - date.day + 1) <= 3),
            "n_a_rows": n_a,
            "n_etm_rows": n_etm,
            "n_tickets": len(tails),
            "duplicate_ticket_occurrences": ticket_occurrences - len(tails),
            "creator_bot_frac": creator_counts["etm-bot"] / max(n_etm, 1),
            "creator_sub_frac": creator_counts["субагент"] / max(n_etm, 1),
            "creator_mixed": int(len(set(creators)) > 1),
            "creator_none": int(not len(creators)),
            "kind_выкуп_frac": float(kind_lower.eq("выкуп").sum()) / max(n_etm, 1),
            "kind_возврат_frac": float(kind_lower.eq("возврат").sum()) / max(n_etm, 1),
            "kind_войд_frac": float(kind_lower.eq("войд").sum()) / max(n_etm, 1),
            "kind_оплата_frac": float(kind_lower.eq("оплата").sum()) / max(n_etm, 1),
            "n_raw_agents": len(a_agent_names | e_agent_names),
            "raw_agent_token_jaccard": agent_overlap,
            "text_token_jaccard": _jaccard(a_tokens, e_tokens),
            "doc_ref_shared": int(bool(doc_refs_a & doc_refs_e)),
            "n_registry_rows": len(reg_rows),
            "n_registry_parties": int(reg_rows.get("party", pd.Series(dtype=object)).nunique()),
            "n_registry_employees": int(reg_rows.get("employee", pd.Series(dtype=object)).nunique()),
            "n_registry_pnrs": int(reg_rows.get("pnr", pd.Series(dtype=object)).nunique()),
            "n_registry_routes": int(reg_rows.get("route", pd.Series(dtype=object)).nunique()),
            "registry_has_sale": int(reg_rows.get("kind", pd.Series(dtype=object))
                                      .astype(str).str.lower().eq("продажа").any()),
            "registry_has_return": int(reg_rows.get("kind", pd.Series(dtype=object))
                                        .astype(str).str.lower().eq("возврат").any()),
            "registry_date_gap_min": float(gap_days.min()) if len(gap_days) else -1.0,
            "registry_party_agent_jaccard_max": max(party_sims, default=0.0),
            "registry_parseable_frac": len(reg_amounts) / max(len(reg_rows), 1),
            "registry_amount_sum": reg_total,
            "registry_min_record_diff_1c": float((reg_amounts - abs_a).abs().min())
            if len(reg_amounts) else -1.0,
            "registry_min_record_diff_etm": float((reg_amounts - abs_e).abs().min())
            if len(reg_amounts) else -1.0,
            "registry_total_diff_1c": abs(reg_total - abs_a) if len(reg_amounts) else -1.0,
            "registry_total_diff_etm": abs(reg_total - abs_e) if len(reg_amounts) else -1.0,
            "registry_eq_1c_001": int(len(reg_amounts) and abs(reg_total - abs_a) < 0.01),
            "registry_eq_etm_001": int(len(reg_amounts) and abs(reg_total - abs_e) < 0.01),
            "registry_eq_1c_2": int(len(reg_amounts) and abs(reg_total - abs_a) < 2.0),
            "registry_eq_etm_2": int(len(reg_amounts) and abs(reg_total - abs_e) < 2.0),
            "raw_agent_category": _category(list(a_agent_names | e_agent_names)),
            "etm_creator_category": _category(erows.get(
                "creator", pd.Series(dtype=object))),
            "etm_kind_category": _category(erows.get(
                "kind", pd.Series(dtype=object))),
            "etm_currency_category": _category(erows.get(
                "currency", pd.Series(dtype=object))),
            "registry_party_category": _category(reg_rows.get(
                "party", pd.Series(dtype=object))),
            "registry_employee_category": _category(reg_rows.get(
                "employee", pd.Series(dtype=object))),
            "registry_kind_category": _category(reg_rows.get(
                "kind", pd.Series(dtype=object))),
            "registry_airline_category": _category(reg_rows.get(
                "airline", pd.Series(dtype=object))),
            "registry_currency_category": _currency_category(registry_pay_cells),
        })
    return pd.DataFrame(
        output, columns=SOURCE_FEATURES + SOURCE_CATEGORICAL_FEATURES
    ).replace([np.inf, -np.inf], np.nan).fillna(-1)
